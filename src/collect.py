import argparse
import calendar
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.github_client import GitHub, hash_id, load_env

CONFIG_EXT = {".xml", ".yml", ".yaml", ".json", ".properties", ".toml", ".gradle", ".conf", ".ini", ".j2", ".sh"}
DOC_EXT = {".md", ".txt", ".rst", ".adoc"}
CODE_EXT = {".java", ".bal", ".ts", ".tsx", ".js", ".jsx", ".go", ".py", ".kt", ".scss", ".css", ".html", ".sql"}
DEP_FILES = {"pom.xml", "build.gradle", "package.json", "pnpm-lock.yaml", "package-lock.json",
             "yarn.lock", "gradle.properties", "ballerina.toml", "dependencies.toml"}


def month_windows(months_back):
    today = date.today()
    y, m = today.year, today.month
    for _ in range(months_back):
        last = calendar.monthrange(y, m)[1]
        yield f"{y:04d}-{m:02d}-01..{y:04d}-{m:02d}-{last:02d}"
        m -= 1
        if m == 0:
            y, m = y - 1, 12


def list_runs(gh, repo, workflow, months_back, cap, conclusions):
    runs = []
    for window in month_windows(months_back):
        page = 1
        while True:
            data = gh.get(f"/repos/{repo}/actions/workflows/{workflow}/runs",
                          {"created": window, "per_page": 100, "page": page},
                          use_cache=not window.startswith(date.today().strftime("%Y-%m")))
            batch = (data or {}).get("workflow_runs", [])
            runs += [r for r in batch if r.get("status") == "completed" and r.get("conclusion") in conclusions]
            if len(batch) < 100:
                break
            page += 1
        print(f"  {repo} {window}: {len(runs)} runs so far", flush=True)
        if len(runs) >= cap:
            break
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs[:cap]


def file_kind(name):
    low = name.lower()
    base = low.rsplit("/", 1)[-1]
    ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
    if base in DEP_FILES:
        return "dep"
    if "/test/" in low or "/tests/" in low or "test" in base or "/__tests__/" in low or ".spec." in base:
        return "test"
    if ".github/" in low:
        return "ci"
    if ext in DOC_EXT:
        return "doc"
    if ext in CONFIG_EXT:
        return "config"
    if ext in CODE_EXT:
        return "code"
    return "other"


def commit_features(gh, repo, sha):
    c = gh.get(f"/repos/{repo}/commits/{sha}")
    if not c:
        return {"commit_found": 0}
    files = c.get("files") or []
    kinds = {k: 0 for k in ("code", "test", "config", "doc", "dep", "ci", "other")}
    dirs = set()
    for f in files:
        kinds[file_kind(f["filename"])] += 1
        dirs.add(f["filename"].split("/")[0])
    stats = c.get("stats") or {}
    msg = (c.get("commit") or {}).get("message") or ""
    return {
        "commit_found": 1,
        "additions": stats.get("additions", 0),
        "deletions": stats.get("deletions", 0),
        "files_changed": len(files),
        "top_dirs_touched": len(dirs),
        "files_added": sum(f.get("status") == "added" for f in files),
        "files_removed": sum(f.get("status") == "removed" for f in files),
        **{f"n_{k}_files": v for k, v in kinds.items()},
        "is_merge_commit": int(len(c.get("parents") or []) > 1),
        "commit_msg_len": len(msg),
        "commit_msg_lines": msg.count("\n") + 1,
        "commit_time": (c.get("commit") or {}).get("committer", {}).get("date"),
    }


def pr_features(gh, repo, sha):
    prs = gh.get(f"/repos/{repo}/commits/{sha}/pulls") or []
    if not prs:
        return {"pr_number": None}
    pr = prs[0]
    return {
        "pr_number": pr.get("number"),
        "pr_author_association": pr.get("author_association"),
        "pr_created_at": pr.get("created_at"),
        "pr_base_ref": (pr.get("base") or {}).get("ref"),
        "pr_is_draft": int(bool(pr.get("draft"))),
    }


def run_row(run, repo, salt):
    actor = (run.get("actor") or {}).get("login", "")
    trig = (run.get("triggering_actor") or {}).get("login", "")
    head_repo = (run.get("head_repository") or {}).get("full_name", "")
    return {
        "repo": repo,
        "workflow": run.get("name"),
        "run_id": run["id"],
        "run_number": run.get("run_number"),
        "run_attempt": run.get("run_attempt"),
        "event": run.get("event"),
        "head_branch_hash": hash_id(run.get("head_branch") or "", salt),
        "head_sha": run.get("head_sha"),
        "is_fork": int(bool(head_repo) and head_repo.lower() != repo.lower()),
        "actor_hash": hash_id(actor, salt),
        "triggering_actor_hash": hash_id(trig, salt),
        "is_bot": int(actor.endswith("[bot]") or "bot" in actor.lower()),
        "title": run.get("display_title"),
        "created_at": run.get("created_at"),
        "run_started_at": run.get("run_started_at"),
        "updated_at": run.get("updated_at"),
        "conclusion": run.get("conclusion"),
    }


def collect_repo(gh, item, cfg, salt, out_dir):
    repo, workflow = item["repo"], item["workflow"]
    print(f"[{repo}] listing runs", flush=True)
    runs = list_runs(gh, repo, workflow, cfg["months_back"], cfg["max_runs_per_repo"], set(cfg["conclusions"]))
    rows = []
    for i, run in enumerate(runs, 1):
        row = run_row(run, repo, salt)
        row.update(commit_features(gh, repo, run["head_sha"]))
        if run.get("event") in ("pull_request", "pull_request_target"):
            row.update(pr_features(gh, repo, run["head_sha"]))
        rows.append(row)
        if i % 100 == 0:
            print(f"  [{repo}] {i}/{len(runs)} enriched, api calls={gh.calls}", flush=True)
    df = pd.DataFrame(rows)
    out = out_dir / (repo.replace("/", "__") + ".csv")
    df.to_csv(out, index=False)
    print(f"[{repo}] saved {len(df)} rows -> {out}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--repo", help="collect only this repo")
    ap.add_argument("--limit", type=int, help="override max runs per repo")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.params).read_text())["collect"]
    if args.limit:
        cfg["max_runs_per_repo"] = args.limit
    env = load_env()
    salt = env.get("HASH_SALT") or "ci-failure-prediction"
    gh = GitHub(env.get("GITHUB_TOKEN"))

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [r for r in cfg["repos"] if not args.repo or r["repo"] == args.repo]

    frames = [collect_repo(gh, item, cfg, salt, out_dir) for item in items]
    if not args.repo:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(out_dir / "runs.csv", index=False)
        print(f"total rows: {len(all_df)}, failure rate: {(all_df.conclusion == 'failure').mean():.3f}")
    print(f"finished at {datetime.now(timezone.utc).isoformat()} api calls={gh.calls}")


if __name__ == "__main__":
    main()
