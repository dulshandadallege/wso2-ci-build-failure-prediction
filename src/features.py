import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "failed"

NUMERIC = [
    "additions", "deletions", "log_churn", "files_changed", "top_dirs_touched",
    "files_added", "files_removed", "n_code_files", "n_test_files", "n_config_files",
    "n_doc_files", "n_dep_files", "n_ci_files", "test_file_ratio", "commit_msg_len",
    "title_len", "hour", "weekday", "commit_age_hours", "pr_age_hours",
    "repo_fail_rate_20", "repo_recent_fail_streak", "author_prior_runs", "author_fail_rate",
    "pr_prior_runs", "pr_prior_failures", "minutes_since_prev_pr_run", "sha_prior_runs",
]
BINARY = [
    "is_fork", "is_bot", "is_rerun", "is_merge_commit", "is_weekend", "docs_only",
    "touches_deps", "touches_ci", "title_dependency", "title_fix", "title_revert",
    "title_test", "repo_prev_failed", "pr_prev_failed", "sha_prev_failed", "pr_first_run",
]
CATEGORICAL = ["repo", "event", "author_association"]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def load_raw(raw_dir="data/raw"):
    files = sorted(p for p in Path(raw_dir).glob("*.csv") if p.name != "runs.csv")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    return df.drop_duplicates("run_id")


def _title_flag(titles, pattern):
    return titles.str.contains(pattern, flags=re.I, regex=True).astype(int)


def add_static_features(df):
    df = df.copy()
    for col in ("created_at", "updated_at", "commit_time", "pr_created_at"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df[TARGET] = (df["conclusion"] == "failure").astype(int)

    count_cols = ["additions", "deletions", "files_changed", "top_dirs_touched", "files_added",
                  "files_removed", "n_code_files", "n_test_files", "n_config_files", "n_doc_files",
                  "n_dep_files", "n_ci_files", "n_other_files", "commit_msg_len"]
    df[count_cols] = df[count_cols].fillna(0)

    df["log_churn"] = np.log1p(df["additions"] + df["deletions"])
    df["test_file_ratio"] = df["n_test_files"] / df["files_changed"].clip(lower=1)
    df["docs_only"] = ((df["files_changed"] > 0) & (df["n_doc_files"] == df["files_changed"])).astype(int)
    df["touches_deps"] = (df["n_dep_files"] > 0).astype(int)
    df["touches_ci"] = (df["n_ci_files"] > 0).astype(int)
    df["is_rerun"] = (df["run_attempt"] > 1).astype(int)

    local = df["created_at"].dt.tz_convert("Asia/Colombo")
    df["hour"] = local.dt.hour
    df["weekday"] = local.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["commit_age_hours"] = ((df["created_at"] - df["commit_time"]).dt.total_seconds() / 3600).clip(lower=0)
    if "pr_created_at" in df:
        df["pr_age_hours"] = ((df["created_at"] - df["pr_created_at"]).dt.total_seconds() / 3600).clip(lower=0)
    else:
        df["pr_age_hours"] = np.nan

    titles = df["title"].fillna("")
    df["title_len"] = titles.str.len()
    df["title_dependency"] = _title_flag(titles, r"\b(?:bump|upgrade|update .*version|dependenc|deps?)\b")
    df["title_fix"] = _title_flag(titles, r"\b(?:fix|bug|issue|resolve)")
    df["title_revert"] = _title_flag(titles, r"\brevert")
    df["title_test"] = _title_flag(titles, r"\btest")

    assoc = df.get("pr_author_association")
    df["author_association"] = assoc.fillna("UNKNOWN") if assoc is not None else "UNKNOWN"
    if "pr_key" not in df:
        df["pr_key"] = df["head_branch_hash"]
    return df


def add_history_features(df, window=20, prior_weight=5):
    """History is built only from runs that had already finished when the current run was created."""
    df = df.sort_values("created_at").reset_index(drop=True)
    out = {k: np.full(len(df), np.nan) for k in (
        "repo_fail_rate_20", "repo_recent_fail_streak", "repo_prev_failed", "author_prior_runs",
        "author_fail_rate", "pr_prior_runs", "pr_prior_failures", "pr_prev_failed",
        "minutes_since_prev_pr_run", "sha_prior_runs", "sha_prev_failed")}

    for repo, idx in df.groupby("repo").groups.items():
        part = df.loc[idx].sort_values("created_at")
        finished = part.sort_values("updated_at")
        fin_iter = iter(finished.itertuples())
        pending = next(fin_iter, None)

        recent = deque(maxlen=window)
        streak = 0
        author = defaultdict(lambda: [0, 0])
        pr = defaultdict(lambda: [0, 0, None])
        sha = defaultdict(lambda: [0, None])
        pr_started = {}
        overall = [0, 0]

        for row in part.itertuples():
            while pending is not None and pending.updated_at < row.created_at:
                f = pending.failed
                recent.append(f)
                streak = streak + 1 if f else 0
                overall[0] += 1
                overall[1] += f
                a = author[pending.actor_hash]
                a[0] += 1
                a[1] += f
                p = pr[pending.pr_key]
                p[0] += 1
                p[1] += f
                p[2] = f
                s = sha[pending.head_sha]
                s[0] += 1
                s[1] = f
                pending = next(fin_iter, None)

            i = row.Index
            base = overall[1] / overall[0] if overall[0] else 0.25
            out["repo_fail_rate_20"][i] = np.mean(recent) if recent else base
            out["repo_recent_fail_streak"][i] = streak
            out["repo_prev_failed"][i] = recent[-1] if recent else 0
            n, k = author[row.actor_hash]
            out["author_prior_runs"][i] = n
            out["author_fail_rate"][i] = (k + prior_weight * base) / (n + prior_weight)
            pn, pk, plast = pr[row.pr_key]
            out["pr_prior_runs"][i] = pn
            out["pr_prior_failures"][i] = pk
            out["pr_prev_failed"][i] = plast if plast is not None else 0
            prev_start = pr_started.get(row.pr_key)
            out["minutes_since_prev_pr_run"][i] = (
                (row.created_at - prev_start).total_seconds() / 60 if prev_start is not None else -1)
            pr_started[row.pr_key] = row.created_at
            sn, slast = sha[row.head_sha]
            out["sha_prior_runs"][i] = sn
            out["sha_prev_failed"][i] = slast if slast is not None else 0

    for k, v in out.items():
        df[k] = v
    df["pr_first_run"] = (df["pr_prior_runs"] == 0).astype(int)
    return df


def build_dataset(raw_dir="data/raw"):
    df = add_static_features(load_raw(raw_dir))
    df = add_history_features(df)
    return df


def time_split(df, train=0.7, val=0.15):
    df = df.sort_values("created_at").reset_index(drop=True)
    a, b = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


if __name__ == "__main__":
    data = build_dataset()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_csv("data/processed/features.csv", index=False)
    print(data.shape, data[TARGET].mean().round(3))
