import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, add_history_features, add_static_features, time_split


def make_runs(rows):
    base = {
        "repo": "org/repo", "workflow": "PR Build", "run_attempt": 1, "event": "pull_request",
        "head_branch_hash": "b", "is_fork": 1, "is_bot": 0, "title": "Fix login", "additions": 10,
        "deletions": 2, "files_changed": 2, "top_dirs_touched": 1, "files_added": 0, "files_removed": 0,
        "n_code_files": 1, "n_test_files": 1, "n_config_files": 0, "n_doc_files": 0, "n_dep_files": 0,
        "n_ci_files": 0, "n_other_files": 0, "is_merge_commit": 0, "commit_msg_len": 30,
        "pr_author_association": "CONTRIBUTOR",
    }
    out = []
    for i, r in enumerate(rows):
        row = {**base, "run_id": i, "head_sha": f"sha{i}", **r}
        row.setdefault("commit_time", row["created_at"])
        row.setdefault("pr_created_at", row["created_at"])
        out.append(row)
    return pd.DataFrame(out)


def test_static_features():
    df = add_static_features(make_runs([
        {"created_at": "2026-01-05T04:00:00Z", "updated_at": "2026-01-05T04:30:00Z", "conclusion": "failure",
         "actor_hash": "a", "pr_key": "p1", "n_doc_files": 2, "files_changed": 2, "n_code_files": 0,
         "n_test_files": 0, "title": "Bump lodash from 4.17.20 to 4.17.21"},
    ]))
    row = df.iloc[0]
    assert row.failed == 1
    assert row.docs_only == 1
    assert row.title_dependency == 1
    assert row.log_churn == pytest.approx(np.log1p(12))
    assert row.hour == 9


def test_history_ignores_runs_still_in_progress():
    df = make_runs([
        {"created_at": "2026-01-01T10:00:00Z", "updated_at": "2026-01-01T11:00:00Z", "conclusion": "failure",
         "actor_hash": "a", "pr_key": "p1"},
        {"created_at": "2026-01-01T10:30:00Z", "updated_at": "2026-01-01T10:50:00Z", "conclusion": "success",
         "actor_hash": "a", "pr_key": "p1"},
        {"created_at": "2026-01-01T12:00:00Z", "updated_at": "2026-01-01T12:30:00Z", "conclusion": "success",
         "actor_hash": "a", "pr_key": "p1"},
    ])
    out = add_history_features(add_static_features(df)).set_index("run_id")
    assert out.loc[1, "pr_prior_runs"] == 0
    assert out.loc[1, "pr_prev_failed"] == 0
    assert out.loc[2, "pr_prior_runs"] == 2
    assert out.loc[2, "pr_prior_failures"] == 1
    assert out.loc[2, "author_prior_runs"] == 2


def test_history_is_separated_by_repo_and_pr():
    df = make_runs([
        {"created_at": "2026-01-01T10:00:00Z", "updated_at": "2026-01-01T10:10:00Z", "conclusion": "failure",
         "actor_hash": "a", "pr_key": "p1", "repo": "org/one"},
        {"created_at": "2026-01-01T11:00:00Z", "updated_at": "2026-01-01T11:10:00Z", "conclusion": "success",
         "actor_hash": "a", "pr_key": "p1", "repo": "org/two"},
        {"created_at": "2026-01-01T12:00:00Z", "updated_at": "2026-01-01T12:10:00Z", "conclusion": "success",
         "actor_hash": "b", "pr_key": "p2", "repo": "org/one"},
    ])
    out = add_history_features(add_static_features(df)).set_index("run_id")
    assert out.loc[1, "pr_prior_runs"] == 0
    assert out.loc[2, "pr_prior_runs"] == 0
    assert out.loc[2, "repo_prev_failed"] == 1
    assert out.loc[2, "repo_recent_fail_streak"] == 1


def test_author_rate_is_smoothed():
    rows = [{"created_at": f"2026-01-0{d}T10:00:00Z", "updated_at": f"2026-01-0{d}T10:10:00Z",
             "conclusion": c, "actor_hash": a, "pr_key": f"p{d}"}
            for d, c, a in ((1, "success", "b"), (2, "success", "b"), (3, "failure", "a"), (4, "success", "a"))]
    out = add_history_features(add_static_features(make_runs(rows))).sort_values("created_at")
    rate = out.iloc[3]["author_fail_rate"]
    assert 1 / 3 < rate < 1


def test_time_split_is_ordered_and_complete():
    rows = [{"created_at": f"2026-01-{d:02d}T10:00:00Z", "updated_at": f"2026-01-{d:02d}T10:10:00Z",
             "conclusion": "success" if d % 3 else "failure", "actor_hash": "a", "pr_key": "p"} for d in range(1, 21)]
    df = add_history_features(add_static_features(make_runs(rows)))
    train, val, test = time_split(df, 0.7, 0.15)
    assert len(train) + len(val) + len(test) == len(df)
    assert train.created_at.max() <= val.created_at.min() <= test.created_at.min()
    assert set(FEATURES) <= set(df.columns)
