import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import BINARY, CATEGORICAL, NUMERIC  # noqa: E402

st.set_page_config(page_title="CI Build Failure Predictor", layout="wide")


@st.cache_resource
def load():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    model = bundle["model"].named_steps["model"]
    explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else None
    return bundle, explainer


bundle, explainer = load()
prep = bundle["model"].named_steps["prep"]
repos = list(prep.named_transformers_["cat"].categories_[CATEGORICAL.index("repo")])

st.title("CI Build Failure Predictor")
st.caption("Estimates the chance that a GitHub Actions PR build fails, from change size, context and recent history.")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Change")
    repo = st.selectbox("Repository", repos)
    additions = st.number_input("Lines added", 0, 100000, 40)
    deletions = st.number_input("Lines deleted", 0, 100000, 10)
    files_changed = st.number_input("Files changed", 0, 3000, 3)
    n_code = st.number_input("Code files", 0, 3000, 2)
    n_test = st.number_input("Test files", 0, 3000, 0)
    n_dep = st.number_input("Dependency files (pom.xml, package.json...)", 0, 100, 0)
    n_doc = st.number_input("Doc files", 0, 3000, 0)
with c2:
    st.subheader("Context")
    title = st.text_input("PR title", "Fix token validation for expired sessions")
    association = st.selectbox("Author association", ["CONTRIBUTOR", "MEMBER", "NONE", "UNKNOWN"])
    is_fork = st.checkbox("PR from a fork", True)
    is_bot = st.checkbox("Bot author", False)
    is_rerun = st.checkbox("Re-run attempt", False)
    hour = st.slider("Hour (Sri Lanka time)", 0, 23, 14)
    weekday = st.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    pr_age = st.number_input("PR age (hours)", 0.0, 10000.0, 2.0)
with c3:
    st.subheader("History")
    pr_prior = st.number_input("Earlier runs on this PR", 0, 200, 0)
    pr_prior_fail = st.number_input("Earlier failed runs on this PR", 0, 200, 0)
    pr_prev_failed = st.checkbox("Last run on this PR failed", False)
    author_runs = st.number_input("Author's earlier runs", 0, 5000, 20)
    author_rate = st.slider("Author's failure rate", 0.0, 1.0, 0.25)
    repo_rate = st.slider("Repo failure rate (last 20 runs)", 0.0, 1.0, 0.25)
    streak = st.number_input("Current repo failure streak", 0, 50, 0)

title_low = title.lower()
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(weekday)
row = {
    "additions": additions, "deletions": deletions, "log_churn": np.log1p(additions + deletions),
    "files_changed": files_changed, "top_dirs_touched": min(files_changed, 3),
    "files_added": 0, "files_removed": 0, "n_code_files": n_code, "n_test_files": n_test,
    "n_config_files": 0, "n_doc_files": n_doc, "n_dep_files": n_dep, "n_ci_files": 0,
    "test_file_ratio": n_test / max(files_changed, 1), "commit_msg_len": len(title) + 20,
    "title_len": len(title), "hour": hour, "weekday": day, "commit_age_hours": 0.5,
    "pr_age_hours": pr_age, "repo_fail_rate_20": repo_rate, "repo_recent_fail_streak": streak,
    "author_prior_runs": author_runs, "author_fail_rate": author_rate, "pr_prior_runs": pr_prior,
    "pr_prior_failures": pr_prior_fail, "minutes_since_prev_pr_run": 60 if pr_prior else -1,
    "sha_prior_runs": int(is_rerun),
    "is_fork": int(is_fork), "is_bot": int(is_bot), "is_rerun": int(is_rerun), "is_merge_commit": 0,
    "is_weekend": int(day >= 5), "docs_only": int(files_changed > 0 and n_doc == files_changed),
    "touches_deps": int(n_dep > 0), "touches_ci": 0,
    "title_dependency": int(any(w in title_low for w in ("bump", "upgrade", "dependenc"))),
    "title_fix": int(any(w in title_low for w in ("fix", "bug", "issue", "resolve"))),
    "title_revert": int("revert" in title_low), "title_test": int("test" in title_low),
    "repo_prev_failed": int(streak > 0), "pr_prev_failed": int(pr_prev_failed),
    "sha_prev_failed": int(is_rerun and pr_prev_failed), "pr_first_run": int(pr_prior == 0),
    "repo": repo, "event": "pull_request", "author_association": association,
}
X = pd.DataFrame([row])[NUMERIC + BINARY + CATEGORICAL]
X[NUMERIC + BINARY] = X[NUMERIC + BINARY].astype(float)
p = float(bundle["model"].predict_proba(X)[0, 1])
risky = p >= bundle["threshold"]

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Failure probability", f"{p:.0%}")
m2.metric("Decision threshold", f"{bundle['threshold']:.2f}")
m3.metric("Prediction", "Likely to FAIL" if risky else "Likely to pass")

if explainer is not None:
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    raw = explainer.shap_values(Xt)
    values = raw[1] if isinstance(raw, list) else np.asarray(raw)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    st.subheader("Why this prediction")
    fig = plt.figure()
    shap.plots.waterfall(shap.Explanation(values[0], base_values=base, data=Xt.iloc[0].to_numpy(),
                                          feature_names=list(Xt.columns)), max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)
    st.caption("Values are in log-odds. Red pushes toward failure, blue toward success.")
