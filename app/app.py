import json
import os
import sys
import urllib.request
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

st.set_page_config(page_title="CI Build Failure Predictor", page_icon="🛠️", layout="wide")

ACCENT = "#F26722"
MODEL_NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest",
               "logistic_regression": "Logistic Regression"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LABELS = {
    "pr_prev_failed": "Last run on this PR failed",
    "author_fail_rate": "Author's failure rate",
    "author_prior_runs": "Author's earlier runs",
    "pr_prior_runs": "Earlier runs on this PR",
    "pr_prior_failures": "Earlier failed runs on this PR",
    "repo_fail_rate_20": "Repo failure rate (last 20)",
    "repo_recent_fail_streak": "Repo failure streak",
    "repo_prev_failed": "Previous repo run failed",
    "log_churn": "Change size (log lines)",
    "additions": "Lines added",
    "deletions": "Lines deleted",
    "files_changed": "Files changed",
    "n_code_files": "Code files",
    "n_test_files": "Test files",
    "n_dep_files": "Dependency files",
    "n_doc_files": "Doc files",
    "pr_age_hours": "PR age (hours)",
    "missingindicator_pr_age_hours": "PR details unavailable",
    "minutes_since_prev_pr_run": "Minutes since last PR run",
    "weekday": "Weekday",
    "hour": "Hour of day",
    "title_len": "PR title length",
    "commit_msg_len": "Commit message length",
    "is_fork": "PR from a fork",
    "is_rerun": "Re-run attempt",
    "is_bot": "Bot author",
    "touches_deps": "Touches dependencies",
    "title_fix": "Title mentions a fix",
    "title_dependency": "Title mentions dependencies",
    "pr_first_run": "First run on this PR",
    "commit_age_hours": "Commit age (hours)",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: linear-gradient(180deg, #FFF7F2 0%, #F7F8FA 260px); }}
.block-container {{ padding-top: 2rem; max-width: 1250px; }}
.hero {{ display: flex; gap: 22px; align-items: center; background: #fff; border-radius: 18px;
        padding: 26px 30px; border: 1px solid #F1E4DC; box-shadow: 0 6px 24px rgba(242,103,34,.08); }}
.hero h1 {{ font-size: 2.05rem; font-weight: 800; color: #1F2430; margin: 0; padding: 0; }}
.hero p {{ color: #5B6170; margin: 6px 0 12px 0; font-size: 1rem; }}
.chip {{ display: inline-block; font-size: .78rem; font-weight: 600; padding: 4px 11px; margin-right: 6px;
        border-radius: 999px; background: #FFF1E8; color: #B54A12; border: 1px solid #FBD5BF; }}
.chip.grey {{ background: #F1F3F6; color: #4A5060; border-color: #E2E5EA; }}
.section-title {{ font-weight: 700; color: #1F2430; font-size: 1.05rem; margin-bottom: .2rem; }}
.section-sub {{ color: #7A8090; font-size: .85rem; margin-bottom: .6rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 16px !important;
        border: 1px solid #ECEEF2 !important; box-shadow: 0 2px 10px rgba(20,30,50,.04); }}
.stButton > button, .stFormSubmitButton > button {{ background: {ACCENT}; color: #fff; border: 0; font-weight: 700;
        border-radius: 12px; padding: .65rem 1.4rem; width: 100%; font-size: 1rem; }}
.stFormSubmitButton > button:hover {{ background: #D9561A; color: #fff; }}
.result {{ background: #fff; border-radius: 18px; padding: 24px 28px; border: 1px solid #ECEEF2;
        box-shadow: 0 6px 24px rgba(20,30,50,.06); }}
.prob {{ font-size: 3.2rem; font-weight: 800; line-height: 1; }}
.badge {{ display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: .9rem; }}
.bar {{ position: relative; height: 14px; border-radius: 999px; margin: 18px 0 6px 0;
        background: linear-gradient(90deg, #2FB67C 0%, #F5B83D 50%, #E5484D 100%); }}
.marker {{ position: absolute; top: -7px; width: 4px; height: 28px; background: #1F2430; border-radius: 3px; }}
.thr {{ position: absolute; top: -4px; width: 2px; height: 22px; background: rgba(31,36,48,.35); }}
.scale {{ display: flex; justify-content: space-between; color: #7A8090; font-size: .78rem; }}
.factor {{ display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid #F1F2F5;
        font-size: .93rem; color: #2A2F3A; }}
.up {{ color: #E5484D; font-weight: 700; }}
.down {{ color: #2FB67C; font-weight: 700; }}
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #ECEEF2; }}
.kpi {{ background: #FAFBFC; border: 1px solid #ECEEF2; border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; }}
.kpi b {{ font-size: 1.25rem; color: #1F2430; }}
.kpi span {{ display: block; color: #7A8090; font-size: .78rem; }}
.note {{ color: #8A8F9C; font-size: .78rem; }}
</style>
""", unsafe_allow_html=True)

ICON = f"""
<svg width="76" height="76" viewBox="0 0 76 76" xmlns="http://www.w3.org/2000/svg">
  <rect width="76" height="76" rx="20" fill="#FFF1E8"/>
  <path d="M16 38h12M36 38h8M52 38h8" stroke="{ACCENT}" stroke-width="3" stroke-linecap="round"/>
  <circle cx="16" cy="38" r="5" fill="{ACCENT}"/>
  <rect x="26" y="29" width="14" height="18" rx="4" fill="#fff" stroke="{ACCENT}" stroke-width="3"/>
  <rect x="42" y="29" width="14" height="18" rx="4" fill="#fff" stroke="#1F2430" stroke-width="3"/>
  <path d="M46 34l6 8M52 34l-6 8" stroke="#E5484D" stroke-width="2.6" stroke-linecap="round"/>
  <path d="M29.5 38.5l3 3 5-6" stroke="#2FB67C" stroke-width="2.6" fill="none" stroke-linecap="round"/>
  <circle cx="60" cy="38" r="5" fill="#1F2430"/>
</svg>"""


RELEASE_URL = os.environ.get(
    "MODEL_RELEASE_URL",
    "https://github.com/dulshandadallege/wso2-ci-build-failure-prediction/releases/latest/download",
)


def fetch_artifacts():
    targets = {"model.joblib": ROOT / "models" / "model.joblib", "metrics.json": ROOT / "reports" / "metrics.json"}
    for name, path in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", path)


@st.cache_resource
def load():
    fetch_artifacts()
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    model = bundle["model"].named_steps["model"]
    explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else None
    metrics_path = ROOT / "reports" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return bundle, explainer, metrics


def shown_value(name, value):
    if name in BINARY or "_" in name and name.split("_")[0] in ("repo", "author", "event") and name not in NUMERIC:
        return "Yes" if value >= 0.5 else "No"
    if name.startswith("missingindicator_"):
        return "Yes" if value >= 0.5 else "No"
    if name == "weekday":
        return WEEKDAYS[int(value)]
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def label(name):
    if name in LABELS:
        return LABELS[name]
    for prefix, text in (("repo_", "Repo: "), ("author_association_", "Author is "), ("event_", "Event: ")):
        if name.startswith(prefix):
            return text + name[len(prefix):].split("/")[-1]
    return name.replace("_", " ").capitalize()


bundle, explainer, metrics = load()
prep = bundle["model"].named_steps["prep"]
repos = list(prep.named_transformers_["cat"].categories_[CATEGORICAL.index("repo")])
threshold = bundle["threshold"]

st.markdown(f"""
<div class="hero">{ICON}
  <div>
    <h1>CI Build Failure Predictor</h1>
    <p>Estimate how likely a GitHub Actions pull-request build is to fail before it finishes, and see why.</p>
    <span class="chip">WSO2 &amp; Ballerina public repositories</span>
    <span class="chip grey">{MODEL_NAMES.get(metrics.get('model', bundle.get('name')), 'ML model')}</span>
    <span class="chip grey">Explained with SHAP</span>
  </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model card")
    test = metrics.get("test", {})
    rows = metrics.get("rows", {})
    for name, value in (("ROC-AUC (test)", test.get("roc_auc")), ("PR-AUC (test)", test.get("pr_auc")),
                        ("Recall at threshold", test.get("recall")), ("Decision threshold", threshold)):
        if value is not None:
            st.markdown(f'<div class="kpi"><b>{value:.2f}</b><span>{name}</span></div>', unsafe_allow_html=True)
    if rows:
        st.markdown(f'<div class="kpi"><b>{sum(rows.values()):,}</b><span>PR build runs from {len(repos)} repositories</span></div>',
                    unsafe_allow_html=True)
    st.markdown("### Repositories")
    st.markdown("\n".join(f"- `{r.split('/')[-1]}`" for r in repos))
    st.markdown('<p class="note">Independent academic project built on public GitHub Actions data. '
                'Not affiliated with or endorsed by WSO2 LLC.</p>', unsafe_allow_html=True)

st.write("")
with st.form("inputs"):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title">📦 Change</div><div class="section-sub">What the commit touches</div>',
                        unsafe_allow_html=True)
            repo = st.selectbox("Repository", repos, format_func=lambda r: r.split("/")[-1])
            a, b = st.columns(2)
            additions = a.number_input("Lines added", 0, 100000, 40)
            deletions = b.number_input("Lines deleted", 0, 100000, 10)
            a, b = st.columns(2)
            files_changed = a.number_input("Files changed", 0, 3000, 3)
            n_code = b.number_input("Code files", 0, 3000, 2)
            a, b = st.columns(2)
            n_test = a.number_input("Test files", 0, 3000, 0)
            n_doc = b.number_input("Doc files", 0, 3000, 0)
            n_dep = st.number_input("Dependency files (pom.xml, package.json...)", 0, 100, 0)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">🧭 Context</div><div class="section-sub">Who, when and how</div>',
                        unsafe_allow_html=True)
            title = st.text_input("PR title", "Fix token validation for expired sessions")
            association = st.selectbox("Author association", ["CONTRIBUTOR", "MEMBER", "NONE", "UNKNOWN"])
            a, b = st.columns(2)
            weekday = a.selectbox("Weekday", WEEKDAYS)
            pr_age = b.number_input("PR age (hours)", 0.0, 10000.0, 2.0)
            hour = st.slider("Hour (Sri Lanka time)", 0, 23, 14)
            is_fork = st.checkbox("PR opened from a fork", True)
            a, b = st.columns(2)
            is_bot = a.checkbox("Bot author", False)
            is_rerun = b.checkbox("Re-run attempt", False)
    with c3:
        with st.container(border=True):
            st.markdown('<div class="section-title">📈 History</div><div class="section-sub">What happened before this run</div>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            pr_prior = a.number_input("Earlier PR runs", 0, 200, 0)
            pr_prior_fail = b.number_input("Earlier PR failures", 0, 200, 0)
            pr_prev_failed = st.checkbox("Last run on this PR failed", False)
            author_runs = st.number_input("Author's earlier runs", 0, 5000, 20)
            author_rate = st.slider("Author's failure rate", 0.0, 1.0, 0.25)
            repo_rate = st.slider("Repo failure rate (last 20 runs)", 0.0, 1.0, 0.25)
            streak = st.number_input("Current repo failure streak", 0, 50, 0)
    submitted = st.form_submit_button("Predict build outcome")

title_low = title.lower()
day = WEEKDAYS.index(weekday)
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

if p >= threshold:
    level, color, bg, verdict = "High risk", "#E5484D", "#FDECEC", "Likely to fail"
elif p >= threshold * 0.65:
    level, color, bg, verdict = "Medium risk", "#C98A12", "#FFF6E0", "Could go either way"
else:
    level, color, bg, verdict = "Low risk", "#2FB67C", "#E8F7F0", "Likely to pass"

st.write("")
left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown(f"""
<div class="result">
  <div style="color:#7A8090;font-weight:600;font-size:.85rem;letter-spacing:.04em">FAILURE PROBABILITY</div>
  <div style="display:flex;align-items:center;gap:16px;margin-top:8px">
    <div class="prob" style="color:{color}">{p:.0%}</div>
    <span class="badge" style="background:{bg};color:{color}">{level}</span>
  </div>
  <div style="color:#2A2F3A;font-weight:600;margin-top:10px">{verdict}</div>
  <div class="bar">
    <div class="thr" style="left:{threshold * 100:.1f}%"></div>
    <div class="marker" style="left:calc({p * 100:.1f}% - 2px)"></div>
  </div>
  <div class="scale"><span>0%</span><span>threshold {threshold:.0%}</span><span>100%</span></div>
</div>""", unsafe_allow_html=True)

    if explainer is not None:
        Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
        raw = explainer.shap_values(Xt)
        values = raw[1] if isinstance(raw, list) else np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
        contrib = pd.Series(values[0], index=Xt.columns)
        contrib = contrib[contrib.abs() > 1e-6].sort_values(key=np.abs, ascending=False).head(6)
        items = "".join(
            f'<div class="factor"><span>{label(k)} <span style="color:#8A8F9C">· {shown_value(k, Xt.iloc[0][k])}</span></span>'
            f'<span class="{"up" if v > 0 else "down"}">{"▲ raises" if v > 0 else "▼ lowers"} risk</span></div>'
            for k, v in contrib.items())
        st.markdown(f'<div class="result" style="margin-top:16px"><div class="section-title">Top factors</div>{items}</div>',
                    unsafe_allow_html=True)

with right:
    if explainer is not None:
        with st.container(border=True):
            st.markdown('<div class="section-title">Why this prediction</div>'
                        '<div class="section-sub">SHAP contributions in log-odds. Red pushes toward failure, blue toward success.</div>',
                        unsafe_allow_html=True)
            base = float(np.atleast_1d(explainer.expected_value)[-1])
            plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
            plt.figure()
            shap.plots.waterfall(shap.Explanation(values[0], base_values=base, data=Xt.iloc[0].to_numpy(),
                                                  feature_names=[label(c) for c in Xt.columns]),
                                 max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
