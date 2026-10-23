# CI Build Failure Prediction

Predicts whether a GitHub Actions pull-request build will fail, at the moment it starts. The model is trained on real CI history from public WSO2 and Ballerina repositories, a Sri Lankan open-source ecosystem.
## Results

Time-ordered split: train on the oldest 70% of runs, validate on the next 15%, test on the newest 15% (30 Sep – 23 Oct 2026).

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| Prior baseline | 0.50 | 0.32 | – | – |
| Logistic regression | 0.67 | 0.49 | 0.49 | 0.71 |
| Random forest | 0.68 | 0.46 | 0.55 | 0.90 |
| **LightGBM (selected)** | **0.70** | **0.51** | **0.53** | **0.68** |
| XGBoost | 0.69 | 0.48 | 0.51 | 0.58 |

The strongest drivers, from SHAP, are whether the previous run on the same PR failed, the author's past failure rate and experience, the repository, and the size of the change.

## Data

2,948 completed PR builds (success or failure) from six repositories:

- `wso2/carbon-identity-framework`, `wso2/identity-apps`, `wso2/carbon-apimgt`
- `ballerina-platform/module-ballerina-http`, `module-ballerina-grpc`, `module-ballerinax-kafka`

Only public run metadata and commit statistics are collected; there is no source code and no logs. Usernames and branch names are stored as salted hashes. A copy of the collected dataset is in `data/dataset/ci_runs.csv`, so everything can be reproduced without a GitHub token.

## Project layout

```
src/collect.py        GitHub API collector (rate-limit aware, cached, parallel)
src/github_client.py  API client and hashing helpers
src/features.py       Leakage-free feature engineering and time split
src/train.py          Model search, threshold tuning, test evaluation
src/explain.py        SHAP figures
app/app.py            Streamlit front end
tests/                Unit tests and model quality gate
CI_Build_Failure_Prediction.ipynb   Full analysis notebook
```

## Run locally

```bash
conda create -n ci-failure python=3.12 -y
conda activate ci-failure
pip install -r requirements-dev.txt

python -m src.train
python -m src.explain
streamlit run app/app.py
```

To collect fresh data, add a read-only token to `.env` (see `.env.example`) and run `python -m src.collect`.

## Docker

```bash
python -m src.train
docker build -t ci-failure-app .
docker run -p 8501:8501 ci-failure-app
```

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

1. lint with ruff, run the unit tests and the model quality gate,
2. retrain the model, fail if test ROC-AUC drops below 0.62 or PR-AUC isn't at least 1.3× the baseline,
3. build the Docker image and check that the app's health endpoint responds.

## Limitations

- Public repositories only. The WSO2 history covers about four months because of the 700-run cap per repo.
- Labels include flaky tests and runner problems, which metadata cannot predict.
- Commit statistics describe the triggering commit, not the whole pull request.
