# CI Build Failure Prediction

Predicts whether a GitHub Actions PR build will fail, using public CI history from WSO2 and Ballerina repositories.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add a read-only GitHub token
```

## Data collection

```bash
python -m src.collect                       # all repos in params.yaml
python -m src.collect --repo wso2/identity-apps --limit 50
```

Only run metadata and commit statistics are collected. Actor and branch names are stored as salted hashes. Raw data and API cache stay local and are not committed.
