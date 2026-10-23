import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import BINARY, CATEGORICAL, NUMERIC, TARGET, build_dataset, time_split

warnings.filterwarnings("ignore")


def preprocessor(scale):
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("bin", SimpleImputer(strategy="constant", fill_value=0), BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)


def candidates(pos_weight, seed):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"), True,
            {"model__C": np.logspace(-3, 1, 20)}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=seed), False,
            {"model__n_estimators": [200, 400, 600], "model__max_depth": [4, 6, 8, 12, None],
             "model__min_samples_leaf": [2, 5, 10, 20], "model__max_features": ["sqrt", 0.3, 0.5]}),
        "lightgbm": (
            LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1, n_jobs=1), False,
            {"model__n_estimators": [200, 400, 700], "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
             "model__num_leaves": [7, 15, 31], "model__min_child_samples": [10, 20, 40],
             "model__subsample": [0.7, 0.9, 1.0], "model__subsample_freq": [1],
             "model__colsample_bytree": [0.6, 0.8, 1.0], "model__reg_lambda": [0, 1, 5]}),
        "xgboost": (
            XGBClassifier(scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=seed, n_jobs=1), False,
            {"model__n_estimators": [200, 400, 700], "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
             "model__max_depth": [3, 4, 5, 6], "model__min_child_weight": [1, 3, 5],
             "model__subsample": [0.7, 0.9, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0],
             "model__reg_lambda": [0, 1, 5]}),
    }


def scores(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


def best_threshold(y, p):
    grid = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["train"]
    seed = params["seed"]
    df = build_dataset()
    train, val, test = time_split(df, params["train_size"], params["val_size"])
    X_cols = NUMERIC + BINARY + CATEGORICAL
    Xtr, ytr = train[X_cols], train[TARGET]
    Xva, yva = val[X_cols], val[TARGET]
    Xte, yte = test[X_cols], test[TARGET]
    print(f"train {len(train)} ({ytr.mean():.3f})  val {len(val)} ({yva.mean():.3f})  test {len(test)} ({yte.mean():.3f})")

    cv = TimeSeriesSplit(n_splits=params["cv_splits"])
    pos_weight = (1 - ytr.mean()) / ytr.mean()
    results, fitted = [], {}

    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    pv = dummy.predict_proba(Xva)[:, 1]
    results.append({"model": "baseline_prior", "cv_pr_auc": np.nan, "threshold": 0.5,
                    **{f"val_{k}": v for k, v in scores(yva, pv, 0.5).items()}})

    for name, (est, scale, grid) in candidates(pos_weight, seed).items():
        pipe = Pipeline([("prep", preprocessor(scale)), ("model", est)])
        search = RandomizedSearchCV(pipe, grid, n_iter=params["n_iter"], scoring="average_precision",
                                    cv=cv, random_state=seed, n_jobs=-1)
        search.fit(Xtr, ytr)
        pv = search.predict_proba(Xva)[:, 1]
        t = best_threshold(yva, pv)
        fitted[name] = (search.best_estimator_, t)
        row = {"model": name, "cv_pr_auc": search.best_score_, "threshold": t,
               **{f"val_{k}": v for k, v in scores(yva, pv, t).items()}}
        results.append(row)
        print(f"{name:20s} cv_pr_auc={row['cv_pr_auc']:.3f} val_pr_auc={row['val_pr_auc']:.3f} val_f1={row['val_f1']:.3f}")

    table = pd.DataFrame(results)
    best = table.dropna(subset=["cv_pr_auc"]).sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    model, threshold = fitted[best]

    final = clone(model).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
    pt = final.predict_proba(Xte)[:, 1]
    test_scores = scores(yte, pt, threshold)
    print(f"selected {best}: test {json.dumps({k: round(v, 3) for k, v in test_scores.items()})}")

    test_table = []
    for name, (m, t) in fitted.items():
        refit = clone(m).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        test_table.append({"model": name, **scores(yte, refit.predict_proba(Xte)[:, 1], t)})
    test_table.append({"model": "baseline_prior", **scores(yte, np.full(len(yte), ytr.mean()), 0.5)})

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump({"model": final, "threshold": threshold, "features": X_cols, "name": best}, "models/model.joblib")
    table.round(4).to_csv("reports/validation_results.csv", index=False)
    pd.DataFrame(test_table).round(4).to_csv("reports/test_results.csv", index=False)
    Path("reports/metrics.json").write_text(json.dumps({
        "model": best, "threshold": threshold,
        "test": {k: round(v, 4) for k, v in test_scores.items()},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "failure_rate": {"train": round(ytr.mean(), 4), "val": round(yva.mean(), 4), "test": round(yte.mean(), 4)},
        "test_period": [str(test.created_at.min()), str(test.created_at.max())],
    }, indent=2))


if __name__ == "__main__":
    main()
