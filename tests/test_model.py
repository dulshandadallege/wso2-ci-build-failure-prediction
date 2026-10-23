import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features import BINARY, DATASET, FEATURES, NUMERIC, TARGET, build_dataset, time_split
from src.train import best_threshold, preprocessor

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def splits():
    df = build_dataset("data/missing")
    return time_split(df, 0.70, 0.15)


@pytest.fixture(scope="module")
def model(splits):
    train, val, _ = splits
    pipe = Pipeline([
        ("prep", preprocessor(False)),
        ("model", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                                 class_weight="balanced", random_state=42, verbose=-1)),
    ])
    fit_df = pd.concat([train, val])
    return pipe.fit(fit_df[FEATURES], fit_df[TARGET])


def test_dataset_shape(splits):
    train, val, test = splits
    total = len(train) + len(val) + len(test)
    assert total > 2500
    assert 0.2 < train[TARGET].mean() < 0.45
    assert train.created_at.max() <= test.created_at.min()


def test_quality_gate(model, splits):
    _, _, test = splits
    p = model.predict_proba(test[FEATURES])[:, 1]
    base = test[TARGET].mean()
    assert roc_auc_score(test[TARGET], p) >= 0.62
    assert average_precision_score(test[TARGET], p) >= base * 1.3


def test_predictions_are_probabilities(model, splits):
    _, _, test = splits
    p = model.predict_proba(test[FEATURES].head(50))[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_model_handles_integer_inputs(model, splits):
    _, _, test = splits
    row = test[FEATURES].head(1).copy()
    num = NUMERIC + BINARY
    row[num] = row[num].fillna(0).round().astype(int).astype(float)
    assert model.predict_proba(row).shape == (1, 2)


def test_best_threshold_in_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.35])
    t = best_threshold(y, p)
    assert 0.1 <= t <= 0.9
