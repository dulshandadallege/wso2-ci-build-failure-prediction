from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

from src.features import TARGET, build_dataset, time_split

OUT = Path("reports/figures")


def transformed(bundle, X):
    prep = bundle["model"].named_steps["prep"]
    return pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)


def explainer_for(bundle, background):
    model = bundle["model"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, background)


def positive_class(values):
    if isinstance(values, list):
        return values[1]
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(Path("params.yaml").read_text())["train"]
    bundle = joblib.load("models/model.joblib")
    df = build_dataset()
    train, _, test = time_split(df, params["train_size"], params["val_size"])
    Xte = transformed(bundle, test[bundle["features"]])
    Xbg = transformed(bundle, train[bundle["features"]].sample(min(300, len(train)), random_state=0))

    exp = explainer_for(bundle, Xbg)
    raw = exp.shap_values(Xte)
    values = positive_class(raw if isinstance(raw, list) else np.asarray(raw))
    base = float(np.atleast_1d(exp.expected_value)[-1])

    shap.summary_plot(values, Xte, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    shap.summary_plot(values, Xte, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT / "shap_importance.png", dpi=150)
    plt.close()

    proba = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    y = test[TARGET].to_numpy()
    cases = {
        "failed_run": int(np.argmax(np.where(y == 1, proba, -1))),
        "passed_run": int(np.argmin(np.where(y == 0, proba, 2))),
    }
    for label, i in cases.items():
        e = shap.Explanation(values[i], base_values=base, data=Xte.iloc[i].to_numpy(),
                             feature_names=list(Xte.columns))
        shap.plots.waterfall(e, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(OUT / f"shap_waterfall_{label}.png", dpi=150)
        plt.close()

    imp = pd.Series(np.abs(values).mean(axis=0), index=Xte.columns).sort_values(ascending=False)
    imp.round(4).to_csv("reports/shap_importance.csv", header=["mean_abs_shap"])
    print(imp.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
