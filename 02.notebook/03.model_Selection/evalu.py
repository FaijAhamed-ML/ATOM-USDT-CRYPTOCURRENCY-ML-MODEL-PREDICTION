

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate, validation_curve
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


tscv = TimeSeriesSplit(n_splits=5)
REG_SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
CLF_SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

def cv_scores(X_train, model, y, task):
    scoring = REG_SCORING if task == "regression" else CLF_SCORING
    res = cross_validate(model, X_train, y, cv=tscv, scoring=scoring, n_jobs=-1)
    out = {"task": task}
    for k in scoring:
        v = res[f"test_{k}"]
        v = -v if k in ("rmse", "mae") else v
        out[k] = v.mean(); out[k + "_std"] = v.std()
    return out



def plot_curve(model, X_train, y, param, values, scoring, title, fname, flip=False):
    tr, va = validation_curve(model, X_train, y, param_name=param, param_range=values, cv=tscv, scoring=scoring, n_jobs=-1)
    s = -1 if flip else 1
    xs = np.arange(len(values)); labels = [str(v) for v in values]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for arr, name in ((s * tr, "train"), (s * va, "validation")):
        ax.plot(xs, arr.mean(1), marker="o", label=name)
        ax.fill_between(xs, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), alpha=0.15)
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_xlabel(param); ax.set_title(title); ax.legend()
    plt.tight_layout(); fig.savefig("../../03.results/eda_visualizations/" + fname, dpi=130); plt.show()
    return pd.DataFrame({"value": labels, "train": (s * tr).mean(1), "validation": (s * va).mean(1)})