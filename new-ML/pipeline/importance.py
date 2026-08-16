import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from pipeline.config import (SEED, FEATURE_COLS, FAMILY_COL, TABLES_DIR)
from pipeline.models import fit_model, predict_model
from pipeline.metrics import pooled_metrics
from pipeline.plots import save_fig


def _permutation_rmse(model, Xp, y, col):
    """Pooled RMSE after shuffling one column of the feature matrix."""
    Xp_perm = {"num": Xp["num"].copy(), "family": Xp["family"].copy()}
    if col == FAMILY_COL:
        vals = Xp_perm["family"].to_numpy()
        np.random.shuffle(vals)
        Xp_perm["family"] = pd.Series(vals)
    else:
        idx = FEATURE_COLS.index(col)
        vals = Xp_perm["num"][:, idx].copy()
        np.random.shuffle(vals)
        Xp_perm["num"][:, idx] = vals
    return pooled_metrics(y, predict_model(model, Xp_perm))["RMSE"]


def permutation_importance(model, Xp, y, n_repeats=10, seed=SEED):
    np.random.seed(seed)
    baseline = pooled_metrics(y, predict_model(model, Xp))["RMSE"]
    cols = FEATURE_COLS + [FAMILY_COL]
    scores = {}
    for col in cols:
        increases = [_permutation_rmse(model, Xp, y, col) - baseline
                     for _ in range(n_repeats)]
        scores[col] = float(np.mean(increases))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def plot_permutation_importance(imp, out_dir=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(imp)
    ax.barh(range(len(names)), list(imp.values()))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("RMSE increase (Hz)")
    ax.set_title("Permutation feature importance (selected model)")
    fig.tight_layout()
    return save_fig(fig, "feature_importance.png", out_dir)


def save_importance_table(imp, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature": list(imp), "importance": list(imp.values())})
    path = out_dir / "feature_importance.csv"
    df.to_csv(path, index=False)
    return path


def plot_shap(model, Xp, out_dir=None, max_display=8):
    df = pd.DataFrame(Xp["num"], columns=FEATURE_COLS)
    df[FAMILY_COL] = Xp["family"].to_numpy()
    try:
        explainer = shap.TreeExplainer(model)
        sample = df.iloc[:200]
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception:
        sample = df.sample(min(100, len(df)), random_state=SEED)
        explainer = shap.KernelExplainer(
            lambda x: predict_model(model, {"num": x[:, :7],
                                            "family": pd.Series(x[:, 7])}),
            df.to_numpy()[:50])
        shap_values = explainer.shap_values(sample.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, max_display=max_display, show=False)
    fig.tight_layout()
    return save_fig(fig, "shap_summary.png", out_dir)
