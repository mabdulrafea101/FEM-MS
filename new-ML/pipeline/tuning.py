import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pipeline.config import SEED, TABLES_DIR
from pipeline.models import run_cv
from pipeline.plots import save_fig

CATBOOST_GRID = [
    {"name": "default", "params": {"iterations": 100, "learning_rate": 0.1, "depth": 6}},
    {"name": "deeper", "params": {"iterations": 200, "learning_rate": 0.1, "depth": 8}},
    {"name": "regularized", "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3}},
    {"name": "shallow", "params": {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5}},
    {"name": "fast", "params": {"iterations": 50, "learning_rate": 0.2, "depth": 6}},
]


def evaluate_candidates(grid, Xp, y, n_folds=5, seed=SEED):
    rows = []
    for cand in grid:
        model = CatBoostRegressor(loss_function="MultiRMSE", random_state=seed,
                                  verbose=False, allow_writing_files=False,
                                  **cand["params"])
        cv = run_cv(model, Xp, y, n_folds=n_folds, seed=seed)
        rows.append({"name": cand["name"], "RMSE_mean": cv["RMSE_mean"],
                     "RMSE_std": cv["RMSE_std"], "folds": cv["folds"],
                     **cand["params"]})
    return pd.DataFrame(rows)


def paired_ttest(cv_default, cv_best):
    t, p = stats.ttest_rel(cv_default, cv_best)
    diff = np.asarray(cv_default) - np.asarray(cv_best)
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return {"t": float(t), "p": float(p), "cohens_d": float(d)}


def plot_hyperparameter_importance(grid_results, out_dir=None):
    params = [c for c in ["iterations", "learning_rate", "depth", "l2_leaf_reg"]
              if c in grid_results.columns]
    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
    for ax, p in zip(axes, params):
        ax.scatter(grid_results[p], grid_results["RMSE_mean"])
        ax.set_xlabel(p); ax.set_ylabel("CV RMSE (Hz)")
    fig.tight_layout()
    return save_fig(fig, "hyperparameter_importance.png", out_dir)


def plot_hyperparam_ttest(ttest_result, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5,
            f"Paired t-test (default vs optimized)\n"
            f"t = {ttest_result['t']:.2f}, p = {ttest_result['p']:.4f}\n"
            f"Cohen's d = {ttest_result['cohens_d']:.2f}",
            ha="center", va="center", fontsize=13)
    ax.axis("off")
    return save_fig(fig, "hyperparam_ttest.png", out_dir)


def save_hyperparam_table(grid_results, ttest_result, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = grid_results.sort_values("RMSE_mean").iloc[0]
    default = grid_results[grid_results["name"] == "default"]
    default = default.iloc[0] if not default.empty else grid_results.iloc[0]
    rows = [{"config": "default", "CV_RMSE_mean": default["RMSE_mean"],
             "CV_RMSE_std": default["RMSE_std"]},
            {"config": "optimized", "CV_RMSE_mean": best["RMSE_mean"],
             "CV_RMSE_std": best["RMSE_std"]}]
    df = pd.DataFrame(rows)
    df["t_stat"] = ttest_result["t"]
    df["p_value"] = ttest_result["p"]
    df["cohens_d"] = ttest_result["cohens_d"]
    path = out_dir / "hyperparam_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path
