import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import SEED, TABLES_DIR
from pipeline.models import fit_model, predict_model, run_cv
from pipeline.metrics import pooled_metrics
from pipeline.plots import save_fig


def compute_learning_curve(model, Xp, y, sizes=(0.1, 0.25, 0.5, 0.75, 1.0),
                           n_folds=5, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = Xp["num"].shape[0]
    rows = []
    for frac in sizes:
        n_sub = max(10, int(n * frac))
        idx = rng.choice(n, size=n_sub, replace=False)
        Xp_sub = {"num": Xp["num"][idx],
                  "family": Xp["family"].iloc[idx].reset_index(drop=True)}
        y_sub = y[idx]
        fit_model(model, Xp_sub, y_sub)
        train_rmse = pooled_metrics(y_sub, predict_model(model, Xp_sub))["RMSE"]
        val_rmse = run_cv(model, Xp_sub, y_sub, n_folds=n_folds, seed=seed)["RMSE_mean"]
        rows.append({"size": frac, "train_rmse": train_rmse, "val_rmse": val_rmse})
    return pd.DataFrame(rows)


def plot_learning_curve(lc_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lc_df["size"], lc_df["train_rmse"], "o-", label="train")
    ax.plot(lc_df["size"], lc_df["val_rmse"], "s-", label="validation")
    ax.set_xlabel("Training fraction"); ax.set_ylabel("Pooled RMSE (Hz)")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "learning_curve_analysis.png", out_dir)


def save_learning_curve(lc_df, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "learning_curve_results.csv"
    lc_df.round(4).to_csv(path, index=False)
    return path


def extrapolation_test(model, Xp_short, y_short, Xp_long, y_long):
    fit_model(model, Xp_short, y_short)
    train_rmse = pooled_metrics(y_short, predict_model(model, Xp_short))["RMSE"]
    extrap_pred = predict_model(model, Xp_long)
    extrap_rmse = pooled_metrics(y_long, extrap_pred)["RMSE"]
    extrap_r2 = pooled_metrics(y_long, extrap_pred)["R2"]
    return {"train_rmse": train_rmse, "extrap_rmse": extrap_rmse,
            "extrap_r2": extrap_r2}


def plot_extrapolation(result, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["train (SHORT)", "extrapolated (LONG)"]
    values = [result["train_rmse"], result["extrap_rmse"]]
    ax.bar(labels, values)
    ax.set_ylabel("Pooled RMSE (Hz)")
    ax.set_title(f"Extrapolation: train SHORT -> test LONG (R2={result['extrap_r2']:.3f})")
    fig.tight_layout()
    return save_fig(fig, "extrapolation_test.png", out_dir)
