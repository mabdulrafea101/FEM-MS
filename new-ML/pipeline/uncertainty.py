import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import (N_BOOTSTRAP, SEED, TARGET_COLS, TABLES_DIR)
from pipeline.models import fit_model, predict_model
from pipeline.plots import save_fig


def bootstrap_predictions(model, Xp_dev, y_dev, Xp_held, n=N_BOOTSTRAP, seed=SEED):
    """Refit on n bootstrap resamples of the dev set; return 95% CI per point."""
    rng = np.random.default_rng(seed)
    n_dev = Xp_dev["num"].shape[0]
    y_dev = np.asarray(y_dev)
    preds = []
    for _ in range(n):
        idx = rng.integers(0, n_dev, size=n_dev)
        Xp_boot = {"num": Xp_dev["num"][idx],
                   "family": Xp_dev["family"].iloc[idx].reset_index(drop=True)}
        fit_model(model, Xp_boot, y_dev[idx])
        base = predict_model(model, Xp_held)
        resid = y_dev[idx] - predict_model(model, Xp_boot)
        r_idx = rng.integers(0, n_dev, size=Xp_held["num"].shape[0])
        preds.append(base + resid[r_idx])
    arr = np.stack(preds)  # (n, n_held, 5)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    return lo, hi


def coverage_rate(lo, hi, y_true):
    y_true = np.asarray(y_true)
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def bootstrap_stats(lo, hi, y_true, out_dir=TABLES_DIR):
    y_true = np.asarray(y_true)
    rows = []
    for i, col in enumerate(TARGET_COLS):
        width = hi[:, i] - lo[:, i]
        rows.append({
            "Mode": col,
            "Mean_CI_Width_Hz": float(width.mean()),
            "Median_CI_Width_Hz": float(np.median(width)),
            "Std_CI_Width_Hz": float(width.std()),
            "Coverage_95pct": coverage_rate(lo[:, [i]], hi[:, [i]], y_true[:, [i]]),
            "Mean_Pred_Std_Hz": float((width / (2 * 1.96)).mean()),
        })
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bootstrap_stats.csv"
    df.round(4).to_csv(path, index=False)
    return path


def plot_bootstrap_ci(pred_mean, lo, hi, y_true, out_dir=None):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(10, 6))
    n = min(200, len(y_true))
    idx = np.argsort(pred_mean[:n, 0])[:n]
    x = np.arange(n)
    ax.fill_between(x, lo[idx, 0], hi[idx, 0], alpha=0.3, label="95% CI")
    ax.plot(x, pred_mean[idx, 0], "k.", ms=3, label="prediction")
    ax.plot(x, y_true[idx, 0], "r.", ms=2, label="actual")
    ax.set_xlabel("Sorted held-out samples"); ax.set_ylabel("B1 (Hz)")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "bootstrap_ci.png", out_dir)


def plot_coverage(lo, hi, y_true, out_dir=None):
    y_true = np.asarray(y_true)
    inside = (y_true[:, 0] >= lo[:, 0]) & (y_true[:, 0] <= hi[:, 0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.arange(len(y_true)), y_true[:, 0], c=inside, cmap="bwr", s=10)
    ax.set_xlabel("Held-out case"); ax.set_ylabel("B1 (Hz)")
    ax.set_title(f"95% CI coverage: {inside.mean():.1%}")
    fig.tight_layout()
    return save_fig(fig, "coverage_analysis.png", out_dir)
