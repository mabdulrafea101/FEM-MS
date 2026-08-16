import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.models import predict_model
from pipeline.metrics import mode_metrics, macro_summary
from pipeline.config import TARGET_COLS
from pipeline.plots import save_fig


def plot_model_comparison(results_df, out_dir=None):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names = results_df["Model"]
    axes[0].bar(names, results_df["Test_MAE"]); axes[0].set_title("Test MAE (Hz)")
    axes[1].bar(names, results_df["Test_RMSE"]); axes[1].set_title("Test RMSE (Hz)")
    axes[2].bar(names, results_df["Test_R2"]); axes[2].set_title("Test R2")
    axes[3].bar(names, results_df["Train_Time_s"]); axes[3].set_title("Train time (s)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_fig(fig, "model_comparison.png", out_dir)


def plot_prediction_vs_actual(models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pred = predict_model(model, Xp_test)[:, 0]
        ax.scatter(y_test[:, 0], pred, s=8, alpha=0.5)
        lim = [min(y_test[:, 0].min(), pred.min()),
               max(y_test[:, 0].max(), pred.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("Actual B1 (Hz)"); ax.set_ylabel("Predicted B1 (Hz)")
    fig.tight_layout()
    return save_fig(fig, "prediction_vs_actual.png", out_dir)


def plot_residuals(models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pred = predict_model(model, Xp_test)
        resid = y_test[:, 0] - pred[:, 0]
        ax.scatter(pred[:, 0], resid, s=8, alpha=0.5)
        ax.axhline(0, color="r", lw=1)
        ax.set_title(f"{name} (B1)")
        ax.set_xlabel("Predicted (Hz)"); ax.set_ylabel("Residual (Hz)")
    fig.tight_layout()
    return save_fig(fig, "residual_plots.png", out_dir)


def plot_per_mode_metrics(best_name, models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    pred = predict_model(models[best_name], Xp_test)
    per_mode = mode_metrics(y_test, pred)
    macro = macro_summary(per_mode)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(per_mode)), per_mode["R2"])
    ax.set_xticks(range(len(per_mode)))
    ax.set_xticklabels(TARGET_COLS)
    ax.axhline(macro["R2"], color="r", ls="--", label=f"macro R2 = {macro['R2']:.4f}")
    ax.set_title(f"{best_name} — per-mode R2")
    ax.legend()
    fig.tight_layout()
    out_dir = Path(out_dir) if out_dir else None
    fig_path = save_fig(fig, "per_mode_metrics.png", out_dir)
    csv_dir = Path(out_dir) if out_dir else Path("outputs/tables")
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_mode.round(4).to_csv(csv_dir / "per_mode_metrics.csv")
    return fig_path
