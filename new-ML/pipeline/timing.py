import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import TABLES_DIR, APDL_SOLVE_SECONDS
from pipeline.models import predict_model
from pipeline.plots import save_fig


def measure_inference(model, Xp, n_reps=5):
    """Mean inference time in microseconds per case."""
    n_cases = Xp["num"].shape[0]
    predict_model(model, Xp)  # warmup
    best = np.inf
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_model(model, Xp)
        best = min(best, (time.perf_counter() - t0) / n_cases * 1e6)
    return float(best)


def timing_table(results_df, models, Xp, apdl_seconds=APDL_SOLVE_SECONDS,
                 out_dir=TABLES_DIR):
    rows = []
    for name, model in models.items():
        train_s = float(results_df.loc[results_df["Model"] == name,
                                       "Train_Time_s"].iloc[0])
        us = measure_inference(model, Xp)
        speedup = apdl_seconds * 1e6 / us if us > 0 else np.inf
        rows.append({"Model": name, "Train_Time_s": train_s,
                     "Inference_us_per_case": us,
                     "Speedup_vs_APDL": speedup})
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path


def plot_timing(table_df, out_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(table_df["Model"], table_df["Train_Time_s"])
    axes[0].set_ylabel("Training time (s)"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(table_df["Model"], table_df["Inference_us_per_case"])
    axes[1].set_ylabel("Inference (µs/case)"); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_fig(fig, "timing_comparison.png", out_dir)
