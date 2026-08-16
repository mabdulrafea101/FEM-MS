import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from pipeline.config import TABLES_DIR
from pipeline.plots import save_fig


def friedman_test(metric_matrix):
    """metric_matrix: rows = models, cols = metrics (raw values, lower = better).

    Models are ranked within each metric column; the Friedman chi2 is computed
    from those ranks as chi2 = n_metrics*(n_models-1)*Kendall's W, with
    n_models-1 degrees of freedom, so all outputs are mutually consistent.
    """
    metric_matrix = np.asarray(metric_matrix, dtype=float)
    ranks = np.apply_along_axis(stats.rankdata, 0, metric_matrix)
    n_models, n_metrics = ranks.shape
    row_sums = ranks.sum(axis=1)
    mean_rank = n_metrics * (n_models + 1) / 2.0
    S = np.sum((row_sums - mean_rank) ** 2)
    w = 12.0 * S / (n_metrics**2 * (n_models**3 - n_models))
    chi2 = n_metrics * (n_models - 1) * w
    p = stats.chi2.sf(chi2, df=n_models - 1)
    return {"chi2": float(chi2), "p": float(p), "kendall_w": float(w)}


def anova_family(drop_df, modes=("B1", "B2", "B3")):
    out = {}
    for m in modes:
        sub = drop_df[drop_df["mode"] == m]
        ff = sub.loc[sub["family"] == "FF", "drop_pct"].to_numpy()
        ss = sub.loc[sub["family"] == "SS", "drop_pct"].to_numpy()
        if len(ff) == 0 or len(ss) == 0:
            continue
        f, p = stats.f_oneway(ff, ss)
        n, k = len(ff) + len(ss), 2
        grand = np.concatenate([ff, ss]).mean()
        ss_between = len(ff) * (ff.mean() - grand) ** 2 + len(ss) * (ss.mean() - grand) ** 2
        ss_total = np.sum((np.concatenate([ff, ss]) - grand) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        out[m] = {"F": float(f), "p": float(p), "eta2": float(eta2)}
    return out


def plot_anova_family(anova_result, drop_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in ["FF", "SS"]:
        means = [drop_df[(drop_df["family"] == fam) & (drop_df["mode"] == m)]["drop_pct"].mean()
                 for m in anova_result]
        ax.bar([f"{m} {fam}" for m in anova_result], means, alpha=0.7)
    ax.set_ylabel("Mean crack-induced drop (%)")
    ax.set_title("ANOVA: family effect on frequency drop (FF vs SS)")
    fig.tight_layout()
    return save_fig(fig, "anova_family_drop.png", out_dir)


def plot_friedman(friedman_result, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5,
            f"Friedman test\nchi2 = {friedman_result['chi2']:.2f}\n"
            f"p = {friedman_result['p']:.4f}\n"
            f"Kendall W = {friedman_result['kendall_w']:.3f}",
            ha="center", va="center", fontsize=13)
    ax.axis("off")
    return save_fig(fig, "model_comparison_friedman.png", out_dir)


def save_stats_summary(friedman_result, anova_result, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"test": "Friedman", **{f"friedman_{k}": v for k, v in friedman_result.items()}}]
    for m, vals in anova_result.items():
        rows.append({"test": f"ANOVA_{m}", **vals})
    df = pd.DataFrame(rows)
    path = out_dir / "statistical_tests_summary.csv"
    df.to_csv(path, index=False)
    return path
