import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import TARGET_COLS, TABLES_DIR
from pipeline.plots import save_fig

BETA_L = np.array([4.730041, 7.853205, 10.995608, 14.137166, 17.278760])


def ebt_frequencies(L, b, h, fc, rho=2400.0, n_modes=5):
    """Undamaged fixed-fixed Euler-Bernoulli frequencies (Hz)."""
    E = 4700.0 * np.sqrt(fc) * 1e6  # ACI 318-19, Pa
    I = b * h**3 / 12.0
    A = b * h
    base = np.sqrt(E * I / (rho * A))
    return BETA_L[:n_modes] ** 2 / (2.0 * np.pi * L**2) * base


def crack_drop_pct(df):
    """Per case x mode: ANSYS (cracked) vs EBT (pristine) frequency drop %."""
    rows = []
    for _, r in df.iterrows():
        ebt = ebt_frequencies(r["L_mm"] / 1000.0, r["b_mm"] / 1000.0,
                              r["h_mm"] / 1000.0, r["fc_MPa"])
        for i, col in enumerate(TARGET_COLS):
            rows.append({"case_id": r["case_id"], "family": r["family"],
                         "mode": f"B{i+1}", "ansys_hz": r[col],
                         "ebt_hz": ebt[i],
                         "drop_pct": (1.0 - r[col] / ebt[i]) * 100.0})
    return pd.DataFrame(rows)


def log_log_slope(df):
    x = np.log(df["L_mm"].to_numpy())
    y = np.log(df["f1_hz"].to_numpy())
    return float(np.polyfit(x, y, 1)[0])


def plot_ebt_validation(drop_df, out_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam in ["FF", "SS"]:
        sub = drop_df[drop_df["family"] == fam]
        axes[0].boxplot([sub.loc[sub["mode"] == m, "drop_pct"] for m in ["B1", "B2", "B3", "B4", "B5"]],
                        labels=["B1", "B2", "B3", "B4", "B5"], widths=0.6)
        axes[1].hist(sub["drop_pct"], bins=40, alpha=0.5, label=fam)
    axes[0].set_title("Crack-induced frequency drop by mode")
    axes[0].set_ylabel("Drop (%)")
    axes[1].set_title("Drop distribution per family")
    axes[1].set_xlabel("Drop (%)")
    axes[1].legend()
    fig.tight_layout()
    return save_fig(fig, "ebt_validation.png", out_dir)


def plot_mode_ratios(df, out_dir=None):
    ratios = pd.DataFrame({
        "f2/f1": df["f2_hz"] / df["f1_hz"],
        "f3/f1": df["f3_hz"] / df["f1_hz"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ratios["f2/f1"], bins=50, alpha=0.7)
    axes[0].axvline(2.7566, color="r", ls="--", label="pristine 2.757")
    axes[0].set_xlabel("f2/f1"); axes[0].legend()
    axes[1].hist(ratios["f3/f1"], bins=50, alpha=0.7)
    axes[1].axvline(5.4039, color="r", ls="--", label="pristine 5.404")
    axes[1].set_xlabel("f3/f1"); axes[1].legend()
    fig.suptitle("Mode-ratio distributions (deviation = damage)")
    fig.tight_layout()
    return save_fig(fig, "mode_ratios.png", out_dir)


def plot_crack_drop(drop_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in ["FF", "SS"]:
        sub = drop_df[drop_df["family"] == fam]
        for m in ["B1", "B5"]:
            s = sub[sub["mode"] == m]
            ax.scatter(s["ansys_hz"], s["drop_pct"], s=6, alpha=0.5,
                       label=f"{fam} {m}")
    ax.set_xlabel("ANSYS frequency (Hz)")
    ax.set_ylabel("Drop vs pristine EBT (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "crack_drop_vs_depth.png", out_dir)


def save_ebt_table(drop_df, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = drop_df.groupby(["family", "mode"])["drop_pct"].agg(
        ["mean", "std"]).round(2).reset_index()
    path = out_dir / "ebt_validation_summary.csv"
    summary.to_csv(path, index=False)
    return path
