import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline.uncertainty import (bootstrap_predictions, coverage_rate,
                                  bootstrap_stats, plot_bootstrap_ci,
                                  plot_coverage)


def test_coverage_rate_near_95_on_linear_data():
    rng = np.random.default_rng(42)
    n = 300
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([3 * Xnum[:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * 150 + ["SS"] * 150)}
    model = LinearRegression()
    lo, hi = bootstrap_predictions(model, X, y, X, n=30)
    assert lo.shape == (n, 5) and hi.shape == (n, 5)
    assert (lo <= hi).all()
    rate = coverage_rate(lo, hi, y)
    assert 0.85 <= rate <= 1.0


def test_bootstrap_stats_and_plots(tmp_path):
    rng = np.random.default_rng(1)
    n = 100
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([2 * Xnum[:, 0] + rng.normal(0, 0.3, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * 50 + ["SS"] * 50)}
    model = LinearRegression()
    lo, hi = bootstrap_predictions(model, X, y, X, n=20)
    stats_path = bootstrap_stats(lo, hi, y, out_dir=tmp_path)
    assert stats_path.name == "bootstrap_stats.csv"
    df = pd.read_csv(stats_path)
    assert len(df) == 5
    assert plot_bootstrap_ci((lo + hi) / 2, lo, hi, y, tmp_path).name == "bootstrap_ci.png"
    assert plot_coverage(lo, hi, y, tmp_path).name == "coverage_analysis.png"
