import numpy as np
import pandas as pd
import pytest
from pipeline.stats_tests import (friedman_test, anova_family,
                                 plot_anova_family, plot_friedman,
                                 save_stats_summary)


def test_friedman_significant_for_clear_winner():
    # rows = models, cols = metrics; model A always best rank
    ranks = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                     dtype=float)
    res = friedman_test(ranks)
    assert res["p"] < 0.05          # max attainable p for 5x3 perfect separation is 0.0173
    assert 0.9 < res["kendall_w"] <= 1.0
    assert res["chi2"] == pytest.approx(12.0, abs=0.1)


def test_friedman_trivial_ranks_no_signal():
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 6, size=(5, 10)).astype(float)
    res = friedman_test(ranks)
    assert res["kendall_w"] < 0.5


def test_anova_family_detects_separation():
    rng = np.random.default_rng(0)
    rows = []
    for i, fam in enumerate(["FF", "SS"]):
        mean = 20.0 if fam == "FF" else 8.0
        for _ in range(200):
            rows.append({"family": fam, "mode": "B1",
                         "drop_pct": rng.normal(mean, 1.5)})
    drop_df = pd.DataFrame(rows)
    res = anova_family(drop_df, modes=("B1",))
    assert res["B1"]["p"] < 0.001
    assert res["B1"]["eta2"] > 0.5


def test_plots_and_table(tmp_path):
    res_f = friedman_test(np.array([[1, 2], [2, 1]], dtype=float))
    rng = np.random.default_rng(0)
    drop_df = pd.DataFrame({"family": ["FF"] * 100 + ["SS"] * 100,
                            "mode": ["B1"] * 200,
                            "drop_pct": rng.normal(10, 1, 200)})
    res_a = anova_family(drop_df, modes=("B1",))
    assert plot_anova_family(res_a, drop_df, tmp_path).name == "anova_family_drop.png"
    assert plot_friedman(res_f, tmp_path).name == "model_comparison_friedman.png"
    assert save_stats_summary(res_f, res_a, tmp_path).name == "statistical_tests_summary.csv"
