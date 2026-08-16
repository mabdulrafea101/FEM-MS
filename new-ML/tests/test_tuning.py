import numpy as np
import pandas as pd
from pipeline.tuning import (CATBOOST_GRID, evaluate_candidates, paired_ttest,
                             plot_hyperparameter_importance,
                             plot_hyperparam_ttest, save_hyperparam_table)


def test_grid_has_five_named_candidates():
    assert len(CATBOOST_GRID) == 5
    assert all("name" in c and "params" in c for c in CATBOOST_GRID)


def test_evaluate_candidates_returns_table():
    rng = np.random.default_rng(0)
    n = 150
    X = {"num": rng.uniform(0, 1, (n, 7)),
         "family": pd.Series(["FF"] * 75 + ["SS"] * 75)}
    y = np.column_stack([5 * X["num"][:, 0] + rng.normal(0, 0.1, n) for _ in range(5)])
    df = evaluate_candidates(CATBOOST_GRID[:2], X, y, n_folds=3)
    assert list(df["name"]) == ["default", CATBOOST_GRID[1]["name"]]
    assert {"name", "RMSE_mean", "RMSE_std"}.issubset(df.columns)


def test_paired_ttest_detects_improvement():
    rng = np.random.default_rng(0)
    default = rng.normal(5.0, 0.3, 5)
    best = rng.normal(4.0, 0.3, 5)
    res = paired_ttest(default, best)
    assert res["p"] < 0.05
    assert res["cohens_d"] > 0


def test_plots_and_table(tmp_path):
    grid_results = pd.DataFrame({"name": ["a", "b"], "RMSE_mean": [1.0, 0.5],
                                 "RMSE_std": [0.1, 0.2],
                                 "depth": [6, 8], "learning_rate": [0.1, 0.05],
                                 "iterations": [100, 200]})
    ttest_result = {"t": 4.43, "p": 0.011, "cohens_d": 1.98}
    assert plot_hyperparameter_importance(grid_results, tmp_path).name == "hyperparameter_importance.png"
    assert plot_hyperparam_ttest(ttest_result, tmp_path).name == "hyperparam_ttest.png"
    assert save_hyperparam_table(grid_results, ttest_result, tmp_path).name == "hyperparam_comparison.csv"
