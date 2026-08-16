import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline.learning import (compute_learning_curve, plot_learning_curve,
                               save_learning_curve, extrapolation_test,
                               plot_extrapolation)


def _linear_setup(n=400):
    rng = np.random.default_rng(0)
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([3 * Xnum[:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * (n // 2) + ["SS"] * (n // 2))}
    return X, y


def test_learning_curve_returns_sizes():
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 0.5, 1.0), n_folds=3)
    assert list(lc["size"]) == [0.25, 0.5, 1.0]
    assert {"size", "train_rmse", "val_rmse"}.issubset(lc.columns)


def test_learning_curve_improves_with_size():
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 1.0), n_folds=3)
    assert lc.iloc[1]["val_rmse"] < lc.iloc[0]["val_rmse"]


def test_extrapolation_sane():
    X, y = _linear_setup()
    model = LinearRegression()
    half = len(y) // 2
    Xs = {"num": X["num"][:half], "family": X["family"][:half]}
    Xl = {"num": X["num"][half:], "family": X["family"][half:]}
    res = extrapolation_test(model, Xs, y[:half], Xl, y[half:])
    assert {"train_rmse", "extrap_rmse", "extrap_r2"}.issubset(res)
    assert res["extrap_r2"] > 0.5


def test_plots(tmp_path):
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 0.5, 1.0), n_folds=3)
    assert plot_learning_curve(lc, tmp_path).name == "learning_curve_analysis.png"
    assert save_learning_curve(lc, tmp_path).name == "learning_curve_results.csv"
    half = len(y) // 2
    res = extrapolation_test(model, {"num": X["num"][:half], "family": X["family"][:half]},
                             y[:half], {"num": X["num"][half:], "family": X["family"][half:]},
                             y[half:])
    assert plot_extrapolation(res, tmp_path).name == "extrapolation_test.png"
