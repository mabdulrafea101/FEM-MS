import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from pipeline.models import (build_models, fit_model, predict_model, run_cv,
                             fit_and_evaluate)
from pipeline.config import SEED, N_FOLDS


def _synthetic(seed=SEED, n=200):
    rng = np.random.default_rng(seed)
    X_num = rng.uniform(0, 1, (n, 7))
    X = {"num": X_num, "family": pd.Series(["FF"] * (n // 2) + ["SS"] * (n // 2))}
    y = np.column_stack([10 * X_num[:, 0] + 2 * X_num[:, 1] + rng.normal(0, 0.1, n)
                         for _ in range(5)])
    return X, y


def test_build_models_returns_five():
    models = build_models()
    assert list(models) == ["Linear Regression", "Random Forest", "XGBoost",
                            "CatBoost", "SVR"]


def test_fit_predict_shape():
    X, y = _synthetic()
    models = build_models()
    for name, model in models.items():
        fit_model(model, X, y)
        pred = predict_model(model, X)
        assert pred.shape == (200, 5), name
        assert np.isfinite(pred).all(), name


def test_run_cv_returns_expected_keys():
    X, y = _synthetic()
    model = build_models()["Linear Regression"]
    res = run_cv(model, X, y)
    assert set(res) == {"RMSE_mean", "RMSE_std", "folds"}
    assert len(res["folds"]) == N_FOLDS


def test_catboost_and_linear_both_fit_synthetic_data():
    X, y = _synthetic()
    cb = build_models()["CatBoost"]
    lr = build_models()["Linear Regression"]
    assert run_cv(cb, X, y)["RMSE_mean"] < 1.0
    assert run_cv(lr, X, y)["RMSE_mean"] < 1.0


def test_fit_and_evaluate_returns_table(tmp_path, monkeypatch):
    X, y = _synthetic()
    models = build_models()
    X_train = {"num": X["num"][:150], "family": X["family"][:150]}
    y_train = y[:150]
    X_test = {"num": X["num"][150:], "family": X["family"][150:]}
    y_test = y[150:]
    df = fit_and_evaluate(models, X_train, y_train, X_test, y_test)
    assert len(df) == 5
    assert {"Model", "Test_R2", "Train_Time_s"}.issubset(df.columns)
