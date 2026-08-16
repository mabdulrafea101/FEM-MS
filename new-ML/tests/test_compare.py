import numpy as np
import pandas as pd
from pipeline.models import build_models, fit_model, predict_model
from pipeline.compare import (plot_model_comparison, plot_prediction_vs_actual,
                              plot_residuals, plot_per_mode_metrics)


def _tiny_setup():
    rng = np.random.default_rng(1)
    n = 60
    X = {"num": rng.uniform(0, 1, (n, 7)),
         "family": pd.Series(["FF"] * 30 + ["SS"] * 30)}
    y = np.column_stack([5 * X["num"][:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    models = build_models()
    for m in models.values():
        fit_model(m, X, y)
    return models, X, y


def test_plot_model_comparison(tmp_path):
    df = pd.DataFrame({"Model": ["A", "B"], "Test_MAE": [1.0, 2.0],
                       "Test_RMSE": [2.0, 3.0], "Test_R2": [0.9, 0.8],
                       "Train_Time_s": [0.1, 0.2]})
    assert plot_model_comparison(df, tmp_path).name == "model_comparison.png"


def test_plot_prediction_vs_actual(tmp_path):
    models, X, y = _tiny_setup()
    assert plot_prediction_vs_actual(models, X, y, tmp_path).name == "prediction_vs_actual.png"


def test_plot_residuals(tmp_path):
    models, X, y = _tiny_setup()
    assert plot_residuals(models, X, y, tmp_path).name == "residual_plots.png"


def test_plot_per_mode_metrics(tmp_path):
    models, X, y = _tiny_setup()
    path = plot_per_mode_metrics("CatBoost", models, X, y, tmp_path)
    assert path.name == "per_mode_metrics.png"
    assert (tmp_path / "per_mode_metrics.csv").exists()
