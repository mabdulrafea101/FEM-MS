import numpy as np
import pandas as pd
import pytest
from pipeline.metrics import mode_metrics, pooled_metrics, macro_summary
from pipeline.config import TARGET_COLS


def test_pooled_metrics_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 31.0])
    m = pooled_metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(1.3333, abs=1e-3)
    assert m["RMSE"] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert m["R2"] == pytest.approx(0.97, abs=1e-3)
    assert m["MAPE"] == pytest.approx(9.4444, abs=1e-3)


def test_mode_metrics_returns_per_target_rows():
    y_true = np.column_stack([np.arange(10, 60, 10.0)] * 5)
    y_pred = y_true + 1.0
    df = mode_metrics(y_true, y_pred)
    assert list(df.index) == TARGET_COLS
    assert list(df.columns) == ["MAE", "RMSE", "R2", "MAPE"]
    assert (df["MAE"] == 1.0).all()


def test_macro_summary():
    df = pd.DataFrame({"MAE": [1.0, 3.0], "RMSE": [2.0, 4.0], "R2": [0.9, 0.7]})
    s = macro_summary(df)
    assert s["MAE"] == pytest.approx(2.0)
    assert s["R2"] == pytest.approx(0.8)
