import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline.config import TARGET_COLS


def _metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100.0,
    }


def mode_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = {}
    for i, col in enumerate(TARGET_COLS):
        rows[col] = _metrics(y_true[:, i], y_pred[:, i])
    return pd.DataFrame(rows).T


def pooled_metrics(y_true, y_pred):
    return _metrics(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())


def macro_summary(mode_df):
    return {col: float(mode_df[col].mean()) for col in mode_df.columns}
