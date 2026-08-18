import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from catboost import CatBoostRegressor
from pipeline.config import SEED, N_FOLDS, TABLES_DIR
from pipeline.metrics import pooled_metrics


def build_models(seed=SEED):
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=seed),
        "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=seed)),
        "CatBoost": CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=6,
            loss_function="MultiRMSE", random_state=seed, verbose=False,
            allow_writing_files=False, thread_count=2),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma="scale")),
    }


def _catboost_df(Xp):
    df = pd.DataFrame(Xp["num"])
    df["family"] = Xp["family"].to_numpy()
    return df


def fit_model(model, Xp, y):
    if isinstance(model, CatBoostRegressor):
        model.fit(_catboost_df(Xp), y, cat_features=["family"])
    else:
        model.fit(Xp["num"], y)


def predict_model(model, Xp):
    if isinstance(model, CatBoostRegressor):
        return np.asarray(model.predict(_catboost_df(Xp)))
    return np.asarray(model.predict(Xp["num"]))


def run_cv(model, Xp, y, n_folds=N_FOLDS, seed=SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rmse = []
    y = np.asarray(y)
    for tr_idx, va_idx in kf.split(Xp["num"]):
        Xp_tr = {"num": Xp["num"][tr_idx], "family": Xp["family"].iloc[tr_idx]}
        Xp_va = {"num": Xp["num"][va_idx], "family": Xp["family"].iloc[va_idx]}
        fit_model(model, Xp_tr, y[tr_idx])
        pred = predict_model(model, Xp_va)
        rmse.append(pooled_metrics(y[va_idx], pred)["RMSE"])
    return {"RMSE_mean": float(np.mean(rmse)),
            "RMSE_std": float(np.std(rmse)),
            "folds": rmse}


def fit_and_evaluate(models, Xp_train, y_train, Xp_test, y_test,
                     out_dir=TABLES_DIR):
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        fit_model(model, Xp_train, y_train)
        train_time = time.perf_counter() - t0
        tr = pooled_metrics(y_train, predict_model(model, Xp_train))
        te = pooled_metrics(y_test, predict_model(model, Xp_test))
        rows.append({"Model": name,
                     "Train_MAE": tr["MAE"], "Train_RMSE": tr["RMSE"],
                     "Train_R2": tr["R2"], "Train_MAPE": tr["MAPE"],
                     "Test_MAE": te["MAE"], "Test_RMSE": te["RMSE"],
                     "Test_R2": te["R2"], "Test_MAPE": te["MAPE"],
                     "Train_Time_s": train_time})
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    return df
