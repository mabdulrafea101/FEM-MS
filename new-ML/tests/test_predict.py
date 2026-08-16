import json
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pipeline.predict import FrequencyPredictor, save_artifacts, das_benchmark_table, DOMAIN


def _fitted_predictor():
    rng = np.random.default_rng(0)
    n = 200
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([Xnum[:, 0] * 100 for _ in range(5)])
    scaler = StandardScaler().fit(Xnum)
    Xs = scaler.transform(Xnum)
    model = LinearRegression().fit(Xs, y)
    return FrequencyPredictor(model, scaler)


def test_predict_accepts_valid_input():
    pred = _fitted_predictor()
    inputs = {"L_mm": 5000.0, "b_mm": 300.0, "h_mm": 500.0, "fc_MPa": 35.0,
              "rho_percent": 1.2, "crack1_depth_mm": 200.0,
              "crack2_depth_mm": 200.0, "family": "FF"}
    out = pred.predict(inputs)
    assert out.shape == (5,)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("override", [
    {"L_mm": 9000.0}, {"fc_MPa": 60.0}, {"family": "FX"},
    {"crack1_depth_mm": -5.0}])
def test_predict_rejects_out_of_domain(override):
    pred = _fitted_predictor()
    inputs = {"L_mm": 5000.0, "b_mm": 300.0, "h_mm": 500.0, "fc_MPa": 35.0,
              "rho_percent": 1.2, "crack1_depth_mm": 200.0,
              "crack2_depth_mm": 200.0, "family": "FF"}
    inputs.update(override)
    with pytest.raises(ValueError):
        pred.predict(inputs)


def test_predict_rejects_missing_key():
    pred = _fitted_predictor()
    with pytest.raises(ValueError):
        pred.predict({"L_mm": 5000.0})


def test_save_artifacts(tmp_path):
    pred = _fitted_predictor()
    meta = save_artifacts(pred.model, pred.scaler, out_dir=tmp_path)
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "scaler.pkl").exists()
    md = json.loads((tmp_path / "feature_metadata.json").read_text())
    assert md["family_mode"] == "onehot"
    assert "domain" in md


def test_das_benchmark_table(tmp_path):
    per_mode = pd.Series([0.99] * 5, index=[f"f{i}_hz" for i in range(1, 6)])
    path = das_benchmark_table(0.99, per_mode, out_dir=tmp_path)
    assert path.name == "das_benchmark_comparison.csv"
    df = pd.read_csv(path)
    assert "Das_2023" in set(df["Reference"])
