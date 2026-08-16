import numpy as np
import pandas as pd
from pipeline.prepare import select_features, split_data, make_xy, Preprocessor
from pipeline.config import FEATURE_COLS, FAMILY_COL, TARGET_COLS, DEV_SIZE


def _sample_df(n=100):
    rng = np.random.default_rng(0)
    ff = pd.DataFrame({"case_id": [f"FF-{i}" for i in range(n // 2)],
                       "combination_code": ["FF"] * (n // 2),
                       "L_mm": rng.uniform(3250, 8000, n // 2),
                       "b_mm": rng.uniform(250, 400, n // 2),
                       "h_mm": rng.uniform(325, 700, n // 2),
                       "fc_MPa": rng.uniform(25, 45, n // 2),
                       "rho_percent": rng.uniform(0.8, 2.0, n // 2),
                       "crack1_depth_mm": rng.uniform(50, 350, n // 2),
                       "crack2_depth_mm": rng.uniform(50, 350, n // 2),
                       "crack1_angle_deg": 90.0, "crack2_angle_deg": 90.0,
                       "Ec_MPa": 25000.0, "preanalysis_qc": "PASS"})
    ss = ff.copy()
    ss["case_id"] = [f"SS-{i}" for i in range(n // 2)]
    ss["combination_code"] = "SS"
    ss["crack1_angle_deg"] = 45.0
    ss["crack2_angle_deg"] = 135.0
    df = pd.concat([ff, ss], ignore_index=True)
    df["family"] = df["combination_code"]
    for i, col in enumerate(TARGET_COLS):
        df[col] = 50.0 * (i + 1) + rng.uniform(-2, 2, n)
    return df


def test_select_features_keeps_only_8_plus_targets():
    df = _sample_df()
    out = select_features(df)
    expected = FEATURE_COLS + [FAMILY_COL] + TARGET_COLS
    assert list(out.columns) == expected
    assert "Ec_MPa" not in out.columns
    assert "bend_1_mode" not in out.columns


def test_split_data_800_200():
    df = _sample_df(1000)
    dev, held = split_data(df)
    assert len(dev) == DEV_SIZE and len(held) == 1000 - DEV_SIZE
    assert not dev.index.intersection(held.index).size


def test_split_is_stratified_by_family():
    df = _sample_df(1000)
    dev, held = split_data(df)
    assert (dev["family"].value_counts().to_dict() ==
            {"FF": 400, "SS": 400})
    assert (held["family"].value_counts().to_dict() ==
            {"FF": 100, "SS": 100})


def test_make_xy_shapes():
    df = _sample_df(100)
    X, y = make_xy(select_features(df))
    assert X.shape == (100, 8)
    assert y.shape == (100, 5)


def test_preprocessor_onehot():
    df = _sample_df(100)
    X, _ = make_xy(select_features(df))
    prep = Preprocessor(family_mode="onehot").fit(X)
    out = prep.transform(X)
    assert out["num"].shape == (100, 8)  # 7 scaled + 1 one-hot
    assert set(out["family"]) == {"FF", "SS"}


def test_preprocessor_native():
    df = _sample_df(100)
    X, _ = make_xy(select_features(df))
    prep = Preprocessor(family_mode="native").fit(X)
    out = prep.transform(X)
    assert out["num"].shape == (100, 7)  # scaled numerics only
    assert list(out["family"].iloc[:2]) in (["FF", "FF"], ["SS", "SS"], ["FF", "SS"], ["SS", "FF"])
