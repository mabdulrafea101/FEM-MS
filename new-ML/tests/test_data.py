import pandas as pd
import pytest
from pipeline.data import load_dataset, run_qc
from pipeline.config import DATA_PATH, TARGET_COLS


def test_load_dataset_shape_and_family():
    df = load_dataset()
    assert df.shape == (1000, 41)  # 40 original columns + family
    assert set(df["family"].unique()) == {"FF", "SS"}
    assert df["family"].value_counts().to_dict() == {"FF": 500, "SS": 500}


def test_load_dataset_has_targets():
    df = load_dataset()
    for col in TARGET_COLS:
        assert col in df.columns
        assert (df[col] > 0).all()


def test_run_qc_passes_on_real_data():
    df = load_dataset()
    checks = run_qc(df)
    assert checks["total_cases"] == 1000
    assert checks["missing_values"] == 0
    assert checks["qc_all_pass"] is True


def test_run_qc_raises_on_bad_data():
    df = pd.DataFrame({"case_id": [1, 2], "preanalysis_qc": ["PASS", "PASS"],
                       "family": ["FF", "FF"]})
    with pytest.raises(AssertionError):
        run_qc(df)
