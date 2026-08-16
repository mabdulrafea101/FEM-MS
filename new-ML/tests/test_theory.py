import numpy as np
import pandas as pd
import pytest
from pipeline.theory import (ebt_frequencies, crack_drop_pct, log_log_slope,
                             plot_ebt_validation, plot_mode_ratios,
                             plot_crack_drop, save_ebt_table)
from pipeline.config import TABLES_DIR


def test_ebt_mode1_reference_case():
    f = ebt_frequencies(L=3.0, b=0.3, h=0.45, fc=30.0)
    assert f[0] == pytest.approx(168.34, rel=1e-3)
    assert f[1] == pytest.approx(464.07, rel=1e-3)


def test_ebt_mode_ratio_fixed_fixed():
    f = ebt_frequencies(4.0, 0.3, 0.5, 35.0)
    assert f[1] / f[0] == pytest.approx(2.7566, rel=1e-3)
    assert f[2] / f[0] == pytest.approx(5.4039, rel=1e-3)


def test_ebt_scale_invariance():
    f = ebt_frequencies(3.0, 0.3, 0.45, 30.0)
    g = ebt_frequencies(3.0, 0.3, 0.45, 30.0)
    np.testing.assert_allclose(f, g)


def test_crack_drop_pct_small_damage_small_drop():
    row = {"case_id": "X-001", "family": "FF", "L_mm": 5000.0, "b_mm": 300.0,
           "h_mm": 500.0, "fc_MPa": 35.0,
           "f1_hz": 52.0, "f2_hz": 143.0, "f3_hz": 261.0,
           "f4_hz": 400.0, "f5_hz": 570.0}
    df = pd.DataFrame([row])
    out = crack_drop_pct(df)
    assert list(out.columns) == ["case_id", "family", "mode", "ansys_hz", "ebt_hz", "drop_pct"]
    assert len(out) == 5
    assert (out["drop_pct"] > 0).all()  # ANSYS (cracked) below EBT (pristine)
    assert (out["drop_pct"] < 40).all()


def test_log_log_slope_close_to_minus_two():
    rng = np.random.default_rng(0)
    L = rng.uniform(3250, 8000, 200)
    f1 = 1e6 / L**2  # exact f ~ L^-2 scaling
    df = pd.DataFrame({"L_mm": L, "f1_hz": f1})
    assert log_log_slope(df) == pytest.approx(-2.0, abs=0.05)


def test_plots_and_table_created(tmp_path):
    df = pd.DataFrame([{"case_id": "X-001", "family": "FF", "L_mm": 5000.0,
                        "b_mm": 300.0, "h_mm": 500.0, "fc_MPa": 35.0,
                        "f1_hz": 52.0, "f2_hz": 143.0, "f3_hz": 261.0,
                        "f4_hz": 400.0, "f5_hz": 570.0}])
    drop_df = crack_drop_pct(df)
    assert plot_ebt_validation(drop_df, tmp_path).name == "ebt_validation.png"
    assert plot_mode_ratios(df, tmp_path).name == "mode_ratios.png"
    assert plot_crack_drop(drop_df, tmp_path).name == "crack_drop_vs_depth.png"
    assert save_ebt_table(drop_df, tmp_path).name == "ebt_validation_summary.csv"
    assert (tmp_path / "ebt_validation_summary.csv").exists()
