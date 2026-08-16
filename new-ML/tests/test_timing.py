import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline.timing import measure_inference, timing_table, plot_timing


def test_measure_inference_returns_microseconds():
    rng = np.random.default_rng(0)
    X = {"num": rng.uniform(0, 1, (100, 7)),
         "family": pd.Series(["FF"] * 50 + ["SS"] * 50)}
    y = np.column_stack([X["num"][:, 0] * 3 for _ in range(5)])
    model = LinearRegression().fit(X["num"], y)
    us = measure_inference(model, X)
    assert us > 0
    assert us < 5000  # sane bound: under 5 ms/case


def test_timing_table_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    X = {"num": rng.uniform(0, 1, (50, 7)),
         "family": pd.Series(["FF"] * 25 + ["SS"] * 25)}
    y = np.column_stack([X["num"][:, 0] for _ in range(5)])
    model = LinearRegression().fit(X["num"], y)
    results_df = pd.DataFrame({"Model": ["Linear Regression"],
                               "Train_Time_s": [0.01]})
    table_path = timing_table(results_df, {"Linear Regression": model}, X,
                              apdl_seconds=360.0, out_dir=tmp_path)
    assert table_path.name == "timing_comparison.csv"
    df = pd.read_csv(table_path)
    assert df.loc[0, "Speedup_vs_APDL"] > 1
    assert plot_timing(df, tmp_path).name == "timing_comparison.png"
