import numpy as np
import pandas as pd
from pipeline.models import build_models, fit_model
from pipeline.importance import (permutation_importance,
                                 plot_permutation_importance,
                                 save_importance_table, plot_shap)


def _tiny_setup():
    rng = np.random.default_rng(3)
    n = 120
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([10 * Xnum[:, 0] + Xnum[:, 1] + rng.normal(0, 0.1, n)
                         for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * 60 + ["SS"] * 60)}
    model = build_models()["Random Forest"]
    fit_model(model, X, y)
    return model, X, y


def test_permutation_importance_ranks_dominant_feature_first():
    model, X, y = _tiny_setup()
    imp = permutation_importance(model, X, y, n_repeats=3)
    assert len(imp) == 8
    assert list(imp)[0] == "L_mm"  # dominant linear feature


def test_importance_plots_and_table(tmp_path):
    model, X, y = _tiny_setup()
    imp = permutation_importance(model, X, y, n_repeats=2)
    assert plot_permutation_importance(imp, tmp_path).name == "feature_importance.png"
    assert save_importance_table(imp, tmp_path).name == "feature_importance.csv"


def test_plot_shap_runs(tmp_path):
    model, X, y = _tiny_setup()
    path = plot_shap(model, X, tmp_path, max_display=5)
    assert path.name == "shap_summary.png"
