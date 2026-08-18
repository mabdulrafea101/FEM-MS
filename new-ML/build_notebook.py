"""Build model_training_ansys.ipynb from the pipeline package (18 stages)."""
import nbformat as nbf
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NB_PATH = PROJECT / "model_training_ansys.ipynb"

HEADER = """# Prediction of Natural Frequencies of Fixed RC Beams — ANSYS Dataset

Single-notebook ML pipeline for the revised thesis. Reads the frozen 1,000-case ANSYS
dataset, validates it against Euler-Bernoulli theory, trains and compares five regression
models (multi-output, B1-B5), and writes every artifact to `outputs/`."""

STAGES = []


def stage(title, markdown, code):
    STAGES.append((title, markdown, code))


stage("1. Setup and Configuration", """
## Stage 1 — Setup and Configuration

- Working directory: `Project/`
- Seed: 42 everywhere
- All outputs: `outputs/` (figures/, tables/, models/, logs/)
""", """
import sys
from pathlib import Path
PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
import logging
logging.basicConfig(level=logging.INFO,
                    filename="outputs/logs/training.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")
from pipeline.config import (SEED, DATA_PATH, OUTPUT_DIR, FIGURES_DIR,
                             TABLES_DIR, MODELS_DIR, LOGS_DIR,
                             FEATURE_COLS, FAMILY_COL, TARGET_COLS,
                             SKIPPED_FIELDS)
for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Output dir:", OUTPUT_DIR)
print("Skipped fields documented:", len(SKIPPED_FIELDS))
""")

stage("2. Data Loading and QC", """
## Stage 2 — Data Loading and QC (Ch. 3.9 audit)

The frozen dataset: 1,000 cases (500 FF, 500 SS), all QC PASS, no missing values.
""", """
from pipeline.data import load_dataset, run_qc
df = load_dataset()
qc = run_qc(df)
print(qc)
""")

stage("3. FEM/Theoretical Validation (EBT checks)", """
## Stage 3 — FEM/Theoretical Validation

Dataset-level checks against undamaged fixed-fixed Euler-Bernoulli theory
(Eq. 2, Ch. 2.2.1). ANSYS (cracked) must lie below EBT (pristine); drop %
is the crack-induced frequency reduction; f vs L slope should be ~ -2.
""", """
from pipeline.theory import (crack_drop_pct, log_log_slope, plot_ebt_validation,
                             plot_mode_ratios, plot_crack_drop, save_ebt_table)
drop_df = crack_drop_pct(df)
slope = log_log_slope(df)
print(f"log-log slope of f1 vs L: {slope:.3f} (theory: -2.00)")
plot_ebt_validation(drop_df)
plot_mode_ratios(df)
plot_crack_drop(drop_df)
print("Saved:", save_ebt_table(drop_df))
""")

stage("4. EDA and Feature Selection", """
## Stage 4 — EDA and Feature Selection (Skipped vs Added)

**Skipped fields (with reasons):** crack locations and angles are constant within
each family (FF: 0.45L/0.55L at 90 deg; SS: 0.1L/0.9L at 45/135 deg), so they carry
no information beyond the family label; solver indices (bend_N_mode) are leakage;
Ec/As/equivalent diameter are derived; mesh/QC fields are metadata.

**Added/retained (8-field matrix):** L_mm, b_mm, h_mm, fc_MPa, rho_percent,
crack1_depth_mm, crack2_depth_mm, family.
""", """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pipeline.prepare import select_features
from pipeline.plots import save_fig

# Family-constant configuration table (for the thesis)
config_rows = []
for fam, (loc1, loc2, a1, a2) in {"FF": ("0.45L", "0.55L", 90, 90),
                                   "SS": ("0.1L", "0.9L", 45, 135)}.items():
    config_rows.append({"family": fam, "crack1_location": loc1,
                        "crack2_location": loc2, "crack1_angle_deg": a1,
                        "crack2_angle_deg": a2})
pd.DataFrame(config_rows).to_csv("outputs/tables/family_configuration.csv", index=False)

# Distribution of retained features per family
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.ravel(), FEATURE_COLS):
    for fam in ["FF", "SS"]:
        ax.hist(df.loc[df[FAMILY_COL] == fam, col], bins=30, alpha=0.5, label=fam)
    ax.set_title(col)
axes.ravel()[-1].axis("off")
fig.legend(loc="upper right")
fig.tight_layout()
save_fig(fig, "parameter_distributions.png")

# Correlation matrix (7 features + 5 targets)
corr = select_features(df).corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax, center=0)
fig.tight_layout()
save_fig(fig, "correlation_matrix.png")
print("Feature matrix:", select_features(df).shape)
""")

stage("5. Split (800/200)", """
## Stage 5 — Train/Test Split (Ch. 3.10.3)

800 development / 200 held-out, stratified by family, seed 42. The held-out set is
not examined until model selection is complete.
""", """
from pipeline.prepare import split_data, make_xy
dev, held = split_data(df)
X_dev, y_dev = make_xy(select_features(dev))
X_held, y_held = make_xy(select_features(held))
print("dev:", X_dev.shape, "held:", X_held.shape)
print(dev[FAMILY_COL].value_counts().to_dict())
""")

stage("6. Preprocessing", """
## Stage 6 — Preprocessing

Step-by-step (documented for the thesis):
1. Load 12-field raw matrix
2. Drop skipped fields (see Stage 4) -> 8-field matrix
3. Split 800/200 stratified by family (Stage 5)
4. StandardScaler fit on dev only (7 continuous features)
5. Family: one-hot for LR/RF/XGB/SVR; native categorical for CatBoost
6. Verify shapes/dtypes/no-NaN
""", """
from pipeline.prepare import Preprocessor
prep = Preprocessor(family_mode="onehot").fit(X_dev)
Xp_dev = prep.transform(X_dev)
Xp_held = prep.transform(X_held)
print("dev num:", Xp_dev["num"].shape, "held num:", Xp_held["num"].shape)
assert not np.isnan(Xp_dev["num"]).any()
import joblib
joblib.dump(prep.scaler, "outputs/models/scaler.pkl")
print("scaler saved to outputs/models/scaler.pkl")
""")

stage("7. Model Development (five algorithms)", """
## Stage 7 — Model Development (Ch. 3.10.1-3.10.2)

- Linear Regression (baseline)
- Random Forest (100 trees, unlimited depth)
- XGBoost (lr=0.1, depth=6) via MultiOutputRegressor
- CatBoost (MultiRMSE, native categorical family)
- SVR (RBF) via MultiOutputRegressor
""", """
from pipeline.models import build_models, fit_model, predict_model
models = build_models()
for name, model in models.items():
    fit_model(model, Xp_dev, y_dev.to_numpy())
    print(name, "fitted")
""")

stage("8. Evaluation and 5-Fold CV", """
## Stage 8 — Evaluation (Ch. 3.10.5)

Per-mode MAE/RMSE/R2/MAPE + macro + pooled; 5-fold CV within dev set (pooled RMSE).
Held-out is NOT used here.
""", """
from pipeline.models import run_cv, fit_and_evaluate
from pipeline.metrics import mode_metrics, macro_summary
cv_rows = {}
for name, model in models.items():
    cv = run_cv(model, Xp_dev, y_dev.to_numpy())
    cv_rows[name] = cv
    print(f"{name}: CV pooled RMSE = {cv['RMSE_mean']:.3f} +- {cv['RMSE_std']:.3f} Hz")
""")

stage("9. Model Comparison (dev-based selection)", """
## Stage 9 — Model Comparison

Rank models by pooled CV RMSE (selection on dev only). Tables and figures are
written to outputs/.
""", """
import pandas as pd
from pipeline.compare import (plot_model_comparison, plot_prediction_vs_actual,
                              plot_residuals, plot_per_mode_metrics)
results_dev = pd.DataFrame([{"Model": name, "CV_RMSE": cv["RMSE_mean"],
                             "CV_RMSE_std": cv["RMSE_std"]}
                            for name, cv in cv_rows.items()])
results_dev.sort_values("CV_RMSE").to_csv("outputs/tables/cv_ranking.csv", index=False)
print(results_dev.sort_values("CV_RMSE"))
""")

stage("10. Held-out Evaluation (once, after selection)", """
## Stage 10 — Held-out Evaluation (Ch. 3.10.3)

The selected model (lowest CV pooled RMSE) is evaluated exactly once on the 200
held-out cases. Metrics are also computed for all five models for the comparison
tables (train/dev basis), but only the selected model's held-out numbers are
reported as the headline result.
""", """
from pipeline.models import fit_and_evaluate
from pipeline.metrics import mode_metrics, macro_summary
best_name = results_dev.sort_values("CV_RMSE").iloc[0]["Model"]
print("Selected model:", best_name)
comparison = fit_and_evaluate(models, Xp_dev, y_dev.to_numpy(),
                              Xp_held, y_held.to_numpy())
comparison["CV_RMSE"] = comparison["Model"].map(lambda n: cv_rows[n]["RMSE_mean"])
comparison.to_csv("outputs/tables/model_comparison.csv", index=False)
print(comparison[["Model", "Test_MAE", "Test_RMSE", "Test_R2"]].to_string(index=False))
per_mode = mode_metrics(y_held.to_numpy(),
                        predict_model(models[best_name], Xp_held))
print("Selected model held-out per-mode metrics:"); print(per_mode.round(4))
print("Macro:", macro_summary(per_mode))
""")

stage("11. Residuals, Prediction vs Actual, Per-mode Plots", """
## Stage 11 — Residual / Scatter Analysis

Residual and prediction-vs-actual plots for all five models; per-mode metrics for
the selected model. All saved to outputs/figures/ and outputs/tables/.
""", """
plot_prediction_vs_actual(models, Xp_held, y_held.to_numpy())
plot_residuals(models, Xp_held, y_held.to_numpy())
plot_per_mode_metrics(best_name, models, Xp_held, y_held.to_numpy())
print("saved prediction_vs_actual.png, residual_plots.png, per_mode_metrics.png/csv")
""")

stage("12. Statistical Tests (Friedman + Family ANOVA)", """
## Stage 12 — Statistical Tests

- Friedman test: model performance ranking across metrics (kept from earlier phase)
- One-way ANOVA: does crack-induced frequency drop (%) differ between FF and SS
  families per mode? (Replaces the old damage-location ANOVA: crack locations are
  fixed per family, so location itself is not testable.)
""", """
from pipeline.stats_tests import (friedman_test, anova_family,
                                 plot_anova_family, plot_friedman,
                                 save_stats_summary)
import numpy as np
# Friedman: rows = models, cols = metrics (lower is better). R2 is negated so
# that all three columns share the same "lower is better" direction.
metric_matrix = np.column_stack([
    comparison["Test_MAE"].to_numpy(),
    comparison["Test_RMSE"].to_numpy(),
    -comparison["Test_R2"].to_numpy(),
])
friedman = friedman_test(metric_matrix)
print("Friedman:", friedman)
anova_res = anova_family(drop_df)
print("ANOVA:", anova_res)
plot_anova_family(anova_res, drop_df)
plot_friedman(friedman)
save_stats_summary(friedman, anova_res)
""")

stage("13. Computational Time", """
## Stage 13 — Computational Time

Training time per model and inference time per case, compared against the wall-clock
cost of one automated APDL solve (configurable; default 360 s/case from the earlier
phase — replace with the measured value if available).
""", """
from pipeline.timing import timing_table, plot_timing
tt = timing_table(comparison, models, Xp_held)
plot_timing(pd.read_csv("outputs/tables/timing_comparison.csv"))
print(tt)
""")

stage("14. Bootstrap Uncertainty (Ch. 3.10.6)", """
## Stage 14 — Bootstrap Uncertainty (Ch. 3.10.6)

Selected model only, after selection. 100 bootstrap resamples of the 800-case dev
set; 95% CI around each of the 200 held-out predictions (per mode); empirical
coverage compared against nominal 95%.
""", """
from pipeline.uncertainty import (bootstrap_predictions, bootstrap_stats,
                                  plot_bootstrap_ci, plot_coverage)
lo, hi = bootstrap_predictions(models[best_name], Xp_dev, y_dev.to_numpy(),
                               Xp_held)
pred_mean = (lo + hi) / 2.0
bootstrap_stats(lo, hi, y_held.to_numpy())
plot_bootstrap_ci(pred_mean, lo, hi, y_held.to_numpy())
plot_coverage(lo, hi, y_held.to_numpy())
from pipeline.uncertainty import coverage_rate
print("Pooled coverage:", coverage_rate(lo, hi, y_held.to_numpy()))
""")

stage("15. Hyperparameter Optimization (prespecified grid)", """
## Stage 15 — Hyperparameter Optimization (Ch. 3.10.3)

Small prespecified candidate grid evaluated under the same 5-fold CV protocol
(explicitly not the open RandomizedSearchCV of the earlier phase). Paired t-test
default vs optimized; statistical vs practical significance.
""", """
from pipeline.tuning import (CATBOOST_GRID, evaluate_candidates, paired_ttest,
                             plot_hyperparameter_importance,
                             plot_hyperparam_ttest, save_hyperparam_table)
grid_results = evaluate_candidates(CATBOOST_GRID, Xp_dev, y_dev.to_numpy())
print(grid_results.sort_values("RMSE_mean")[["name", "RMSE_mean", "RMSE_std"]])
default_folds = grid_results.loc[grid_results["name"] == "default", "folds"].iloc[0]
best_row = grid_results.sort_values("RMSE_mean").iloc[0]
ttest = paired_ttest(default_folds, best_row["folds"])
print("Paired t-test (default vs optimized, per fold):", ttest)
plot_hyperparameter_importance(grid_results)
plot_hyperparam_ttest(ttest)
save_hyperparam_table(grid_results, ttest)
""")

stage("16. Feature Importance (permutation + SHAP)", """
## Stage 16 — Feature Importance

Permutation importance (model-agnostic, pooled-RMSE increase, Ch. 3.10.4) plus SHAP
summary for the selected model (supplementary; note the distinction in the thesis).
""", """
from pipeline.importance import (permutation_importance,
                                 plot_permutation_importance,
                                 save_importance_table, plot_shap)
imp = permutation_importance(models[best_name], Xp_held, y_held.to_numpy(),
                             n_repeats=10)
print(imp)
plot_permutation_importance(imp)
save_importance_table(imp)
plot_shap(models[best_name], Xp_held)
""")

stage("17. Learning Curves and Extrapolation", """
## Stage 17 — Learning Curves and Extrapolation

Learning curve for the selected model; extrapolation test: train on SHORT beams,
evaluate on LONG beams (real length axis).
""", """
from pipeline.learning import (compute_learning_curve, plot_learning_curve,
                               save_learning_curve, extrapolation_test,
                               plot_extrapolation)
lc = compute_learning_curve(models[best_name], Xp_dev, y_dev.to_numpy())
print(lc)
plot_learning_curve(lc)
save_learning_curve(lc)
X_short = make_xy(select_features(dev[dev["length_class"] == "SHORT"]))
X_long = make_xy(select_features(dev[dev["length_class"] == "LONG"]))
prep_s = Preprocessor(family_mode="onehot").fit(X_short[0])
Xp_s = prep_s.transform(X_short[0])
Xp_l = prep_s.transform(X_long[0])  # same scaler: no LONG statistics leak
from sklearn.base import clone
extrap = extrapolation_test(clone(models[best_name]), Xp_s, X_short[1].to_numpy(),
                            Xp_l, X_long[1].to_numpy())
print("Extrapolation:", extrap)
plot_extrapolation(extrap)
""")

stage("18. Das Benchmark, Final Model, and Engineering Interpretation", """
## Stage 18 — Das (2023) Benchmark, Final Model, and Interpretation

- Conceptual comparison vs Das (2023): 98.78-98.88% R2 for steel/aluminum beams
- Save best model + scaler + metadata with domain-enforced prediction interface
- Engineering interpretation: dominant parameters, damage detectability, limits
""", """
from pipeline.predict import (das_benchmark_table, save_artifacts,
                              FrequencyPredictor, DOMAIN)
from pipeline.metrics import pooled_metrics
per_mode_r2 = mode_metrics(y_held.to_numpy(),
                           predict_model(models[best_name], Xp_held))["R2"]
pooled_r2 = pooled_metrics(y_held.to_numpy(),
                           predict_model(models[best_name], Xp_held))["R2"]
print(f"Selected model pooled R2 on held-out: {pooled_r2:.4f}")
das_benchmark_table(pooled_r2, per_mode_r2)
save_artifacts(models[best_name], prep.scaler)
predictor = FrequencyPredictor(models[best_name], prep.scaler)
demo = {"L_mm": 5000.0, "b_mm": 300.0, "h_mm": 500.0, "fc_MPa": 35.0,
        "rho_percent": 1.2, "crack1_depth_mm": 200.0,
        "crack2_depth_mm": 200.0, "family": "FF"}
print("Example prediction:", predictor.predict(demo).round(2))
print()
print("DONE - all outputs under outputs/ (figures/ tables/ models/ logs/)")
""")

if __name__ == "__main__":
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(HEADER)]
    for title, md, code in STAGES:
        cells.append(nbf.v4.new_markdown_cell(md))
        cells.append(nbf.v4.new_code_cell(code))
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (venv12)", "language": "python",
        "name": "python3"}
    nbf.write(nb, str(NB_PATH))
    print(f"Wrote {len(cells)} cells to {NB_PATH}")
