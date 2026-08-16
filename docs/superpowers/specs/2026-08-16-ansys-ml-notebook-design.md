# Design: Single-Notebook ANSYS ML Pipeline for the Revised Thesis

**Date:** 2026-08-16
**Status:** Draft for review
**Scope:** One Jupyter notebook (`model_training_ansys.ipynb`) that replaces the old notebook + all `scripts/` outputs, trained on the new 1000-case ANSYS dataset, writing everything into a single consolidated output folder. The revised thesis (ful_thesis.md) will be rewritten to reference only this folder.

---

## 1. Goal and Success Criteria

- A single, reproducible notebook that covers the entire ML workflow described in revised Chapter 3: dataset QC → theoretical validation → preprocessing → five regression models → 5-fold CV → comparison → residual/scatter → computational-time → bootstrap uncertainty → hyperparameter optimization → feature importance → learning curves/extrapolation → final model + engineering interpretation.
- All figures, tables, models, and logs land in one folder: `Project/outputs/` (subfolders `figures/`, `tables/`, `models/`, `logs/`).
- The thesis text tables (4.19, 4.21, 4.25, etc.) and image paths are regenerated/rewritten from these outputs.

## 2. Data

- **Input:** `Project/data/rc_beam_ansys_dataset.xlsx` — a copy of `new-chapters/RC_Beam_1000_Updated_Frequencies_Merged-CL.xlsx` (sheet `ML Dataset`), so the notebook is self-contained.
- **Records:** 1000 cases, 500 FF + 500 SS, all QC PASS, no missing values, no duplicates.
- **FF family:** flexural cracks at 0.45L & 0.55L, both 90° (mid-span, max-moment zone).
- **SS family:** shear cracks at 0.1L & 0.9L, 45°/135° (near supports).
- All beams are **fixed-fixed** (boundary condition); FF/SS is a **crack family**, not a boundary condition.
- **Targets:** f1–f5 (B1–B5 physical bending frequencies, shape-classified in ANSYS post-processing).
- **Environment:** `.venv12` (adds `openpyxl`, already installed).

## 3. Notebook Stages

### Stage 1 — Setup & Data QC
- Paths, seed=42, logging to `outputs/logs/training.log`.
- Load `ML Dataset` sheet; assert 1000 rows, 0 missing, QC PASS, FF/SS = 500/500, unique case IDs.
- Solver mode indices (`bend_1_mode`…`bend_5_mode`) retained as diagnostic metadata only — never model inputs.

### Stage 2 — FEM/Theoretical Validation (dataset-level EBT checks)
Replaces all old Python-FEM validation scripts (`validate_gautam_2016.py`, `validate_fem_das2023.py`, `validate_massenzio_2005.py`, `comprehensive_validation.py`). FEM validation itself is ANSYS-side (Ch. 3.4–3.6) and is not re-implemented in Python.

- For each case, compute the undamaged fixed-fixed EBT frequencies B1–B5:
  `f_n = (β_n L)² / (2π L²) · √(EI/ρA)`, with βL = 4.730041, 7.853205, 10.995608, 14.137166, 17.278760; E from ACI 318-19 (`4700√f'c` MPa, consistent with the dataset's `Ec_MPa`), ρ = 2400 kg/m³, I = bh³/12.
- Compare ANSYS (cracked) vs EBT (pristine) → per-mode crack-induced frequency drop (%).
- Scaling-law checks: log–log slope of f vs L ≈ −2; mode-ratio distributions (cracked beams deviate from pristine 2.76 ratio; compare per family).
- Outputs: `figures/ebt_validation.png`, `figures/mode_ratios.png`, `figures/crack_drop_vs_depth.png`, `tables/ebt_validation_summary.csv`.

### Stage 3 — EDA
- Parameter distributions per family (`parameter_distributions.png`), correlation matrix of 7 features + 5 targets (`correlation_matrix.png`), crack-depth vs frequency scatter per family (`crack_depth_vs_frequency.png`).
- Locations/angles shown here as per-family constants — not used as features (see Stage 4).

### Stage 4 — Leakage-Controlled 8-Field Feature Matrix
- **Features (8):** `L_mm`, `b_mm`, `h_mm`, `fc_MPa`, `rho_percent`, `crack1_depth_mm`, `crack2_depth_mm`, `family` (FF/SS).
- **Dropped (with reason):**
  - `case_id`, `dataset_role`, `length_class` (redundant with L), `combination_code/name`, `crack1/2_type` (redundant with family) — administrative/derived.
  - `Ec_MPa`, `As_mm2`, `equivalent_diameter_mm`, `slenderness_L_h`, `width_depth_b_h` — deterministically derived from retained fields.
  - `Concrete Cover` — constant 40 mm (no information).
  - `supports` — constant (all fixed-fixed).
  - `mesh_size_mm`, `length_divisions`, `height_divisions`, `concrete_element`, `rebar_element`, `modes_extracted`, `preanalysis_qc` — solver/QC metadata.
  - `bend_1_mode`…`bend_5_mode` — leakage (derived from the solved frequencies).
  - `frequency_source` — provenance only.
- Crack locations/angles are **constant within each family** (500 identical pairs) and perfectly collinear with `family`; including them would make LR/SVR rank-deficient and corrupt feature importance. They are therefore represented by the family label only (per Ch. 3.10.1).
- **Feature-decision documentation (in-notebook):** a dedicated markdown section ("Feature Selection — Skipped vs Added") documents, for the thesis: (a) the full 12-field raw matrix, (b) which fields were skipped and why (locations/angles = family constants, zero additional information; derived/QC/leakage fields), (c) which fields were added/retained (8-field matrix), and (d) the full preprocessing steps applied. This documentation lives in notebook markdown cells and is summarized in `DECISIONS.md`; no separate mapping CSV is produced.

### Stage 5 — Train/Test Split
- 800 development / 200 held-out, stratified by family, `random_state=42` (Ch. 3.10.3).
- Held-out set is never examined (not targets, not predictors, not even as a group) until model selection is complete.

### Stage 6 — Preprocessing
- `StandardScaler` fitted on the 800 dev cases only (applied to the 7 continuous features).
- `family` → one-hot for LR/RF/XGB/SVR; native categorical (`cat_features`) for CatBoost.
- **Preprocessing documentation (in-notebook):** a markdown block records the exact step-by-step preprocessing pipeline for reproducibility and thesis text: (1) load 12-field raw matrix, (2) drop skipped fields with per-column reason (Stage 4), (3) retain 8-field matrix, (4) split 800/200 stratified by family (seed 42), (5) fit `StandardScaler` on dev only, (6) encode family per model type, (7) verify shapes/dtypes/no-NaN after each step.

### Stage 7 — Models (multi-output, all 5 targets jointly)
| Model | Configuration |
|---|---|
| Linear Regression | baseline |
| Random Forest | 100 estimators, unlimited depth (Breiman 2001) |
| XGBoost | lr=0.1, max_depth=6, wrapped in `MultiOutputRegressor` |
| CatBoost | `loss_function='MultiRMSE'`, `cat_features=[family]` |
| SVR | RBF, C tuned via CV, wrapped in `MultiOutputRegressor` |

Rationale (Ch. 3.10.1): CatBoost natively handles multi-target (MultiRMSE); LR/RF use native multi-output; XGB/SVR are single-output and wrapped so all five share the same folds, splits, and evaluation protocol.

### Stage 8 — Evaluation
- Per-mode MAE, RMSE, R², MAPE + macro-average (unweighted mean over 5 outputs) + pooled (flattened predictions/targets).
- 5-fold CV within the 800 dev cases: mean pooled validation RMSE, stability (std) per model.
- Held-out evaluation **once**, for the selected model only, after selection.

### Stage 9 — Comparison, Residuals, Statistical Tests
- Tables: `tables/model_comparison.csv`, `tables/per_mode_metrics.csv`.
- Figures: `model_comparison.png`, `prediction_vs_actual.png`, `residual_plots.png` (all 5 models).
- **Friedman test** (kept from old `statistical_tests.py`): non-parametric comparison of the 5 models across metrics (χ², Kendall's W).
- **One-way ANOVA — crack family effect (NEW, replaces damage-location ANOVA):** test whether the crack-induced frequency drop (%) differs between FF and SS families, per mode (B1, B2, B3). Report F, p, η². Figure `anova_family_drop.png`; results appended to `statistical_tests_summary.csv`.
  - Physical interpretation: FF cracks sit at max-moment zones → stronger B1/B2 reduction; SS cracks near supports → different modal sensitivity pattern.
  - *Why the old ANOVA was dropped:* crack location is fixed within each family; there is no within-family location variation to test.

### Stage 10 — Computational Time
- Training time per model; inference time (µs per case, mean over repeated runs).
- Comparison vs per-case APDL wall-clock cost (configurable constant `APDL_SOLVE_SECONDS`, default 360 s; to be replaced with the real measured value if available).
- Outputs: `tables/timing_comparison.csv`, `figures/timing_comparison.png`.

### Stage 11 — Bootstrap Uncertainty (per Ch. 3.10.6)
- Applied **only to the selected model**, once selection is complete.
- Refit the selected algorithm on **100 bootstrap resamples of the 800-case dev set** (`random_state=42`).
- From the spread of predictions at each held-out point, construct a **95% confidence interval** around that point's prediction, per mode.
- **Calibration:** nominal 95% coverage vs empirical coverage over the 200 held-out cases.
- Outputs per mode: `tables/bootstrap_stats.csv`, `figures/bootstrap_ci.png`, `figures/coverage_analysis.png`.

### Stage 12 — Hyperparameter Optimization (selected model)
- Small **prespecified candidate grid** (~5 configurations) evaluated under the same 5-fold CV protocol (Ch. 3.10.3 — explicitly not open-ended RandomizedSearchCV as in the old phase).
- Paired t-test default vs optimized across CV folds; ΔR² practical-significance discussion (statistical vs practical significance).
- Outputs: `tables/hyperparam_comparison.csv`, `figures/hyperparameter_importance.png`, `figures/hyperparam_ttest.png`.

### Stage 13 — Feature Importance
- **Permutation importance** (increase in pooled validation error when an input is shuffled), per Ch. 3.10.4 — model-agnostic, common scale.
- **SHAP summary** for the selected model (kept by user decision; note in interpretation that Ch. 3.10.4 reports permutation importance; SHAP is supplementary).
- Outputs: `tables/feature_importance.csv`, `figures/feature_importance.png`, `figures/shap_summary.png`.

### Stage 14 — Learning Curves & Extrapolation
- Learning curve for the selected model (`figures/learning_curve_analysis.png`, `tables/learning_curve_results.csv`).
- Extrapolation test: train on SHORT cases only, evaluate on LONG cases (real length axis; `length_class` used only for this test, not as a feature). Outputs: `figures/extrapolation_test.png`, `tables/extrapolation_results.csv`.

### Stage 15 — Das (2023) Benchmark Comparison
- Table comparing final pooled R² and per-mode R² against Das (2023) 98.78–98.88% (steel/aluminum beams, SVM/RF) with caveats (different material, boundary condition, crack families). `tables/das_benchmark_comparison.csv`.

### Stage 16 — Final Model & Prediction Interface
- Save `outputs/models/best_model.pkl`, `outputs/models/scaler.pkl`, `outputs/models/feature_metadata.json`.
- `predict_frequencies()` with **domain enforcement** (Ch. 3.10.3): rejects out-of-range L, b, h, fc, ρ, crack depths; rejects family ∉ {FF, SS}; rejects intermediate crack locations/angles/mixed families.
- Final summary cell printing all headline numbers.

## 4. Output Folder Structure

```
Project/outputs/
├── figures/      # all PNGs (flat, thesis-friendly)
├── tables/       # all CSVs
├── models/       # best_model.pkl, scaler.pkl, feature_metadata.json
└── logs/         # training.log
```

Thesis integration: rewrite all image/table paths in `ful_thesis.md` from `docs/figures/...` and `simulation/outputs/...` to `outputs/figures/...` / `outputs/tables/...`; replace thesis table values with the new CSVs.

## 5. Decisions Ledger (what we use / don't use / why)

Documented in `DECISIONS.md` at the project root (generated as part of this work) and summarized here:

### Kept (used)
- Five models: LR, RF, XGBoost, CatBoost, SVR (RQ2 — same set as old thesis).
- 5-fold cross-validation; MAE/RMSE/R² metrics; now extended to 5 modes + MAPE + macro/pooled aggregation.
- Training/inference time comparison (extended; now vs APDL wall-clock).
- Residual/scatter analysis for all models.
- Bootstrap uncertainty quantification (now per Ch. 3.10.6 — selected model only, 100 resamples, calibration check).
- Hyperparameter optimization (changed protocol — see below).
- Friedman model-comparison test; paired t-test (default vs optimized).
- Permutation feature importance (per Ch. 3.10.4).
- EBT theoretical consistency checks (adapted from FEM validation scripts to dataset-level checks).
- Learning curves + extrapolation (extrapolation axis changed to length).

### Changed (why)
- Dataset: 3000 synthetic CSV → 1000-case ANSYS xlsx (real crack geometry from validated formulation).
- Features: 6 (incl. Damage_Type/Severity) → 8-field matrix (family + 2 crack depths) per Ch. 3.10.1; locations/angles dropped as features (constant per family → collinear).
- Targets: 2 modes → 5 bending modes; multi-output joint models (CatBoost MultiRMSE; XGB/SVR via MultiOutputRegressor).
- Split: 80/20 → 800/200 dev/held-out with strict one-time held-out evaluation (Ch. 3.10.3).
- Hyperparameter tuning: RandomizedSearchCV ×50 → small prespecified candidate grid (Ch. 3.10.3 explicitly rejects the open search of the earlier phase).
- Validation figures: Python-FEM reimplementations (gautam/das/massenzio/comprehensive) → dataset-level EBT checks; FEM validation is ANSYS-side (Ch. 3.4–3.6).
- Outputs: 3+ folders → single `Project/outputs/`.

### Dropped (why)
- ANOVA on crack location — locations fixed per family; replaced by family ANOVA (Stage 9).
- `validate_rc_beam.py` / `rc_validation` figures — superseded by ANSYS benchmark and dataset-level checks.
- Monte Carlo FEM material-uncertainty propagation (§4.2.6 of old thesis) — FEM-side analysis; belongs to ANSYS-side work, not the notebook.
- Old statistical-tests figures not applicable: `damage_location_anova.png` (see above).
- Old `simulation/` tree outputs (except as reference) — consolidated into `outputs/`.
- Leakage columns (bend modes, Ec, As, cover, mesh, QC, solver metadata) — removed from features (Stage 4).

## 6. Open Items
- Real APDL wall-clock solve time per case (replace `APDL_SOLVE_SECONDS` default) — ask thesis author.
- Whether the old `simulation/` and `scripts/` files should be deleted or kept as reference after consolidation.

## 7. Verification
- Notebook runs end-to-end with `.venv12`; all outputs land in `outputs/`.
- Assertions: 1000 rows, 500/500 families, 0 missing, 800/200 split sizes, per-mode metrics match table CSVs, coverage ~95%.
- After run: grep thesis for stale image paths; verify every thesis-referenced figure exists in `outputs/figures/`.
- In-notebook documentation present: feature-selection (skipped vs added) section and step-by-step preprocessing record.
