# Decisions Ledger — Revised ML Pipeline (ANSYS dataset)

See `docs/superpowers/specs/2026-08-16-ansys-ml-notebook-design.md` for the full spec.

## Kept (used)
- Five models: Linear Regression, Random Forest, XGBoost, CatBoost, SVR.
- 5-fold cross-validation; MAE/RMSE/R2 metrics (now 5 modes + MAPE + macro/pooled).
- Training/inference time comparison (now vs APDL wall-clock).
- Residual/scatter analysis; Friedman model comparison; paired t-test (default vs optimized).
- Permutation feature importance (Ch. 3.10.4); bootstrap uncertainty (Ch. 3.10.6).
- Learning curves + extrapolation (axis changed to length SHORT->LONG).

## Changed (why)
- Dataset: 3000 synthetic CSV -> 1000-case ANSYS xlsx (validated formulation, real crack geometry).
- Features: 6 -> 8-field matrix (7 continuous + family); crack locations/angles dropped
  (constant per family, perfectly collinear with family — zero additional information).
- Targets: 2 modes -> 5 bending modes (B1-B5), trained jointly (CatBoost MultiRMSE;
  XGB/SVR via MultiOutputRegressor).
- Split: 80/20 -> 800/200 dev/held-out, held-out examined once after selection (Ch. 3.10.3).
- Tuning: RandomizedSearchCV x50 -> small prespecified candidate grid (Ch. 3.10.3).
- Validation: Python-FEM reimplementations (gautam/das/massenzio scripts) -> dataset-level
  Euler-Bernoulli checks; FEM validation is ANSYS-side (Ch. 3.4-3.6).
- Outputs: 3+ folders -> single Project/outputs/ (figures/ tables/ models/ logs/).

## Dropped (why)
- ANOVA on crack location: locations fixed per family; replaced by family ANOVA (FF vs SS
  on crack-induced frequency drop).
- validate_rc_beam.py / rc_validation figures: superseded by ANSYS benchmark + dataset checks.
- Monte Carlo FEM material-uncertainty propagation (old 4.2.6): FEM-side; ANSYS-side scope.
- Leakage/solver/QC columns: removed from features (see SKIPPED_FIELDS in pipeline/config.py).
- family is not a source column: Task 2 derives df["family"] = df["combination_code"] on load (500 FF + 500 SS).
- Old simulation/ outputs: consolidated into outputs/ (kept in repo as reference).

## Open items
- Real APDL wall-clock solve time per case (default 360 s used until provided).
- Whether old scripts/ and simulation/ are deleted or kept as reference after consolidation.
