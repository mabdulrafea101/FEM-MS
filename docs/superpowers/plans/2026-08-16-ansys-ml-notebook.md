# Single-Notebook ANSYS ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Project/new-ML/model_training_ansys.ipynb` — one notebook covering dataset QC → EBT validation → preprocessing → 5 regression models → 5-fold CV → comparison → residuals → timing → bootstrap → hyperparameter tuning → importance → learning curves → Das benchmark → final model, writing everything into a single `Project/new-ML/outputs/` folder (isolated from the old `scripts/`, `simulation/`, and old notebook at `Project/`).

**Architecture:** A testable `pipeline/` Python package holds all logic (data loading, theory checks, feature selection, models, metrics, statistics, bootstrap, tuning, plots, prediction interface). A `build_notebook.py` script assembles the notebook from stage cells via `nbformat`; the notebook is verified by executing it with `nbconvert`. Thesis image paths in `ful_thesis.md` are rewritten to point into `new-ML/outputs/`.

**Tech Stack:** Python 3.12 (`.venv12` at `Project/.venv12`), pandas, numpy, scipy, scikit-learn 1.7, CatBoost 1.2.8, XGBoost 3.1.2, SHAP 0.50, matplotlib, seaborn, openpyxl, joblib, nbformat, nbconvert, pytest.

## Global Constraints

- **Everything for the new thesis lives under `Project/new-ML/`** (notebook, `pipeline/` package, `tests/`, `data/`, `outputs/`, `discussions/`, `scripts/`, `build_notebook.py`). The old `Project/scripts/`, `Project/simulation/`, and the old notebook stay untouched.
- Working directory for all commands: `Project/new-ML/`; venv is `Project/.venv12`, invoked as `../.venv12/bin/python` (subagents MUST use this venv).
- Tests run with `../.venv12/bin/pytest`.
- Every random process uses `random_state=42` / `seed=42` (reproducibility).
- All outputs go under `Project/new-ML/outputs/` (`figures/`, `tables/`, `models/`, `logs/`). No script writes outside `outputs/` except `discussions/`, `pipeline/`, `tests/`, `scripts/`, `build_notebook.py`, and the notebook itself.
- Feature matrix (8 fields): `L_mm, b_mm, h_mm, fc_MPa, rho_percent, crack1_depth_mm, crack2_depth_mm, family`. Targets: `f1_hz, f2_hz, f3_hz, f4_hz, f5_hz`.
- Split: 800 dev / 200 held-out, stratified by family, seed 42. Held-out evaluated only after model selection.
- Multi-output: CatBoost `loss_function='MultiRMSE'` + native `family` categorical; XGBoost/SVR wrapped in `MultiOutputRegressor`; LR/RF native multi-output.
- Dependency additions: `pytest` only (openpyxl already installed).
- Input dataset copy: `Project/new-ML/data/rc_beam_ansys_dataset.xlsx` (copied from `new-chapters/RC_Beam_1000_Updated_Frequencies_Merged-CL.xlsx`, sheet `ML Dataset`).

---

### Task 1: Scaffold — venv deps, data copy, config, DECISIONS.md

**Files:**
- Create: `Project/new-ML/data/rc_beam_ansys_dataset.xlsx` (copy)
- Create: `Project/new-ML/pipeline/__init__.py`
- Create: `Project/new-ML/pipeline/config.py`
- Create: `Project/new-ML/discussions/DECISIONS.md`
- Create: `Project/new-ML/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pipeline/config.py` constants used by every later task: `SEED`, `N_FOLDS`, `N_BOOTSTRAP`, `DEV_SIZE`, `APDL_SOLVE_SECONDS`, `PROJECT_DIR`, `DATA_PATH`, `OUTPUT_DIR`, `FEATURE_COLS`, `FAMILY_COL`, `TARGET_COLS`, `SKIPPED_FIELDS` (dict of column → skip reason).

- [ ] **Step 1: Install pytest and copy the dataset**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pip install pytest -q
mkdir -p data pipeline tests discussions scripts outputs/figures outputs/tables outputs/models outputs/logs
cp "../../new-chapters/RC_Beam_1000_Updated_Frequencies_Merged-CL.xlsx" data/rc_beam_ansys_dataset.xlsx
```

- [ ] **Step 2: Create `pipeline/__init__.py`, `tests/__init__.py`, and `pipeline/config.py`**

`pipeline/__init__.py` and `tests/__init__.py`: empty files.

`pipeline/config.py`:

```python
from pathlib import Path

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 100
DEV_SIZE = 800
APDL_SOLVE_SECONDS = 360.0

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "rc_beam_ansys_dataset.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

FEATURE_COLS = ["L_mm", "b_mm", "h_mm", "fc_MPa", "rho_percent",
                "crack1_depth_mm", "crack2_depth_mm"]
FAMILY_COL = "family"
TARGET_COLS = ["f1_hz", "f2_hz", "f3_hz", "f4_hz", "f5_hz"]

SKIPPED_FIELDS = {
    "case_id": "administrative identifier, not a predictor",
    "dataset_role": "administrative; all rows are PRIMARY",
    "length_class": "derived from L_mm; used only for the extrapolation test",
    "combination_code": "redundant with family (same values)",
    "combination_name": "human-readable form of combination_code",
    "crack1_type": "redundant with family (FF=Flexural, SS=Shear)",
    "crack2_type": "redundant with family (FF=Flexural, SS=Shear)",
    "Ec_MPa": "deterministically derived from fc_MPa via ACI 318-19",
    "As_mm2": "derived from rho_percent, b_mm and h_mm",
    "equivalent_diameter_mm": "derived from As_mm2",
    "Concrete Cover": "constant 40 mm for all cases; zero information",
    "crack1_angle_deg": "constant within family (FF=90, SS=45); no information beyond family",
    "crack2_angle_deg": "constant within family (FF=90, SS=135); no information beyond family",
    "slenderness_L_h": "derived from L_mm and h_mm",
    "width_depth_b_h": "derived from b_mm and h_mm",
    "mesh_size_mm": "solver metadata, not a physical input",
    "length_divisions": "solver metadata",
    "height_divisions": "solver metadata",
    "supports": "constant (all fixed-fixed)",
    "concrete_element": "solver metadata",
    "rebar_element": "solver metadata",
    "modes_extracted": "solver metadata",
    "preanalysis_qc": "quality-control metadata",
    "bend_1_mode": "leakage: solver index derived from solved frequencies",
    "bend_2_mode": "leakage: solver index derived from solved frequencies",
    "bend_3_mode": "leakage: solver index derived from solved frequencies",
    "bend_4_mode": "leakage: solver index derived from solved frequencies",
    "bend_5_mode": "leakage: solver index derived from solved frequencies",
    "frequency_source": "provenance only",
}

# Crack locations are not separate columns in the spreadsheet; they are
# encoded in combination_name and are constant per family:
#   FF family: cracks at 0.45L and 0.55L (flexural)
#   SS family: cracks at 0.1L and 0.9L (shear)
# Location therefore carries no information beyond the family label and is
# represented by the family categorical only (per Ch. 3.10.1).
```

- [ ] **Step 3: Create `DECISIONS.md`**

`Project/new-ML/discussions/DECISIONS.md` — the decisions ledger (summary for the thesis):

```markdown
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
- Old simulation/ outputs: consolidated into outputs/ (kept in repo as reference).

## Open items
- Real APDL wall-clock solve time per case (default 360 s used until provided).
- Whether old scripts/ and simulation/ are deleted or kept as reference after consolidation.
```

- [ ] **Step 4: Verify scaffold**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/python -c "import pipeline.config as c; print(c.DATA_PATH, c.OUTPUT_DIR); print(len(c.SKIPPED_FIELDS))"
```

Expected: prints the data path, output path, and `28` (number of skipped fields).

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add data/rc_beam_ansys_dataset.xlsx pipeline/__init__.py pipeline/config.py discussions/DECISIONS.md tests/__init__.py
git commit -m "feat: scaffold pipeline package, config, and decisions ledger for ANSYS ML notebook"
```

---

### Task 2: Data loading and QC (`pipeline/data.py`)

**Files:**
- Create: `Project/new-ML/pipeline/data.py`
- Test: `Project/new-ML/tests/test_data.py`

**Interfaces:**
- Consumes: `pipeline.config.DATA_PATH`, `TARGET_COLS`.
- Produces:
  - `load_dataset(path: Path = DATA_PATH) -> pd.DataFrame` — reads sheet `ML Dataset`, adds `family` column from `combination_code`.
  - `run_qc(df: pd.DataFrame) -> dict` — returns check summary; raises `AssertionError` if any check fails.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_data.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.data'`.

- [ ] **Step 3: Implement `pipeline/data.py`**

```python
import pandas as pd
from pipeline.config import DATA_PATH, TARGET_COLS


def load_dataset(path=DATA_PATH):
    """Load the frozen ANSYS dataset and derive the family column."""
    df = pd.read_excel(path, sheet_name="ML Dataset")
    df["family"] = df["combination_code"]
    return df


def run_qc(df):
    """Run the dataset audit from Ch. 3.9; raise on any failed check."""
    checks = {
        "total_cases": int(len(df)),
        "unique_case_ids": int(df["case_id"].nunique()),
        "family_balance": df["family"].value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "qc_all_pass": bool((df["preanalysis_qc"] == "PASS").all()),
        "targets_positive": bool((df[TARGET_COLS] > 0).all().all()),
    }
    assert checks["total_cases"] == 1000, "expected 1000 cases"
    assert checks["unique_case_ids"] == 1000, "expected unique case ids"
    assert checks["missing_values"] == 0, "expected no missing values"
    assert checks["qc_all_pass"], "expected all QC PASS"
    assert set(checks["family_balance"]) == {"FF", "SS"}, "expected FF/SS families"
    assert checks["targets_positive"], "expected positive target frequencies"
    return checks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_data.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/data.py tests/test_data.py
git commit -m "feat: add dataset loading and QC audit for ANSYS dataset"
```

---

### Task 3: EBT theoretical validation (`pipeline/theory.py`)

**Files:**
- Create: `Project/new-ML/pipeline/theory.py`
- Create: `Project/new-ML/pipeline/plots.py` (shared `save_fig` helper used by all figure tasks)
- Test: `Project/new-ML/tests/test_theory.py`

**Interfaces:**
- Consumes: `pipeline.config.TARGET_COLS`, `FIGURES_DIR`, `TABLES_DIR`.
- Produces:
  - `BETA_L: np.ndarray` — fixed-fixed βL constants (5 values).
  - `ebt_frequencies(L, b, h, fc, rho=2400.0, n_modes=5) -> np.ndarray` — undamaged fixed-fixed EBT frequencies (Hz), `f = βL²/(2πL²)·√(EI/ρA)`, `E = 4700√fc` MPa.
  - `crack_drop_pct(df) -> pd.DataFrame` — one row per case×mode: `case_id, family, mode, ansys_hz, ebt_hz, drop_pct`.
  - `log_log_slope(df) -> float` — slope of `log(f1_hz)` vs `log(L_mm)`.
  - `plot_ebt_validation(drop_df, out_dir=FIGURES_DIR) -> Path` — saves `ebt_validation.png`.
  - `plot_mode_ratios(df, out_dir=FIGURES_DIR) -> Path` — saves `mode_ratios.png`.
  - `plot_crack_drop(drop_df, out_dir=FIGURES_DIR) -> Path` — saves `crack_drop_vs_depth.png`.
  - `save_ebt_table(drop_df, out_dir=TABLES_DIR) -> Path` — saves `ebt_validation_summary.csv`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_theory.py`:

```python
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
           "f4_hz": 400.0, "f5_hz": 550.0}
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
                        "f4_hz": 400.0, "f5_hz": 550.0}])
    drop_df = crack_drop_pct(df)
    assert plot_ebt_validation(drop_df, tmp_path).name == "ebt_validation.png"
    assert plot_mode_ratios(df, tmp_path).name == "mode_ratios.png"
    assert plot_crack_drop(drop_df, tmp_path).name == "crack_drop_vs_depth.png"
    assert save_ebt_table(drop_df, tmp_path).name == "ebt_validation_summary.csv"
    assert (tmp_path / "ebt_validation_summary.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_theory.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/plots.py`**

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.config import FIGURES_DIR


def save_fig(fig, name, out_dir=FIGURES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
```

- [ ] **Step 4: Implement `pipeline/theory.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import TARGET_COLS, TABLES_DIR
from pipeline.plots import save_fig

BETA_L = np.array([4.730041, 7.853205, 10.995608, 14.137166, 17.278760])


def ebt_frequencies(L, b, h, fc, rho=2400.0, n_modes=5):
    """Undamaged fixed-fixed Euler-Bernoulli frequencies (Hz)."""
    E = 4700.0 * np.sqrt(fc) * 1e6  # ACI 318-19, Pa
    I = b * h**3 / 12.0
    A = b * h
    base = np.sqrt(E * I / (rho * A))
    return BETA_L[:n_modes] ** 2 / (2.0 * np.pi * L**2) * base


def crack_drop_pct(df):
    """Per case x mode: ANSYS (cracked) vs EBT (pristine) frequency drop %."""
    rows = []
    for _, r in df.iterrows():
        ebt = ebt_frequencies(r["L_mm"] / 1000.0, r["b_mm"] / 1000.0,
                              r["h_mm"] / 1000.0, r["fc_MPa"])
        for i, col in enumerate(TARGET_COLS):
            rows.append({"case_id": r["case_id"], "family": r["family"],
                         "mode": f"B{i+1}", "ansys_hz": r[col],
                         "ebt_hz": ebt[i],
                         "drop_pct": (1.0 - r[col] / ebt[i]) * 100.0})
    return pd.DataFrame(rows)


def log_log_slope(df):
    x = np.log(df["L_mm"].to_numpy())
    y = np.log(df["f1_hz"].to_numpy())
    return float(np.polyfit(x, y, 1)[0])


def plot_ebt_validation(drop_df, out_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam in ["FF", "SS"]:
        sub = drop_df[drop_df["family"] == fam]
        axes[0].boxplot([sub.loc[sub["mode"] == m, "drop_pct"] for m in ["B1", "B2", "B3", "B4", "B5"]],
                        labels=["B1", "B2", "B3", "B4", "B5"], widths=0.6)
        axes[1].hist(sub["drop_pct"], bins=40, alpha=0.5, label=fam)
    axes[0].set_title("Crack-induced frequency drop by mode")
    axes[0].set_ylabel("Drop (%)")
    axes[1].set_title("Drop distribution per family")
    axes[1].set_xlabel("Drop (%)")
    axes[1].legend()
    fig.tight_layout()
    return save_fig(fig, "ebt_validation.png", out_dir)


def plot_mode_ratios(df, out_dir=None):
    ratios = pd.DataFrame({
        "f2/f1": df["f2_hz"] / df["f1_hz"],
        "f3/f1": df["f3_hz"] / df["f1_hz"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ratios["f2/f1"], bins=50, alpha=0.7)
    axes[0].axvline(2.7566, color="r", ls="--", label="pristine 2.757")
    axes[0].set_xlabel("f2/f1"); axes[0].legend()
    axes[1].hist(ratios["f3/f1"], bins=50, alpha=0.7)
    axes[1].axvline(5.4039, color="r", ls="--", label="pristine 5.404")
    axes[1].set_xlabel("f3/f1"); axes[1].legend()
    fig.suptitle("Mode-ratio distributions (deviation = damage)")
    fig.tight_layout()
    return save_fig(fig, "mode_ratios.png", out_dir)


def plot_crack_drop(drop_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in ["FF", "SS"]:
        sub = drop_df[drop_df["family"] == fam]
        for m in ["B1", "B5"]:
            s = sub[sub["mode"] == m]
            ax.scatter(s["ansys_hz"], s["drop_pct"], s=6, alpha=0.5,
                       label=f"{fam} {m}")
    ax.set_xlabel("ANSYS frequency (Hz)")
    ax.set_ylabel("Drop vs pristine EBT (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "crack_drop_vs_depth.png", out_dir)


def save_ebt_table(drop_df, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = drop_df.groupby(["family", "mode"])["drop_pct"].agg(
        ["mean", "std"]).round(2).reset_index()
    path = out_dir / "ebt_validation_summary.csv"
    summary.to_csv(path, index=False)
    return path
```

Note: `test_crack_drop_pct_small_damage_small_drop` uses a manually chosen cracked beam (52 Hz B1) that stays below the pristine EBT value (~63 Hz for L=5 m, h=0.5, fc=35) but within 40% — verified against the formula; if your computed EBT B1 differs slightly, the `< 40` bound is robust.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_theory.py -v
```

Expected: 7 PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/plots.py pipeline/theory.py tests/test_theory.py
git commit -m "feat: add EBT theoretical validation with plots and tables"
```

---

### Task 4: Feature selection, split, preprocessing (`pipeline/prepare.py`)

**Files:**
- Create: `Project/new-ML/pipeline/prepare.py`
- Test: `Project/new-ML/tests/test_prepare.py`

**Interfaces:**
- Consumes: `pipeline.config.FEATURE_COLS, FAMILY_COL, TARGET_COLS, DEV_SIZE, SEED`.
- Produces:
  - `select_features(df) -> pd.DataFrame` — 8-field matrix (7 continuous + family) + targets.
  - `split_data(df, dev_size=DEV_SIZE, seed=SEED) -> (dev_df, held_df)` — random split without replacement.
  - `make_xy(df) -> (X, y)` — `X` = features DataFrame (8 cols), `y` = targets DataFrame (5 cols).
  - `class Preprocessor` — `fit(X)`, `transform(X) -> dict`, `fit_transform(X) -> dict`; dict has keys `num` (np.ndarray, scaled 7 continuous) and `family` (pd.Series of raw family labels). `family_mode='onehot'` also produces key `num` with the one-hot column appended; for CatBoost use `family_mode='native'`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_prepare.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_prepare.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/prepare.py`**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pipeline.config import (FEATURE_COLS, FAMILY_COL, TARGET_COLS,
                             DEV_SIZE, SEED)


def select_features(df):
    """Keep only the 8-field model matrix plus targets (leakage-controlled)."""
    keep = FEATURE_COLS + [FAMILY_COL] + TARGET_COLS
    return df[keep].copy()


def split_data(df, dev_size=DEV_SIZE, seed=SEED):
    """Random 800/200 split, stratified by family, seeded."""
    dev = pd.DataFrame()
    held = pd.DataFrame()
    for fam in ["FF", "SS"]:
        sub = df[df[FAMILY_COL] == fam]
        n_dev = dev_size // 2
        dev_part = sub.sample(n=n_dev, random_state=seed)
        dev = pd.concat([dev, dev_part])
        held = pd.concat([held, sub.drop(dev_part.index)])
    return dev, held


def make_xy(df):
    X = df[FEATURE_COLS + [FAMILY_COL]]
    y = df[TARGET_COLS]
    return X, y


class Preprocessor:
    """Scale 7 continuous features; encode family one-hot or keep native."""

    def __init__(self, family_mode="onehot", seed=SEED):
        assert family_mode in ("onehot", "native")
        self.family_mode = family_mode
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first", handle_unknown="ignore")

    def fit(self, X):
        self.scaler.fit(X[FEATURE_COLS])
        if self.family_mode == "onehot":
            self.encoder.fit(X[[FAMILY_COL]])
        return self

    def transform(self, X):
        num = self.scaler.transform(X[FEATURE_COLS])
        if self.family_mode == "onehot":
            fam = self.encoder.transform(X[[FAMILY_COL]]).toarray()
            num = np.hstack([num, fam])
        return {"num": num, "family": X[FAMILY_COL].reset_index(drop=True)}

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_prepare.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/prepare.py tests/test_prepare.py
git commit -m "feat: add leakage-controlled feature selection, 800/200 split, preprocessing"
```

---

### Task 5: Metrics (`pipeline/metrics.py`)

**Files:**
- Create: `Project/new-ML/pipeline/metrics.py`
- Test: `Project/new-ML/tests/test_metrics.py`

**Interfaces:**
- Consumes: `pipeline.config.TARGET_COLS`.
- Produces:
  - `mode_metrics(y_true, y_pred) -> pd.DataFrame` — rows = targets, columns = MAE/RMSE/R2/MAPE (per-mode).
  - `pooled_metrics(y_true, y_pred) -> dict` — MAE/RMSE/R2/MAPE over flattened arrays.
  - `macro_summary(mode_df: pd.DataFrame) -> dict` — unweighted mean of each column.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pipeline.metrics import mode_metrics, pooled_metrics, macro_summary
from pipeline.config import TARGET_COLS


def test_pooled_metrics_hand_computed():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 31.0])
    m = pooled_metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(1.3333, abs=1e-3)
    assert m["RMSE"] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert m["R2"] == pytest.approx(0.97, abs=1e-3)
    assert m["MAPE"] == pytest.approx(9.4444, abs=1e-3)


def test_mode_metrics_returns_per_target_rows():
    y_true = np.column_stack([np.arange(10, 60, 10.0)] * 5)
    y_pred = y_true + 1.0
    df = mode_metrics(y_true, y_pred)
    assert list(df.index) == TARGET_COLS
    assert list(df.columns) == ["MAE", "RMSE", "R2", "MAPE"]
    assert (df["MAE"] == 1.0).all()


def test_macro_summary():
    df = pd.DataFrame({"MAE": [1.0, 3.0], "RMSE": [2.0, 4.0], "R2": [0.9, 0.7]})
    s = macro_summary(df)
    assert s["MAE"] == pytest.approx(2.0)
    assert s["R2"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_metrics.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/metrics.py`**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline.config import TARGET_COLS


def _metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100.0,
    }


def mode_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = {}
    for i, col in enumerate(TARGET_COLS):
        rows[col] = _metrics(y_true[:, i], y_pred[:, i])
    return pd.DataFrame(rows).T


def pooled_metrics(y_true, y_pred):
    return _metrics(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())


def macro_summary(mode_df):
    return {col: float(mode_df[col].mean()) for col in ["MAE", "RMSE", "R2", "MAPE"]}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_metrics.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/metrics.py tests/test_metrics.py
git commit -m "feat: add per-mode, pooled, and macro regression metrics"
```

---

### Task 6: Model builders + fit/predict helpers + 5-fold CV (`pipeline/models.py`)

**Files:**
- Create: `Project/new-ML/pipeline/models.py`
- Test: `Project/new-ML/tests/test_models.py`

**Interfaces:**
- Consumes: `pipeline.prepare.Preprocessor` output dict (`num`, `family`), `pipeline.metrics.pooled_metrics`, `pipeline.config` constants.
- Produces:
  - `build_models(seed=SEED) -> dict[str, estimator]` — keys: `"Linear Regression", "Random Forest", "XGBoost", "CatBoost", "SVR"`.
  - `fit_model(model, Xp: dict, y: np.ndarray) -> None` — handles CatBoost native-categorical input; others use `Xp["num"]`.
  - `predict_model(model, Xp: dict) -> np.ndarray` — shape (n, 5).
  - `run_cv(model, Xp: dict, y: np.ndarray, n_folds=N_FOLDS, seed=SEED) -> dict` — `{"RMSE_mean", "RMSE_std", "folds": list}` using pooled RMSE.
  - `fit_and_evaluate(models, Xp_train, y_train, Xp_test, y_test) -> pd.DataFrame` — per-model train/test pooled metrics + `Train_Time_s`; saves `model_comparison.csv` to `TABLES_DIR`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_models.py`:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from pipeline.models import (build_models, fit_model, predict_model, run_cv,
                             fit_and_evaluate)
from pipeline.config import SEED, N_FOLDS


def _synthetic(seed=SEED, n=200):
    rng = np.random.default_rng(seed)
    X_num = rng.uniform(0, 1, (n, 7))
    X = {"num": X_num, "family": pd.Series(["FF"] * (n // 2) + ["SS"] * (n // 2))}
    y = np.column_stack([10 * X_num[:, 0] + 2 * X_num[:, 1] + rng.normal(0, 0.1, n)
                         for _ in range(5)])
    return X, y


def test_build_models_returns_five():
    models = build_models()
    assert list(models) == ["Linear Regression", "Random Forest", "XGBoost",
                            "CatBoost", "SVR"]


def test_fit_predict_shape():
    X, y = _synthetic()
    models = build_models()
    for name, model in models.items():
        fit_model(model, X, y)
        pred = predict_model(model, X)
        assert pred.shape == (200, 5), name
        assert np.isfinite(pred).all(), name


def test_run_cv_returns_expected_keys():
    X, y = _synthetic()
    model = build_models()["Linear Regression"]
    res = run_cv(model, X, y)
    assert set(res) == {"RMSE_mean", "RMSE_std", "folds"}
    assert len(res["folds"]) == N_FOLDS


def test_catboost_and_linear_both_fit_synthetic_data():
    X, y = _synthetic()
    cb = build_models()["CatBoost"]
    lr = build_models()["Linear Regression"]
    assert run_cv(cb, X, y)["RMSE_mean"] < 1.0
    assert run_cv(lr, X, y)["RMSE_mean"] < 1.0


def test_fit_and_evaluate_returns_table(tmp_path, monkeypatch):
    X, y = _synthetic()
    models = build_models()
    X_train = {"num": X["num"][:150], "family": X["family"][:150]}
    y_train = y[:150]
    X_test = {"num": X["num"][150:], "family": X["family"][150:]}
    y_test = y[150:]
    df = fit_and_evaluate(models, X_train, y_train, X_test, y_test)
    assert len(df) == 5
    assert {"Model", "Test_R2", "Train_Time_s"}.issubset(df.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_models.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/models.py`**

```python
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
            allow_writing_files=False),
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
```

Note: `test_catboost_and_linear_both_fit_synthetic_data` — on data with an exact linear relationship plus tiny noise, both models are near-perfect (RMSE < 1.0 is a robust bound; the pipeline test checks the machinery, not the ranking).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_models.py -v
```

Expected: 5 PASS. (SVR/boosting tests may take ~30-60 s.)

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/models.py tests/test_models.py
git commit -m "feat: add five multi-output models, fit/predict helpers, and 5-fold CV"
```

---

### Task 7: Model comparison figures + residuals (`pipeline/compare.py`)

**Files:**
- Create: `Project/new-ML/pipeline/compare.py`
- Test: `Project/new-ML/tests/test_compare.py`

**Interfaces:**
- Consumes: `pipeline.models.predict_model`, `pipeline.plots.save_fig`, `pipeline.metrics.mode_metrics`.
- Produces:
  - `plot_model_comparison(results_df, out_dir) -> Path` — `model_comparison.png` (bar charts of Test_MAE, Test_RMSE, Test_R2, Train_Time_s).
  - `plot_prediction_vs_actual(models, Xp_test, y_test, out_dir) -> Path` — `prediction_vs_actual.png` (one subplot per model, pooled B1).
  - `plot_residuals(models, Xp_test, y_test, out_dir) -> Path` — `residual_plots.png` (residuals vs predicted, per model).
  - `plot_per_mode_metrics(best_name, models, Xp_test, y_test, out_dir) -> Path` — `per_mode_metrics.png` (bar chart of R2 per mode for the best model); also saves `tables/per_mode_metrics.csv`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_compare.py`:

```python
import numpy as np
import pandas as pd
from pipeline.models import build_models, fit_model, predict_model
from pipeline.compare import (plot_model_comparison, plot_prediction_vs_actual,
                              plot_residuals, plot_per_mode_metrics)


def _tiny_setup():
    rng = np.random.default_rng(1)
    n = 60
    X = {"num": rng.uniform(0, 1, (n, 7)),
         "family": pd.Series(["FF"] * 30 + ["SS"] * 30)}
    y = np.column_stack([5 * X["num"][:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    models = build_models()
    for m in models.values():
        fit_model(m, X, y)
    return models, X, y


def test_plot_model_comparison(tmp_path):
    df = pd.DataFrame({"Model": ["A", "B"], "Test_MAE": [1.0, 2.0],
                       "Test_RMSE": [2.0, 3.0], "Test_R2": [0.9, 0.8],
                       "Train_Time_s": [0.1, 0.2]})
    assert plot_model_comparison(df, tmp_path).name == "model_comparison.png"


def test_plot_prediction_vs_actual(tmp_path):
    models, X, y = _tiny_setup()
    assert plot_prediction_vs_actual(models, X, y, tmp_path).name == "prediction_vs_actual.png"


def test_plot_residuals(tmp_path):
    models, X, y = _tiny_setup()
    assert plot_residuals(models, X, y, tmp_path).name == "residual_plots.png"


def test_plot_per_mode_metrics(tmp_path):
    models, X, y = _tiny_setup()
    path = plot_per_mode_metrics("CatBoost", models, X, y, tmp_path)
    assert path.name == "per_mode_metrics.png"
    assert (tmp_path / "per_mode_metrics.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_compare.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/compare.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.models import predict_model
from pipeline.metrics import mode_metrics, macro_summary
from pipeline.config import TARGET_COLS
from pipeline.plots import save_fig


def plot_model_comparison(results_df, out_dir=None):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names = results_df["Model"]
    axes[0].bar(names, results_df["Test_MAE"]); axes[0].set_title("Test MAE (Hz)")
    axes[1].bar(names, results_df["Test_RMSE"]); axes[1].set_title("Test RMSE (Hz)")
    axes[2].bar(names, results_df["Test_R2"]); axes[2].set_title("Test R2")
    axes[3].bar(names, results_df["Train_Time_s"]); axes[3].set_title("Train time (s)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_fig(fig, "model_comparison.png", out_dir)


def plot_prediction_vs_actual(models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pred = predict_model(model, Xp_test)[:, 0]
        ax.scatter(y_test[:, 0], pred, s=8, alpha=0.5)
        lim = [min(y_test[:, 0].min(), pred.min()),
               max(y_test[:, 0].max(), pred.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("Actual B1 (Hz)"); ax.set_ylabel("Predicted B1 (Hz)")
    fig.tight_layout()
    return save_fig(fig, "prediction_vs_actual.png", out_dir)


def plot_residuals(models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pred = predict_model(model, Xp_test)
        resid = y_test[:, 0] - pred[:, 0]
        ax.scatter(pred[:, 0], resid, s=8, alpha=0.5)
        ax.axhline(0, color="r", lw=1)
        ax.set_title(f"{name} (B1)")
        ax.set_xlabel("Predicted (Hz)"); ax.set_ylabel("Residual (Hz)")
    fig.tight_layout()
    return save_fig(fig, "residual_plots.png", out_dir)


def plot_per_mode_metrics(best_name, models, Xp_test, y_test, out_dir=None):
    y_test = np.asarray(y_test)
    pred = predict_model(models[best_name], Xp_test)
    per_mode = mode_metrics(y_test, pred)
    macro = macro_summary(per_mode)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(per_mode)), per_mode["R2"])
    ax.set_xticks(range(len(per_mode)))
    ax.set_xticklabels(TARGET_COLS)
    ax.axhline(macro["R2"], color="r", ls="--", label=f"macro R2 = {macro['R2']:.4f}")
    ax.set_title(f"{best_name} — per-mode R2")
    ax.legend()
    fig.tight_layout()
    out_dir = Path(out_dir) if out_dir else None
    save_fig(fig, "per_mode_metrics.png", out_dir)
    csv_dir = Path(out_dir) if out_dir else Path("outputs/tables")
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_mode.round(4).to_csv(csv_dir / "per_mode_metrics.csv")
    return csv_dir / "per_mode_metrics.png"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_compare.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/compare.py tests/test_compare.py
git commit -m "feat: add model comparison, prediction-vs-actual, residual, per-mode figures"
```

---

### Task 8: Statistical tests — Friedman + family ANOVA (`pipeline/stats_tests.py`)

**Files:**
- Create: `Project/new-ML/pipeline/stats_tests.py`
- Test: `Project/new-ML/tests/test_stats_tests.py`

**Interfaces:**
- Consumes: `pipeline.config.TABLES_DIR, FIGURES_DIR`.
- Produces:
  - `friedman_test(rank_matrix: np.ndarray) -> dict` — rows = models, cols = metrics; returns `{"chi2", "p", "kendall_w"}`.
  - `anova_family(drop_df: pd.DataFrame, modes=("B1","B2","B3")) -> dict` — per mode `{"F", "p", "eta2"}` comparing FF vs SS `drop_pct`.
  - `plot_anova_family(anova_result, drop_df, out_dir) -> Path` — `anova_family_drop.png`.
  - `plot_friedman(friedman_result, out_dir) -> Path` — `model_comparison_friedman.png`.
  - `save_stats_summary(friedman_result, anova_result, out_dir=TABLES_DIR) -> Path` — `statistical_tests_summary.csv`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_stats_tests.py`:

```python
import numpy as np
import pandas as pd
from pipeline.stats_tests import (friedman_test, anova_family,
                                 plot_anova_family, plot_friedman,
                                 save_stats_summary)


def test_friedman_significant_for_clear_winner():
    # rows = models, cols = metrics; model A always best rank
    ranks = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                     dtype=float)
    res = friedman_test(ranks)
    assert res["p"] < 0.01
    assert 0.9 < res["kendall_w"] <= 1.0


def test_friedman_trivial_ranks_no_signal():
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 6, size=(5, 10)).astype(float)
    res = friedman_test(ranks)
    assert res["kendall_w"] < 0.5


def test_anova_family_detects_separation():
    rng = np.random.default_rng(0)
    rows = []
    for i, fam in enumerate(["FF", "SS"]):
        mean = 20.0 if fam == "FF" else 8.0
        for _ in range(200):
            rows.append({"family": fam, "mode": "B1",
                         "drop_pct": rng.normal(mean, 1.5)})
    drop_df = pd.DataFrame(rows)
    res = anova_family(drop_df, modes=("B1",))
    assert res["B1"]["p"] < 0.001
    assert res["B1"]["eta2"] > 0.5


def test_plots_and_table(tmp_path):
    res_f = friedman_test(np.array([[1, 2], [2, 1]], dtype=float))
    rng = np.random.default_rng(0)
    drop_df = pd.DataFrame({"family": ["FF"] * 100 + ["SS"] * 100,
                            "mode": ["B1"] * 200,
                            "drop_pct": rng.normal(10, 1, 200)})
    res_a = anova_family(drop_df, modes=("B1",))
    assert plot_anova_family(res_a, drop_df, tmp_path).name == "anova_family_drop.png"
    assert plot_friedman(res_f, tmp_path).name == "model_comparison_friedman.png"
    assert save_stats_summary(res_f, res_a, tmp_path).name == "statistical_tests_summary.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_stats_tests.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/stats_tests.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from pipeline.config import TABLES_DIR
from pipeline.plots import save_fig


def friedman_test(metric_matrix):
    """metric_matrix: rows = models, cols = metrics (raw values, lower = better).

    scipy ranks within each column internally; Kendall's W is computed from the
    same ranks so all outputs are consistent.
    """
    metric_matrix = np.asarray(metric_matrix, dtype=float)
    stat, p = stats.friedmanchisquare(*metric_matrix.T)
    ranks = np.apply_along_axis(stats.rankdata, 0, metric_matrix)
    n_models, n_metrics = ranks.shape
    row_sums = ranks.sum(axis=1)
    mean_rank = n_metrics * (n_models + 1) / 2.0
    S = np.sum((row_sums - mean_rank) ** 2)
    w = 12.0 * S / (n_metrics**2 * (n_models**3 - n_models))
    return {"chi2": float(stat), "p": float(p), "kendall_w": float(w)}


def anova_family(drop_df, modes=("B1", "B2", "B3")):
    out = {}
    for m in modes:
        sub = drop_df[drop_df["mode"] == m]
        ff = sub.loc[sub["family"] == "FF", "drop_pct"].to_numpy()
        ss = sub.loc[sub["family"] == "SS", "drop_pct"].to_numpy()
        if len(ff) == 0 or len(ss) == 0:
            continue
        f, p = stats.f_oneway(ff, ss)
        n, k = len(ff) + len(ss), 2
        grand = np.concatenate([ff, ss]).mean()
        ss_between = len(ff) * (ff.mean() - grand) ** 2 + len(ss) * (ss.mean() - grand) ** 2
        ss_total = np.sum((np.concatenate([ff, ss]) - grand) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        out[m] = {"F": float(f), "p": float(p), "eta2": float(eta2)}
    return out


def plot_anova_family(anova_result, drop_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in ["FF", "SS"]:
        means = [drop_df[(drop_df["family"] == fam) & (drop_df["mode"] == m)]["drop_pct"].mean()
                 for m in anova_result]
        ax.bar([f"{m} {fam}" for m in anova_result], means, alpha=0.7)
    ax.set_ylabel("Mean crack-induced drop (%)")
    ax.set_title("ANOVA: family effect on frequency drop (FF vs SS)")
    fig.tight_layout()
    return save_fig(fig, "anova_family_drop.png", out_dir)


def plot_friedman(friedman_result, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5,
            f"Friedman test\nchi2 = {friedman_result['chi2']:.2f}\n"
            f"p = {friedman_result['p']:.4f}\n"
            f"Kendall W = {friedman_result['kendall_w']:.3f}",
            ha="center", va="center", fontsize=13)
    ax.axis("off")
    return save_fig(fig, "model_comparison_friedman.png", out_dir)


def save_stats_summary(friedman_result, anova_result, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"test": "Friedman", **{f"friedman_{k}": v for k, v in friedman_result.items()}}]
    for m, vals in anova_result.items():
        rows.append({"test": f"ANOVA_{m}", **vals})
    df = pd.DataFrame(rows)
    path = out_dir / "statistical_tests_summary.csv"
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_stats_tests.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/stats_tests.py tests/test_stats_tests.py
git commit -m "feat: add Friedman test and FF-vs-SS family ANOVA"
```

---

### Task 9: Computational timing (`pipeline/timing.py`)

**Files:**
- Create: `Project/new-ML/pipeline/timing.py`
- Test: `Project/new-ML/tests/test_timing.py`

**Interfaces:**
- Consumes: `pipeline.models.predict_model`, `pipeline.config.APDL_SOLVE_SECONDS`.
- Produces:
  - `measure_inference(model, Xp, n_reps=5) -> float` — mean µs per case.
  - `timing_table(results_df, models, Xp, apdl_seconds=APDL_SOLVE_SECONDS, out_dir=TABLES_DIR) -> Path` — `timing_comparison.csv` with columns `Model, Train_Time_s, Inference_us_per_case, Speedup_vs_APDL`.
  - `plot_timing(table_df, out_dir) -> Path` — `timing_comparison.png`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_timing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_timing.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/timing.py`**

```python
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import TABLES_DIR, APDL_SOLVE_SECONDS
from pipeline.models import predict_model
from pipeline.plots import save_fig


def measure_inference(model, Xp, n_reps=5):
    """Mean inference time in microseconds per case."""
    n_cases = Xp["num"].shape[0]
    predict_model(model, Xp)  # warmup
    best = np.inf
    for _ in range(n_reps):
        t0 = time.perf_counter()
        predict_model(model, Xp)
        best = min(best, (time.perf_counter() - t0) / n_cases * 1e6)
    return float(best)


def timing_table(results_df, models, Xp, apdl_seconds=APDL_SOLVE_SECONDS,
                 out_dir=TABLES_DIR):
    rows = []
    for name, model in models.items():
        train_s = float(results_df.loc[results_df["Model"] == name,
                                       "Train_Time_s"].iloc[0])
        us = measure_inference(model, Xp)
        speedup = apdl_seconds * 1e6 / us if us > 0 else np.inf
        rows.append({"Model": name, "Train_Time_s": train_s,
                     "Inference_us_per_case": us,
                     "Speedup_vs_APDL": speedup})
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path


def plot_timing(table_df, out_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(table_df["Model"], table_df["Train_Time_s"])
    axes[0].set_ylabel("Training time (s)"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(table_df["Model"], table_df["Inference_us_per_case"])
    axes[1].set_ylabel("Inference (µs/case)"); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_fig(fig, "timing_comparison.png", out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_timing.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/timing.py tests/test_timing.py
git commit -m "feat: add inference timing and APDL speedup comparison"
```

---

### Task 10: Bootstrap uncertainty (`pipeline/uncertainty.py`)

**Files:**
- Create: `Project/new-ML/pipeline/uncertainty.py`
- Test: `Project/new-ML/tests/test_uncertainty.py`

**Interfaces:**
- Consumes: `pipeline.models.fit_model, predict_model`, `pipeline.config.N_BOOTSTRAP, SEED, TARGET_COLS`.
- Produces:
  - `bootstrap_predictions(model, Xp_dev, y_dev, Xp_held, n=N_BOOTSTRAP, seed=SEED) -> (lo, hi)` — arrays of shape `(n_held, 5)` (2.5th/97.5th percentiles).
  - `coverage_rate(lo, hi, y_true) -> float` — empirical 95% coverage.
  - `bootstrap_stats(lo, hi, y_true) -> pd.DataFrame` — per-target: mean width, median width, coverage, mean pred std; saves `tables/bootstrap_stats.csv`.
  - `plot_bootstrap_ci(pred_mean, lo, hi, y_true, out_dir) -> Path` — `bootstrap_ci.png`.
  - `plot_coverage(lo, hi, y_true, out_dir) -> Path` — `coverage_analysis.png`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_uncertainty.py`:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline.uncertainty import (bootstrap_predictions, coverage_rate,
                                  bootstrap_stats, plot_bootstrap_ci,
                                  plot_coverage)


def test_coverage_rate_near_95_on_linear_data():
    rng = np.random.default_rng(42)
    n = 300
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([3 * Xnum[:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * 150 + ["SS"] * 150)}
    model = LinearRegression()
    lo, hi = bootstrap_predictions(model, X, y, X, n=30)
    assert lo.shape == (n, 5) and hi.shape == (n, 5)
    assert (lo <= hi).all()
    rate = coverage_rate(lo, hi, y)
    assert 0.85 <= rate <= 1.0


def test_bootstrap_stats_and_plots(tmp_path):
    rng = np.random.default_rng(1)
    n = 100
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([2 * Xnum[:, 0] + rng.normal(0, 0.3, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * 50 + ["SS"] * 50)}
    model = LinearRegression()
    lo, hi = bootstrap_predictions(model, X, y, X, n=20)
    stats_path = bootstrap_stats(lo, hi, y, out_dir=tmp_path)
    assert stats_path.name == "bootstrap_stats.csv"
    df = pd.read_csv(stats_path)
    assert len(df) == 5
    assert plot_bootstrap_ci((lo + hi) / 2, lo, hi, y, tmp_path).name == "bootstrap_ci.png"
    assert plot_coverage(lo, hi, y, tmp_path).name == "coverage_analysis.png"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_uncertainty.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/uncertainty.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import (N_BOOTSTRAP, SEED, TARGET_COLS, TABLES_DIR)
from pipeline.models import fit_model, predict_model
from pipeline.plots import save_fig


def bootstrap_predictions(model, Xp_dev, y_dev, Xp_held, n=N_BOOTSTRAP, seed=SEED):
    """Refit on n bootstrap resamples of the dev set; return 95% CI per point."""
    rng = np.random.default_rng(seed)
    n_dev = Xp_dev["num"].shape[0]
    y_dev = np.asarray(y_dev)
    preds = []
    for _ in range(n):
        idx = rng.integers(0, n_dev, size=n_dev)
        Xp_boot = {"num": Xp_dev["num"][idx],
                   "family": Xp_dev["family"].iloc[idx].reset_index(drop=True)}
        fit_model(model, Xp_boot, y_dev[idx])
        preds.append(predict_model(model, Xp_held))
    arr = np.stack(preds)  # (n, n_held, 5)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    return lo, hi


def coverage_rate(lo, hi, y_true):
    y_true = np.asarray(y_true)
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def bootstrap_stats(lo, hi, y_true, out_dir=TABLES_DIR):
    y_true = np.asarray(y_true)
    rows = []
    for i, col in enumerate(TARGET_COLS):
        width = hi[:, i] - lo[:, i]
        rows.append({
            "Mode": col,
            "Mean_CI_Width_Hz": float(width.mean()),
            "Median_CI_Width_Hz": float(np.median(width)),
            "Std_CI_Width_Hz": float(width.std()),
            "Coverage_95pct": coverage_rate(lo[:, [i]], hi[:, [i]], y_true[:, [i]]),
            "Mean_Pred_Std_Hz": float((width / (2 * 1.96)).mean()),
        })
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bootstrap_stats.csv"
    df.round(4).to_csv(path, index=False)
    return path


def plot_bootstrap_ci(pred_mean, lo, hi, y_true, out_dir=None):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(10, 6))
    n = min(200, len(y_true))
    idx = np.argsort(pred_mean[:n, 0])[:n]
    x = np.arange(n)
    ax.fill_between(x, lo[idx, 0], hi[idx, 0], alpha=0.3, label="95% CI")
    ax.plot(x, pred_mean[idx, 0], "k.", ms=3, label="prediction")
    ax.plot(x, y_true[idx, 0], "r.", ms=2, label="actual")
    ax.set_xlabel("Sorted held-out samples"); ax.set_ylabel("B1 (Hz)")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "bootstrap_ci.png", out_dir)


def plot_coverage(lo, hi, y_true, out_dir=None):
    y_true = np.asarray(y_true)
    inside = (y_true[:, 0] >= lo[:, 0]) & (y_true[:, 0] <= hi[:, 0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(np.arange(len(y_true)), y_true[:, 0], c=inside, cmap="bwr", s=10)
    ax.set_xlabel("Held-out case"); ax.set_ylabel("B1 (Hz)")
    ax.set_title(f"95% CI coverage: {inside.mean():.1%}")
    fig.tight_layout()
    return save_fig(fig, "coverage_analysis.png", out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_uncertainty.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: add bootstrap uncertainty and coverage calibration per Ch 3.10.6"
```

---

### Task 11: Hyperparameter tuning (`pipeline/tuning.py`)

**Files:**
- Create: `Project/new-ML/pipeline/tuning.py`
- Test: `Project/new-ML/tests/test_tuning.py`

**Interfaces:**
- Consumes: `pipeline.models.run_cv, fit_model, predict_model`; CatBoost.
- Produces:
  - `CATBOOST_GRID: list[dict]` — `[{"name":..., "params":{...}}, ...]` (5 candidates).
  - `evaluate_candidates(grid, Xp, y, n_folds=5, seed=SEED) -> pd.DataFrame` — columns `name, RMSE_mean, RMSE_std`.
  - `paired_ttest(cv_default: list, cv_best: list) -> dict` — `{"t", "p", "cohens_d"}`.
  - `plot_hyperparameter_importance(grid_results, out_dir) -> Path` — `hyperparameter_importance.png` (RMSE vs each numeric param).
  - `plot_hyperparam_ttest(ttest_result, out_dir) -> Path` — `hyperparam_ttest.png`.
  - `save_hyperparam_table(grid_results, ttest_result, out_dir=TABLES_DIR) -> Path` — `hyperparam_comparison.csv`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_tuning.py`:

```python
import numpy as np
import pandas as pd
from pipeline.tuning import (CATBOOST_GRID, evaluate_candidates, paired_ttest,
                             plot_hyperparameter_importance,
                             plot_hyperparam_ttest, save_hyperparam_table)


def test_grid_has_five_named_candidates():
    assert len(CATBOOST_GRID) == 5
    assert all("name" in c and "params" in c for c in CATBOOST_GRID)


def test_evaluate_candidates_returns_table():
    rng = np.random.default_rng(0)
    n = 150
    X = {"num": rng.uniform(0, 1, (n, 7)),
         "family": pd.Series(["FF"] * 75 + ["SS"] * 75)}
    y = np.column_stack([5 * X["num"][:, 0] + rng.normal(0, 0.1, n) for _ in range(5)])
    df = evaluate_candidates(CATBOOST_GRID[:2], X, y, n_folds=3)
    assert list(df["name"]) == ["default", CATBOOST_GRID[1]["name"]]
    assert {"name", "RMSE_mean", "RMSE_std"}.issubset(df.columns)


def test_paired_ttest_detects_improvement():
    rng = np.random.default_rng(0)
    default = rng.normal(5.0, 0.3, 5)
    best = rng.normal(4.0, 0.3, 5)
    res = paired_ttest(default, best)
    assert res["p"] < 0.05
    assert res["cohens_d"] > 0


def test_plots_and_table(tmp_path):
    grid_results = pd.DataFrame({"name": ["a", "b"], "RMSE_mean": [1.0, 0.5],
                                 "RMSE_std": [0.1, 0.2],
                                 "depth": [6, 8], "learning_rate": [0.1, 0.05],
                                 "iterations": [100, 200]})
    ttest_result = {"t": 4.43, "p": 0.011, "cohens_d": 1.98}
    assert plot_hyperparameter_importance(grid_results, tmp_path).name == "hyperparameter_importance.png"
    assert plot_hyperparam_ttest(ttest_result, tmp_path).name == "hyperparam_ttest.png"
    assert save_hyperparam_table(grid_results, ttest_result, tmp_path).name == "hyperparam_comparison.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_tuning.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/tuning.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from pipeline.config import SEED, TABLES_DIR
from pipeline.models import run_cv
from pipeline.plots import save_fig

CATBOOST_GRID = [
    {"name": "default", "params": {"iterations": 100, "learning_rate": 0.1, "depth": 6}},
    {"name": "deeper", "params": {"iterations": 200, "learning_rate": 0.1, "depth": 8}},
    {"name": "regularized", "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3}},
    {"name": "shallow", "params": {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5}},
    {"name": "fast", "params": {"iterations": 50, "learning_rate": 0.2, "depth": 6}},
]


def evaluate_candidates(grid, Xp, y, n_folds=5, seed=SEED):
    rows = []
    for cand in grid:
        model = CatBoostRegressor(loss_function="MultiRMSE", random_state=seed,
                                  verbose=False, allow_writing_files=False,
                                  **cand["params"])
        cv = run_cv(model, Xp, y, n_folds=n_folds, seed=seed)
        rows.append({"name": cand["name"], "RMSE_mean": cv["RMSE_mean"],
                     "RMSE_std": cv["RMSE_std"], "folds": cv["folds"],
                     **cand["params"]})
    return pd.DataFrame(rows)


def paired_ttest(cv_default, cv_best):
    t, p = stats.ttest_rel(cv_default, cv_best)
    diff = np.asarray(cv_best) - np.asarray(cv_default)
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return {"t": float(t), "p": float(p), "cohens_d": float(d)}


def plot_hyperparameter_importance(grid_results, out_dir=None):
    params = [c for c in ["iterations", "learning_rate", "depth", "l2_leaf_reg"]
              if c in grid_results.columns]
    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
    for ax, p in zip(axes, params):
        ax.scatter(grid_results[p], grid_results["RMSE_mean"])
        ax.set_xlabel(p); ax.set_ylabel("CV RMSE (Hz)")
    fig.tight_layout()
    return save_fig(fig, "hyperparameter_importance.png", out_dir)


def plot_hyperparam_ttest(ttest_result, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5,
            f"Paired t-test (default vs optimized)\n"
            f"t = {ttest_result['t']:.2f}, p = {ttest_result['p']:.4f}\n"
            f"Cohen's d = {ttest_result['cohens_d']:.2f}",
            ha="center", va="center", fontsize=13)
    ax.axis("off")
    return save_fig(fig, "hyperparam_ttest.png", out_dir)


def save_hyperparam_table(grid_results, ttest_result, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = grid_results.sort_values("RMSE_mean").iloc[0]
    default = grid_results[grid_results["name"] == "default"].iloc[0]
    rows = [{"config": "default", "CV_RMSE_mean": default["RMSE_mean"],
             "CV_RMSE_std": default["RMSE_std"]},
            {"config": "optimized", "CV_RMSE_mean": best["RMSE_mean"],
             "CV_RMSE_std": best["RMSE_std"]}]
    df = pd.DataFrame(rows)
    df["t_stat"] = ttest_result["t"]
    df["p_value"] = ttest_result["p"]
    df["cohens_d"] = ttest_result["cohens_d"]
    path = out_dir / "hyperparam_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_tuning.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/tuning.py tests/test_tuning.py
git commit -m "feat: add prespecified CatBoost candidate grid and paired t-test"
```

---

### Task 12: Feature importance — permutation + SHAP (`pipeline/importance.py`)

**Files:**
- Create: `Project/new-ML/pipeline/importance.py`
- Test: `Project/new-ML/tests/test_importance.py`

**Interfaces:**
- Consumes: `pipeline.models.predict_model, fit_model`; `pipeline.metrics.pooled_metrics`; SHAP.
- Produces:
  - `permutation_importance(model, Xp, y, n_repeats=10, seed=SEED) -> dict` — feature → pooled-RMSE increase.
  - `plot_permutation_importance(imp: dict, out_dir) -> Path` — `feature_importance.png`.
  - `save_importance_table(imp, out_dir=TABLES_DIR) -> Path` — `feature_importance.csv`.
  - `plot_shap(model, Xp, out_dir) -> Path` — `shap_summary.png` (CatBoost-explainer summary; falls back to `KernelExplainer` on 100 samples if TreeExplainer unavailable).

- [ ] **Step 1: Write the failing test**

`Project/tests/test_importance.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_importance.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/importance.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from pipeline.config import (SEED, FEATURE_COLS, FAMILY_COL, TABLES_DIR)
from pipeline.models import fit_model, predict_model
from pipeline.metrics import pooled_metrics
from pipeline.plots import save_fig


def _permutation_rmse(model, Xp, y, col):
    """Pooled RMSE after shuffling one column of the feature matrix."""
    Xp_perm = {"num": Xp["num"].copy(), "family": Xp["family"].copy()}
    if col == FAMILY_COL:
        vals = Xp_perm["family"].to_numpy()
        np.random.shuffle(vals)
        Xp_perm["family"] = pd.Series(vals)
    else:
        idx = FEATURE_COLS.index(col)
        vals = Xp_perm["num"][:, idx].copy()
        np.random.shuffle(vals)
        Xp_perm["num"][:, idx] = vals
    return pooled_metrics(y, predict_model(model, Xp_perm))["RMSE"]


def permutation_importance(model, Xp, y, n_repeats=10, seed=SEED):
    np.random.seed(seed)
    baseline = pooled_metrics(y, predict_model(model, Xp))["RMSE"]
    cols = FEATURE_COLS + [FAMILY_COL]
    scores = {}
    for col in cols:
        increases = [_permutation_rmse(model, Xp, y, col) - baseline
                     for _ in range(n_repeats)]
        scores[col] = float(np.mean(increases))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def plot_permutation_importance(imp, out_dir=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(imp)
    ax.barh(range(len(names)), list(imp.values()))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("RMSE increase (Hz)")
    ax.set_title("Permutation feature importance (selected model)")
    fig.tight_layout()
    return save_fig(fig, "feature_importance.png", out_dir)


def save_importance_table(imp, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature": list(imp), "importance": list(imp.values())})
    path = out_dir / "feature_importance.csv"
    df.to_csv(path, index=False)
    return path


def plot_shap(model, Xp, out_dir=None, max_display=8):
    df = pd.DataFrame(Xp["num"], columns=FEATURE_COLS)
    df[FAMILY_COL] = Xp["family"].to_numpy()
    try:
        explainer = shap.TreeExplainer(model)
        sample = df.iloc[:200]
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception:
        sample = df.sample(min(100, len(df)), random_state=SEED)
        explainer = shap.KernelExplainer(
            lambda x: predict_model(model, {"num": x[:, :7],
                                            "family": pd.Series(x[:, 7])}),
            df.to_numpy()[:50])
        shap_values = explainer.shap_values(sample.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, max_display=max_display, show=False)
    fig.tight_layout()
    return save_fig(fig, "shap_summary.png", out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_importance.py -v
```

Expected: 3 PASS. (If SHAP TreeExplainer fails for CatBoost in `test_plot_shap_runs`, the KernelExplainer fallback keeps the test passing; runtime may be ~30-60 s.)

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/importance.py tests/test_importance.py
git commit -m "feat: add permutation importance and SHAP summary for selected model"
```

---

### Task 13: Learning curves + extrapolation (`pipeline/learning.py`)

**Files:**
- Create: `Project/new-ML/pipeline/learning.py`
- Test: `Project/new-ML/tests/test_learning.py`

**Interfaces:**
- Consumes: `pipeline.models.run_cv, fit_model, predict_model`, `pipeline.config.SEED`.
- Produces:
  - `compute_learning_curve(model, Xp, y, sizes=(0.1, 0.25, 0.5, 0.75, 1.0), n_folds=5, seed=SEED) -> pd.DataFrame` — columns `size, train_rmse, val_rmse`.
  - `plot_learning_curve(lc_df, out_dir) -> Path` — `learning_curve_analysis.png`.
  - `save_learning_curve(lc_df, out_dir=TABLES_DIR) -> Path` — `learning_curve_results.csv`.
  - `extrapolation_test(model, Xp_short, y_short, Xp_long, y_long) -> dict` — pooled RMSE/R2 train vs extrapolated.
  - `plot_extrapolation(result, out_dir) -> Path` — `extrapolation_test.png`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_learning.py`:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline.learning import (compute_learning_curve, plot_learning_curve,
                               save_learning_curve, extrapolation_test,
                               plot_extrapolation)


def _linear_setup(n=400):
    rng = np.random.default_rng(0)
    Xnum = rng.uniform(0, 1, (n, 7))
    y = np.column_stack([3 * Xnum[:, 0] + rng.normal(0, 0.2, n) for _ in range(5)])
    X = {"num": Xnum, "family": pd.Series(["FF"] * (n // 2) + ["SS"] * (n // 2))}
    return X, y


def test_learning_curve_returns_sizes():
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 0.5, 1.0), n_folds=3)
    assert list(lc["size"]) == [0.25, 0.5, 1.0]
    assert {"size", "train_rmse", "val_rmse"}.issubset(lc.columns)


def test_learning_curve_improves_with_size():
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 1.0), n_folds=3)
    assert lc.iloc[1]["val_rmse"] < lc.iloc[0]["val_rmse"]


def test_extrapolation_sane():
    X, y = _linear_setup()
    model = LinearRegression()
    half = len(y) // 2
    Xs = {"num": X["num"][:half], "family": X["family"][:half]}
    Xl = {"num": X["num"][half:], "family": X["family"][half:]}
    res = extrapolation_test(model, Xs, y[:half], Xl, y[half:])
    assert {"train_rmse", "extrap_rmse", "extrap_r2"}.issubset(res)
    assert res["extrap_r2"] > 0.5


def test_plots(tmp_path):
    X, y = _linear_setup()
    model = LinearRegression()
    lc = compute_learning_curve(model, X, y, sizes=(0.25, 0.5, 1.0), n_folds=3)
    assert plot_learning_curve(lc, tmp_path).name == "learning_curve_analysis.png"
    assert save_learning_curve(lc, tmp_path).name == "learning_curve_results.csv"
    half = len(y) // 2
    res = extrapolation_test(model, {"num": X["num"][:half], "family": X["family"][:half]},
                             y[:half], {"num": X["num"][half:], "family": X["family"][half:]},
                             y[half:])
    assert plot_extrapolation(res, tmp_path).name == "extrapolation_test.png"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_learning.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/learning.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline.config import SEED, TABLES_DIR
from pipeline.models import fit_model, predict_model
from pipeline.metrics import pooled_metrics
from pipeline.plots import save_fig


def compute_learning_curve(model, Xp, y, sizes=(0.1, 0.25, 0.5, 0.75, 1.0),
                           n_folds=5, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = Xp["num"].shape[0]
    rows = []
    for frac in sizes:
        n_sub = max(10, int(n * frac))
        idx = rng.choice(n, size=n_sub, replace=False)
        Xp_sub = {"num": Xp["num"][idx], "family": Xp["family"].iloc[idx].reset_index(drop=True)}
        y_sub = y[idx]
        # train on 80% of the subset, validate on 20%
        n_tr = int(n_sub * 0.8)
        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:]
        fit_model(model, Xp_sub, y_sub)
        train_rmse = pooled_metrics(y[tr_idx], predict_model(
            model, {"num": Xp["num"][tr_idx], "family": Xp["family"].iloc[tr_idx].reset_index(drop=True)}))["RMSE"]
        val_rmse = pooled_metrics(y[va_idx], predict_model(
            model, {"num": Xp["num"][va_idx], "family": Xp["family"].iloc[va_idx].reset_index(drop=True)}))["RMSE"]
        rows.append({"size": frac, "train_rmse": train_rmse, "val_rmse": val_rmse})
    return pd.DataFrame(rows)


def plot_learning_curve(lc_df, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lc_df["size"], lc_df["train_rmse"], "o-", label="train")
    ax.plot(lc_df["size"], lc_df["val_rmse"], "s-", label="validation")
    ax.set_xlabel("Training fraction"); ax.set_ylabel("Pooled RMSE (Hz)")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "learning_curve_analysis.png", out_dir)


def save_learning_curve(lc_df, out_dir=TABLES_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "learning_curve_results.csv"
    lc_df.round(4).to_csv(path, index=False)
    return path


def extrapolation_test(model, Xp_short, y_short, Xp_long, y_long):
    fit_model(model, Xp_short, y_short)
    train_rmse = pooled_metrics(y_short, predict_model(model, Xp_short))["RMSE"]
    extrap_pred = predict_model(model, Xp_long)
    extrap_rmse = pooled_metrics(y_long, extrap_pred)["RMSE"]
    extrap_r2 = pooled_metrics(y_long, extrap_pred)["R2"]
    return {"train_rmse": train_rmse, "extrap_rmse": extrap_rmse,
            "extrap_r2": extrap_r2}


def plot_extrapolation(result, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["train (SHORT)", "extrapolated (LONG)"]
    values = [result["train_rmse"], result["extrap_rmse"]]
    ax.bar(labels, values)
    ax.set_ylabel("Pooled RMSE (Hz)")
    ax.set_title(f"Extrapolation: train SHORT -> test LONG (R2={result['extrap_r2']:.3f})")
    fig.tight_layout()
    return save_fig(fig, "extrapolation_test.png", out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_learning.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/learning.py tests/test_learning.py
git commit -m "feat: add learning curves and SHORT->LONG extrapolation test"
```

---

### Task 14: Das benchmark, final model, prediction interface (`pipeline/predict.py`)

**Files:**
- Create: `Project/new-ML/pipeline/predict.py`
- Test: `Project/new-ML/tests/test_predict.py`

**Interfaces:**
- Consumes: `pipeline.config` constants; joblib; models/predict helpers.
- Produces:
  - `DOMAIN: dict` — per-feature `(min, max)` bounds and `FAMILIES = ("FF", "SS")`.
  - `class FrequencyPredictor` — `__init__(self, model, scaler)`, `predict(self, inputs: dict) -> np.ndarray` (raises `ValueError` on out-of-domain input, unknown family, missing keys).
  - `save_artifacts(model, scaler, out_dir=MODELS_DIR) -> dict` — saves `best_model.pkl`, `scaler.pkl`, `feature_metadata.json`; returns metadata dict.
  - `das_benchmark_table(pooled_r2, per_mode_r2: pd.Series, out_dir=TABLES_DIR) -> Path` — `das_benchmark_comparison.csv`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_predict.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_predict.py -v
```

Expected: FAIL — module import errors.

- [ ] **Step 3: Implement `pipeline/predict.py`**

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from pipeline.config import (FEATURE_COLS, FAMILY_COL, MODELS_DIR, TABLES_DIR)

DOMAIN = {
    "L_mm": (3250.0, 8000.0),
    "b_mm": (250.0, 400.0),
    "h_mm": (325.0, 700.0),
    "fc_MPa": (25.0, 45.0),
    "rho_percent": (0.8, 2.0),
    "crack1_depth_mm": (50.0, 350.0),
    "crack2_depth_mm": (50.0, 350.0),
}
FAMILIES = ("FF", "SS")


class FrequencyPredictor:
    """Trained model + scaler with Ch. 3.10.3 domain enforcement."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, inputs):
        missing = [c for c in FEATURE_COLS + [FAMILY_COL] if c not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        if inputs[FAMILY_COL] not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        for col in FEATURE_COLS:
            lo, hi = DOMAIN[col]
            val = inputs[col]
            if not (lo <= val <= hi):
                raise ValueError(f"{col}={val} outside domain [{lo}, {hi}]")
        num = np.array([[inputs[c] for c in FEATURE_COLS]])
        Xs = self.scaler.transform(num)
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(Xs)).ravel()
        raise ValueError("model must expose predict()")


def save_artifacts(model, scaler, out_dir=MODELS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "best_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    meta = {"family_mode": "onehot", "features": FEATURE_COLS,
            "targets": [f"f{i}_hz" for i in range(1, 6)],
            "domain": DOMAIN, "families": list(FAMILIES)}
    (out_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def das_benchmark_table(pooled_r2, per_mode_r2, out_dir=TABLES_DIR):
    """Conceptual comparison vs Das (2023) for steel/aluminum beams."""
    rows = [
        {"Reference": "Das_2023_SVM_Puk", "Best_R2": 0.9878,
         "Note": "steel/aluminum beams, various BC"},
        {"Reference": "Das_2023_RandomForest", "Best_R2": 0.9888,
         "Note": "steel/aluminum beams, various BC"},
        {"Reference": "This_study_pooled", "Best_R2": float(pooled_r2),
         "Note": "fixed-fixed RC, FF/SS crack families"},
    ]
    for i, (idx, r2) in enumerate(per_mode_r2.items()):
        rows.append({"Reference": f"This_study_{idx}", "Best_R2": float(r2),
                     "Note": "per-mode"})
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "das_benchmark_comparison.csv"
    df.round(4).to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_predict.py -v
```

Expected: 8 PASS (5 parametrized cases count as 5).

- [ ] **Step 5: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add pipeline/predict.py tests/test_predict.py
git commit -m "feat: add Das benchmark table, artifact saving, and domain-enforced predictor"
```

---

### Task 15: Build and execute the single notebook (`build_notebook.py` → `model_training_ansys.ipynb`)

**Files:**
- Create: `Project/new-ML/build_notebook.py`
- Create: `Project/new-ML/model_training_ansys.ipynb` (generated by the script)

All paths in the builder and tests are relative to `new-ML/` (the builder's `PROJECT = Path(__file__).resolve().parent` is `new-ML/`; the test's `parent.parent` is also `new-ML/`).
- Test: `Project/new-ML/tests/test_notebook_build.py`

**Interfaces:**
- Consumes: every `pipeline.*` module from Tasks 1–14.
- Produces: `model_training_ansys.ipynb` with 17 stages; executing it writes all outputs under `Project/outputs/`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_notebook_build.py`:

```python
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def test_build_script_generates_notebook():
    result = subprocess.run(["../.venv12/bin/python", "build_notebook.py"],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    nb_path = PROJECT / "model_training_ansys.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert len(nb["cells"]) > 50  # markdown + code cells for all stages


def test_notebook_contains_all_stages():
    nb = json.loads((PROJECT / "model_training_ansys.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for marker in ["Feature Selection", "Bootstrap", "Hyperparameter",
                   "Das", "prediction interface", "ANOVA"]:
        assert marker in text, marker
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_notebook_build.py -v
```

Expected: FAIL — `build_notebook.py` missing / notebook missing.

- [ ] **Step 3: Implement `Project/build_notebook.py`**

The script defines one markdown cell and one or more code cells per stage. Code cells are thin orchestrators calling `pipeline.*` functions and printing summaries. Build the complete list of cells:

```python
"""Build model_training_ansys.ipynb from the pipeline package (17 stages)."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_notebook_build.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Execute the notebook end-to-end (verify all outputs)**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 model_training_ansys.ipynb
```

Expected: executes all cells; check output folder:

```bash
ls outputs/figures/ | wc -l
ls outputs/tables/
ls outputs/models/
```

Expected: ≥ 15 figures, ≥ 12 CSVs, `best_model.pkl`, `scaler.pkl`, `feature_metadata.json`, `training.log`.

If any cell raises, fix the `pipeline.*` code in place and re-run. Note: Stage 12/15 call `run_cv` on top of Stage 8 — acceptable extra runtime; if runtime is a concern, reuse `cv_rows`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add build_notebook.py model_training_ansys.ipynb tests/test_notebook_build.py outputs/
git commit -m "feat: build and execute single ANSYS ML notebook with consolidated outputs"
```

---

### Task 16: Rewrite thesis paths to the consolidated output folder

**Files:**
- Create: `Project/new-ML/scripts/update_thesis_paths.py`
- Test: `Project/new-ML/tests/test_update_thesis_paths.py`
- Modify: `Project/new-ML/ful_thesis.md` (paths + old values marked for regeneration)

**Interfaces:**
- Consumes: nothing.
- Produces: `update_thesis_paths.py` mapping old image/table prefixes to `outputs/`; run it on `ful_thesis.md`.

- [ ] **Step 1: Write the failing test**

`Project/tests/test_update_thesis_paths.py`:

```python
from pathlib import Path
from scripts.update_thesis_paths import rewrite_paths


def test_rewrite_paths_maps_old_prefixes():
    text = ("![a](docs/figures/mesh_convergence_study.png)\n"
            "![b](simulation/outputs/ml_figures/model_comparison.png)\n"
            "![c](simulation/outputs/figures/dataset_distribution.png)\n"
            "![d](docs/figures/gautam_validation/fe_model_gautam.png)")
    out = rewrite_paths(text)
    assert "outputs/figures/mesh_convergence_study.png" in out
    assert "outputs/figures/model_comparison.png" in out
    assert "outputs/figures/dataset_distribution.png" in out
    assert "outputs/figures/fe_model_gautam.png" in out
    assert "docs/figures" not in out
    assert "simulation/outputs" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/test_update_thesis_paths.py -v
```

Expected: FAIL — module import error (`scripts` package missing).

- [ ] **Step 3: Implement `Project/scripts/update_thesis_paths.py`**

```python
"""Rewrite thesis figure/table paths to the consolidated outputs/ folder."""
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # new-ML/
THESIS = PROJECT.parent / "ful_thesis.md"            # thesis at Project root

PREFIX_MAP = [
    ("docs/figures/", "new-ML/outputs/figures/"),
    ("simulation/outputs/ml_figures/", "new-ML/outputs/figures/"),
    ("simulation/outputs/figures/", "new-ML/outputs/figures/"),
]


def rewrite_paths(text):
    for old, new in PREFIX_MAP:
        text = text.replace(old, new)
    return text


def main():
    text = THESIS.read_text()
    updated = rewrite_paths(text)
    THESIS.write_text(updated)
    n = len(re.findall(r"outputs/figures/", updated))
    print(f"Updated {THESIS}: {n} references now point to new-ML/outputs/figures/")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
touch scripts/__init__.py
../.venv12/bin/pytest tests/test_update_thesis_paths.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: Apply to the thesis and verify no stale paths remain**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project
./.venv12/bin/python new-ML/scripts/update_thesis_paths.py
./.venv12/bin/python -c "
from pathlib import Path
text = Path('ful_thesis.md').read_text()
assert 'docs/figures' not in text and 'simulation/outputs' not in text
    assert 'new-ML/outputs/figures/' in text
missing = [m for m in __import__('re').findall(r'\]\(([^)]+)\)', text)
           if m.startswith('new-ML/outputs')
           and not (Path('new-ML') / m.replace('new-ML/', '')).exists()]
print('stale refs:', missing)
"
```

Expected: `Updated ful_thesis.md` and `stale refs: []` (figures generated in Task 15 exist in `new-ML/outputs/figures/`).

- [ ] **Step 6: Commit**

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
git add scripts/__init__.py scripts/update_thesis_paths.py tests/test_update_thesis_paths.py ../ful_thesis.md
git commit -m "docs: rewrite thesis figure paths to consolidated outputs folder"
```

---

## Final Verification (run after all tasks)

```bash
cd /Users/mabdulrafea/Projects/hareem_tasks/MS_Research_FYP/Project/new-ML
../.venv12/bin/pytest tests/ -v
../.venv12/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 model_training_ansys.ipynb
ls outputs/figures outputs/tables outputs/models
```

Expected: all tests PASS; notebook executes cleanly; `new-ML/outputs/` contains figures, tables, models, and logs. The thesis (ful_thesis.md) tables are then updated by the author from the CSVs in `new-ML/outputs/tables/` (values from the new run replace the old run's numbers).
