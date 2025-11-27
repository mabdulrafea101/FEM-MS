# Project Specifications

## 1. Libraries and Packages

### 1.1 Simulation (`simulation/src`)

- **numpy** – core numerical operations, array handling, and linear algebra helpers.
- **scipy.linalg** – solves the generalized eigenvalue problem for the FEM model.
- **dataclasses** – provides the `@dataclass` decorator for the `SimulationResult` container.
- **logging** – records the progress of dataset generation and FEM solves.
- **os** – filesystem utilities for creating log directories and locating files.

### 1.2 Data Generation (`simulation/src/generate_dataset.py`)

- **numpy** – generates random damage parameters and stores beam geometry values.
- **pandas** – builds a `DataFrame` from the simulated results and writes the CSV dataset.
- **scipy.stats.qmc** – Latin‑Hypercube Sampling (LHS) to produce a space‑filling set of beam parameters.
- **logging** – logs the start, progress, and any errors during dataset creation.
- **os** – creates the `logs` folder and resolves the output CSV path.
- **BeamFEM** (from `fem_core.py`) – the finite‑element model used to compute natural frequencies for each sampled beam.

### 1.3 Visualization (`simulation/src/visualize_results.py`)

- **matplotlib.pyplot** – plots mode shapes, frequency distributions, and severity impact figures.
- **pandas** – reads the generated CSV for statistical visualisations.
- **numpy** – normalises mode‑shape amplitudes for comparison.
- **os** – builds the output directory for figures.
- **BeamFEM** – recomputes eigenvalues for a few representative damage cases to illustrate the FEM response.

### 1.4 Machine‑Learning (`model_training.ipynb`)

- **numpy** – array manipulation and numerical utilities.
- **pandas** – loads the dataset, performs exploratory data analysis, and stores intermediate results.
- **matplotlib.pyplot** & **seaborn** – visualise data distributions, model predictions, and residuals.
- **logging**, **pathlib**, **warnings** – initialise a reproducible training environment and handle noisy warnings.
- **scikit‑learn** (`model_selection`, `preprocessing`, `linear_model`, `ensemble`, `svm`, `metrics`, `inspection`) – provides train‑test splitting, scaling, five regression algorithms, cross‑validation, evaluation metrics (MAE, RMSE, R²), and permutation‑importance analysis.
- **xgboost** – gradient‑boosted trees implementation for high‑performance regression.
- **catboost** – another gradient‑boosted framework that handles categorical features natively (not used here but kept for completeness).
- **shap** – SHAP values for model‑agnostic feature importance visualisation.
- **joblib** – persists trained models to disk for later inference.

## 2. Simulation Workflow

1. **Define beam geometry & material** – length, width, depth, concrete strength, density, and number of FEM elements.
2. **Instantiate `BeamFEM`** – optionally specify a damage type (`none`, `corrosion`, `crack`, `random`) and its parameters.
3. **Assemble global stiffness & mass matrices** – using element‑level matrices derived from Euler‑Bernoulli beam theory.
4. **Apply fixed‑fixed boundary conditions** – remove the DOFs of the two fixed ends.
5. **Solve the generalized eigenvalue problem** – `K·u = ω²·M·u` to obtain natural frequencies and mode shapes.
6. **Generate a synthetic dataset** (`generate_dataset.py`):
   - Perform Latin‑Hypercube Sampling for the four geometric/material variables.
   - Loop over predefined damage scenarios (pristine, corrosion, crack, random) and compute the FEM response for each sample.
   - Store the beam parameters, damage description, severity metric, and the first two natural frequencies in a CSV file (`simulation/data/beam_vibration_dataset.csv`).
7. **Visualise results** (`visualize_results.py`):
   - Plot mode‑shape comparisons for selected damage cases.
   - Visualise the frequency distribution per damage type.
   - Examine the relationship between damage severity and the first natural frequency.

## 3. Machine‑Learning Workflow

1. **Load the dataset** – `pandas.read_csv` reads the CSV generated in step 6.
2. **Log basic information** – shape, column names, and data‑type summary.
3. **Check for missing values** – abort if any NaNs are found.
4. **Pre‑process**:
   - Encode the categorical `Damage_Type` (one‑hot or label encoding).
   - Split into training and test sets (`train_test_split`).
   - Standardise numeric features with `StandardScaler`.
5. **Model training** – fit five regressors on the training data:
   - `LinearRegression`
   - `RandomForestRegressor`
   - `XGBRegressor`
   - `CatBoostRegressor`
   - `SVR`
6. **Model evaluation** – compute MAE, RMSE, and R² on the test set for each model; optionally perform k‑fold cross‑validation.
7. **Feature‑importance analysis**:
   - Permutation importance (`sklearn.inspection.permutation_importance`).
   - SHAP values (`shap` library) for the tree‑based models.
8. **Visualise performance** – bar plots of metric scores, scatter plots of predicted vs. actual frequencies, and residual distributions.
9. **Persist models** – `joblib.dump` stores each trained model for later inference.

---

_All paths are relative to the project root. The specifications above capture the essential libraries, their rationale, and the end‑to‑end workflow for both the FEM simulation and the subsequent machine‑learning model training._
