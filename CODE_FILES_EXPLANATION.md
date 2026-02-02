# Complete Code Files Documentation

## All Coding Files Used in the MS Research FYP Thesis

**Thesis Title:** Prediction of Natural Frequencies of Fixed Reinforced Concrete Beams Using Machine Learning: A Finite Element Validated Approach

---

## Project Structure Overview

```
Project/
├── simulation/                          # Main simulation module
│   ├── src/
│   │   ├── fem_core.py                 # Core FEM engine
│   │   ├── generate_dataset.py         # Dataset generation
│   │   └── visualize_results.py        # Visualization utilities
│   ├── data/
│   │   └── beam_vibration_dataset.csv  # Generated dataset
│   ├── models/
│   │   ├── best_model_CatBoost.pkl     # Trained ML model
│   │   └── scaler.pkl                  # Feature scaler
│   ├── outputs/
│   │   ├── figures/                    # FEM output figures
│   │   └── ml_figures/                 # ML training figures
│   ├── logs/
│   │   └── generation.log              # Dataset generation log
│   └── tests/
│       └── validation.py               # Basic tests
│
├── scripts/                             # Validation and analysis scripts
│   ├── validate_fem_das2023.py         # Das (2023) validation
│   ├── validate_gautam_2016.py         # Gautam (2016) validation
│   ├── validate_rc_beam.py             # RC beam validation
│   ├── validate_massenzio_2005.py      # Massenzio crack validation
│   ├── comprehensive_validation.py     # Combined validation
│   ├── statistical_tests.py            # Statistical analysis
│   ├── learning_curve_analysis.py      # ML learning curves
│   ├── generate_uncertainty_viz.py     # Uncertainty quantification
│   └── hyperparameter_tuning.py        # Hyperparameter optimization
│
├── model_training.ipynb                 # Jupyter notebook for ML
├── docs/
│   └── figures/                        # Validation figures
│
└── ful_thesis.md                       # Main thesis document
```

---

## 1. FEM Core Engine: `fem_core.py`

**Location:** `simulation/src/fem_core.py`

**Lines of Code:** ~220

**Purpose:** Implements the complete Finite Element Method for Euler-Bernoulli beam vibration analysis with damage modeling.

### Key Components

#### 1.1 BeamFEM Class Initialization

```python
class BeamFEM:
    def __init__(self, L, b, h, fc, damage_type='none', damage_params=None,
                 n_elements=20, rho=2400):
        """
        Initialize a reinforced concrete beam for FEM analysis.

        Parameters:
        -----------
        L : float
            Beam length in meters (valid range: 3-8 m)
        b : float
            Beam width in meters (valid range: 0.2-0.5 m)
        h : float
            Beam depth in meters (valid range: 0.3-0.8 m)
        fc : float
            Concrete compressive strength in MPa (valid range: 25-50 MPa)
        damage_type : str
            Type of damage: 'none', 'corrosion', 'crack', 'random'
        damage_params : dict
            Parameters specific to damage type
        n_elements : int
            Number of finite elements (default: 20)
        rho : float
            Concrete density in kg/m³ (default: 2400)
        """
        self.L = L
        self.b = b
        self.h = h
        self.fc = fc

        # Material properties (ACI 318-19 - Eq. 3)
        self.E = 4700 * np.sqrt(self.fc) * 1e6  # Pa

        # Section properties
        self.I = b * h**3 / 12  # Moment of inertia (Eq. 12)
        self.A = b * h          # Cross-sectional area
        self.rho = rho

        # FEM setup
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.dof_per_node = 2  # [v, θ] at each node
        self.total_dof = self.n_nodes * self.dof_per_node

        # Mesh generation
        self.node_coords = np.linspace(0, L, self.n_nodes)
        self.element_lengths = np.diff(self.node_coords)

        # Damage configuration
        self.damage_type = damage_type
        self.damage_params = damage_params or {}

        # Calculate stiffness profile (with damage)
        self._calculate_stiffness_profile()
```

#### 1.2 Damage Modeling - Stiffness Profile Calculation

```python
def _calculate_stiffness_profile(self):
    """
    Calculate the stiffness (I_effective) profile along the beam.

    Implements damage models:
    - Corrosion: Eq. 15, 16 (Rodriguez et al., 1997)
    - Crack: Eq. 17 (Dimarogonas, 1996)
    - Random: Eq. 18 (Thesis formulation)
    """
    # Start with uniform moment of inertia
    I_profile = np.ones(self.n_elements) * self.I

    if self.damage_type == 'corrosion':
        # Uniform corrosion reduces stiffness throughout beam
        # Eq. 16: α = min(1.6 × C/100, 0.9)
        level = self.damage_params.get('level', 0)  # Corrosion %
        alpha = min(1.6 * (level / 100.0), 0.9)
        # Eq. 15: I_corroded = I_original × (1 - α)
        I_profile *= (1 - alpha)

    elif self.damage_type == 'crack':
        # Localized crack at specified location
        loc = self.damage_params.get('location', self.L / 2)
        severity = self.damage_params.get('severity', 0.5)
        width = self.damage_params.get('width', self.L / 10)

        # Find elements within crack zone
        elem_centers = self.node_coords[:-1] + self.element_lengths / 2
        mask = np.abs(elem_centers - loc) <= (width / 2)

        # Eq. 17: I_effective = I_original × (1 - β)
        I_profile[mask] *= (1 - severity)

    elif self.damage_type == 'random':
        # Multiple random cracks at random locations
        count = self.damage_params.get('count', 3)
        sev_min, sev_max = self.damage_params.get('severity_range', (0.1, 0.4))

        # Randomly select elements
        indices = np.random.choice(self.n_elements, count, replace=False)
        severities = np.random.uniform(sev_min, sev_max, count)

        # Eq. 18: Apply damage to selected elements
        for idx, sev in zip(indices, severities):
            I_profile[idx] *= (1 - sev)

    self.I_profile = I_profile
```

#### 1.3 Element Matrix Assembly

```python
def _element_matrices(self, elem_idx):
    """
    Compute element stiffness and mass matrices.

    Implements:
    - Eq. 13: Euler-Bernoulli stiffness matrix
    - Eq. 14: Consistent mass matrix

    Source: Zienkiewicz & Taylor (2000), Bathe (2014)
    """
    le = self.element_lengths[elem_idx]
    I_elem = self.I_profile[elem_idx]
    E = self.E
    rho = self.rho
    A = self.A

    # Element stiffness matrix (Eq. 13)
    # [k]e = (EI/Le³) × [standard coefficients]
    k_factor = (E * I_elem) / (le**3)
    k = k_factor * np.array([
        [12,      6*le,    -12,      6*le   ],
        [6*le,    4*le**2, -6*le,    2*le**2],
        [-12,    -6*le,     12,     -6*le   ],
        [6*le,    2*le**2, -6*le,    4*le**2]
    ])

    # Element mass matrix (Eq. 14)
    # [m]e = (ρALe/420) × [standard coefficients]
    m_factor = (rho * A * le) / 420
    m = m_factor * np.array([
        [156,     22*le,    54,     -13*le  ],
        [22*le,   4*le**2,  13*le,  -3*le**2],
        [54,      13*le,    156,    -22*le  ],
        [-13*le, -3*le**2, -22*le,   4*le**2]
    ])

    return k, m
```

#### 1.4 Eigenvalue Problem Solution

```python
def solve_eigenvalues(self):
    """
    Solve the generalized eigenvalue problem for natural frequencies.

    Implements Eq. 10: [K]{u} = ω²[M]{u}

    Returns:
    --------
    Result object with:
        - frequencies: Natural frequencies in Hz
        - mode_shapes: Eigenvectors (displacement DOFs only)
        - nodes: Node coordinates
    """
    # Assemble global matrices
    K, M = self.assemble_global_matrices()

    # Apply fixed-fixed boundary conditions
    # Remove DOFs: [v₀, θ₀, v_n, θ_n]
    K_red, M_red, free_dof = self.apply_boundary_conditions(K, M)

    # Solve eigenvalue problem using scipy
    # scipy.linalg.eigh returns: eigenvalues (ω²), eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(K_red, M_red)

    # Convert to frequencies (Eq. 11: f = ω/2π)
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

    # Extract mode shapes (displacement DOFs only)
    n_modes = min(5, len(frequencies))

    return Result(
        frequencies=frequencies[:n_modes],
        mode_shapes=eigenvectors[:, :n_modes],
        nodes=self.node_coords
    )
```

---

## 2. Dataset Generator: `generate_dataset.py`

**Location:** `simulation/src/generate_dataset.py`

**Lines of Code:** ~130

**Purpose:** Generates 3,000 beam samples using Latin Hypercube Sampling for uniform parameter space coverage.

### Key Code

```python
import numpy as np
import pandas as pd
from scipy.stats import qmc
from fem_core import BeamFEM

def generate_dataset():
    """
    Generate dataset of beam frequencies for ML training.

    Sampling Method: Latin Hypercube Sampling (LHS)
    Total Samples: 3,000
      - 1,000 Pristine (no damage)
      - 700 Corroded (5-30% corrosion)
      - 700 Cracked (single crack, 10-70% severity)
      - 600 Random damage (2-4 cracks)

    Parameter Ranges:
      - Length: 3.0-8.0 m
      - Width: 0.2-0.5 m
      - Depth: 0.3-0.8 m
      - Concrete Strength: 25-50 MPa
    """
    # Define scenarios
    scenarios = [
        {'type': 'none', 'count': 1000},
        {'type': 'corrosion', 'count': 700},
        {'type': 'crack', 'count': 700},
        {'type': 'random', 'count': 600}
    ]

    total_samples = sum(s['count'] for s in scenarios)

    # Latin Hypercube Sampling for parameter coverage
    sampler = qmc.LatinHypercube(d=4, seed=42)
    sample = sampler.random(n=total_samples)

    # Scale to actual parameter ranges
    l_bounds = [3.0, 8.0]
    b_bounds = [0.2, 0.5]
    h_bounds = [0.3, 0.8]
    fc_bounds = [25, 50]

    lengths = qmc.scale(sample[:, 0:1], l_bounds[0], l_bounds[1]).flatten()
    widths = qmc.scale(sample[:, 1:2], b_bounds[0], b_bounds[1]).flatten()
    depths = qmc.scale(sample[:, 2:3], h_bounds[0], h_bounds[1]).flatten()
    strengths = qmc.scale(sample[:, 3:4], fc_bounds[0], fc_bounds[1]).flatten()

    data = []
    current_idx = 0

    for scenario in scenarios:
        sType = scenario['type']
        count = scenario['count']

        for i in range(count):
            idx = current_idx + i
            L = lengths[idx]
            b = widths[idx]
            h = depths[idx]
            fc = strengths[idx]

            # Configure damage parameters
            damage_params = {}
            severity_metric = 0.0

            if sType == 'corrosion':
                level = np.random.uniform(5, 30)  # 5-30% corrosion
                damage_params = {'level': level}
                severity_metric = level

            elif sType == 'crack':
                loc = np.random.uniform(0.1*L, 0.9*L)
                sev = np.random.uniform(0.1, 0.7)  # 10-70% severity
                width = np.random.uniform(0.1, 0.5)
                damage_params = {'location': loc, 'severity': sev, 'width': width}
                severity_metric = sev * 100

            elif sType == 'random':
                cnt = np.random.randint(2, 5)
                damage_params = {'count': cnt, 'severity_range': (0.1, 0.5)}
                severity_metric = 0.3 * cnt * 100

            # Run FEM simulation
            beam = BeamFEM(L, b, h, fc, damage_type=sType, damage_params=damage_params)
            res = beam.solve_eigenvalues()

            # Store results
            row = {
                'ID': idx,
                'Length': L,
                'Width': b,
                'Depth': h,
                'Conc_Strength': fc,
                'Damage_Type': sType,
                'Damage_Severity': severity_metric,
                'Freq_Mode_1': res.frequencies[0],
                'Freq_Mode_2': res.frequencies[1]
            }
            data.append(row)

        current_idx += count

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('beam_vibration_dataset.csv', index=False)

    return df
```

---

## 3. Validation Scripts

### 3.1 Gautam 2016 Validation: `validate_gautam_2016.py`

**Purpose:** Validates FEM against published ANSYS results for fixed-fixed steel beam.

```python
# Reference values from Gautam et al. (2016) Table 5
TARGET_ANALYTICAL = {'f1': 132.04, 'f2': 357.30, 'f3': 687.72}
TARGET_ANSYS = {'f1': 132.04, 'f2': 357.80, 'f3': 687.19}

# Beam parameters from Table 4
L = 2.0            # Beam length (m)
b = 0.3            # Width (m)
h = 0.1            # Height (m)
E = 20.5e10        # Elastic modulus (Pa) = 205 GPa
rho = 7830         # Density (kg/m³)

def calculate_theoretical_frequencies(E, I, rho, A, L, n_modes=3):
    """
    Calculate theoretical frequencies using Euler-Bernoulli theory.

    Eq. 2: f_n = (λ_n² / 2πL²) √(EI/ρA)

    λ_n values for fixed-fixed beam:
    Mode 1: 4.730041
    Mode 2: 7.853205
    Mode 3: 10.995608
    """
    lambda_n = [4.730041, 7.853205, 10.995608]

    frequencies = []
    for i in range(n_modes):
        lam = lambda_n[i]
        f_n = (lam**2 / (2 * np.pi * L**2)) * np.sqrt(E * I / (rho * A))
        frequencies.append(f_n)

    return frequencies
```

### 3.2 Das 2023 Validation: `validate_fem_das2023.py`

**Purpose:** Validates FEM against Das (2023) cantilever beam ANSYS results.

```python
# Reference: Das (2023) Table 3 - Aluminum beam
# Case A: h/L = 1/48
BEAM_CONFIG = {
    'L': 1.2,           # Length (m)
    'b': 0.025,         # Width (m)
    'h': 0.025,         # Height (m)
    'E': 72e9,          # Elastic modulus (Pa) = 72 GPa
    'rho': 2810,        # Density (kg/m³)
    'BC': 'cantilever'  # Boundary condition
}

TARGET_DAS_ANSYS = {
    'f1': 13.552,
    'f2': 84.816,
    'f3': 237.030,
    'f4': 463.260,
    'f5': 763.260
}
```

### 3.3 RC Beam Validation: `validate_rc_beam.py`

**Purpose:** Validates corrosion-frequency relationship against Zhang et al. (2020).

```python
# Zhang et al. (2020) experimental findings:
# - ~0.8% frequency reduction per 1% corrosion
# - Second mode more sensitive than first mode
# - Nonlinear decay pattern

def validate_corrosion_sensitivity():
    """
    Validate that FEM produces correct corrosion-frequency sensitivity.

    Expected: ~0.8% frequency drop per 1% corrosion increase
    """
    L, b, h, fc = 5.0, 0.3, 0.5, 35

    # Pristine beam
    beam_0 = BeamFEM(L, b, h, fc, damage_type='none')
    f_pristine = beam_0.solve_eigenvalues().frequencies[0]

    # Test corrosion levels
    corrosion_levels = [0, 5, 10, 15, 20]
    sensitivities = []

    for c in corrosion_levels[1:]:
        beam = BeamFEM(L, b, h, fc, damage_type='corrosion',
                       damage_params={'level': c})
        f_damaged = beam.solve_eigenvalues().frequencies[0]

        freq_drop_pct = (f_pristine - f_damaged) / f_pristine * 100
        sensitivity = freq_drop_pct / c  # % freq drop per % corrosion

        sensitivities.append(sensitivity)

    avg_sensitivity = np.mean(sensitivities)
    # Expected: ~0.8
    return avg_sensitivity
```

---

## 4. ML Training Notebook: `model_training.ipynb`

**Total Cells:** 46

### 4.1 Key Cells Summary

| Cell Range | Purpose | Key Functions |
|------------|---------|---------------|
| 1-5 | Setup | Import libraries, configure logging |
| 6-9 | Data Loading | Load CSV, describe statistics |
| 10-14 | EDA | Histograms, correlation matrix, scatter plots |
| 15-18 | Preprocessing | Train/test split, StandardScaler |
| 19-30 | Model Training | Train 5 models with evaluation |
| 31-36 | Comparison | Bar charts, prediction plots, residuals |
| 37-40 | Interpretation | Feature importance, SHAP analysis |
| 41-42 | Save Model | Export CatBoost and scaler |
| 43-46 | Summary | Prediction interface, final summary |

### 4.2 Model Training Code

```python
# Cell 20: Evaluation function
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, use_scaling=True):
    """Train and evaluate a model with comprehensive metrics."""

    # Train
    model.fit(X_train_data, y_tr)

    # Predictions
    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    # Metrics
    train_mae = mean_absolute_error(y_tr, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
    train_r2 = r2_score(y_tr, y_train_pred)

    test_mae = mean_absolute_error(y_te, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_test_pred))
    test_r2 = r2_score(y_te, y_test_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_data, y_tr, cv=5, scoring='r2')

    return model, y_test_pred

# Cell 28: CatBoost training
cb_model = CatBoostRegressor(
    iterations=200,
    depth=8,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)
cb_trained, cb_predictions = evaluate_model(
    'CatBoost', cb_model, X_train_scaled, y_train, X_test_scaled, y_test
)

# Cell 40: SHAP analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols)

# Cell 42: Save best model
joblib.dump(best_model, 'simulation/models/best_model_CatBoost.pkl')
joblib.dump(scaler, 'simulation/models/scaler.pkl')
```

---

## 5. Visualization: `visualize_results.py`

**Purpose:** Generate thesis figures showing mode shapes and frequency distributions.

```python
def plot_mode_shapes():
    """
    Compare mode shapes for normal, corroded, and cracked beams.

    Output: simulation/outputs/figures/mode_shape_comparison.png
    """
    L, b, h, fc = 5.0, 0.3, 0.5, 30

    # Three beam conditions
    beam_norm = BeamFEM(L, b, h, fc, damage_type='none')
    beam_corr = BeamFEM(L, b, h, fc, damage_type='corrosion',
                        damage_params={'level': 20})
    beam_crack = BeamFEM(L, b, h, fc, damage_type='crack',
                         damage_params={'location': L/2, 'severity': 0.5})

    res_norm = beam_norm.solve_eigenvalues()
    res_corr = beam_corr.solve_eigenvalues()
    res_crack = beam_crack.solve_eigenvalues()

    # Plot Mode 1 comparison
    plt.figure(figsize=(12, 6))
    plt.plot(res_norm.nodes, normalize(res_norm.mode_shapes[:, 0]),
             'b-', label=f'Normal (f={res_norm.frequencies[0]:.2f} Hz)')
    plt.plot(res_corr.nodes, normalize(res_corr.mode_shapes[:, 0]),
             'r--', label=f'Corroded 20% (f={res_corr.frequencies[0]:.2f} Hz)')
    plt.plot(res_crack.nodes, normalize(res_crack.mode_shapes[:, 0]),
             'g-.', label=f'Cracked (f={res_crack.frequencies[0]:.2f} Hz)')

    plt.xlabel('Position along beam (m)')
    plt.ylabel('Normalized displacement')
    plt.legend()
    plt.savefig('mode_shape_comparison.png')
```

---

## 6. Output Files Summary

### 6.1 Data Files

| File | Location | Description |
|------|----------|-------------|
| `beam_vibration_dataset.csv` | simulation/data/ | 3,000 samples, 9 columns |
| `best_model_CatBoost.pkl` | simulation/models/ | Trained CatBoost model |
| `scaler.pkl` | simulation/models/ | StandardScaler for features |

### 6.2 Figure Outputs

| Figure | Location | Used In |
|--------|----------|---------|
| `parameter_distributions.png` | outputs/ml_figures/ | Chapter 4.3 |
| `correlation_matrix.png` | outputs/ml_figures/ | Chapter 4.3 |
| `damage_vs_frequency.png` | outputs/ml_figures/ | Chapter 4.4 |
| `model_comparison.png` | outputs/ml_figures/ | Chapter 4.8 |
| `prediction_vs_actual.png` | outputs/ml_figures/ | Chapter 4.8 |
| `residual_plots.png` | outputs/ml_figures/ | Chapter 4.8 |
| `feature_importance.png` | outputs/ml_figures/ | Chapter 4.8 |
| `shap_summary.png` | outputs/ml_figures/ | Chapter 4.8 |
| `mode_shape_comparison.png` | outputs/figures/ | Chapter 4.4 |

### 6.3 Log Files

| File | Location | Contents |
|------|----------|----------|
| `generation.log` | simulation/logs/ | Dataset generation progress |
| `ml_training.log` | simulation/logs/ | ML training progress |

---

## 7. Dependencies

```txt
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
catboost>=1.0.0
shap>=0.40.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Model persistence
joblib>=1.1.0

# Jupyter
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

*Document created: January 2025*
*For MS Research FYP - Code Documentation*
