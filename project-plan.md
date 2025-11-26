# Project Plan: Predictive Modeling of Natural Frequency Shifts in Corroded Fixed RC Beams

This document outlines the plan for simulating the effects of corrosion on the natural frequency of Fixed-Fixed Reinforced Concrete (RC) beams and developing a predictive machine learning model.

## Part 1: Simulation Code Files

The simulation will be implemented in Python using Matrix Structural Analysis.

### 1. `fem_core.py` (The Physics Engine)

This file contains the core class and functions for the Finite Element Model.

- **Class:** `BeamFEM`
- **Dependencies:** `numpy`, `scipy.linalg`
- **Key Methods:**
  - `__init__(self, length, width, depth, concrete_strength, corrosion_level)`: Initializes beam properties.
  - `calculate_stiffness_reduction(self)`: Implements the "Stiffness Reduction Method".
    - **Formula:** $EI_{corroded} = EI_{original} \times (1 - \alpha)$
    - $\alpha$ is the damage factor proportional to `corrosion_level`.
  - `element_matrices(self, L_e, E, I, A, rho)`: Computes local stiffness ($k$) and mass ($m$) matrices for a beam element.
  - `assemble_global_matrices(self)`: Assembles local matrices into Global Stiffness ($[K]$) and Global Mass ($[M]$) matrices.
    - **Discretization:** 20 elements.
  - `apply_boundary_conditions(self)`: Applies Fixed-Fixed boundary conditions.
    - **Action:** Removes rows and columns corresponding to locked Degrees of Freedom (DOFs) at Node 0 and Node N (translation and rotation).
  - `solve_eigenvalues(self)`: Solves the Equation of Motion: $[K]\{u\} - \omega^2 [M]\{u\} = 0$.
    - **Algorithm:** `scipy.linalg.eigh(K, M)`
    - **Output:** Natural Frequencies ($f = \frac{\sqrt{\text{Eigenvalue}}}{2\pi}$) for Mode 1 and Mode 2.

### 2. `validation.py` (Quality Assurance)

This script runs specific tests to verify the FEM engine before dataset generation.

- **Test A (Theoretical):**
  - **Scenario:** Standard steel beam ($L=3m$).
  - **Benchmark:** Euler-Bernoulli formula for Fixed-Fixed beams.
  - **Success Criteria:** Error < 1%.
- **Test B (Experimental):**
  - **Scenario:** Sivasuriyan paper dimensions.
  - **Adjustment:** Temporarily modify BCs to match the paper (likely Simply Supported) if needed.
  - **Success Criteria:** Match "Mode 1" frequency.

### 3. `generate_dataset.py` (Data Factory)

This script generates the dataset for machine learning.

- **Sampling Strategy:** Latin Hypercube Sampling (LHS) using `scipy.stats.qmc`.
- **Dataset Structure:**
  - **Total Samples:** 2,000
  - **Pristine Data (1,500 rows):** Vary $L, b, h, f'c$ with `Corrosion = 0`.
  - **Corroded Data (500 rows):** Re-run 500 random beams from the pristine set with `Corrosion` values (e.g., 5%, 10%, 15%, 20%).
- **Output:** `beam_vibration_dataset.csv`
  - **Columns:** `ID`, `Length`, `Width`, `Depth`, `Conc_Strength`, `Corrosion_Level`, `Freq_Mode_1`, `Freq_Mode_2`.
- **Logging:** Implement logging to track progress and capture any simulation failures.

---

## Part 2: Model Training and Validation

This section describes the structure of the Jupyter Notebook (`model_training.ipynb`) for developing the predictive model.

### 1. Data Loading and Preprocessing

- **Load Data:** Read `beam_vibration_dataset.csv`.
- **Exploratory Data Analysis (EDA):**
  - Visualize distributions of input variables (Length, Width, Depth, Strength, Corrosion).
  - Plot correlations between `Corrosion_Level` and `Freq_Mode_1` to observe the physical relationship.
- **Data Splitting:**
  - Split data into Training (80%) and Testing (20%) sets.
  - Ensure stratified sampling if necessary to maintain corrosion distribution across splits.

### 2. Model Selection and Training

- **Algorithms to Try:**
  - Linear Regression (Baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - Support Vector Regression (SVR)
- **Training Loop:**
  - Train each model on the training set.
  - Use Cross-Validation (e.g., 5-fold) to tune hyperparameters and ensure robustness.

### 3. Model Evaluation

- **Metrics:**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared ($R^2$)
- **Comparison:** Compare models based on performance metrics on the test set to select the best performer.

### 4. Feature Importance Analysis

- Analyze which features (Length, Depth, Corrosion, etc.) have the most impact on the predicted frequency.
- **Method:** SHAP values or Permutation Importance.

### 5. Prediction Interface

- Create a simple function that takes beam parameters as input and outputs the predicted natural frequency.
- **Example:** `predict_frequency(length, width, depth, strength, corrosion)`
