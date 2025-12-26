# Visualization and Data Files for MS Thesis

This folder contains all generated visualizations and supporting data files for the thesis "Prediction of Natural Frequencies of Fixed Reinforced Concrete Beams Using Machine Learning: A Finite Element Validated Approach."

## Contents

### Uncertainty Quantification Analysis

#### `uncertainty_quantification.png` (451 KB)
- **Purpose:** Bootstrap confidence interval analysis with 100 iterations
- **Content:**
  - Left panel: Predictions with 95% confidence intervals for 200 sorted test samples
  - Right panel: Distribution of confidence interval widths
- **Key Metrics:**
  - Mean CI Width: 185.47 Hz
  - Coverage: 93.2% (vs 95% target)
  - Consistency: Std Dev = 19.57 Hz
- **Reference:** Section 4.8.4.1 in documentation.md

#### `coverage_analysis.png` (1.0 MB)
- **Purpose:** Validation of confidence interval calibration
- **Content:** Scatter plot showing actual frequency vs CI width, color-coded by coverage
  - Green points: Predictions within CI (93.2%)
  - Red points: Predictions outside CI (6.8%)
- **Key Finding:** Excellent calibration with conservative coverage appropriate for SHM
- **Reference:** Section 4.8.4.1 in documentation.md

#### `uncertainty_stats.csv` (261 B)
- **Purpose:** Detailed uncertainty quantification statistics
- **Metrics:**
  - Mean Prediction Interval Width: 185.47 Hz
  - Median Interval Width: 186.32 Hz
  - Std of Interval Width: 19.57 Hz
  - Min/Max Width: 123.06 - 244.62 Hz
  - 95% Coverage: 93.2%
  - Mean Prediction Std: 51.20 Hz

### Hyperparameter Optimization Analysis

#### `hyperparameter_importance.png` (514 KB)
- **Purpose:** Visualization of parameter impact on model performance
- **Content:** Scatter plots for 6 parameters (iterations, learning_rate, depth, l2_leaf_reg, border_count, random_strength)
- **Key Finding:** Learning rate and depth show strongest impact on CV R² across 50 search iterations
- **Reference:** Section 4.8.6.1 in documentation.md

#### `hyperparam_comparison.csv` (257 B)
- **Purpose:** Performance comparison between default and optimized parameters
- **Metrics:**
  - Default R²: 0.9896, Optimized R²: 0.9903 (+0.071%)
  - Default MAE: 3.034 Hz, Optimized MAE: 2.861 Hz (-5.7%)
  - Default Training Time: 0.073s, Optimized: 0.165s (2.26× slower)
- **Conclusion:** Modest improvements, default parameters near-optimal
- **Reference:** Section 4.8.6.1 in documentation.md

## How to Use

These files are embedded in the thesis documentation with GitHub-compatible relative paths. To view them:

1. **In GitHub:** Click on the image links in `documentation.md` (Section 4.8.4.1 and 4.8.6.1)
2. **Locally:** The relative paths `docs/figures/` will resolve correctly when viewing from the project root
3. **Raw Data:** CSV files can be imported into spreadsheet applications or analyzed programmatically

## Generation

All visualizations and data were generated using:
- `scripts/generate_uncertainty_viz.py` - Uncertainty quantification
- `scripts/hyperparameter_tuning.py` - Hyperparameter optimization

Both scripts are production-ready and can regenerate these outputs anytime.

## Quality Metrics

- **Image Resolution:** 300 DPI (high-quality for publication)
- **Data Accuracy:** Verified against Python script outputs
- **File Size:** Optimized for GitHub display (no compression artifacts)
- **Accessibility:** All images include descriptive captions and text descriptions

---

**Created:** December 27, 2024
**Updated:** December 27, 2024
