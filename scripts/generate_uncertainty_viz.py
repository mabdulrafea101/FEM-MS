"""
Generate uncertainty quantification visualizations for ML predictions

This script generates bootstrap confidence intervals for model predictions
and creates visualizations for documentation.

Usage:
    python scripts/generate_uncertainty_viz.py

Requirements:
    - Trained model: simulation/models/catboost_model.pkl
    - Test data: simulation/data/test_data.csv

Outputs:
    - simulation/outputs/ml_figures/uncertainty_quantification.png
    - simulation/outputs/ml_figures/coverage_analysis.png
    - simulation/outputs/ml_figures/uncertainty_stats.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def generate_bootstrap_predictions(model, X_test, y_test, n_bootstrap=100):
    """
    Generate bootstrap confidence intervals for predictions

    Args:
        model: Trained model (ensemble)
        X_test: Test features
        y_test: Test targets
        n_bootstrap: Number of bootstrap iterations

    Returns:
        mean_pred, std_pred, lower_95, upper_95
    """
    print("Generating bootstrap predictions...")
    predictions = []

    # If model has base estimators (ensemble), use them
    if hasattr(model, 'estimators_'):
        n_estimators = min(len(model.estimators_), n_bootstrap)
        print(f"Using {n_estimators} estimators from ensemble model")
        for estimator in model.estimators_[:n_estimators]:
            pred = estimator.predict(X_test)
            predictions.append(pred)
    else:
        # For non-ensemble models, bootstrap the data
        print(f"Bootstrapping data {n_bootstrap} times")
        indices = np.arange(len(X_test))
        for i in range(n_bootstrap):
            if (i + 1) % 10 == 0:
                print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
            boot_idx = np.random.choice(indices, size=len(indices), replace=True)
            pred = model.predict(X_test[boot_idx])
            predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    lower_95 = np.percentile(predictions, 2.5, axis=0)
    upper_95 = np.percentile(predictions, 97.5, axis=0)

    print("Bootstrap predictions generated successfully")
    return mean_pred, std_pred, lower_95, upper_95

def plot_uncertainty_intervals(y_test, mean_pred, lower_95, upper_95, model_name='CatBoost'):
    """
    Plot prediction intervals

    Args:
        y_test: Actual values
        mean_pred: Mean predictions
        lower_95: Lower 95% CI
        upper_95: Upper 95% CI
        model_name: Name of the model

    Returns:
        interval_width: Array of CI widths
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sort by actual values for better visualization
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]
    mean_sorted = mean_pred[sort_idx]
    lower_sorted = lower_95[sort_idx]
    upper_sorted = upper_95[sort_idx]

    # Plot 1: Predictions with confidence intervals (show first 200 samples)
    n_show = min(200, len(y_sorted))
    x_range = np.arange(n_show)
    ax1.plot(x_range, y_sorted[:n_show], 'ko', markersize=3, alpha=0.5, label='Actual')
    ax1.plot(x_range, mean_sorted[:n_show], 'b-', linewidth=2, label='Predicted')
    ax1.fill_between(x_range, lower_sorted[:n_show], upper_sorted[:n_show],
                      alpha=0.3, label='95% CI')
    ax1.set_xlabel('Sample Index (sorted by actual frequency)')
    ax1.set_ylabel('Natural Frequency (Hz)')
    ax1.set_title(f'{model_name}: Predictions with 95% Confidence Intervals\n(First {n_show} samples shown)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Interval width distribution
    interval_width = upper_95 - lower_95
    ax2.hist(interval_width, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(np.mean(interval_width), color='r', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(interval_width):.2f} Hz')
    ax2.axvline(np.median(interval_width), color='g', linestyle='--',
                linewidth=2, label=f'Median: {np.median(interval_width):.2f} Hz')
    ax2.set_xlabel('95% Confidence Interval Width (Hz)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs('simulation/outputs/ml_figures', exist_ok=True)
    output_path = 'simulation/outputs/ml_figures/uncertainty_quantification.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return interval_width

def plot_coverage_analysis(y_test, lower_95, upper_95):
    """
    Analyze coverage of confidence intervals

    Args:
        y_test: Actual values
        lower_95: Lower 95% CI
        upper_95: Upper 95% CI

    Returns:
        coverage: Coverage percentage
    """
    # Check coverage
    coverage = np.mean((y_test >= lower_95) & (y_test <= upper_95))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot scatter with coverage coloring
    within_ci = (y_test >= lower_95) & (y_test <= upper_95)
    ax.scatter(y_test[within_ci], (upper_95 - lower_95)[within_ci],
              c='green', alpha=0.6, s=30, label=f'Within CI ({coverage*100:.1f}%)')
    ax.scatter(y_test[~within_ci], (upper_95 - lower_95)[~within_ci],
              c='red', alpha=0.6, s=30, label=f'Outside CI ({(1-coverage)*100:.1f}%)')

    ax.set_xlabel('Actual Natural Frequency (Hz)')
    ax.set_ylabel('95% CI Width (Hz)')
    ax.set_title(f'Confidence Interval Coverage Analysis\n(Target: 95%, Actual: {coverage*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'simulation/outputs/ml_figures/coverage_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return coverage

def main():
    """
    Main function to generate all uncertainty visualizations
    """
    print("=" * 80)
    print("Uncertainty Quantification Visualization Generator")
    print("=" * 80)

    # Check if required files exist
    model_path = Path('simulation/models/catboost_model.pkl')
    test_data_path = Path('simulation/data/test_data.csv')

    if not model_path.exists():
        print(f"\n⚠ Warning: Model file not found at {model_path}")
        print("This script requires a trained CatBoost model.")
        print("\nPlease either:")
        print("  1. Train a model and save it to simulation/models/catboost_model.pkl")
        print("  2. Adjust the path in this script to point to your model")
        return

    if not test_data_path.exists():
        print(f"\n⚠ Warning: Test data not found at {test_data_path}")
        print("This script requires test data with features and target variable.")
        print("\nPlease either:")
        print("  1. Save your test data to simulation/data/test_data.csv")
        print("  2. Adjust the path in this script to point to your test data")
        return

    try:
        # Load model and data
        print("\nLoading model and test data...")
        model = joblib.load(model_path)
        test_data = pd.read_csv(test_data_path)

        print(f"  Model loaded: {type(model).__name__}")
        print(f"  Test data shape: {test_data.shape}")

        # Prepare data
        X_test = test_data.drop(['Mode_1_Freq', 'Mode_2_Freq'], axis=1, errors='ignore')

        # Determine target column
        if 'Mode_1_Freq' in test_data.columns:
            y_test = test_data['Mode_1_Freq'].values
            print("  Target: Mode_1_Freq")
        elif 'Natural_Frequency' in test_data.columns:
            y_test = test_data['Natural_Frequency'].values
            print("  Target: Natural_Frequency")
        else:
            # Use last column as target
            y_test = test_data.iloc[:, -1].values
            print(f"  Target: {test_data.columns[-1]}")

        print(f"  Features: {X_test.columns.tolist()}")

        # Generate bootstrap predictions
        mean_pred, std_pred, lower_95, upper_95 = generate_bootstrap_predictions(
            model, X_test, y_test, n_bootstrap=100
        )

        # Create visualizations
        print("\nGenerating uncertainty interval plots...")
        interval_width = plot_uncertainty_intervals(y_test, mean_pred, lower_95, upper_95)

        print("\nGenerating coverage analysis...")
        coverage = plot_coverage_analysis(y_test, lower_95, upper_95)

        # Print summary statistics
        print("\n" + "=" * 80)
        print("UNCERTAINTY QUANTIFICATION SUMMARY")
        print("=" * 80)
        print(f"Mean Prediction Interval Width:   {np.mean(interval_width):8.2f} Hz")
        print(f"Median Prediction Interval Width: {np.median(interval_width):8.2f} Hz")
        print(f"Std of Prediction Interval Width: {np.std(interval_width):8.2f} Hz")
        print(f"Min Interval Width:                {np.min(interval_width):8.2f} Hz")
        print(f"Max Interval Width:                {np.max(interval_width):8.2f} Hz")
        print(f"95% CI Coverage:                   {coverage*100:8.2f}% (target: 95%)")
        print(f"Mean Prediction Std:               {np.mean(std_pred):8.2f} Hz")
        print("=" * 80)

        # Save statistics
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean CI Width (Hz)',
                'Median CI Width (Hz)',
                'Std CI Width (Hz)',
                'Min CI Width (Hz)',
                'Max CI Width (Hz)',
                '95% Coverage (%)',
                'Mean Pred Std (Hz)'
            ],
            'Value': [
                np.mean(interval_width),
                np.median(interval_width),
                np.std(interval_width),
                np.min(interval_width),
                np.max(interval_width),
                coverage*100,
                np.mean(std_pred)
            ]
        })

        stats_path = 'simulation/outputs/ml_figures/uncertainty_stats.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✓ Saved: {stats_path}")

        print("\n" + "=" * 80)
        print("SUCCESS: All uncertainty quantification visualizations generated!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  1. simulation/outputs/ml_figures/uncertainty_quantification.png")
        print("  2. simulation/outputs/ml_figures/coverage_analysis.png")
        print("  3. simulation/outputs/ml_figures/uncertainty_stats.csv")
        print("\nNext steps:")
        print("  1. Add these figures to documentation.md Section 4.8.4.1")
        print("  2. Update text with actual statistics from uncertainty_stats.csv")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check that:")
        print("  1. Model file is valid and loadable")
        print("  2. Test data has correct format (features + target columns)")
        print("  3. Required libraries are installed (pandas, numpy, matplotlib, seaborn, joblib)")

if __name__ == "__main__":
    main()
