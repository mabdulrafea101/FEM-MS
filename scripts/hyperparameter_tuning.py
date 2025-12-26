"""
Systematic hyperparameter optimization for CatBoost model

This script performs randomized search for optimal hyperparameters
and compares default vs optimized model performance.

Usage:
    python scripts/hyperparameter_tuning.py

Requirements:
    - Training data: simulation/data/train_data.csv
    - Test data: simulation/data/test_data.csv

Outputs:
    - simulation/outputs/ml_figures/hyperparameter_importance.png
    - simulation/outputs/ml_figures/hyperparam_comparison.csv
    - simulation/outputs/ml_figures/optimization_results.txt
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from pathlib import Path

sns.set_style("whitegrid")

def define_param_grid():
    """
    Define hyperparameter search space for CatBoost

    Returns:
        param_distributions: Dictionary of parameter distributions
    """
    param_distributions = {
        'iterations': randint(50, 500),           # Number of boosting iterations
        'learning_rate': uniform(0.01, 0.3),     # Step size shrinkage
        'depth': randint(4, 10),                  # Tree depth
        'l2_leaf_reg': uniform(1, 10),           # L2 regularization
        'border_count': randint(32, 255),        # Number of splits for numerical features
        'random_strength': uniform(0, 10)        # Randomness for scoring splits
    }
    return param_distributions

def run_hyperparameter_search(X_train, y_train, n_iter=50, cv=5):
    """
    Run randomized hyperparameter search

    Args:
        X_train: Training features
        y_train: Training targets
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds

    Returns:
        random_search: Fitted RandomizedSearchCV object
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    # Base model
    base_model = CatBoostRegressor(
        random_seed=42,
        verbose=False,
        thread_count=-1
    )

    # Parameter grid
    param_dist = define_param_grid()

    print("\nParameter Search Space:")
    for param, dist in param_dist.items():
        print(f"  {param:20s}: {dist}")

    # Randomized search
    print(f"\nRunning randomized search:")
    print(f"  Iterations: {n_iter}")
    print(f"  Cross-validation folds: {cv}")
    print(f"  Scoring metric: R²")

    start_time = time.time()

    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

    # Results
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    for param, value in random_search.best_params_.items():
        if isinstance(value, float):
            print(f"{param:20s}: {value:.4f}")
        else:
            print(f"{param:20s}: {value}")

    print(f"\nBest CV R² Score: {random_search.best_score_:.6f}")
    print(f"Std of CV R² Score: {random_search.cv_results_['std_test_score'][random_search.best_index_]:.6f}")

    return random_search

def plot_hyperparameter_importance(cv_results, output_dir='simulation/outputs/ml_figures'):
    """
    Visualize hyperparameter importance

    Args:
        cv_results: Cross-validation results dictionary
        output_dir: Directory to save figures
    """
    results_df = pd.DataFrame(cv_results)

    # Extract parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]

    # Determine layout
    n_params = len(param_cols)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, param_col in enumerate(param_cols):
        ax = axes[idx]
        param_name = param_col.replace('param_', '')

        # Convert to numeric
        x = pd.to_numeric(results_df[param_col], errors='coerce')
        y = results_df['mean_test_score']

        # Remove NaN values
        mask = ~np.isnan(x)
        x_clean = x[mask]
        y_clean = y[mask]

        # Scatter plot
        ax.scatter(x_clean, y_clean, alpha=0.6, s=50, c=y_clean,
                  cmap='viridis', edgecolors='k', linewidth=0.5)
        ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean CV R²', fontsize=11)
        ax.set_title(f'Impact of {param_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(vmin=y_clean.min(),
                                                       vmax=y_clean.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='R²')

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/hyperparameter_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def compare_default_vs_optimized(X_train, y_train, X_test, y_test, best_params):
    """
    Compare default vs optimized hyperparameters

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        best_params: Best parameters from search

    Returns:
        comparison: DataFrame with comparison results
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    print("\n" + "=" * 80)
    print("TRAINING DEFAULT vs OPTIMIZED MODELS")
    print("=" * 80)

    # Default model
    print("\nTraining default model...")
    default_model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )
    start_time = time.time()
    default_model.fit(X_train, y_train)
    default_train_time = time.time() - start_time
    default_pred = default_model.predict(X_test)

    # Optimized model
    print("Training optimized model...")
    optimized_model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        verbose=False
    )
    start_time = time.time()
    optimized_model.fit(X_train, y_train)
    optimized_train_time = time.time() - start_time
    optimized_pred = optimized_model.predict(X_test)

    # Comparison
    comparison = pd.DataFrame({
        'Model': ['Default', 'Optimized'],
        'R²': [
            r2_score(y_test, default_pred),
            r2_score(y_test, optimized_pred)
        ],
        'MAE (Hz)': [
            mean_absolute_error(y_test, default_pred),
            mean_absolute_error(y_test, optimized_pred)
        ],
        'RMSE (Hz)': [
            np.sqrt(mean_squared_error(y_test, default_pred)),
            np.sqrt(mean_squared_error(y_test, optimized_pred))
        ],
        'Train Time (s)': [
            default_train_time,
            optimized_train_time
        ]
    })

    # Calculate improvement
    comparison['R² Improvement (%)'] = [
        0.0,
        ((comparison.loc[1, 'R²'] - comparison.loc[0, 'R²']) /
         comparison.loc[0, 'R²']) * 100
    ]

    print("\n" + "=" * 80)
    print("DEFAULT vs OPTIMIZED COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print("=" * 80)

    output_path = 'simulation/outputs/ml_figures/hyperparam_comparison.csv'
    comparison.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

    return comparison

def save_optimization_results(random_search, comparison, output_dir='simulation/outputs/ml_figures'):
    """
    Save optimization results to text file

    Args:
        random_search: RandomizedSearchCV object
        comparison: Comparison DataFrame
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/optimization_results.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("BEST PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        for param, value in random_search.best_params_.items():
            if isinstance(value, float):
                f.write(f"{param:20s}: {value:.4f}\n")
            else:
                f.write(f"{param:20s}: {value}\n")

        f.write(f"\nBest CV R² Score: {random_search.best_score_:.6f}\n")
        f.write(f"Std of CV R² Score: {random_search.cv_results_['std_test_score'][random_search.best_index_]:.6f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DEFAULT vs OPTIMIZED COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")

        # Top 5 parameter combinations
        f.write("=" * 80 + "\n")
        f.write("TOP 5 PARAMETER COMBINATIONS\n")
        f.write("=" * 80 + "\n\n")

        cv_results_df = pd.DataFrame(random_search.cv_results_)
        top_5 = cv_results_df.nsmallest(5, 'rank_test_score')[
            ['rank_test_score', 'mean_test_score', 'std_test_score'] +
            [col for col in cv_results_df.columns if col.startswith('param_')]
        ]

        f.write(top_5.to_string(index=False))
        f.write("\n")

    print(f"✓ Saved: {output_path}")

def main():
    """
    Main function
    """
    print("=" * 80)
    print("CatBoost Hyperparameter Optimization")
    print("=" * 80)

    # Check if required files exist
    train_path = Path('simulation/data/train_data.csv')
    test_path = Path('simulation/data/test_data.csv')

    if not train_path.exists():
        print(f"\n⚠ Warning: Training data not found at {train_path}")
        print("This script requires training data.")
        print("\nPlease save your training data to simulation/data/train_data.csv")
        return

    if not test_path.exists():
        print(f"\n⚠ Warning: Test data not found at {test_path}")
        print("This script requires test data.")
        print("\nPlease save your test data to simulation/data/test_data.csv")
        return

    try:
        # Load data
        print("\nLoading data...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        print(f"  Training data shape: {train_data.shape}")
        print(f"  Test data shape: {test_data.shape}")

        # Prepare features and targets
        X_train = train_data.drop(['Mode_1_Freq', 'Mode_2_Freq'], axis=1, errors='ignore')
        X_test = test_data.drop(['Mode_1_Freq', 'Mode_2_Freq'], axis=1, errors='ignore')

        if 'Mode_1_Freq' in train_data.columns:
            y_train = train_data['Mode_1_Freq'].values
            y_test = test_data['Mode_1_Freq'].values
        else:
            y_train = train_data.iloc[:, -1].values
            y_test = test_data.iloc[:, -1].values

        print(f"  Features: {X_train.columns.tolist()}")

        # Run hyperparameter search
        search = run_hyperparameter_search(X_train, y_train, n_iter=50, cv=5)

        # Visualize
        print("\nGenerating hyperparameter importance plots...")
        plot_hyperparameter_importance(search.cv_results_)

        # Compare default vs optimized
        comparison = compare_default_vs_optimized(
            X_train, y_train, X_test, y_test,
            search.best_params_
        )

        # Save results
        print("\nSaving optimization results...")
        save_optimization_results(search, comparison)

        print("\n" + "=" * 80)
        print("SUCCESS: Hyperparameter optimization completed!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  1. simulation/outputs/ml_figures/hyperparameter_importance.png")
        print("  2. simulation/outputs/ml_figures/hyperparam_comparison.csv")
        print("  3. simulation/outputs/ml_figures/optimization_results.txt")
        print("\nNext steps:")
        print("  1. Review optimization_results.txt for best parameters")
        print("  2. Consider using optimized parameters in final model")
        print("  3. Add comparison table to documentation if improvement is significant")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check that:")
        print("  1. Data files are in correct format")
        print("  2. Required libraries are installed")

if __name__ == "__main__":
    main()
