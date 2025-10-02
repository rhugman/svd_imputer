"""
Basic usage example for svd_imputer package.

This example demonstrates how to use the Imputer class to impute
missing values in time series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from svd_imputer import Imputer


def create_sample_data():
    """Create sample time series data with missing values."""
    # Create date range
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create synthetic time series
    t = np.arange(100)
    series_a = 10 + 2 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.5, 100)
    series_b = 20 + 3 * np.cos(2 * np.pi * t / 20) + np.random.normal(0, 0.7, 100)
    series_c = 15 + 1.5 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 0.4, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Site_A': series_a,
        'Site_B': series_b,
        'Site_C': series_c
    }, index=dates)
    
    # Introduce missing values randomly (10% of data)
    rng = np.random.default_rng(42)
    for col in df.columns:
        missing_idx = rng.choice(df.index, size=10, replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


def example_1_automatic_rank():
    """Example 1: Automatic rank estimation (default behavior)."""
    print("=" * 60)
    print("Example 1: Automatic Rank Estimation")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Missing percentage: {df.isna().sum().sum() / df.size * 100:.1f}%")
    
    # Create imputer with automatic rank estimation
    imputer = Imputer(variance_threshold=0.95, verbose=True)
    
    # Fit and transform
    print("\nFitting and transforming...")
    df_imputed = imputer.fit_transform(df)
    
    print(f"\nImputed data shape: {df_imputed.shape}")
    print(f"Remaining missing values: {df_imputed.isna().sum().sum()}")
    print(f"Estimated rank: {imputer.rank_}")
    
    return df, df_imputed


def example_2_fixed_rank():
    """Example 2: Using a fixed rank."""
    print("\n" + "=" * 60)
    print("Example 2: Fixed Rank")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Create imputer with fixed rank
    imputer = Imputer(rank=2, verbose=True)
    
    # Fit and transform
    print("\nFitting and transforming...")
    df_imputed = imputer.fit_transform(df)
    
    print(f"\nImputed data shape: {df_imputed.shape}")
    print(f"Remaining missing values: {df_imputed.isna().sum().sum()}")
    
    return df, df_imputed


def example_3_with_scaling():
    """Example 3: Using StandardScaler for normalization."""
    print("\n" + "=" * 60)
    print("Example 3: With StandardScaler")
    print("=" * 60)
    
    from sklearn.preprocessing import StandardScaler
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Create imputer with scaling
    scaler = StandardScaler()
    imputer = Imputer(
        variance_threshold=0.95,
        scaler=scaler,
        verbose=True
    )
    
    # Fit and transform
    print("\nFitting and transforming with scaling...")
    df_imputed = imputer.fit_transform(df)
    
    print(f"\nImputed data shape: {df_imputed.shape}")
    print(f"Remaining missing values: {df_imputed.isna().sum().sum()}")
    
    return df, df_imputed


def example_4_real_world_workflow():
    """Example 4: Real-world workflow with data loading and visualization."""
    print("\n" + "=" * 60)
    print("Example 4: Real-World Workflow")
    print("=" * 60)
    
    # Simulate loading data from CSV
    df = create_sample_data()
    
    print("\n1. Data loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    print("\n2. Data validation")
    print(f"   Index type: {type(df.index).__name__}")
    print(f"   Is sorted: {df.index.is_monotonic_increasing}")
    print(f"   Missing values per column:")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        print(f"     {col}: {n_missing} ({pct_missing:.1f}%)")
    
    print("\n3. Imputation")
    imputer = Imputer(variance_threshold=0.95, verbose=True)
    df_imputed = imputer.fit_transform(df)
    
    print("\n4. Results")
    print(f"   Imputation complete!")
    print(f"   Rank used: {imputer.rank_}")
    
    # Optional: Visualize one series
    print("\n5. Visualization (optional - requires matplotlib)")
    try:
        plot_comparison(df, df_imputed, column='Site_A')
    except Exception as e:
        print(f"   Skipping plot: {e}")
    
    return df, df_imputed


def plot_comparison(df_original, df_imputed, column):
    """Plot original vs imputed data for a single column."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot imputed (full line)
    ax.plot(df_imputed.index, df_imputed[column], 
            label='Imputed', color='blue', linewidth=2, alpha=0.7)
    
    # Plot original observed values (scatter)
    ax.scatter(df_original.index, df_original[column], 
               label='Original (observed)', color='red', s=20, zorder=5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'Time Series Imputation: {column}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'imputation_example_{column}.png', dpi=150)
    print(f"   Plot saved as 'imputation_example_{column}.png'")
    plt.close()


def example_5_separate_fit_transform():
    """Example 5: Using fit() and transform() separately."""
    print("\n" + "=" * 60)
    print("Example 5: Separate fit() and transform()")
    print("=" * 60)
    
    # Create training and test data
    df_train = create_sample_data()
    df_test = create_sample_data()  # In practice, this would be different data
    
    print(f"\nTraining data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    # Fit on training data
    print("\nFitting on training data...")
    imputer = Imputer(variance_threshold=0.95, verbose=True)
    imputer.fit(df_train)
    
    # Transform both datasets
    print("\nTransforming training data...")
    df_train_imputed = imputer.transform(df_train)
    
    print("\nTransforming test data...")
    df_test_imputed = imputer.transform(df_test)
    
    print(f"\nBoth datasets imputed using rank: {imputer.rank_}")
    
    return df_train_imputed, df_test_imputed


if __name__ == '__main__':
    print("\n")
    print("*" * 60)
    print("SVD Time Series Imputer - Examples")
    print("*" * 60)
    
    # Run examples
    df1, df1_imputed = example_1_automatic_rank()
    df2, df2_imputed = example_2_fixed_rank()
    df3, df3_imputed = example_3_with_scaling()
    df4, df4_imputed = example_4_real_world_workflow()
    df_train, df_test = example_5_separate_fit_transform()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
