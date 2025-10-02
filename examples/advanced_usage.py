"""
Advanced example: Working with real CSV data files.

This example demonstrates how to:
1. Load multiple CSV files with time series data
2. Process and combine them into a single DataFrame
3. Handle different datetime formats
4. Resample data to daily frequency
5. Impute missing values
6. Visualize results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from svd_imputer import Imputer


def load_and_process_csv_files(data_dir='data'):
    """
    Load and process multiple CSV files from a directory.
    
    This function handles:
    - Multiple datetime formats
    - Column name standardization
    - Removing invalid dates
    - Setting datetime index
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all time series
    """
    print(f"Loading data from: {data_dir}")
    
    df_dict = {}
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        print(f"  Processing: {filename}")
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Standardize column names (lowercase)
            df.columns = [c.lower() for c in df.columns]
            
            # Remove completely empty rows
            df.dropna(how='all', inplace=True)
            
            # Remove invalid date entries
            if 'date' in df.columns:
                df = df.loc[df['date'] != 'NaN-NaN-NaN NaN:NaN:NaN']
                df.set_index('date', inplace=True)
            
            # Strip whitespace from index
            if isinstance(df.index, pd.Index):
                df.index = df.index.str.strip()
            
            # Try multiple datetime formats
            try:
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
                except ValueError:
                    # Use automatic format detection
                    df.index = pd.to_datetime(df.index)
            
            # If multiple columns, select the main data column
            # Common names: 'wl', 'wl_corr', 'value', etc.
            if isinstance(df, pd.DataFrame) and len(df.columns) > 1:
                for col_name in ['wl_corr', 'wl', 'value', 'data']:
                    if col_name in df.columns:
                        df = df[col_name]
                        break
            
            # Convert to Series if it's a DataFrame with one column
            df = df.squeeze()
            
            # Remove NaN values for this series
            df = df.dropna()
            
            # Set series name based on filename
            series_name = os.path.splitext(filename)[0].lower()
            df.name = series_name
            
            df_dict[series_name] = df
            print(f"    Shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
            
        except Exception as e:
            print(f"    Error processing {filename}: {e}")
            continue
    
    # Combine all series into a single DataFrame
    if not df_dict:
        raise ValueError("No valid data files found")
    
    data = pd.DataFrame(df_dict)
    
    # Sort by index
    data = data.sort_index()
    
    print(f"\nCombined data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def resample_to_daily(df, method='mean'):
    """
    Resample time series to daily frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    method : str, optional
        Aggregation method ('mean', 'median', 'sum', etc.)
        
    Returns
    -------
    pd.DataFrame
        Resampled DataFrame
    """
    print(f"\nResampling to daily frequency using '{method}'...")
    
    if method == 'mean':
        df_daily = df.resample('D').mean()
    elif method == 'median':
        df_daily = df.resample('D').median()
    elif method == 'sum':
        df_daily = df.resample('D').sum()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Remove rows that are all NaN
    df_daily = df_daily.dropna(how='all')
    
    print(f"Resampled shape: {df_daily.shape}")
    print(f"Missing values: {df_daily.isna().sum().sum()} "
          f"({df_daily.isna().sum().sum() / df_daily.size * 100:.1f}%)")
    
    return df_daily


def impute_and_fill_gaps(df, variance_threshold=0.95, fill_dates=True):
    """
    Impute missing values and optionally fill date gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    variance_threshold : float
        Variance threshold for rank estimation
    fill_dates : bool
        Whether to create continuous date range
        
    Returns
    -------
    pd.DataFrame
        Imputed DataFrame (with continuous dates if fill_dates=True)
    """
    print(f"\n{'='*60}")
    print("Imputation")
    print(f"{'='*60}")
    
    # Impute existing data
    imputer = Imputer(variance_threshold=variance_threshold, verbose=True)
    df_imputed = imputer.fit_transform(df)
    
    if fill_dates:
        print("\nFilling date gaps to create continuous time series...")
        # Create complete date range
        new_index = pd.date_range(
            start=df_imputed.index.min(),
            end=df_imputed.index.max(),
            freq='D'
        )
        
        # Create new DataFrame with complete date range
        df_complete = pd.DataFrame(index=new_index, columns=df_imputed.columns, dtype=float)
        
        # Fill in the imputed values
        df_complete.loc[df_imputed.index, :] = df_imputed.values
        
        # Impute the newly created gaps
        print(f"New missing values from date gaps: {df_complete.isna().sum().sum()}")
        if df_complete.isna().sum().sum() > 0:
            print("Imputing date gaps...")
            imputer2 = Imputer(rank=imputer.rank_, verbose=False)
            df_complete = imputer2.fit_transform(df_complete)
        
        print(f"Final shape: {df_complete.shape}")
        print(f"Final missing values: {df_complete.isna().sum().sum()}")
        
        return df_complete
    
    return df_imputed


def visualize_results(df_original, df_imputed, output_dir='output'):
    """
    Create visualization plots comparing original and imputed data.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Original data with missing values
    df_imputed : pd.DataFrame
        Imputed data
    output_dir : str
        Directory to save plots
    """
    print(f"\n{'='*60}")
    print("Visualization")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each column
    for col in df_imputed.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot imputed line
        ax.plot(df_imputed.index, df_imputed[col],
                label='Imputed', color='blue', linewidth=1.5, alpha=0.7)
        
        # Plot original observed points
        ax.scatter(df_original.index, df_original[col],
                   label='Original', color='red', s=10, zorder=5, alpha=0.6)
        
        ax.set_title(f'Time Series Imputation: {col}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'imputation_{col}.png')
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
        plt.close()
    
    # Create summary plot (all series)
    fig, axes = plt.subplots(len(df_imputed.columns), 1,
                             figsize=(12, 3*len(df_imputed.columns)))
    
    if len(df_imputed.columns) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, df_imputed.columns):
        ax.plot(df_imputed.index, df_imputed[col],
                label='Imputed', color='blue', linewidth=1, alpha=0.7)
        ax.scatter(df_original.index, df_original[col],
                   label='Original', color='red', s=5, alpha=0.5)
        ax.set_title(col)
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'imputation_summary.png')
    plt.savefig(summary_path, dpi=150)
    print(f"  Saved: {summary_path}")
    plt.close()


def main():
    """
    Main workflow for processing real data files.
    """
    print("\n" + "*" * 60)
    print("Advanced Example: Real CSV Data Processing")
    print("*" * 60 + "\n")
    
    # Step 1: Load data
    try:
        data = load_and_process_csv_files(data_dir='data')
    except FileNotFoundError:
        print("\nNote: 'data' directory not found. Creating sample data instead...")
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        data = pd.DataFrame({
            'site_a': np.sin(np.arange(365) * 2 * np.pi / 30) + np.random.normal(0, 0.1, 365),
            'site_b': np.cos(np.arange(365) * 2 * np.pi / 20) + np.random.normal(0, 0.1, 365)
        }, index=dates)
        # Add missing values
        rng = np.random.default_rng(42)
        for col in data.columns:
            missing_idx = rng.choice(data.index, size=30, replace=False)
            data.loc[missing_idx, col] = np.nan
    
    # Step 2: Optionally resample to daily (if data is high-frequency)
    if len(data) > 1000:  # If more than ~3 years of daily data, likely high-frequency
        data_daily = resample_to_daily(data, method='mean')
    else:
        data_daily = data.copy()
    
    # Step 3: Impute missing values
    data_imputed = impute_and_fill_gaps(
        data_daily,
        variance_threshold=0.95,
        fill_dates=True
    )
    
    # Step 4: Visualize results
    try:
        visualize_results(data_daily, data_imputed, output_dir='output')
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    # Step 5: Save results
    output_file = 'output/imputed_data.csv'
    os.makedirs('output', exist_ok=True)
    data_imputed.to_csv(output_file)
    print(f"\n{'='*60}")
    print(f"Imputed data saved to: {output_file}")
    print(f"{'='*60}")
    
    return data_daily, data_imputed


if __name__ == '__main__':
    df_original, df_imputed = main()
    print("\nWorkflow completed successfully!")
