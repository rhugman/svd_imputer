"""
Test uncertainty estimation functionality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from svd_imputer import Imputer

# Create sample data
print("Creating sample data...")
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'A': np.sin(np.arange(100) * 2 * np.pi / 30) + np.random.normal(0, 0.1, 100),
    'B': np.cos(np.arange(100) * 2 * np.pi / 20) + np.random.normal(0, 0.1, 100)
}, index=dates)

# Add missing values
rng = np.random.default_rng(42)
for col in df.columns:
    missing_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[missing_idx, col] = np.nan

print(f"Data shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")

# Test 1: No uncertainty (backward compatibility)
print("\n" + "="*60)
print("Test 1: Standard imputation (no uncertainty)")
print("="*60)
imputer = Imputer(verbose=True)
df_imputed = imputer.fit_transform(df)
print(f"✓ Imputation complete")

# Test 2: Monte Carlo uncertainty
print("\n" + "="*60)
print("Test 2: Monte Carlo uncertainty")
print("="*60)
imputer = Imputer(verbose=True)
df_imputed, unc_mc = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='monte_carlo',
    n_repeats=50,
    mask_strategy='block',
    block_len=5
)
print(f"\n✓ Method: {unc_mc['method']}")
print(f"✓ RMSE: {unc_mc['rmse']:.4f} ± {unc_mc['rmse_std']:.4f}")
print(f"✓ MAE: {unc_mc['mae']:.4f} ± {unc_mc['mae_std']:.4f}")

# Get confidence intervals
df_lower, df_upper = imputer.get_confidence_intervals(df_imputed, unc_mc)
print(f"✓ Confidence intervals computed")

# Test 3: Bootstrap uncertainty
print("\n" + "="*60)
print("Test 3: Bootstrap uncertainty")
print("="*60)
imputer = Imputer(verbose=True)
df_imputed, unc_boot = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='bootstrap',
    n_bootstrap=30,
    seed=42
)
print(f"\n✓ Method: {unc_boot['method']}")
print(f"✓ Confidence: {unc_boot['confidence']}")
df_lower_boot, df_upper_boot = imputer.get_confidence_intervals(df_imputed, unc_boot)
# Calculate average width only where we imputed
missing_in_imputed = df_imputed.index.isin(df[df.isna().any(axis=1)].index)
if missing_in_imputed.any():
    avg_width = (df_upper_boot - df_lower_boot).loc[missing_in_imputed].values.flatten()
    avg_width = avg_width[~np.isnan(avg_width)].mean()
    print(f"✓ Average CI width: {avg_width:.4f}")

# Test 4: Hybrid uncertainty
print("\n" + "="*60)
print("Test 4: Hybrid uncertainty")
print("="*60)
imputer = Imputer(verbose=True)
df_imputed, unc_hybrid = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='hybrid',
    n_repeats=30,
    n_bootstrap=20,
    seed=42
)
print(f"\n✓ Method: {unc_hybrid['method']}")
df_lower_hybrid, df_upper_hybrid = imputer.get_confidence_intervals(df_imputed, unc_hybrid)
print(f"✓ Hybrid confidence intervals computed")

# Test 5: estimate_uncertainty method
print("\n" + "="*60)
print("Test 5: Separate estimate_uncertainty() method")
print("="*60)
imputer = Imputer(verbose=True)
imputer.fit(df)
uncertainty = imputer.estimate_uncertainty(df, n_repeats=50, seed=42)
print(f"\n✓ RMSE: {uncertainty['RMSE']['mean']:.4f}")
print(f"✓ 95% CI: ({uncertainty['RMSE']['95%_CI'][0]:.4f}, {uncertainty['RMSE']['95%_CI'][1]:.4f})")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
