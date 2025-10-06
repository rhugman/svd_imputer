"""
Test uncertainty estimation functionality
"""

import numpy as np
import pandas as pd

from svd_imputer import Imputer


def test_monte_carlo_uncertainty():
    """Test Monte Carlo uncertainty estimation."""
    print("Test 1: Monte Carlo uncertainty estimation... ", end="")

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "A": np.sin(np.arange(50) * 2 * np.pi / 30) + np.random.normal(0, 0.1, 50),
            "B": np.cos(np.arange(50) * 2 * np.pi / 20) + np.random.normal(0, 0.1, 50),
        },
        index=dates,
    )

    # Add missing values
    rng = np.random.default_rng(42)
    for col in df.columns:
        missing_idx = rng.choice(df.index, size=10, replace=False)
        df.loc[missing_idx, col] = np.nan

    # Test Monte Carlo uncertainty
    imputer = Imputer(data=df, verbose=False)
    df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=20)

    # Validate results
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    assert uncertainty is not None, "Uncertainty not returned!"
    assert "method" in uncertainty, "Method not in uncertainty dict!"
    assert uncertainty["method"] == "monte_carlo", f"Expected monte_carlo, got {uncertainty['method']}"
    assert "rmse" in uncertainty, "RMSE not in uncertainty dict!"
    assert "mae" in uncertainty, "MAE not in uncertainty dict!"
    assert uncertainty["rmse"] > 0, "RMSE should be positive!"
    assert uncertainty["mae"] > 0, "MAE should be positive!"

    print("✓ PASSED")


def test_no_uncertainty_backward_compatibility():
    """Test standard imputation without uncertainty (backward compatibility)."""
    print("Test 2: Standard imputation (no uncertainty)... ", end="")

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    df = pd.DataFrame({"A": np.random.randn(30), "B": np.random.randn(30)}, index=dates)

    # Add missing values
    df.iloc[::5, :] = np.nan

    # Test standard imputation
    imputer = Imputer(data=df, verbose=False)
    df_imputed = imputer.fit_transform()

    # Validate results
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    assert isinstance(df_imputed, pd.DataFrame), "Should return DataFrame!"

    print("✓ PASSED")


def test_uncertainty_parameters():
    """Test uncertainty estimation with different parameters."""
    print("Test 3: Uncertainty parameters... ", end="")

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({"A": np.random.randn(40), "B": np.random.randn(40)}, index=dates)

    # Add missing values
    df.iloc[::6, :] = np.nan

    # Test with different n_repeats
    imputer = Imputer(data=df, verbose=False)
    df_imputed, unc1 = imputer.fit_transform(return_uncertainty=True, n_repeats=10)

    # Reset imputer for second test
    imputer2 = Imputer(data=df, verbose=False)
    df_imputed2, unc2 = imputer2.fit_transform(return_uncertainty=True, n_repeats=30)

    # Validate results
    assert unc1["method"] == "monte_carlo"
    assert unc2["method"] == "monte_carlo"
    assert "rmse" in unc1 and "rmse" in unc2
    assert "mae" in unc1 and "mae" in unc2

    print("✓ PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing uncertainty estimation functionality")
    print("=" * 60)

    tests = [
        test_monte_carlo_uncertainty,
        test_no_uncertainty_backward_compatibility,
        test_uncertainty_parameters,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            raise

    print("\n" + "=" * 60)
    print("All uncertainty tests passed! ✓")
    print("=" * 60)
