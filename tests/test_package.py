"""
Simple test to verify the svd_imputer package works correctly.
"""

import numpy as np
import pandas as pd
import pytest

from svd_imputer import Imputer


def test_basic_imputation():
    """Test basic imputation functionality."""
    print("Test 1: Basic imputation... ", end="")

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "A": np.arange(50) + np.random.normal(0, 1, 50),
            "B": np.arange(50) * 2 + np.random.normal(0, 1, 50),
        },
        index=dates,
    )

    # Add missing values
    df.iloc[5:10, 0] = np.nan
    df.iloc[15:20, 1] = np.nan

    # Impute using new data-centric API
    imputer = Imputer(data=df, verbose=False)
    df_imputed = imputer.fit_transform()

    # Check no missing values remain
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    assert df_imputed.shape == df.shape, "Shape mismatch!"

    print("✓ PASSED")


def test_automatic_rank():
    """Test automatic rank estimation."""
    print("Test 2: Automatic rank estimation... ", end="")

    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {"A": np.random.randn(50), "B": np.random.randn(50), "C": np.random.randn(50)},
        index=dates,
    )

    df.iloc[::5, :] = np.nan

    imputer = Imputer(data=df, variance_threshold=0.95, verbose=False)
    df_imputed = imputer.fit_transform()

    assert imputer.rank_ is not None, "Rank not estimated!"
    assert imputer.rank_ >= 1, f"Invalid rank: {imputer.rank_}"
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"

    print(f"✓ PASSED (rank={imputer.rank_})")


def test_fixed_rank():
    """Test with fixed rank."""
    print("Test 3: Fixed rank... ", end="")

    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)

    df.iloc[::3, :] = np.nan

    imputer = Imputer(data=df, rank=1, verbose=False)
    df_imputed = imputer.fit_transform()

    assert imputer.rank_ == 1, f"Rank should be 1, got {imputer.rank_}"
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"

    print("✓ PASSED")


def test_validation_errors():
    """Test data validation."""
    print("Test 4: Data validation... ", end="")

    # Test non-datetime index
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        Imputer(data=df, verbose=False)

    # Test unsorted index
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"A": range(10)}, index=dates[[5, 3, 7, 1, 9, 0, 2, 4, 6, 8]])
    with pytest.raises(ValueError, match="sorted"):
        Imputer(data=df, verbose=False)

    print("✓ PASSED")


def test_fit_transform_separately():
    """Test fit and transform separately."""
    print("Test 5: Separate fit/transform... ", end="")

    dates1 = pd.date_range("2020-01-01", periods=50, freq="D")
    df1 = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates1)
    df1.iloc[::5, :] = np.nan

    # With data-centric design, each imputer is fitted to specific data
    # Fit on df1
    imputer = Imputer(data=df1, verbose=False)
    imputer.fit()
    rank_used = imputer.rank_

    # Transform same data
    df1_imputed = imputer.transform()
    assert df1_imputed.isna().sum().sum() == 0
    assert imputer.rank_ == rank_used, "Rank changed!"

    # Test that we can call fit_transform after fit
    df1_imputed2 = imputer.fit_transform()
    assert df1_imputed2.isna().sum().sum() == 0

    print("✓ PASSED")


def test_uncertainty_estimation():
    """Test uncertainty estimation."""
    print("Test 6: Uncertainty estimation... ", end="")

    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)

    df.iloc[::5, :] = np.nan

    # Test with uncertainty estimation
    imputer = Imputer(data=df, verbose=False)
    df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=20)

    assert df_imputed.isna().sum().sum() == 0
    assert uncertainty is not None
    assert "rmse" in uncertainty
    assert "mae" in uncertainty
    assert "method" in uncertainty
    assert uncertainty["method"] == "monte_carlo"

    print("✓ PASSED")


def test_rank_optimization():
    """Test rank optimization with rank='auto'."""
    print("Test 7: Rank optimization (rank='auto')... ", end="")

    # Create synthetic low-rank data
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=80, freq="D")

    # Generate low-rank structure (rank=2)
    t = np.arange(80)
    component1 = np.sin(2 * np.pi * t / 20)
    component2 = np.cos(2 * np.pi * t / 30) * 0.5

    df = pd.DataFrame(
        {
            "A": component1 + 0.1 * np.random.randn(80),
            "B": component2 + 0.1 * np.random.randn(80),
            "C": 0.8 * component1 + 0.6 * component2 + 0.1 * np.random.randn(80),
        },
        index=dates,
    )

    # Add missing values
    df.iloc[::6, :] = np.nan

    # Test rank optimization
    imputer = Imputer(data=df, rank="auto", verbose=False)
    df_imputed = imputer.fit_transform()

    # Verify results
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    assert imputer.rank_ is not None, "Optimal rank not set!"
    assert imputer.rank_ >= 1, f"Invalid optimal rank: {imputer.rank_}"

    # Check optimization results are available
    opt_results = imputer.get_optimization_results()
    assert opt_results is not None, "Optimization results not stored!"
    assert "optimal_rank" in opt_results, "Missing optimal_rank in results!"
    assert "results_df" in opt_results, "Missing results_df in results!"
    assert opt_results["optimal_rank"] == imputer.rank_, "Rank mismatch!"

    # Results dataframe should have multiple ranks tested
    results_df = opt_results["results_df"]
    assert len(results_df) > 1, "Should test multiple ranks!"
    assert "mean_rmse" in results_df.columns, "Missing mean_rmse column!"

    print(f"✓ PASSED (optimal rank: {imputer.rank_})")


def test_rank_auto_validation():
    """Test rank='auto' parameter validation."""
    print("Test 8: Rank parameter validation... ", end="")

    # Create dummy data for validation
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"A": range(10), "B": range(10)}, index=dates)

    # Test invalid string
    with pytest.raises(ValueError, match="rank must be int, 'auto', or None"):
        Imputer(data=df, rank="invalid")

    # Test valid values
    imputer1 = Imputer(data=df, rank="auto")  # Should work
    imputer2 = Imputer(data=df, rank=3)  # Should work
    imputer3 = Imputer(data=df, rank=None)  # Should work

    assert imputer1.rank == "auto"
    assert imputer2.rank == 3
    assert imputer3.rank is None

    print("✓ PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running svd_imputer Tests")
    print("=" * 60 + "\n")

    tests = [
        test_basic_imputation,
        test_automatic_rank,
        test_fixed_rank,
        test_validation_errors,
        test_fit_transform_separately,
        test_uncertainty_estimation,
        test_rank_optimization,
        test_rank_auto_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
