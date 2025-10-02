"""
Simple test to verify the svd_imputer package works correctly.
"""

import pandas as pd
import numpy as np
from svd_imputer import Imputer


def test_basic_imputation():
    """Test basic imputation functionality."""
    print("Test 1: Basic imputation... ", end="")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'A': np.arange(50) + np.random.normal(0, 1, 50),
        'B': np.arange(50) * 2 + np.random.normal(0, 1, 50)
    }, index=dates)
    
    # Add missing values
    df.iloc[5:10, 0] = np.nan
    df.iloc[15:20, 1] = np.nan
    
    # Impute
    imputer = Imputer(verbose=False)
    df_imputed = imputer.fit_transform(df)
    
    # Check no missing values remain
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    assert df_imputed.shape == df.shape, "Shape mismatch!"
    
    print("✓ PASSED")


def test_automatic_rank():
    """Test automatic rank estimation."""
    print("Test 2: Automatic rank estimation... ", end="")
    
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'A': np.random.randn(50),
        'B': np.random.randn(50),
        'C': np.random.randn(50)
    }, index=dates)
    
    df.iloc[::5, :] = np.nan
    
    imputer = Imputer(variance_threshold=0.95, verbose=False)
    df_imputed = imputer.fit_transform(df)
    
    assert imputer.rank_ is not None, "Rank not estimated!"
    assert imputer.rank_ >= 1, f"Invalid rank: {imputer.rank_}"
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    
    print(f"✓ PASSED (rank={imputer.rank_})")


def test_fixed_rank():
    """Test with fixed rank."""
    print("Test 3: Fixed rank... ", end="")
    
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'A': np.random.randn(50),
        'B': np.random.randn(50)
    }, index=dates)
    
    df.iloc[::3, :] = np.nan
    
    imputer = Imputer(rank=1, verbose=False)
    df_imputed = imputer.fit_transform(df)
    
    assert imputer.rank_ == 1, f"Rank should be 1, got {imputer.rank_}"
    assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
    
    print("✓ PASSED")


def test_validation_errors():
    """Test data validation."""
    print("Test 4: Data validation... ", end="")
    
    # Test non-datetime index
    df = pd.DataFrame({'A': [1, 2, 3]})
    try:
        imputer = Imputer(verbose=False)
        imputer.fit_transform(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "DatetimeIndex" in str(e)
    
    # Test unsorted index
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({'A': range(10)}, index=dates[[5, 3, 7, 1, 9, 0, 2, 4, 6, 8]])
    try:
        imputer = Imputer(verbose=False)
        imputer.fit_transform(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "sorted" in str(e).lower()
    
    print("✓ PASSED")


def test_fit_transform_separately():
    """Test fit and transform separately."""
    print("Test 5: Separate fit/transform... ", end="")
    
    dates1 = pd.date_range('2020-01-01', periods=50, freq='D')
    df1 = pd.DataFrame({
        'A': np.random.randn(50),
        'B': np.random.randn(50)
    }, index=dates1)
    df1.iloc[::5, :] = np.nan
    
    dates2 = pd.date_range('2020-03-01', periods=30, freq='D')
    df2 = pd.DataFrame({
        'A': np.random.randn(30),
        'B': np.random.randn(30)
    }, index=dates2)
    df2.iloc[::4, :] = np.nan
    
    # Fit on df1
    imputer = Imputer(verbose=False)
    imputer.fit(df1)
    rank_used = imputer.rank_
    
    # Transform df1
    df1_imputed = imputer.transform(df1)
    assert df1_imputed.isna().sum().sum() == 0
    
    # Transform df2 (using same rank)
    df2_imputed = imputer.transform(df2)
    assert df2_imputed.isna().sum().sum() == 0
    assert imputer.rank_ == rank_used, "Rank changed!"
    
    print("✓ PASSED")


def test_with_scaler():
    """Test with StandardScaler."""
    print("Test 6: With StandardScaler... ", end="")
    
    from sklearn.preprocessing import StandardScaler
    
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'A': np.random.randn(50) * 100 + 1000,  # Large scale
        'B': np.random.randn(50) * 0.1 + 5     # Small scale
    }, index=dates)
    
    df.iloc[::5, :] = np.nan
    
    imputer = Imputer(scaler=StandardScaler(), verbose=False)
    df_imputed = imputer.fit_transform(df)
    
    assert df_imputed.isna().sum().sum() == 0
    # Check that scale is preserved (not normalized)
    assert df_imputed['A'].mean() > 900  # Should be near 1000
    assert df_imputed['B'].mean() < 10   # Should be near 5
    
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
        test_with_scaler
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


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
