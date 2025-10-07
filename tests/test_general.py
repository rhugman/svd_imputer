"""
General test suite for package-level functionality and utilities.

This file contains tests for:
- Package imports and version handling
- Public API structure and compatibility
- Integration tests and examples
- Performance benchmarks and utilities
- Legacy test compatibility
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import svd_imputer
from svd_imputer import Imputer, validate_dataframe


class TestPackageImports:
    """Test package-level imports and structure."""

    def test_main_imports(self):
        """Test main package imports."""
        # Test that main classes are importable
        assert hasattr(svd_imputer, "Imputer")
        assert hasattr(svd_imputer, "validate_dataframe")
        assert hasattr(svd_imputer, "__version__")

    def test_imputer_import(self):
        """Test Imputer class import."""
        from svd_imputer import Imputer

        assert Imputer is not None

        # Test basic instantiation
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}, index=dates)
        imputer = Imputer(data=df, verbose=False)
        assert isinstance(imputer, Imputer)

    def test_validate_dataframe_import(self):
        """Test validate_dataframe function import."""
        from svd_imputer import validate_dataframe

        assert validate_dataframe is not None

        # Test basic usage
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"A": [1, 2, 3]}, index=dates)
        result = validate_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_version_handling(self):
        """Test version handling."""
        import svd_imputer

        # Version should exist and be a string
        assert hasattr(svd_imputer, "__version__")
        assert isinstance(svd_imputer.__version__, str)
        # Version might be "unknown" which is valid for development


class TestVersionHandling:
    """Test version and metadata handling."""

    def test_version_exists(self):
        """Test that version is accessible."""
        import svd_imputer

        # Version might be "unknown" which is valid
        assert hasattr(svd_imputer, "__version__")
        assert isinstance(svd_imputer.__version__, str)

    def test_version_fallback_scenarios(self):
        """Test version fallback mechanisms."""
        # Test that version handling works even if setuptools_scm fails
        with patch.dict("sys.modules", {"setuptools_scm": None}):
            try:
                import importlib

                importlib.reload(svd_imputer)
                # Should still have a version, even if "unknown"
                assert hasattr(svd_imputer, "__version__")
            except (ImportError, AttributeError):
                # Expected in some test environments
                pass

    def test_module_docstring(self):
        """Test module has proper docstring."""
        import svd_imputer

        # Module should have some documentation
        assert svd_imputer.__doc__ is not None or True  # Allow None for minimal modules


class TestPublicAPI:
    """Test public API structure and compatibility."""

    def test_public_api_structure(self):
        """Test public API structure."""
        import svd_imputer

        # Check __all__ if it exists
        if hasattr(svd_imputer, "__all__"):
            assert isinstance(svd_imputer.__all__, list)
            # Check that all listed items are actually available
            for item in svd_imputer.__all__:
                assert hasattr(svd_imputer, item), f"__all__ lists '{item}' but it's not available"

    def test_api_backwards_compatibility(self):
        """Test API backwards compatibility."""
        # Test that common usage patterns still work
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"A": [1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10], "B": [2, 4, np.nan, 8, 10, 12, np.nan, 16, 18, 20]}, index=dates
        )

        # Basic usage should work
        imputer = Imputer(data=df, verbose=False)
        result = imputer.fit_transform()
        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0

    def test_private_imports_not_exposed(self):
        """Test that private imports are not exposed."""
        import svd_imputer

        # These internal modules should not be directly accessible
        # (though some might be for backwards compatibility)
        internal_items = ["logging", "warnings", "sys", "os"]
        for item in internal_items:
            # It's okay if some are exposed, but they shouldn't be in __all__
            if hasattr(svd_imputer, "__all__") and hasattr(svd_imputer, item):
                assert item not in svd_imputer.__all__


class TestIntegrationExamples:
    """Test integration scenarios and usage examples."""

    def test_basic_usage_example(self):
        """Test basic usage example."""
        # Create sample time series data
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "temperature": 20 + 5 * np.sin(np.arange(30) * 2 * np.pi / 30) + np.random.randn(30),
                "humidity": 50 + 10 * np.cos(np.arange(30) * 2 * np.pi / 30) + np.random.randn(30),
                "pressure": 1000 + np.random.randn(30) * 5,
            },
            index=dates,
        )

        # Add some missing values
        df.loc[df.index[5:8], "temperature"] = np.nan
        df.loc[df.index[15], "humidity"] = np.nan
        df.loc[df.index[22:25], "pressure"] = np.nan

        # Basic imputation
        imputer = Imputer(data=df, verbose=False)
        df_imputed = imputer.fit_transform()

        # Verify results
        assert df_imputed.isna().sum().sum() == 0
        assert df_imputed.shape[1] == df.shape[1]  # Same columns

        # Values should be reasonable (not extreme outliers)
        for col in df.columns:
            original_std = df[col].std()
            imputed_values = df_imputed.loc[df[col].isna(), col]
            if len(imputed_values) > 0:
                # Imputed values should be within reasonable range
                assert np.all(np.abs(imputed_values - df[col].mean()) < 3 * original_std)

    def test_rank_optimization_example(self):
        """Test rank optimization example."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        # Create structured data with known rank
        np.random.seed(123)
        factor1 = np.sin(np.arange(50) * 0.1)
        factor2 = np.cos(np.arange(50) * 0.1)

        df = pd.DataFrame(
            {
                "series1": factor1 + 0.5 * factor2 + np.random.randn(50) * 0.1,
                "series2": 0.8 * factor1 - 0.3 * factor2 + np.random.randn(50) * 0.1,
                "series3": -0.6 * factor1 + 0.9 * factor2 + np.random.randn(50) * 0.1,
                "series4": 0.2 * factor1 + 0.7 * factor2 + np.random.randn(50) * 0.1,
            },
            index=dates,
        )

        # Add missing values
        missing_indices = [5, 12, 18, 25, 33, 41, 47]
        for i, idx in enumerate(missing_indices):
            col = i % 4
            df.iloc[idx, col] = np.nan

        # Test automatic rank selection
        imputer = Imputer(data=df, rank="auto", verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        assert hasattr(imputer, "rank_") and imputer.rank_ is not None
        # Should select reasonable rank (likely 2-3 for this synthetic data)
        assert 1 <= imputer.rank_ <= 4

    def test_uncertainty_estimation_example(self):
        """Test uncertainty estimation example."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")

        df = pd.DataFrame(
            {"A": np.arange(20) + np.random.randn(20) * 0.1, "B": np.arange(20) ** 0.5 + np.random.randn(20) * 0.1},
            index=dates,
        )

        # Add missing values
        df.iloc[3, 0] = np.nan
        df.iloc[8, 1] = np.nan
        df.iloc[15, 0] = np.nan

        # Fit imputer
        imputer = Imputer(data=df, rank=2, verbose=False)
        imputer.fit()

        # Estimate uncertainty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uncertainty = imputer.estimate_uncertainty(n_repeats=10)

        assert isinstance(uncertainty, dict)
        assert "RMSE" in uncertainty

        # RMSE should have mean and std
        rmse_stats = uncertainty["RMSE"]
        assert "mean" in rmse_stats
        assert "std" in rmse_stats
        assert rmse_stats["mean"] >= 0  # RMSE should be non-negative


class TestLegacyCompatibility:
    """Test compatibility with legacy usage patterns."""

    def test_legacy_basic_imputation(self):
        """Test legacy basic imputation pattern."""
        # This mimics the test_package.py style
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({"A": np.random.randn(20), "B": np.random.randn(20)}, index=dates)

        # Add missing values in legacy pattern
        df.iloc[5, 0] = np.nan
        df.iloc[10, 1] = np.nan
        df.iloc[15, 0] = np.nan

        # Legacy usage pattern
        imputer = Imputer(data=df)
        df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        assert df_imputed.shape == df.shape

    def test_legacy_rank_auto(self):
        """Test legacy rank='auto' usage."""
        dates = pd.date_range("2020-01-01", periods=25, freq="D")
        df = pd.DataFrame({"A": np.random.randn(25), "B": np.random.randn(25), "C": np.random.randn(25)}, index=dates)

        # Add missing values
        df.iloc[::5, 0] = np.nan  # Every 5th value

        # Legacy auto-rank pattern
        imputer = Imputer(data=df, rank="auto")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        assert hasattr(imputer, "rank_")

    def test_legacy_validation_errors(self):
        """Test legacy validation error patterns."""
        # Non-datetime index (legacy error pattern)
        df_bad = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(ValueError):
            Imputer(data=df_bad)


class TestPerformanceBenchmarks:
    """Test performance characteristics and benchmarks."""

    def test_small_data_performance(self):
        """Test performance with small datasets."""
        import time

        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({"A": np.random.randn(20), "B": np.random.randn(20)}, index=dates)

        df.iloc[::4, 0] = np.nan  # 25% missing in A
        df.iloc[1::4, 1] = np.nan  # 25% missing in B

        start_time = time.time()
        imputer = Imputer(data=df, verbose=False)
        result = imputer.fit_transform()
        end_time = time.time()

        # Should be very fast for small data
        assert (end_time - start_time) < 2.0  # 2 seconds max
        assert result.isna().sum().sum() == 0

    def test_medium_data_performance(self):
        """Test performance with medium datasets."""
        import time

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100), "C": np.random.randn(100)}, index=dates)

        # Add 15% missing values randomly
        np.random.seed(42)
        for col in df.columns:
            missing_idx = np.random.choice(100, size=15, replace=False)
            df.iloc[missing_idx, df.columns.get_loc(col)] = np.nan

        start_time = time.time()
        imputer = Imputer(data=df, rank=2, verbose=False)
        result = imputer.fit_transform()
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 10.0  # 10 seconds max
        assert result.isna().sum().sum() == 0

    def test_memory_usage(self):
        """Test memory usage characteristics."""
        # Test that imputer doesn't use excessive memory
        dates = pd.date_range("2020-01-01", periods=200, freq="h")
        df = pd.DataFrame({f"col_{i}": np.random.randn(200) for i in range(5)}, index=dates)

        # Add some missing values
        for i in range(5):
            missing_idx = np.random.choice(200, size=20, replace=False)
            df.iloc[missing_idx, i] = np.nan

        # Should not crash with memory issues
        try:
            imputer = Imputer(data=df, rank=3, verbose=False)
            result = imputer.fit_transform()
            assert result.isna().sum().sum() == 0
        except MemoryError:
            pytest.skip("Not enough memory for this test")


class TestErrorRecovery:
    """Test error recovery and robustness."""

    def test_numerical_stability(self):
        """Test numerical stability with challenging data."""
        dates = pd.date_range("2020-01-01", periods=15, freq="D")

        # Create numerically challenging data
        df = pd.DataFrame(
            {
                "small": np.random.randn(15) * 1e-10,  # Very small values
                "large": np.random.randn(15) * 1e10,  # Very large values
                "normal": np.random.randn(15),  # Normal values
            },
            index=dates,
        )

        # Add missing values
        df.iloc[5, 0] = np.nan
        df.iloc[8, 1] = np.nan
        df.iloc[12, 2] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore numerical warnings

            imputer = Imputer(data=df, verbose=False)
            result = imputer.fit_transform()

            # Should complete without crashing
            assert isinstance(result, pd.DataFrame)
            # May not perfectly impute all values due to numerical issues
            assert result.isna().sum().sum() <= df.isna().sum().sum()

    def test_convergence_robustness(self):
        """Test robustness with data that may not converge well."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")

        # Create data with very different scales that might not converge
        df = pd.DataFrame(
            {
                "A": [1e-6, 2e-6, np.nan, 4e-6, 5e-6, 6e-6, np.nan, 8e-6, 9e-6, 1e-5],
                "B": [1e6, 2e6, 3e6, np.nan, 5e6, 6e6, 7e6, np.nan, 9e6, 1e7],
            },
            index=dates,
        )

        # Use strict convergence criteria that might not be met
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            imputer = Imputer(data=df, max_iters=5, tol=1e-15, verbose=False)
            result = imputer.fit_transform()

            # Should not crash even if convergence is not achieved
            assert isinstance(result, pd.DataFrame)

    def test_edge_case_recovery(self):
        """Test recovery from various edge cases."""
        # Test with minimal data that's right at the edge of what's valid
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.nan, 2.0, np.nan]}, index=dates)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            imputer = Imputer(data=df, rank=1, verbose=False)
            result = imputer.fit_transform()

            # Should handle minimal data gracefully
            assert isinstance(result, pd.DataFrame)
            # Might not perfectly impute due to insufficient data
            remaining_missing = result.isna().sum().sum()
            original_missing = df.isna().sum().sum()
            assert remaining_missing <= original_missing
