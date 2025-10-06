"""
Pytest-compatible test suite for SVD Imputer package
"""

import numpy as np
import pandas as pd
import pytest

from svd_imputer import Imputer


class TestSVDImputerBasic:
    """Test basic imputation functionality."""

    def test_basic_imputation(self):
        """Test basic imputation functionality."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({"A": np.random.randn(20), "B": np.random.randn(20)}, index=dates)

        df.iloc[::5, :] = np.nan

        imputer = Imputer(data=df, verbose=False)
        df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"
        assert df_imputed.shape == df.shape, "Shape changed during imputation!"
        assert isinstance(df_imputed, pd.DataFrame), "Should return DataFrame!"

    def test_automatic_rank_estimation(self):
        """Test automatic rank estimation."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "A": np.random.randn(50),
                "B": np.random.randn(50),
                "C": np.random.randn(50),
            },
            index=dates,
        )

        df.iloc[::5, :] = np.nan

        imputer = Imputer(data=df, variance_threshold=0.95, verbose=False)
        df_imputed = imputer.fit_transform()

        assert imputer.rank_ is not None, "Rank not estimated!"
        assert imputer.rank_ >= 1, f"Invalid rank: {imputer.rank_}"
        assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"

    def test_fixed_rank(self):
        """Test with fixed rank."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)

        df.iloc[::3, :] = np.nan

        imputer = Imputer(data=df, rank=1, verbose=False)
        df_imputed = imputer.fit_transform()

        assert imputer.rank_ == 1, f"Rank should be 1, got {imputer.rank_}"
        assert df_imputed.isna().sum().sum() == 0, "Missing values remain!"

    def test_fit_transform_separately(self):
        """Test fit and transform separately."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)

        df.iloc[::5, :] = np.nan

        # With data-centric design, each imputer is fitted to specific data
        imputer = Imputer(data=df, verbose=False)
        imputer.fit()
        rank_used = imputer.rank_

        # Transform same data
        df_imputed = imputer.transform()
        assert df_imputed.isna().sum().sum() == 0
        assert imputer.rank_ == rank_used, "Rank changed!"

        # Test that we can call fit_transform after fit
        df_imputed2 = imputer.fit_transform()
        assert df_imputed2.isna().sum().sum() == 0


class TestSVDImputerValidation:
    """Test data validation."""

    def test_non_datetime_index_error(self):
        """Test non-datetime index raises error."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            Imputer(data=df, verbose=False)

    def test_unsorted_index_error(self):
        """Test unsorted index raises error."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"A": range(10)}, index=dates[[5, 3, 7, 1, 9, 0, 2, 4, 6, 8]])
        with pytest.raises(ValueError, match="sorted"):
            Imputer(data=df, verbose=False)

    def test_invalid_rank_parameter(self):
        """Test invalid rank parameter raises error."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"A": range(10), "B": range(10)}, index=dates)

        with pytest.raises(ValueError, match="rank must be int, 'auto', or None"):
            Imputer(data=df, rank="invalid", verbose=False)

    def test_valid_rank_parameters(self):
        """Test valid rank parameters."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"A": range(10), "B": range(10)}, index=dates)

        # These should all work
        imputer1 = Imputer(data=df, rank="auto", verbose=False)
        imputer2 = Imputer(data=df, rank=3, verbose=False)
        imputer3 = Imputer(data=df, rank=None, verbose=False)

        assert imputer1.rank == "auto"
        assert imputer2.rank == 3
        assert imputer3.rank is None


class TestSVDImputerAdvanced:
    """Test advanced features."""

    @pytest.mark.slow
    def test_rank_optimization(self):
        """Test rank optimization with rank='auto'."""
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

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
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


# Fixtures for common test data
@pytest.fixture
def simple_time_series():
    """Create a simple time series for testing."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    return pd.DataFrame({"A": np.random.randn(30), "B": np.random.randn(30)}, index=dates)


@pytest.fixture
def time_series_with_missing(simple_time_series):
    """Create time series with missing values."""
    df = simple_time_series.copy()
    df.iloc[::5, :] = np.nan
    return df


class TestSVDImputerWithFixtures:
    """Test using pytest fixtures."""

    def test_with_fixture_data(self, time_series_with_missing):
        """Test imputation using fixture data."""
        imputer = Imputer(data=time_series_with_missing, verbose=False)
        df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        assert df_imputed.shape == time_series_with_missing.shape

    @pytest.mark.parametrize("rank", [1, 2, 3, None])
    def test_different_ranks(self, simple_time_series, rank):
        """Test imputation with different rank values."""
        df = simple_time_series.copy()
        df.iloc[::6, :] = np.nan

        imputer = Imputer(data=df, rank=rank, verbose=False)
        df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        if rank is not None:
            assert imputer.rank_ == rank

    @pytest.mark.parametrize("n_repeats", [5, 10, 20])
    def test_uncertainty_different_repeats(self, time_series_with_missing, n_repeats):
        """Test uncertainty estimation with different n_repeats."""
        imputer = Imputer(data=time_series_with_missing, verbose=False)
        df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=n_repeats)

        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"
        assert "rmse" in uncertainty
        assert "mae" in uncertainty


# Integration tests
@pytest.mark.integration
class TestSVDImputerIntegration:
    """Integration tests with real-world scenarios."""

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="H")
        df = pd.DataFrame(
            {
                "sensor1": np.random.randn(1000) + np.sin(np.arange(1000) * 2 * np.pi / 24),
                "sensor2": np.random.randn(1000) + np.cos(np.arange(1000) * 2 * np.pi / 24),
                "sensor3": np.random.randn(1000),
            },
            index=dates,
        )

        # Add 20% missing values
        missing_mask = np.random.random(df.shape) < 0.2
        df = df.mask(missing_mask)

        imputer = Imputer(data=df, verbose=False)
        df_imputed = imputer.fit_transform()

        assert df_imputed.isna().sum().sum() == 0
        # Check that we preserved the general trends
        assert np.corrcoef(df_imputed["sensor1"].dropna(), df["sensor1"].dropna())[0, 1] > 0.5


# Benchmark tests
@pytest.mark.benchmark
class TestSVDImputerBenchmarks:
    """Benchmark tests for performance monitoring."""

    def test_basic_imputation_benchmark(self, benchmark, time_series_with_missing):
        """Benchmark basic imputation."""

        def impute():
            imputer = Imputer(data=time_series_with_missing, verbose=False)
            return imputer.fit_transform()

        result = benchmark(impute)
        assert result.isna().sum().sum() == 0

    def test_uncertainty_benchmark(self, benchmark, time_series_with_missing):
        """Benchmark uncertainty estimation."""

        def impute_with_uncertainty():
            imputer = Imputer(data=time_series_with_missing, verbose=False)
            return imputer.fit_transform(return_uncertainty=True, n_repeats=10)

        df_imputed, uncertainty = benchmark(impute_with_uncertainty)
        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"
