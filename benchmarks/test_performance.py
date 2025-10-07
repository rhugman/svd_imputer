"""
Performance benchmarks for SVD Imputer
"""

import numpy as np
import pandas as pd
import pytest

from svd_imputer import Imputer


class TestPerformanceBenchmarks:
    """Benchmark tests for performance monitoring."""

    @pytest.mark.benchmark(group="basic")
    def test_small_dataset_benchmark(self, benchmark):
        """Benchmark basic imputation on small dataset."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)
        df.iloc[::5, :] = np.nan

        def impute_small():
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform()

        result = benchmark(impute_small)
        assert result.isna().sum().sum() == 0

    @pytest.mark.benchmark(group="basic")
    def test_medium_dataset_benchmark(self, benchmark):
        """Benchmark basic imputation on medium dataset."""
        dates = pd.date_range("2020-01-01", periods=500, freq="H")
        df = pd.DataFrame(
            {
                "A": np.random.randn(500),
                "B": np.random.randn(500),
                "C": np.random.randn(500),
            },
            index=dates,
        )
        df.iloc[::10, :] = np.nan

        def impute_medium():
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform()

        result = benchmark(impute_medium)
        assert result.isna().sum().sum() == 0

    @pytest.mark.benchmark(group="basic")
    def test_large_dataset_benchmark(self, benchmark):
        """Benchmark basic imputation on large dataset."""
        dates = pd.date_range("2020-01-01", periods=2000, freq="15min")
        df = pd.DataFrame(
            {
                "sensor1": np.random.randn(2000) + np.sin(np.arange(2000) * 2 * np.pi / 96),
                "sensor2": np.random.randn(2000) + np.cos(np.arange(2000) * 2 * np.pi / 96),
                "sensor3": np.random.randn(2000),
                "sensor4": np.random.randn(2000),
            },
            index=dates,
        )
        # Add 15% missing values
        missing_mask = np.random.random(df.shape) < 0.15
        df = df.mask(missing_mask)

        def impute_large():
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform()

        result = benchmark(impute_large)
        assert result.isna().sum().sum() == 0

    @pytest.mark.benchmark(group="uncertainty")
    def test_uncertainty_benchmark_small(self, benchmark):
        """Benchmark uncertainty estimation on small dataset."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)}, index=dates)
        df.iloc[::8, :] = np.nan

        def impute_with_uncertainty_small():
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform(return_uncertainty=True, n_repeats=20)

        df_imputed, uncertainty = benchmark(impute_with_uncertainty_small)
        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"

    @pytest.mark.benchmark(group="uncertainty")
    def test_uncertainty_benchmark_medium(self, benchmark):
        """Benchmark uncertainty estimation on medium dataset."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        df = pd.DataFrame(
            {
                "A": np.random.randn(300),
                "B": np.random.randn(300),
                "C": np.random.randn(300),
            },
            index=dates,
        )
        df.iloc[::12, :] = np.nan

        def impute_with_uncertainty_medium():
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform(return_uncertainty=True, n_repeats=15)

        df_imputed, uncertainty = benchmark(impute_with_uncertainty_medium)
        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"

    @pytest.mark.benchmark(group="rank_optimization")
    def test_rank_optimization_benchmark(self, benchmark):
        """Benchmark rank optimization."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")

        # Create structured data
        t = np.arange(200)
        component1 = np.sin(2 * np.pi * t / 30)
        component2 = np.cos(2 * np.pi * t / 20)

        df = pd.DataFrame(
            {
                "A": component1 + 0.1 * np.random.randn(200),
                "B": component2 + 0.1 * np.random.randn(200),
                "C": 0.8 * component1 + 0.6 * component2 + 0.1 * np.random.randn(200),
                "D": np.random.randn(200) * 0.5,
            },
            index=dates,
        )

        df.iloc[::15, :] = np.nan

        def optimize_rank():
            imputer = Imputer(data=df, rank="auto", verbose=False)
            return imputer.fit_transform()

        result = benchmark(optimize_rank)
        assert result.isna().sum().sum() == 0

    @pytest.mark.benchmark(group="different_ranks")
    @pytest.mark.parametrize("rank", [1, 2, 3, 5])
    def test_fixed_rank_benchmark(self, benchmark, rank):
        """Benchmark different fixed ranks."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        df = pd.DataFrame(
            {f"col_{i}": np.random.randn(300) + np.sin(np.arange(300) * 2 * np.pi / (10 + i)) for i in range(6)},
            index=dates,
        )

        df.iloc[::10, :] = np.nan

        def impute_fixed_rank():
            imputer = Imputer(data=df, rank=rank, verbose=False)
            return imputer.fit_transform()

        result = benchmark(impute_fixed_rank)
        assert result.isna().sum().sum() == 0


class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    def test_memory_usage_scaling(self):
        """Test memory usage with increasing dataset size."""
        import tracemalloc

        sizes = [100, 500, 1000]
        memory_usage = []

        for size in sizes:
            dates = pd.date_range("2020-01-01", periods=size, freq="H")
            df = pd.DataFrame(
                {
                    "A": np.random.randn(size),
                    "B": np.random.randn(size),
                    "C": np.random.randn(size),
                },
                index=dates,
            )
            df.iloc[::10, :] = np.nan

            tracemalloc.start()
            imputer = Imputer(data=df, verbose=False)
            df_imputed = imputer.fit_transform()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append(peak / 1024 / 1024)  # Convert to MB
            assert df_imputed.isna().sum().sum() == 0

        # Memory usage should scale reasonably (not exponentially)
        # This is a basic sanity check
        assert memory_usage[-1] < memory_usage[0] * (sizes[-1] / sizes[0]) ** 2

    @pytest.mark.benchmark(group="memory")
    def test_memory_efficiency_benchmark(self, benchmark):
        """Benchmark memory-efficient operation."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="H")
        df = pd.DataFrame(
            {
                "A": np.random.randn(1000),
                "B": np.random.randn(1000),
                "C": np.random.randn(1000),
                "D": np.random.randn(1000),
            },
            index=dates,
        )
        df.iloc[::20, :] = np.nan

        def memory_efficient_impute():
            # Test that we can process data without excessive memory usage
            imputer = Imputer(data=df, verbose=False)
            return imputer.fit_transform()

        result = benchmark.pedantic(memory_efficient_impute, iterations=3, rounds=2)
        assert result.isna().sum().sum() == 0


# Comparison benchmarks (if other methods were available)
class TestComparisonBenchmarks:
    """Benchmarks comparing different approaches within the package."""

    @pytest.mark.benchmark(group="comparison")
    @pytest.mark.parametrize("use_uncertainty", [False, True])
    def test_uncertainty_overhead(self, benchmark, use_uncertainty):
        """Compare performance with and without uncertainty estimation."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {
                "A": np.random.randn(200),
                "B": np.random.randn(200),
                "C": np.random.randn(200),
            },
            index=dates,
        )
        df.iloc[::10, :] = np.nan

        if use_uncertainty:

            def impute_with_uncertainty():
                imputer = Imputer(data=df, verbose=False)
                return imputer.fit_transform(return_uncertainty=True, n_repeats=10)

            df_imputed, uncertainty = benchmark(impute_with_uncertainty)
            assert uncertainty is not None
        else:

            def impute_without_uncertainty():
                imputer = Imputer(data=df, verbose=False)
                return imputer.fit_transform()

            df_imputed = benchmark(impute_without_uncertainty)

        assert df_imputed.isna().sum().sum() == 0
