"""
Pytest-compatible uncertainty estimation tests
"""

import numpy as np
import pandas as pd
import pytest
from svd_imputer import Imputer


class TestUncertaintyEstimation:
    """Test uncertainty estimation functionality."""

    def test_monte_carlo_uncertainty(self):
        """Test Monte Carlo uncertainty estimation."""
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

    def test_no_uncertainty_backward_compatibility(self):
        """Test standard imputation without uncertainty (backward compatibility)."""
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

    @pytest.mark.parametrize("n_repeats", [5, 10, 30])
    def test_uncertainty_parameters(self, n_repeats):
        """Test uncertainty estimation with different parameters."""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=40, freq="D")
        df = pd.DataFrame({"A": np.random.randn(40), "B": np.random.randn(40)}, index=dates)

        # Add missing values
        df.iloc[::6, :] = np.nan

        # Test with different n_repeats
        imputer = Imputer(data=df, verbose=False)
        df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=n_repeats)

        # Validate results
        assert uncertainty["method"] == "monte_carlo"
        assert "rmse" in uncertainty
        assert "mae" in uncertainty
        assert df_imputed.isna().sum().sum() == 0

    def test_uncertainty_return_types(self):
        """Test that uncertainty returns expected data types."""
        dates = pd.date_range("2020-01-01", periods=25, freq="D")
        df = pd.DataFrame({"A": np.random.randn(25), "B": np.random.randn(25)}, index=dates)

        # Add missing values
        df.iloc[::4, :] = np.nan

        imputer = Imputer(data=df, verbose=False)
        df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=15)

        # Check return types
        assert isinstance(df_imputed, pd.DataFrame)
        assert isinstance(uncertainty, dict)
        assert isinstance(uncertainty["rmse"], (int, float))
        assert isinstance(uncertainty["mae"], (int, float))
        assert isinstance(uncertainty["method"], str)

    @pytest.mark.slow
    def test_uncertainty_consistency(self):
        """Test that uncertainty estimation is reasonably consistent."""
        # Create sample data with known structure
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        t = np.arange(100)
        
        df = pd.DataFrame(
            {
                "A": np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(100),
                "B": np.cos(2 * np.pi * t / 15) + 0.1 * np.random.randn(100),
            },
            index=dates,
        )

        # Add structured missing values
        df.iloc[::10, :] = np.nan

        # Run uncertainty estimation multiple times
        uncertainties = []
        for i in range(3):
            imputer = Imputer(data=df, verbose=False)
            _, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=50)
            uncertainties.append(uncertainty["rmse"])

        # Check that results are reasonably consistent
        mean_rmse = np.mean(uncertainties)
        std_rmse = np.std(uncertainties)
        cv = std_rmse / mean_rmse  # Coefficient of variation

        # CV should be less than 50% for consistent results
        assert cv < 0.5, f"Uncertainty estimates too variable: CV = {cv:.3f}"

    def test_uncertainty_with_different_missing_patterns(self):
        """Test uncertainty with different missing data patterns."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        
        patterns = [
            "random",      # Random missing
            "block",       # Consecutive missing blocks
            "periodic",    # Periodic missing pattern
        ]
        
        for pattern in patterns:
            df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)}, index=dates)
            
            if pattern == "random":
                # Random 20% missing
                mask = np.random.random(df.shape) < 0.2
                df = df.mask(mask)
            elif pattern == "block":
                # Missing blocks
                df.iloc[10:15, :] = np.nan
                df.iloc[30:35, :] = np.nan
            elif pattern == "periodic":
                # Every 5th value missing
                df.iloc[::5, :] = np.nan

            if df.isna().any().any():  # Only test if there are missing values
                imputer = Imputer(data=df, verbose=False)
                df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=10)

                assert df_imputed.isna().sum().sum() == 0, f"Missing values remain in {pattern} pattern!"
                assert uncertainty["method"] == "monte_carlo"
                assert uncertainty["rmse"] > 0
                assert uncertainty["mae"] > 0


# Fixtures for uncertainty tests
@pytest.fixture
def uncertainty_test_data():
    """Create test data specifically for uncertainty testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    # Create data with some structure for better uncertainty estimation
    t = np.arange(60)
    return pd.DataFrame(
        {
            "trend": t * 0.1 + np.random.randn(60) * 0.2,
            "seasonal": np.sin(2 * np.pi * t / 10) + np.random.randn(60) * 0.1,
            "noise": np.random.randn(60),
        },
        index=dates,
    )


class TestUncertaintyWithFixtures:
    """Test uncertainty functionality using fixtures."""

    def test_uncertainty_with_fixture_data(self, uncertainty_test_data):
        """Test uncertainty estimation with fixture data."""
        df = uncertainty_test_data.copy()
        df.iloc[::8, :] = np.nan  # Add missing values

        imputer = Imputer(data=df, verbose=False)
        df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=15)

        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"
        assert all(key in uncertainty for key in ["rmse", "mae", "method"])

    @pytest.mark.parametrize("missing_fraction", [0.1, 0.2, 0.3])
    def test_uncertainty_vs_missing_fraction(self, uncertainty_test_data, missing_fraction):
        """Test how uncertainty changes with different amounts of missing data."""
        df = uncertainty_test_data.copy()
        
        # Create random missing pattern
        mask = np.random.random(df.shape) < missing_fraction
        df = df.mask(mask)

        if df.isna().any().any():
            imputer = Imputer(data=df, verbose=False)
            df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=20)

            assert df_imputed.isna().sum().sum() == 0
            assert uncertainty["rmse"] > 0
            
            # Generally, more missing data should lead to higher uncertainty
            # (though this is not always guaranteed due to randomness)
            assert uncertainty["rmse"] < 10.0  # Sanity check


@pytest.mark.integration
class TestUncertaintyIntegration:
    """Integration tests for uncertainty estimation."""

    def test_uncertainty_real_world_scenario(self):
        """Test uncertainty estimation in a realistic scenario."""
        # Simulate sensor data with daily patterns
        dates = pd.date_range("2020-01-01", periods=200, freq="H")
        hours = dates.hour
        
        df = pd.DataFrame({
            "temperature": 20 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.randn(200) * 2,
            "humidity": 50 + 20 * np.cos(2 * np.pi * hours / 24) + np.random.randn(200) * 5,
            "pressure": 1013 + np.random.randn(200) * 3,
        }, index=dates)
        
        # Simulate realistic missing patterns (sensor outages)
        # Random missing
        mask1 = np.random.random((200, 3)) < 0.05
        # Sensor outage blocks
        df.iloc[50:55, 0] = np.nan  # Temperature sensor out
        df.iloc[100:110, 1] = np.nan  # Humidity sensor out
        
        df = df.mask(mask1)
        
        imputer = Imputer(data=df, verbose=False)
        df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True, n_repeats=30)
        
        # Validate results
        assert df_imputed.isna().sum().sum() == 0
        assert uncertainty["method"] == "monte_carlo"
        
        # Check that daily patterns are preserved
        temp_correlation = np.corrcoef(
            df_imputed["temperature"], 
            20 + 10 * np.sin(2 * np.pi * df_imputed.index.hour / 24)
        )[0, 1]
        assert temp_correlation > 0.7, "Daily temperature pattern not preserved"