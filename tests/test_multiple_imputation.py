import numpy as np
import pandas as pd
import pytest
from svd_imputer import Imputer

class TestMultipleImputation:
    """Test Multiple Imputation with Stochastic SVD and Rubin's Rules."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        np.random.seed(42)
        data = {
            "A": np.random.randn(20),
            "B": np.random.randn(20) + 5,
            "C": np.random.randn(20) * 2
        }
        df = pd.DataFrame(data, index=dates)

        # Create scattered NaN values
        df.iloc[1, 0] = np.nan  # A[1] = NaN
        df.iloc[3, 1] = np.nan  # B[3] = NaN
        df.iloc[7, 0] = np.nan  # A[7] = NaN
        df.iloc[12, 1] = np.nan  # B[12] = NaN
        df.iloc[16, 2] = np.nan  # C[16] = NaN

        return df

    def test_multiple_imputation_structure(self, sample_data):
        """Test that fit_transform returns correct structure with return_uncertainty=True."""
        imputer = Imputer(data=sample_data, rank=2, verbose=False)
        
        # Run with return_uncertainty=True
        result = imputer.fit_transform(return_uncertainty=True, n_imputations=5, seed=42)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        df_imputed, df_uncertainty = result
        
        assert isinstance(df_imputed, pd.DataFrame)
        assert isinstance(df_uncertainty, pd.DataFrame)
        
        assert df_imputed.shape == sample_data.shape
        assert df_uncertainty.shape == sample_data.shape
        
        assert df_imputed.isna().sum().sum() == 0
        assert df_uncertainty.isna().sum().sum() == 0

    def test_uncertainty_values(self, sample_data):
        """Test that uncertainty values are reasonable."""
        imputer = Imputer(data=sample_data, rank=2, verbose=False)
        df_imputed, df_uncertainty = imputer.fit_transform(return_uncertainty=True, n_imputations=10, seed=42)
        
        # Uncertainty should be non-negative
        assert (df_uncertainty >= 0).all().all()
        
        # Uncertainty should be strictly positive (unless data is constant or perfect fit)
        # With random noise added, it should be positive
        assert (df_uncertainty > 0).all().all()

    def test_stochasticity(self, sample_data):
        """Test that stochastic imputation produces different results."""
        imputer = Imputer(data=sample_data, rank=2, verbose=False)
        
        # Run twice with different seeds
        df_imputed1, _ = imputer.fit_transform(return_uncertainty=True, n_imputations=1, seed=1)
        df_imputed2, _ = imputer.fit_transform(return_uncertainty=True, n_imputations=1, seed=2)
        
        # Should be different
        assert not df_imputed1.equals(df_imputed2)

    def test_backward_compatibility(self, sample_data):
        """Test that return_uncertainty=False still works as expected."""
        imputer = Imputer(data=sample_data, rank=2, verbose=False)
        result = imputer.fit_transform(return_uncertainty=False)
        
        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0
