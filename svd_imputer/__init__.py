"""
SVD Time Series Imputer
========================

A simple and streamlined package for time series imputation using
Singular Value Decomposition (SVD) with automatic rank estimation.

Main Components:
- Imputer: Main class for time series imputation
- validate_dataframe: Data validation utilities
"""

from .imputer import Imputer
from .preprocessing import validate_dataframe

# Version information
try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs without setuptools_scm
    __version__ = "0.1.0"

__all__ = ["Imputer", "validate_dataframe", "__version__"]
