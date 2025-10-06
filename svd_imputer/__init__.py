"""
SVD Time Series Imputer
========================

A simple and streamlined package for time series imputation using
Singular Value Decomposition (SVD) with automatic rank estimation.

Main Components:
- Imputer: Main class for time series imputation
- preprocessing: Data validation utilities
"""

from .imputer import Imputer
from .preprocessing import prepare_data

# Version information
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installs
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

__all__ = ["Imputer", "prepare_data", "__version__"]

__version__ = "0.1.0"
__all__ = ["Imputer"]
