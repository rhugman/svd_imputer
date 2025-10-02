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

__version__ = "0.1.0"
__all__ = ["Imputer"]
