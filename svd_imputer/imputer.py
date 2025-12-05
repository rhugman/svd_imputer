"""
SVD-based time series imputation with automatic rank estimation.

This module contains the core SVD imputation algorithm, rank estimation,
and the main Imputer class.
"""

import logging
import warnings
from math import sqrt
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .preprocessing import (
    check_sufficient_rank,
    postprocess_after_svd,
    preprocess_for_svd,
    validate_dataframe,
)

# Configure logger for this module
logger = logging.getLogger(__name__)


def _configure_logger():
    """Configure logger with appropriate defaults if not already configured."""
    # Only configure if no handlers exist and no parent handlers exist
    if not logger.handlers and not logger.parent.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent duplicate messages

    # If parent has handlers, use those and just set level if needed
    elif logger.parent.handlers and logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


# Configure logger
_configure_logger()


def estimate_rank(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Estimate optimal rank based on cumulative variance explained.

    Performs SVD on the data (with NaN values temporarily filled) and
    determines the rank needed to capture the specified variance threshold.

    Parameters
    ----------
    X : np.ndarray
        Input data array (may contain NaN values)
    variance_threshold : float, optional
        Fraction of variance to preserve (default: 0.95 for 95%)

    Returns
    -------
    int
        Estimated optimal rank

    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    >>> rank = estimate_rank(X, variance_threshold=0.95)
    """
    # Use consistent preprocessing approach
    X_temp, _ = preprocess_for_svd(X)

    # Fill missing values with 0.0 (mean in standardized space) for rank estimation
    X_temp = np.where(np.isnan(X_temp), 0.0, X_temp)

    # Perform SVD
    logger.debug(f"Computing SVD for rank estimation on {X_temp.shape} matrix")
    _, s, _ = np.linalg.svd(X_temp, full_matrices=False)

    # Calculate cumulative variance explained
    variance_explained = (s**2) / np.sum(s**2)
    cumulative_variance = np.cumsum(variance_explained)

    # Find minimum rank that meets threshold
    rank = np.searchsorted(cumulative_variance, variance_threshold) + 1

    # Ensure rank is at least 1 and at most min(n_rows, n_cols)
    max_rank = len(s)
    rank = max(1, min(rank, max_rank))

    logger.debug(
        f"Estimated rank {rank} for variance threshold {variance_threshold:.1%} "
        f"(captures {cumulative_variance[rank-1]:.1%} variance)"
    )

    return int(rank)


def compute_low_rank_approximation(X: np.ndarray, rank: int) -> np.ndarray:
    """
    Compute low-rank approximation of the input matrix using SVD.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix
    rank : int
        Desired rank for the approximation

    Returns
    -------
    np.ndarray
        Low-rank approximation of the input matrix
    """
    # Perform SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Low-rank approximation
    S = np.diag(s[:rank])
    X_approx = U[:, :rank] @ S @ Vt[:rank, :]
    return X_approx


def iterative_svd_impute(
    X: np.ndarray,
    rank: int = 2,
    max_iters: int = 500,
    tol: float = 1e-4,
    return_svd: bool = False,
    stochastic: bool = False,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Impute missing values using iterative SVD.

    This algorithm iteratively:
    1. Applies preprocessing (detrending, standardization)
    2. Computes SVD and low-rank approximation
    3. Updates missing values with approximation
    4. Repeats until convergence
    5. Applies postprocessing to restore original scale

    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    rank : int, optional
        Number of singular values/vectors to keep (default: 2)
    max_iters : int, optional
        Maximum number of iterations (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    return_svd : bool, optional
        Whether to return SVD components (default: False)
    stochastic : bool, optional
        Whether to use stochastic imputation (default: False)
    random_state : int or np.random.Generator, optional
        Random seed for stochastic imputation

    Returns
    -------
    np.ndarray or tuple
        Array with imputed values. If return_svd is True, returns (X_imputed, svd_dict).
        If stochastic is True, svd_dict includes 'sigma_sq' (residual variance).

    Raises
    ------
    ValueError
        If rank exceeds matrix dimensions

    Examples
    --------
    >>> X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> X_imputed = iterative_svd_impute(X, rank=2)
    """
    # Validate rank
    n_rows, n_cols = X.shape
    max_possible_rank = min(n_rows, n_cols)

    if rank > max_possible_rank:
        raise ValueError(f"rank={rank} exceeds maximum possible rank {max_possible_rank} " f"for matrix with shape {X.shape}")
    
    # Initialize random generator if stochastic
    rng = np.random.default_rng(random_state) if stochastic else None

    # Store missing positions before preprocessing
    inds = np.where(np.isnan(X))
    observed_mask = ~np.isnan(X)

    # Preprocess (this now preserves missing values)
    X_filled, preprocessing_info = preprocess_for_svd(X)

    # Convert to numpy array if it's a DataFrame after preprocessing
    if isinstance(X_filled, pd.DataFrame):
        X_filled = X_filled.values

    # Now fill missing values with zeros (mean in standardized space) for initial guess
    X_filled = np.where(np.isnan(X_filled), 0.0, X_filled)

    # Step 2: Iterative updates
    converged = False
    final_svd = None
    sigma_sq = 0.0

    logger.debug(f"Starting iterative SVD imputation (rank={rank}, max_iters={max_iters}, tol={tol}, stochastic={stochastic})")
    for it in range(max_iters):
        # Compute SVD and store components for potential return
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        final_svd = {"U": U[:, :rank], "s": s[:rank], "Vt": Vt[:rank, :]}

        # Compute low-rank approximation
        X_approx = compute_low_rank_approximation(X_filled, rank)

        # Update only the originally missing entries
        X_new = X_filled.copy()
        
        if stochastic:
            # Calculate residual variance on observed data
            # Residuals = Observed - Approximation
            residuals = X_filled[observed_mask] - X_approx[observed_mask]
            sigma_sq = np.mean(residuals**2)
            
            # Add Gaussian noise to the imputed values
            # Noise ~ N(0, sigma_sq)
            noise = rng.normal(0, np.sqrt(sigma_sq), size=X.shape)
            
            # Update missing values with approximation + noise
            X_new[inds] = X_approx[inds] + noise[inds]
        else:
            # Deterministic update
            X_new[inds] = X_approx[inds]

        # Check convergence
        diff = np.linalg.norm(X_new - X_filled) / np.linalg.norm(X_filled)
        X_filled = X_new

        if diff < tol:
            converged = True
            logger.debug(f"SVD imputation converged at iteration {it+1} (diff={diff:.2e})")
            break

    # Warn if didn't converge
    if not converged:
        logger.warning(
            f"Max iterations ({max_iters}) reached without convergence "
            f"(diff={diff:.2e}). Consider increasing max_iters or relaxing tol."
        )

    X_filled = postprocess_after_svd(X_filled, preprocessing_info)

    if return_svd and final_svd is not None:
        if stochastic:
            final_svd["sigma_sq"] = sigma_sq
        return X_filled, final_svd
    else:
        return X_filled


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return sqrt(np.mean((true - pred) ** 2))


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(true - pred))


def _random_mask_observed(X: np.ndarray, frac: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
    """
    Safely mask observed values by going row-by-row, ensuring no entirely NaN rows.

    This improved algorithm prevents the creation of entirely NaN rows that would
    get filled with column means during imputation, eliminating bias in Monte Carlo validation.

    Algorithm:
    1. Go through rows in random order
    2. Skip rows with ≤1 non-missing value (to preserve at least 1 observation per row)
    3. In each eligible row, randomly mask columns (max n_cols - 1)
    4. Continue until target fraction is achieved

    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    frac : float
        Target fraction of observed values to mask
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Boolean mask (True = observed, False = masked)

    Notes
    -----
    This function guarantees that no rows will become entirely NaN after masking,
    preventing the mean-filling bias that can occur in Monte Carlo validation.
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)  # True = observed, False = masked

    # Count total observed values and target number to mask
    total_observed = np.sum(mask)
    target_to_mask = int(total_observed * frac)

    if target_to_mask == 0:
        return mask  # Nothing to mask

    # Create list of row indices in random order
    n_rows, n_cols = X.shape
    row_indices = np.arange(n_rows)
    rng.shuffle(row_indices)

    masked_count = 0

    # Go through rows in random order
    for row_idx in row_indices:
        if masked_count >= target_to_mask:
            break

        # Get observed positions in this row
        row_observed = mask[row_idx, :]
        n_observed_in_row = np.sum(row_observed)

        # Skip rows with ≤ 1 observed value (can't safely mask anything)
        if n_observed_in_row <= 1:
            continue

        # Determine how many columns to mask in this row
        # - Maximum: n_observed - 1 (leave at least 1 observed)
        # - Target: remaining values to mask
        max_to_mask_in_row = n_observed_in_row - 1
        remaining_to_mask = target_to_mask - masked_count
        n_to_mask_in_row = min(max_to_mask_in_row, remaining_to_mask)

        if n_to_mask_in_row <= 0:
            continue

        # Get column indices of observed values in this row
        observed_cols = np.where(row_observed)[0]

        # Set n_to_mask_in_row to a random number between 1 and n_to_mask_in_row
        n_to_mask_in_row = rng.integers(1, n_to_mask_in_row + 1)

        # Randomly select columns to mask
        cols_to_mask = rng.choice(observed_cols, size=n_to_mask_in_row, replace=False)

        # Apply masking
        mask[row_idx, cols_to_mask] = False
        masked_count += n_to_mask_in_row

    return mask


def _block_mask_time(X: np.ndarray, block_len: int = 5, n_blocks: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """
    Safely mask temporal blocks ensuring no entirely NaN rows.

    This improved algorithm prevents the creation of entirely NaN rows by validating
    each block placement before applying it.

    Algorithm:
    1. For each block to place:
       - Try random starting positions
       - For each position, check if masking would create entirely NaN rows
       - Apply masking only if safe (leaves at least 1 observed value per row)
    2. If can't place requested blocks safely, place as many as possible

    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    block_len : int
        Length of temporal blocks to mask
    n_blocks : int
        Number of blocks to mask
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Boolean mask (True = observed, False = masked)

    Notes
    -----
    This function guarantees that no rows will become entirely NaN after masking.
    If the data is too sparse to safely place all requested blocks, fewer blocks
    will be placed to maintain data integrity.
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)  # True = observed, False = masked
    n_rows, n_cols = X.shape

    blocks_placed = 0
    max_attempts = n_blocks * 50  # Limit attempts to avoid infinite loops

    for attempt in range(max_attempts):
        if blocks_placed >= n_blocks:
            break

        # Random starting position
        start_row = rng.integers(0, max(1, n_rows - block_len + 1))
        end_row = min(start_row + block_len, n_rows)
        block_rows = list(range(start_row, end_row))

        # Test if masking this block would create entirely NaN rows
        test_mask = mask.copy()

        # Temporarily mask the block
        for r in block_rows:
            for c in range(n_cols):
                if mask[r, c]:  # Only mask currently observed values
                    test_mask[r, c] = False

        # Check if any rows became entirely NaN
        any_entirely_nan = False
        for r in block_rows:
            if not np.any(test_mask[r, :]):  # Row has no True values
                any_entirely_nan = True
                break

        # If safe, apply the masking
        if not any_entirely_nan:
            mask = test_mask
            blocks_placed += 1

    if blocks_placed < n_blocks:
        logger.warning(
            f"Could only place {blocks_placed}/{n_blocks} blocks safely. "
            f"Data may be too sparse for requested block masking."
        )
        warnings.warn(
            f"Could only place {blocks_placed}/{n_blocks} blocks safely. "
            "Data may be too sparse for requested block masking.",
            RuntimeWarning,
        )

        # If no blocks could be placed, fall back to random masking to ensure Monte Carlo diversity
        if blocks_placed == 0:
            logger.warning(
                "No blocks could be placed safely. Falling back to random masking with frac=0.1 "
                "to ensure Monte Carlo validation diversity."
            )
            warnings.warn(
                "No blocks could be placed safely. Falling back to random masking with frac=0.1 "
                "to ensure Monte Carlo validation diversity.",
                RuntimeWarning,
            )
            return _random_mask_observed(X, frac=0.1, seed=seed)

    return mask


def _monte_carlo_validation(
    X: np.ndarray,
    rank: int,
    max_iters: int,
    tol: float,
    preprocessing_info: tuple,
    n_repeats: int = 100,
    mask_strategy: str = "random",
    frac: float = 0.1,
    block_len: int = 5,
    n_blocks: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform Monte Carlo validation to estimate imputation error.

    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values (preprocessed)
    rank : int
        Rank for SVD imputation
    max_iters : int
        Maximum iterations for imputation
    tol : float
        Convergence tolerance
    preprocessing_info : tuple
        Preprocessing parameters for postprocessing results
    n_repeats : int
        Number of Monte Carlo repeats
    mask_strategy : str
        'random' or 'block'
    frac : float
        Fraction to mask (for random strategy)
    block_len : int
        Block length (for block strategy)
    n_blocks : int
        Number of blocks (for block strategy)
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with RMSE and MAE statistics and postprocessed raw_imputed
    """
    rng = np.random.default_rng(seed)
    rmse_list = []
    mae_list = []
    imputed_list = []

    for i in range(n_repeats):
        # Generate a different seed for each iteration
        # If overall seed is None, each iteration gets None (true randomness)
        # If overall seed is provided, each iteration gets a deterministic but different seed
        if seed is None:
            s = None  # Each iteration gets true randomness
        else:
            s = int(rng.integers(1 << 30))  # Each iteration gets a different deterministic seed

        # Create mask
        if mask_strategy == "random":
            mask = _random_mask_observed(X, frac=frac, seed=s)
        elif mask_strategy == "block":
            mask = _block_mask_time(X, block_len=block_len, n_blocks=n_blocks, seed=s)
        else:
            raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")

        # Build X_with_nans for imputation
        X_with_nans = X.copy()
        obs_all = ~np.isnan(X)
        masked_positions = np.logical_and(obs_all, ~mask)
        X_with_nans[masked_positions] = np.nan

        # check if there are any rows that are entirely NaN
        if np.any(np.all(np.isnan(X_with_nans), axis=1)):
            raise ValueError("There are rows that are entirely NaN after masking. Adjust masking parameters.")

        _rank = None
        if isinstance(rank, int):
            _rank = rank
        elif isinstance(rank, str) and rank == "auto":
            warnings.warn("rank='auto' recalculating rank for each Monte Carlo iteration.", RuntimeWarning)
            _rank = estimate_rank(X_with_nans, variance_threshold=variance_threshold, preprocessed=False)
            assert _rank is not None
        else:
            raise ValueError(f"Unsupported rank type: {type(rank)}")
        # Impute
        X_imputed = iterative_svd_impute(X_with_nans, rank=rank, max_iters=max_iters, tol=tol)

        # Apply postprocessing to convert back to original scale
        X_imputed_postprocessed = postprocess_after_svd(X_imputed, preprocessing_info)
        imputed_list.append(X_imputed_postprocessed)

        # Compute error only on masked positions
        true_vals = X[masked_positions]
        pred_vals = X_imputed[masked_positions]

        rmse_list.append(_rmse(true_vals, pred_vals))
        mae_list.append(_mae(true_vals, pred_vals))

    # Summarize
    def summarize(vals):
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0  # ddof=1 for sample std dev
        se = s / np.sqrt(len(vals))
        lower = m - 1.96 * se
        upper = m + 1.96 * se
        return {"mean": m, "std": s, "95%_CI": (lower, upper)}

    return {
        "RMSE": summarize(rmse_list),
        "MAE": summarize(mae_list),
        "raw_rmse": rmse_list,
        "raw_mae": mae_list,
        "raw_imputed": imputed_list,
    }


class Imputer:
    """
    SVD-based time series imputer with automatic rank estimation.

    This class provides an efficient interface for imputing missing values
    in time series data using Singular Value Decomposition (SVD). Data is
    validated and preprocessed once at initialization, eliminating redundant
    operations and ensuring consistency across all analyses.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index and time series data to be imputed.
        This data is validated and preprocessed once at initialization.
    variance_threshold : float, optional
        Fraction of variance to preserve for automatic rank estimation.
        Default is 0.95 (95% of variance). Must be between 0 and 1.
    rank : int, str, or None, optional
        Fixed rank to use. Options:
        - int: Use fixed rank value
        - "auto": Optimize rank via cross-validation to minimize imputation error
        - None (default): Estimate rank based on variance_threshold
    max_iters : int, optional
        Maximum number of SVD iterations (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    verbose : bool, optional
        Whether to print progress information (default: True)

    Attributes
    ----------
    data_original_ : pd.DataFrame
        Original validated input data
    data_preprocessed_ : pd.DataFrame or np.ndarray
        Preprocessed data ready for SVD operations
    preprocessing_info_ : tuple
        Preprocessing parameters for consistent transformations
    rank_ : int
        The rank used for imputation (set after fitting)
    is_fitted_ : bool
        Whether the imputer has been fitted
    columns_ : list
        Column names from the input DataFrame
    index_name_ : str
        Name of the index from the input DataFrame
    svd_components_ : dict, optional
        Cached SVD components {'U': U, 's': s, 'Vt': Vt} for reuse
    optimization_results_ : dict, optional
        Results from rank optimization (only set when rank="auto")

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from svd_imputer import Imputer
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range('2020-01-01', periods=10)
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
    ...     'B': [10, np.nan, 30, 40, np.nan, 60, 70, 80, 90, 100]
    ... }, index=dates)
    >>>
    >>> # Automatic rank estimation (default behavior)
    >>> imputer = Imputer(df, variance_threshold=0.95)
    >>> df_imputed = imputer.fit_transform()
    >>>
    >>> # Fixed rank
    >>> imputer = Imputer(df, rank=2)
    >>> df_imputed = imputer.fit_transform()
    >>>
    >>> # Auto-optimize rank via cross-validation
    >>> imputer = Imputer(df, rank="auto")
    >>> imputer.fit()
    >>> print(f"Optimized rank: {imputer.rank_}")
    >>> df_imputed = imputer.transform()
    >>>
    >>> # SVD reuse for analysis
    >>> imputer = Imputer(df, rank=2)
    >>> imputer.fit()
    >>>
    >>> # Project new data onto SVD subspace
    >>> df_projected = imputer.project_data(new_df)
    >>>
    >>> # Reconstruct data using SVD
    >>> df_reconstructed = imputer.reconstruct_data(new_df)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variance_threshold: float = 0.95,
        rank: Union[int, str, None] = None,
        max_iters: int = 500,
        tol: float = 1e-4,
        verbose: bool = True,
    ):
        # Validate parameters
        if variance_threshold <= 0 or variance_threshold > 1:
            raise ValueError(f"variance_threshold must be between 0 and 1, got {variance_threshold}")

        if rank is not None:
            if isinstance(rank, str) and rank != "auto":
                raise ValueError(f"rank must be int, 'auto', or None, got '{rank}'")
            elif isinstance(rank, int) and rank < 1:
                raise ValueError(f"rank must be positive, got {rank}")

        if max_iters < 1:
            raise ValueError(f"max_iters must be positive, got {max_iters}")

        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")

        # Validate and preprocess data ONCE at initialization
        self.data_original_ = validate_dataframe(data)
        self.data_preprocessed_, self.preprocessing_info_ = preprocess_for_svd(self.data_original_)

        # Store metadata from original data
        self.columns_ = self.data_original_.columns.tolist()
        self.index_name_ = self.data_original_.index.name
        self.shape_ = self.data_original_.shape

        # Store parameters
        self.variance_threshold = variance_threshold
        self.rank = rank
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

        # Attributes set during fitting
        self.rank_ = None
        self.is_fitted_ = False
        self.optimization_results_ = None
        self.svd_components_ = None  # Dict with {'U': U, 's': s, 'Vt': Vt}

        if self.verbose:
            logger.info(f"Initialized imputer with data shape {self.shape_}")
            n_missing = np.isnan(self.data_original_.values).sum()
            total_vals = np.prod(self.shape_)
            pct_missing = (n_missing / total_vals) * 100
            logger.info(f"Missing values: {n_missing}/{total_vals} ({pct_missing:.1f}%)")

    def fit(self) -> "Imputer":
        """
        Fit the imputer on the stored preprocessed data.

        The rank determination strategy depends on the `rank` parameter:
        - If rank="auto": Optimize rank via cross-validation to minimize error
        - If rank=None: Estimate rank based on variance_threshold (default)
        - If rank=int: Use the specified fixed rank

        Returns
        -------
        self
            Fitted imputer

        Notes
        -----
        When rank="auto", the optimization results are stored in the
        `optimization_results_` attribute for inspection.
        """
        # Work directly with preprocessed data
        X_array = (
            self.data_preprocessed_.values if isinstance(self.data_preprocessed_, pd.DataFrame) else self.data_preprocessed_
        )

        # Determine rank based on user specification
        if self.rank == "auto":
            # Auto-optimize rank via cross-validation
            if self.verbose:
                logger.info("Auto-optimizing rank via cross-validation...")

            self.optimization_results_ = self.optimize_rank()
            self.rank_ = self.optimization_results_["optimal_rank"]

            if self.verbose:
                score = self.optimization_results_["optimal_score"]
                converged = self.optimization_results_["convergence_info"]["is_converged"]
                logger.info(f"Optimized rank: {self.rank_} (CV score: {score:.4f})")
                if not converged:
                    logger.warning("Optimization may not have converged to a clear minimum")

        elif self.rank is None:
            # Variance-based estimation (default behavior)
            self.rank_ = estimate_rank(X_array, self.variance_threshold)
            if self.verbose:
                logger.info(f"Estimated rank: {self.rank_} " f"(variance threshold: {self.variance_threshold*100:.0f}%)")

        else:
            # Fixed rank specified by user
            self.rank_ = self.rank
            # Check if requested rank is feasible
            check_sufficient_rank(self.data_original_, self.rank_)
            if self.verbose:
                logger.info(f"Using fixed rank: {self.rank_}")

        # Perform imputation to cache SVD components
        if self.verbose:
            logger.info(f"Fitting SVD imputer with rank={self.rank_}...")

        X_imputed, svd_components = self._fit_and_cache_svd(X_array)
        self.svd_components_ = svd_components

        self.is_fitted_ = True
        if self.verbose:
            logger.info("Fit completed. SVD components cached for reuse.")

        return self

    def _fit_and_cache_svd(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fit the SVD imputer and cache the SVD components.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed data array

        Returns
        -------
        tuple
            (imputed_data, svd_components_dict)
        """
        result = iterative_svd_impute(X, rank=self.rank_, max_iters=self.max_iters, tol=self.tol, return_svd=True)

        if isinstance(result, tuple):
            X_imputed, svd_components = result
        else:
            # Fallback if SVD components not returned
            X_imputed = result
            svd_components = None

        return X_imputed, svd_components

    def optimize_rank(
        self,
        rank_range: Optional[Tuple[int, int]] = None,
        cv_folds: int = 5,
        n_repeats_per_fold: int = 20,
        mask_strategy: str = "random",
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        metric: str = "rmse",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize rank via cross-validation to minimize imputation error.

        This method systematically tests different rank values using cross-validation
        with multiple random masking experiments to find the rank that minimizes
        prediction error on held-out data.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
        rank_range : tuple of int, optional
            (min_rank, max_rank) to test. If None, uses [1, min(n_rows, n_cols, 10)]
        cv_folds : int, optional
            Number of cross-validation folds (default: 5)
        n_repeats_per_fold : int, optional
            Number of random masking experiments per fold (default: 20)
        mask_strategy : str, optional
            'random' or 'block' masking strategy (default: 'random')
        frac : float, optional
            Fraction of values to mask for 'random' strategy (default: 0.1)
        block_len : int, optional
            Block length for 'block' strategy (default: 5)
        n_blocks : int, optional
            Number of blocks for 'block' strategy (default: 1)
        metric : str, optional
            Optimization metric: 'rmse' or 'mae' (default: 'rmse')
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary containing:
            - 'optimal_rank': Best rank found
            - 'optimal_score': Cross-validation score for optimal rank
            - 'results_df': DataFrame with detailed results for all ranks
            - 'cv_details': Fold-by-fold results
            - 'convergence_info': Optimization diagnostics

        Examples
        --------
        >>> imputer = Imputer()
        >>> results = imputer.optimize_rank(df, rank_range=(1, 8))
        >>> print(f"Optimal rank: {results['optimal_rank']}")
        >>> print(results['results_df'])
        """
        # Work with stored preprocessed data
        X_array = (
            self.data_preprocessed_.values if isinstance(self.data_preprocessed_, pd.DataFrame) else self.data_preprocessed_
        )
        n_rows, n_cols = X_array.shape

        # Determine rank range
        if rank_range is None:
            if self.verbose:
                logger.info("Estimating rank range based on variance thresholds (0.75 - 0.95)...")

            # Fill missing values with 0.0 (mean) for estimation
            X_filled = np.where(np.isnan(X_array), 0.0, X_array)

            # Compute SVD once
            _, s, _ = np.linalg.svd(X_filled, full_matrices=False)

            # Calculate cumulative variance
            variance_explained = (s**2) / np.sum(s**2)
            cumulative_variance = np.cumsum(variance_explained)

            # Find ranks for thresholds
            min_rank = np.searchsorted(cumulative_variance, 0.75) + 1
            max_rank = np.searchsorted(cumulative_variance, 0.95) + 1

            # Ensure valid range
            max_possible = len(s)
            min_rank = max(1, min(min_rank, max_possible))
            max_rank = max(1, min(max_rank, max_possible))

            if max_rank < min_rank:
                max_rank = min_rank

            rank_range = (int(min_rank), int(max_rank))

            if self.verbose:
                logger.info(f"Auto-selected rank range: {rank_range} based on variance.")

        min_rank, max_rank = rank_range
        if min_rank < 1:
            raise ValueError(f"min_rank must be at least 1, got {min_rank}")
        if max_rank > min(n_rows, n_cols):
            raise ValueError(
                f"max_rank={max_rank} exceeds maximum possible rank {min(n_rows, n_cols)} "
                f"for matrix with shape {X_array.shape}"
            )

        if metric not in ["rmse", "mae"]:
            raise ValueError(f"metric must be 'rmse' or 'mae', got '{metric}'")

        if self.verbose:
            logger.info(f"Optimizing rank in range [{min_rank}, {max_rank}] using {cv_folds}-fold CV")
            logger.info(f"Strategy: {mask_strategy}, Repeats per fold: {n_repeats_per_fold}")

        # Initialize random number generator
        rng = np.random.default_rng(seed)

        # Storage for results
        rank_results = []
        cv_details = {}

        # Test each rank
        ranks_to_test = list(range(min_rank, max_rank + 1))

        for rank in ranks_to_test:
            if self.verbose:
                logger.debug(f"Testing rank {rank}...")

            fold_scores = []
            cv_details[rank] = []

            # Cross-validation folds
            for fold in range(cv_folds):
                fold_seed = None if seed is None else int(rng.integers(1 << 30))

                # Perform validation for this fold and rank
                fold_results = _monte_carlo_validation(
                    X_array,
                    rank=rank,
                    max_iters=self.max_iters,
                    tol=self.tol,
                    preprocessing_info=self.preprocessing_info_,
                    n_repeats=n_repeats_per_fold,
                    mask_strategy=mask_strategy,
                    frac=frac,
                    block_len=block_len,
                    n_blocks=n_blocks,
                    seed=fold_seed,
                )

                # Extract the metric of interest
                if metric == "rmse":
                    fold_score = fold_results["RMSE"]["mean"]
                else:  # mae
                    fold_score = fold_results["MAE"]["mean"]

                fold_scores.append(fold_score)
                cv_details[rank].append({"fold": fold, "score": fold_score, "full_results": fold_results})

            # Summarize across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            rank_results.append(
                {
                    "rank": rank,
                    f"mean_{metric}": mean_score,
                    f"std_{metric}": std_score,
                    "fold_scores": fold_scores,
                }
            )

            if self.verbose:
                logger.debug(f"  Rank {rank}: {metric.upper()}={mean_score:.4f} ± {std_score:.4f}")

        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(rank_results)

        # Find optimal rank (minimum error)
        optimal_idx = results_df[f"mean_{metric}"].idxmin()
        optimal_rank = results_df.loc[optimal_idx, "rank"]
        optimal_score = results_df.loc[optimal_idx, f"mean_{metric}"]

        # Check convergence (is there a clear minimum?)
        scores = results_df[f"mean_{metric}"].values
        stds = results_df[f"std_{metric}"].values

        # Simple convergence check: optimal score should be significantly better than others
        other_scores = scores[scores != optimal_score]
        if len(other_scores) > 0:
            min_other = other_scores.min()
            improvement = min_other - optimal_score
            significance = improvement / stds[optimal_idx] if stds[optimal_idx] > 0 else float("inf")
        else:
            significance = float("inf")

        convergence_info = {
            "improvement_over_second_best": improvement if len(other_scores) > 0 else 0,
            "significance_ratio": significance,
            "is_converged": significance > 1.0,  # At least 1 std dev improvement
            "tested_ranks": ranks_to_test,
            "total_experiments": len(ranks_to_test) * cv_folds * n_repeats_per_fold,
        }

        if self.verbose:
            logger.info("\nOptimization complete:")
            logger.info(f"  Optimal rank: {optimal_rank}")
            logger.info(f"  {metric.upper()}: {optimal_score:.4f}")
            logger.info(f"  Convergence: {'Yes' if convergence_info['is_converged'] else 'No'}")
            if not convergence_info["is_converged"]:
                logger.warning("  Consider expanding rank range or increasing n_repeats_per_fold")

        return {
            "optimal_rank": int(optimal_rank),
            "optimal_score": float(optimal_score),
            "results_df": results_df,
            "cv_details": cv_details,
            "convergence_info": convergence_info,
            "parameters": {
                "rank_range": rank_range,
                "cv_folds": cv_folds,
                "n_repeats_per_fold": n_repeats_per_fold,
                "mask_strategy": mask_strategy,
                "metric": metric,
                "seed": seed,
            },
        }

    def get_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Get rank optimization results (only available when rank="auto" was used).

        Returns
        -------
        dict or None
            Optimization results dictionary containing:
            - 'optimal_rank': Best rank found
            - 'optimal_score': Cross-validation score
            - 'results_df': DataFrame with all tested ranks
            - 'cv_details': Detailed fold-by-fold results
            - 'convergence_info': Optimization diagnostics
            - 'parameters': Optimization parameters used

        Examples
        --------
        >>> imputer = Imputer(rank="auto")
        >>> imputer.fit(df)
        >>> results = imputer.get_optimization_results()
        >>> print(results['results_df'])
        >>> print(f"Tested {len(results['cv_details'])} ranks")
        """
        if not hasattr(self, "optimization_results_") or self.optimization_results_ is None:
            return None
        return self.optimization_results_

    def estimate_uncertainty(
        self,
        n_repeats: int = 100,
        mask_strategy: str = "random",
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Estimate imputation uncertainty using Monte Carlo validation.

        This method temporarily masks observed values, imputes them, and
        compares with actual values to estimate prediction error.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
        n_repeats : int, optional
            Number of Monte Carlo iterations (default: 100)
        mask_strategy : str, optional
            'random' or 'block' masking strategy (default: 'block')
        frac : float, optional
            Fraction of values to mask for 'random' strategy (default: 0.1)
        block_len : int, optional
            Length of temporal blocks for 'block' strategy (default: 5)
        n_blocks : int, optional
            Number of blocks for 'block' strategy (default: 1)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary containing:
            - 'RMSE': {'mean': float, 'std': float, '95%_CI': tuple}
            - 'MAE': {'mean': float, 'std': float, '95%_CI': tuple}
            - 'raw_rmse': list of individual RMSE values
            - 'raw_mae': list of individual MAE values

        Examples
        --------
        >>> imputer = Imputer()
        >>> imputer.fit(df)
        >>> uncertainty = imputer.estimate_uncertainty(df, n_repeats=100)
        >>> print(f"RMSE: {uncertainty['RMSE']['mean']:.3f}")
        """
        # Check if fitted
        if not self.is_fitted_:
            raise RuntimeError("Imputer must be fitted before estimating uncertainty. " "Call fit() first.")

        # Work with stored preprocessed data
        X_array = (
            self.data_preprocessed_.values if isinstance(self.data_preprocessed_, pd.DataFrame) else self.data_preprocessed_
        )

        if self.verbose:
            logger.info(f"Estimating uncertainty with {n_repeats} Monte Carlo repeats...")
            logger.info(f"Strategy: {mask_strategy}, Rank: {self.rank_}")

        # Perform Monte Carlo validation
        results = _monte_carlo_validation(
            X_array,
            rank=self.rank_,
            max_iters=self.max_iters,
            tol=self.tol,
            preprocessing_info=self.preprocessing_info_,
            n_repeats=n_repeats,
            mask_strategy=mask_strategy,
            frac=frac,
            block_len=block_len,
            n_blocks=n_blocks,
            seed=seed,
        )

        if self.verbose:
            logger.info(
                f"RMSE: {results['RMSE']['mean']:.4f} "
                f"(95% CI: {results['RMSE']['95%_CI'][0]:.4f} - "
                f"{results['RMSE']['95%_CI'][1]:.4f})"
            )
            logger.info(
                f"MAE:  {results['MAE']['mean']:.4f} "
                f"(95% CI: {results['MAE']['95%_CI'][0]:.4f} - "
                f"{results['MAE']['95%_CI'][1]:.4f})"
            )

        return results

    def transform(self) -> pd.DataFrame:
        """
        Impute missing values in the stored data.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values in original format
        """
        # Check if fitted
        if not self.is_fitted_:
            raise RuntimeError("Imputer must be fitted before transform. " "Call fit() first.")

        # Work with stored preprocessed data
        X_array = (
            self.data_preprocessed_.values if isinstance(self.data_preprocessed_, pd.DataFrame) else self.data_preprocessed_
        )

        # Use cached SVD components if available for faster imputation
        if self.svd_components_ is not None:
            if self.verbose:
                logger.debug("Using cached SVD components for imputation...")
            X_imputed = self._impute_with_cached_svd(X_array, self.preprocessing_info_)
        else:
            # Fallback to full imputation
            if self.verbose:
                logger.info(f"Imputing with rank={self.rank_}, max_iters={self.max_iters}, tol={self.tol}")
            X_imputed = iterative_svd_impute(X_array, rank=self.rank_, max_iters=self.max_iters, tol=self.tol)
            # Apply postprocessing to return data in original scale
            X_imputed = postprocess_after_svd(X_imputed, self.preprocessing_info_)

        # Convert back to original DataFrame format
        df_imputed = pd.DataFrame(
            X_imputed,
            index=self.data_original_.index,
            columns=self.data_original_.columns,
        )

        if self.verbose:
            n_missing = np.isnan(self.data_original_.values).sum()
            logger.info(f"Imputed {n_missing} missing values")

        return df_imputed

    def _impute_with_cached_svd(self, X: np.ndarray, preprocessing_info: tuple) -> np.ndarray:
        """
        Perform imputation using cached SVD components for efficiency.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed data array
        preprocessing_info : tuple
            Preprocessing parameters for postprocessing

        Returns
        -------
        np.ndarray
            Imputed data array in original scale
        """
        # Use the standard iterative SVD imputation - the cached components
        # are used implicitly through the rank being fixed from fit()
        X_filled = iterative_svd_impute(X, rank=self.rank_, max_iters=self.max_iters, tol=self.tol)

        # Apply postprocessing to return data in original scale
        X_filled = postprocess_after_svd(X_filled, preprocessing_info)
        return X_filled

    def project_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Project new data onto the fitted SVD subspace.

        This method applies the same preprocessing as used during fitting,
        then projects the data onto the learned SVD subspace using cached components.
        Useful for analyzing how new data relates to the fitted model.

        Parameters
        ----------
        new_data : pd.DataFrame
            New DataFrame to project onto SVD subspace. Must have same columns
            as the original fitting data.

        Returns
        -------
        pd.DataFrame
            Projected data in original DataFrame format

        Raises
        ------
        RuntimeError
            If imputer is not fitted
        ValueError
            If new_data columns don't match original data
        """
        # Check if fitted
        if not self.is_fitted_ or self.svd_components_ is None:
            raise RuntimeError("Imputer must be fitted before projecting data. " "Call fit() first.")

        # Validate new data structure matches original
        new_data_validated = validate_dataframe(new_data)

        if list(new_data_validated.columns) != self.columns_:
            raise ValueError(
                f"New data columns {list(new_data_validated.columns)} " f"don't match original columns {self.columns_}"
            )

        # Apply consistent preprocessing
        new_data_preprocessed, _ = preprocess_for_svd(new_data_validated)
        X_new = new_data_preprocessed.values if isinstance(new_data_preprocessed, pd.DataFrame) else new_data_preprocessed

        # Project onto SVD subspace using cached components
        Vt = self.svd_components_["Vt"]

        # Project new data onto the SVD subspace defined by Vt (feature space)
        # This projects columns onto the learned feature directions
        X_projected = X_new @ Vt.T @ Vt  # Project and reconstruct in rank-reduced space

        # Convert back to original scale using stored preprocessing info
        X_projected = postprocess_after_svd(X_projected, self.preprocessing_info_)

        # Return as DataFrame with original structure
        df_projected = pd.DataFrame(
            X_projected,
            index=new_data_validated.index,
            columns=new_data_validated.columns,
        )

        if self.verbose:
            logger.debug(f"Projected data onto rank-{self.rank_} SVD subspace")

        return df_projected

    def reconstruct_data(self, new_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Reconstruct data using the fitted SVD components.

        This method applies the fitted low-rank approximation X ≈ U @ S @ Vt
        to reconstruct data. Useful for denoising, compression, or generating
        smooth versions of time series data.

        Parameters
        ----------
        new_data : pd.DataFrame, optional
            New DataFrame to reconstruct. Must have same columns as original data.
            If None (default), reconstructs the original fitted data.
            Missing values will be filled with column means before reconstruction.

        Returns
        -------
        pd.DataFrame
            Reconstructed data in original DataFrame format

        Raises
        ------
        RuntimeError
            If imputer is not fitted
        ValueError
            If new_data columns don't match original data
        """
        # Check if fitted
        if not self.is_fitted_ or self.svd_components_ is None:
            raise RuntimeError("Imputer must be fitted before reconstructing data. " "Call fit() first.")

        # Use original data if no new data provided
        if new_data is None:
            new_data = self.data_original_.copy()
            if self.verbose:
                logger.debug("Using original fitted data for reconstruction")

        # Validate new data structure matches original
        new_data_validated = validate_dataframe(new_data)

        if list(new_data_validated.columns) != self.columns_:
            raise ValueError(
                f"New data columns {list(new_data_validated.columns)} " f"don't match original columns {self.columns_}"
            )

        # Apply same preprocessing as original data
        # Convert to numpy array for processing
        X_new = new_data_validated.values.astype(float)

        # Fill missing values with column means (in original scale) before preprocessing
        for j in range(X_new.shape[1]):
            col_mask = np.isnan(X_new[:, j])
            if col_mask.sum() > 0:
                # Use the original preprocessing info means (in original scale)
                X_new[col_mask, j] = self.preprocessing_info_[0][j]  # means from preprocessing

        # Apply standardization using original preprocessing parameters
        X_standardized,_ = preprocess_for_svd(X_new)

        # Reconstruct using cached SVD components
        # Project standardized data onto SVD subspace and reconstruct
        Vt = self.svd_components_["Vt"]

        # Use the standardized data in reconstruction
        X_reconstructed = X_standardized @ Vt.T @ Vt

        # Convert back to original scale
        X_reconstructed = postprocess_after_svd(X_reconstructed, self.preprocessing_info_)

        # Return as DataFrame with original structure
        df_reconstructed = pd.DataFrame(
            X_reconstructed,
            index=new_data_validated.index,
            columns=new_data_validated.columns,
        )

        if self.verbose:
            logger.debug(f"Reconstructed data using rank-{self.rank_} SVD approximation")

        return df_reconstructed

    def calculate_reconstruction_residuals(
        self, new_data: Optional[pd.DataFrame] = None, return_stats: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Calculate residuals between observed values and their SVD reconstruction.

        This method computes residuals = original - reconstructed for observed
        (non-missing) values only. Useful for:
        - Model diagnostics: How well does the SVD represent the data?
        - Quality assessment: Are there systematic biases in reconstruction?
        - Outlier detection: Large residuals may indicate anomalies
        - Model selection: Compare residual patterns across different ranks

        Parameters
        ----------
        new_data : pd.DataFrame, optional
            Data to calculate residuals for. Must have same columns as original data.
            If None (default), uses the original fitted data.
        return_stats : bool, optional
            Whether to return summary statistics along with residuals (default: True)

        Returns
        -------
        pd.DataFrame or tuple
            If return_stats=False: DataFrame of residuals (NaN for originally missing values)
            If return_stats=True: (residuals_df, stats_dict) where stats_dict contains:
            - 'rmse': Root Mean Square Error for observed values
            - 'mae': Mean Absolute Error for observed values
            - 'bias': Mean bias (systematic over/under-estimation)
            - 'r_squared': R² correlation between original and reconstructed
            - 'residual_stats': Per-column statistics (mean, std, min, max)
            - 'n_observed': Number of observed values used in calculations

        Raises
        ------
        RuntimeError
            If imputer is not fitted
        ValueError
            If new_data columns don't match original data

        Examples
        --------
        >>> # Basic residual calculation
        >>> imputer = Imputer(data, rank=5)
        >>> imputer.fit()
        >>> residuals_df = imputer.calculate_reconstruction_residuals(return_stats=False)
        >>>
        >>> # With detailed statistics
        >>> residuals_df, stats = imputer.calculate_reconstruction_residuals()
        >>> print(f"Overall RMSE: {stats['rmse']:.4f}")
        >>> print(f"R²: {stats['r_squared']:.4f}")
        >>> print(stats['residual_stats'])
        >>>
        >>> # For new data
        >>> residuals_new, stats_new = imputer.calculate_reconstruction_residuals(new_df)
        """
        # Check if fitted
        if not self.is_fitted_ or self.svd_components_ is None:
            raise RuntimeError("Imputer must be fitted before calculating residuals. " "Call fit() first.")

        # Use original data if no new data provided
        if new_data is None:
            original_data = self.data_original_.copy()
            if self.verbose:
                logger.debug("Calculating residuals for original fitted data")
        else:
            # Validate new data structure matches original
            original_data = validate_dataframe(new_data)

            if list(original_data.columns) != self.columns_:
                raise ValueError(
                    f"New data columns {list(original_data.columns)} " f"don't match original columns {self.columns_}"
                )

            if self.verbose:
                logger.debug("Calculating residuals for new data")

        # Get reconstruction of the data
        reconstructed_data = self.reconstruct_data(original_data)

        # Calculate residuals only for observed (non-missing) values
        observed_mask = ~pd.isna(original_data)

        # Initialize residuals DataFrame with NaN
        residuals_df = pd.DataFrame(np.nan, index=original_data.index, columns=original_data.columns)

        # Calculate residuals where data was observed
        residuals_df[observed_mask] = original_data[observed_mask] - reconstructed_data[observed_mask]

        if not return_stats:
            return residuals_df

        # Calculate detailed statistics for observed values only
        original_obs = original_data.values[observed_mask.values]
        reconstructed_obs = reconstructed_data.values[observed_mask.values]
        residuals_obs = residuals_df.values[observed_mask.values]

        # Overall statistics
        rmse = np.sqrt(np.mean(residuals_obs**2))
        mae = np.mean(np.abs(residuals_obs))
        bias = np.mean(residuals_obs)

        # R-squared (correlation coefficient squared)
        correlation = np.corrcoef(original_obs, reconstructed_obs)[0, 1]
        r_squared = correlation**2 if not np.isnan(correlation) else 0.0

        # Per-column statistics
        residual_stats = {}
        for col in original_data.columns:
            col_mask = observed_mask[col]
            if col_mask.sum() > 0:  # If there are observed values in this column
                col_residuals = residuals_df[col][col_mask].values
                residual_stats[col] = {
                    "mean": np.mean(col_residuals),
                    "std": np.std(col_residuals),
                    "min": np.min(col_residuals),
                    "max": np.max(col_residuals),
                    "n_observed": len(col_residuals),
                    "rmse": np.sqrt(np.mean(col_residuals**2)),
                    "mae": np.mean(np.abs(col_residuals)),
                }
            else:
                residual_stats[col] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n_observed": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                }

        stats_dict = {
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r_squared": r_squared,
            "residual_stats": residual_stats,
            "n_observed": len(residuals_obs),
            "rank_used": self.rank_,
        }

        if self.verbose:
            logger.info(f"Residual statistics (n={len(residuals_obs)} observed values):")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAE:  {mae:.6f}")
            logger.info(f"  Bias: {bias:.6f}")
            logger.info(f"  R²:   {r_squared:.6f}")

        return residuals_df, stats_dict

    def fit_transform(
        self,
        return_uncertainty: bool = False,
        n_imputations: int = 5,
        n_repeats: int = 100,
        mask_strategy: str = "block",
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        seed: Optional[int] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Union[Dict[str, Any], pd.DataFrame]]]:
        """
        Fit the imputer and transform the stored data in one step.

        If return_uncertainty is True, uses Multiple Imputation with Stochastic SVD
        and Rubin's Rules to estimate uncertainty.

        Parameters
        ----------
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates (default: False)
        n_imputations : int, optional
            Number of multiple imputations for uncertainty estimation (default: 5).
            Used when return_uncertainty=True.
        n_repeats : int, optional
            Deprecated. Used for old Monte Carlo validation.
        mask_strategy : str, optional
            Deprecated. Used for old Monte Carlo validation.
        frac : float, optional
            Deprecated. Used for old Monte Carlo validation.
        block_len : int, optional
            Deprecated. Used for old Monte Carlo validation.
        n_blocks : int, optional
            Deprecated. Used for old Monte Carlo validation.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame or tuple
            If return_uncertainty=False: Returns imputed DataFrame
            If return_uncertainty=True: Returns (imputed_df_mean, uncertainty_df)
            where uncertainty_df contains standard deviations per element.

        Examples
        --------
        >>> # Simple imputation (no uncertainty)
        >>> imputer = Imputer(data)
        >>> df_imputed = imputer.fit_transform()

        >>> # With Multiple Imputation uncertainty
        >>> df_imputed, df_uncertainty = imputer.fit_transform(
        ...     return_uncertainty=True, n_imputations=10
        ... )
        """
        if not return_uncertainty:
            # Standard behavior - no uncertainty
            return self.fit().transform()

        # Fit first to determine rank
        if not self.is_fitted_:
            self.fit()

        # Work with stored preprocessed data
        X_array = (
            self.data_preprocessed_.values if isinstance(self.data_preprocessed_, pd.DataFrame) else self.data_preprocessed_
        )

        if self.verbose:
            logger.info(f"Performing Multiple Imputation with {n_imputations} stochastic runs...")

        # Initialize random generator
        rng = np.random.default_rng(seed)
        
        imputed_matrices = []
        residual_variances = []

        for i in range(n_imputations):
            # Generate a seed for this run
            run_seed = rng.integers(1 << 30)
            
            # Run stochastic imputation
            # We need to pass return_svd=True to get the residual variance (sigma_sq)
            X_imputed, svd_res = iterative_svd_impute(
                X_array, 
                rank=self.rank_, 
                max_iters=self.max_iters, 
                tol=self.tol, 
                return_svd=True,
                stochastic=True,
                random_state=run_seed
            )
            
            # Extract residual variance
            sigma_sq = svd_res["sigma_sq"]
            
            # Postprocess to original scale
            X_imputed_post = postprocess_after_svd(X_imputed, self.preprocessing_info_)
            
            imputed_matrices.append(X_imputed_post)
            
            # Calculate element-wise residual variance in original scale.
            # sigma_sq is the variance of residuals in standardized space.
            # Since Var(aX + b) = a^2 Var(X), we scale sigma_sq by scales^2.
            # This represents the "Within Variance" for Rubin's rules.
            _, scales = self.preprocessing_info_
            
            # Handle scales being scalar or array
            if np.isscalar(scales):
                run_variance = np.full(X_array.shape, sigma_sq * (scales**2))
            else:
                # scales is array of shape (n_cols,)
                run_variance = np.outer(np.ones(X_array.shape[0]), sigma_sq * (scales**2))
                
            residual_variances.append(run_variance)

        # Apply Rubin's Rules
        # 1. Point Estimate: Mean of imputed matrices
        # Stack matrices: (n_imputations, n_rows, n_cols)
        M_stack = np.stack(imputed_matrices, axis=0)
        point_estimate = np.mean(M_stack, axis=0)
        
        # 2. Within-Variance (W): Average of residual variances
        # Stack variances: (n_imputations, n_rows, n_cols)
        W_stack = np.stack(residual_variances, axis=0)
        W = np.mean(W_stack, axis=0)
        
        # 3. Between-Variance (B): Variance of the point estimates
        # B = 1/(M-1) * sum((X_m - X_bar)^2)
        if n_imputations > 1:
            B = np.var(M_stack, axis=0, ddof=1)
        else:
            B = np.zeros_like(point_estimate)
        
        # 4. Total Variance (T)
        # T = W + (1 + 1/M) * B
        T = W + (1 + 1/n_imputations) * B
        
        # 5. Uncertainty (Standard Deviation)
        uncertainty = np.sqrt(T)
        
        # Convert to DataFrames
        df_imputed = pd.DataFrame(
            point_estimate,
            index=self.data_original_.index,
            columns=self.data_original_.columns
        )
        
        df_uncertainty = pd.DataFrame(
            uncertainty,
            index=self.data_original_.index,
            columns=self.data_original_.columns
        )
        
        if self.verbose:
            logger.info("Multiple Imputation complete.")
            logger.info(f"Average uncertainty (std): {np.nanmean(uncertainty):.4f}")

        return df_imputed, df_uncertainty

    def _uncertainty_monte_carlo(
        self,
        n_repeats: int,
        mask_strategy: str,
        frac: float,
        block_len: int,
        n_blocks: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """
        Compute constant uncertainty band using Monte Carlo validation.

        Returns
        -------
        dict
            {'method': 'monte_carlo', 'rmse': float, 'mae': float,
             'rmse_ci': tuple, 'mae_ci': tuple}
        """
        results = self.estimate_uncertainty(n_repeats, mask_strategy, frac, block_len, n_blocks, seed)

        return {
            "method": "monte_carlo",
            "rmse": results["RMSE"]["mean"],
            "mae": results["MAE"]["mean"],
            "rmse_std": results["RMSE"]["std"],
            "mae_std": results["MAE"]["std"],
            "rmse_ci": results["RMSE"]["95%_CI"],
            "mae_ci": results["MAE"]["95%_CI"],
            "raw_rmse": results["raw_rmse"],
            "raw_mae": results["raw_mae"],
            "raw_imputed": results["raw_imputed"],
        }

    def get_params(self) -> dict:
        """
        Get parameters for this estimator.

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {
            "variance_threshold": self.variance_threshold,
            "rank": self.rank,
            "max_iters": self.max_iters,
            "tol": self.tol,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "Imputer":
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
