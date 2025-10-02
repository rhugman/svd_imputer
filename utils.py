import numpy as np
import numpy as np
from math import sqrt
from statistics import mean, stdev


def iterative_svd_impute(X, rank=2, max_iters=500, tol=1e-4, mask=None, scaler=None):
    """
    X : numpy array with np.nan for missing values
    rank : number of singular values/vectors to keep
    mask : boolean array (True = observed, False = artificially masked for validation)
    """
    X_filled = X.copy()

    if scaler is not None:
        X_filled = scaler.fit_transform(X_filled)
    
    # Step 1: initialize missing with column means
    col_means = np.nanmean(X_filled, axis=0)
    # ramdom uniform sample between min and max of each column
    #col_mins = np.nanmin(X_filled, axis=0)
    #col_maxs = np.nanmax(X_filled, axis=0)
    #rng = np.random.default_rng(seed=42)
    #col_means = rng.normal(col_mins, col_maxs)
    inds = np.where(np.isnan(X_filled))
    X_filled[inds] = np.take(col_means, inds[1])
    
    # Iterative updates
    for it in range(max_iters):
        # SVD
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        
        # Low-rank approximation
        S = np.diag(s[:rank])
        X_approx = U[:, :rank] @ S @ Vt[:rank, :]
        
        # Update only the originally missing entries
        X_new = X_filled.copy()
        X_new[inds] = X_approx[inds]
        
        # Check convergence
        diff = np.linalg.norm(X_new - X_filled) / np.linalg.norm(X_filled)
        X_filled = X_new
        if diff < tol:
            break
    if scaler is not None:
        X_filled = scaler.inverse_transform(X_filled)

    # If mask is provided, compute validation error
    if mask is not None:
        error = np.mean((X_filled[~mask] - X[~mask])**2) ** 0.5
        return X_filled, error
    
    return X_filled




def rmse(true, pred):
    return sqrt(np.mean((true - pred) ** 2))

def mae(true, pred):
    return np.mean(np.abs(true - pred))

def random_mask_observed(X, frac=0.1, seed=None):
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)            # True = observed
    obs_indices = np.argwhere(mask)
    n_hide = int(len(obs_indices) * frac)
    hide_idx = rng.choice(len(obs_indices), size=n_hide, replace=False)
    for idx in hide_idx:
        r, c = obs_indices[idx]
        mask[r, c] = False
    return mask

def block_mask_time(X, block_len=5, n_blocks=1, seed=None):
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)
    n_rows, _ = X.shape
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n_rows - block_len + 1))
        rows = range(start, start + block_len)
        # hide those rows across all columns that are observed
        for r in rows:
            for c in range(X.shape[1]):
                if not np.isnan(X[r, c]):
                    mask[r, c] = False
    return mask

def monte_carlo_validation(X, impute_func, rank=2,scaler=None, n_repeats=100,
                           mask_strategy='random', frac=0.1,
                           block_len=5, n_blocks=1, seed=None):
    rng = np.random.default_rng(seed)
    rmse_list = []
    mae_list = []
    n_rows, n_cols = X.shape

    for i in range(n_repeats):
        s = None if seed is None else int(rng.integers(1<<30))
        if mask_strategy == 'random':
            mask = random_mask_observed(X, frac=frac, seed=s)
        elif mask_strategy == 'block':
            mask = block_mask_time(X, block_len=block_len, n_blocks=n_blocks, seed=s)
        else:
            raise ValueError("unsupported mask_strategy")

        # Build X_with_nans for imputation: keep original np.nan, plus set masked obs -> np.nan
        X_with_nans = X.copy()
        obs_all = ~np.isnan(X)
        # set entries that are masked for validation to np.nan
        masked_positions = np.logical_and(obs_all, ~mask)
        X_with_nans[masked_positions] = np.nan

        # impute; adapt if impute_func supports mask argument
        result = impute_func(X_with_nans, rank=rank, mask=mask, scaler=scaler)  # try this form
        if isinstance(result, tuple):
            X_imputed, _ = result
        else:
            X_imputed = result

        # compute error only on masked positions (the validation set)
        true_vals = X[masked_positions]
        pred_vals = X_imputed[masked_positions]

        rmse_list.append(rmse(true_vals, pred_vals))
        mae_list.append(mae(true_vals, pred_vals))

    # summarize: mean and 95% CI via normal approximation (mean +/- 1.96*SE)
    def summarize(vals):
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        se = s / np.sqrt(len(vals))
        lower = m - 1.96 * se
        upper = m + 1.96 * se
        return {"mean": m, "std": s, "95%_CI": (lower, upper)}

    return {"RMSE": summarize(rmse_list), "MAE": summarize(mae_list),
            "raw_rmse": rmse_list, "raw_mae": mae_list}
