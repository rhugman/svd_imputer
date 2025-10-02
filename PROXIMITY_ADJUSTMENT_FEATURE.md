# Proximity-Based Uncertainty Adjustment Feature

## Overview

Added a **data-driven proximity-based uncertainty adjustment** feature that automatically learns how prediction error varies with distance to nearest observations, then applies this relationship to scale uncertainty estimates.

## Motivation

Standard uncertainty methods (Monte Carlo, Bootstrap, Hybrid) provide either:
- **Constant uncertainty** across all imputed values (Monte Carlo)
- **Point-wise uncertainty** that varies but may not explicitly account for temporal proximity (Bootstrap)

However, intuitively:
- **Imputed values near observations** should have lower uncertainty
- **Imputed values far from observations** should have higher uncertainty

This feature makes this relationship explicit and data-driven.

## Implementation

### New Methods in `Imputer` class

#### 1. `_learn_distance_uncertainty_relationship(df, n_samples=50)`

**Purpose**: Learn empirical relationship between distance to nearest observation and prediction error

**How it works**:
1. Randomly masks one observed value per column (n_samples times)
2. Imputes the masked value
3. Records:
   - Actual prediction error
   - Distance to nearest remaining observation (in days)
4. Fits a model: `error = f(distance)`
5. Returns a function that maps distance → uncertainty multiplier

**Model fitting hierarchy** (tries in order):
1. **Exponential**: `error ~ a * exp(b * distance)` - preferred for growth patterns
2. **Linear**: `error ~ slope * distance + intercept` - fallback for simple relationships
3. **Step function**: Based on median distance ratio - final fallback
4. **Identity**: Returns 1.0 if insufficient data

#### 2. `_apply_proximity_adjustment(df, uncertainty_df)`

**Purpose**: Apply learned relationship to scale uncertainty estimates

**How it works**:
1. For each missing value:
   - Calculate distance to nearest observation (in days)
   - Apply learned multiplier: `adjusted_uncertainty = base_uncertainty * f(distance)`
2. Stores learned function as `distance_to_uncertainty_fn_` for inspection

### Updated `fit_transform` Method

Added new parameter:

```python
def fit_transform(
    self,
    X: pd.DataFrame,
    return_uncertainty: bool = False,
    uncertainty_method: str = 'monte_carlo',
    adjust_by_proximity: bool = False,  # <-- NEW
    ...
)
```

**Usage**:
```python
imputer = Imputer(scaler=StandardScaler())

df_imputed, unc = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='hybrid',
    adjust_by_proximity=True,  # Activate feature
    n_repeats=100,
    n_bootstrap=50
)

# Inspect learned relationship
print(imputer.distance_to_uncertainty_fn_(30))  # Multiplier at 30 days
```

## Key Features

### 1. **Fully Automatic**
- No user-defined parameters required
- Learns from data characteristics
- Works with any dataset

### 2. **Data-Driven**
- Learns actual error patterns from validation
- Adapts to specific dataset characteristics
- Robust fallbacks for edge cases

### 3. **Inspectable**
- Stores learned function as `distance_to_uncertainty_fn_`
- Can visualize distance-uncertainty relationship
- Provides diagnostic output in verbose mode

### 4. **Compatible**
- Works with all three uncertainty methods (Monte Carlo, Bootstrap, Hybrid)
- Adjusts both std and confidence intervals
- Optional - only activates when `adjust_by_proximity=True`

## Example Usage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from svd_imputer import Imputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Impute with proximity-adjusted uncertainty
imputer = Imputer(scaler=StandardScaler(), verbose=True)

df_imputed, unc = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='hybrid',
    adjust_by_proximity=True,
    n_repeats=100,
    n_bootstrap=50,
    confidence=0.95
)

# Get confidence intervals
df_lower, df_upper = imputer.get_confidence_intervals(df_imputed, unc)

# Visualize learned relationship
d_range = np.linspace(0, 365, 100)
multipliers = [imputer.distance_to_uncertainty_fn_(d) for d in d_range]

plt.plot(d_range, multipliers, linewidth=2)
plt.xlabel('Distance to nearest observation (days)')
plt.ylabel('Uncertainty multiplier')
plt.title('Learned Distance-Uncertainty Relationship')
plt.grid(True, alpha=0.3)
plt.show()

# Compare original vs adjusted uncertainty
print(f"Example multipliers:")
for d in [0, 7, 30, 90, 365]:
    mult = imputer.distance_to_uncertainty_fn_(d)
    print(f"  {d:3d} days: {mult:.3f}x")
```

## Benefits

1. **More realistic uncertainty estimates**
   - Accounts for information decay with distance
   - Reflects actual data structure

2. **Better decision support**
   - Higher uncertainty in data-sparse regions
   - Lower uncertainty near observations

3. **No parameter tuning**
   - Learns automatically from data
   - No manual calibration required

4. **Interpretable**
   - Can visualize and understand learned relationship
   - Clear physical meaning (distance → uncertainty)

## Testing

See `test.ipynb` for comprehensive examples including:
- Learning and visualizing distance-uncertainty relationship
- Comparing original vs proximity-adjusted uncertainty
- Real-world groundwater monitoring data application

## Performance

- **Learning phase**: ~50 validation samples (fast, < 30 seconds typically)
- **Application**: O(n_missing × n_observed) per column (fast, vectorizable)
- **Memory**: Minimal overhead (stores single function)

## Future Enhancements

Potential extensions:
1. **Spatial weighting**: Account for correlation with other series
2. **Adaptive sampling**: Focus validation on critical distance ranges
3. **Ensemble models**: Combine multiple distance-error models
4. **Directional effects**: Different uncertainty for forward vs backward gaps
