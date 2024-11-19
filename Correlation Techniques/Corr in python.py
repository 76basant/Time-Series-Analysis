import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.api import OLS, add_constant

# Example Data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# Function to calculate Pearson Correlation
def calculate_pearson_correlation(x, y):
    pearson_corr, _ = pearsonr(x, y)
    return pearson_corr

# Function to calculate Cross-Correlation
def compute_cross_correlation(x, y, max_lag=None):
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    mean_x, mean_y = np.mean(x), np.mean(y)
    denom = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    cross_corr = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            num = np.sum((x[:lag] - mean_x) * (y[-lag:] - mean_y))
        else:
            num = np.sum((x[lag:] - mean_x) * (y[:n - lag] - mean_y))
        cross_corr[lag] = num / denom
    return cross_corr

# Function to calculate Autocorrelation
def compute_autocorrelation(x, max_lag=None):
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    mean_x = np.mean(x)
    denom = np.sum((x - mean_x) ** 2)
    autocorr = {}
    for lag in range(1, max_lag + 1):
        num = np.sum((x[:n - lag] - mean_x) * (x[lag:] - mean_x))
        autocorr[lag] = num / denom
    return autocorr

# Function to calculate Partial Correlation
def compute_partial_correlation(x, lag):
    n = len(x)
    if lag <= 0 or lag >= n:
        raise ValueError("Lag must be between 1 and n-1")
    
    # Create lagged variables
    lagged_data = np.column_stack([x[i:n - lag + i] for i in range(lag)])
    residuals_target = OLS(x[lag:], add_constant(lagged_data[:, :-1])).fit().resid
    residuals_lagged = OLS(x[:n - lag], add_constant(lagged_data[:, :-1])).fit().resid
    
    # Compute Pearson correlation of residuals
    partial_corr, _ = pearsonr(residuals_target, residuals_lagged)
    return partial_corr

# Visualization Function
def plot_correlations(x, y, cross_corr, autocorr):
    plt.figure(figsize=(14, 6))

    # Cross-correlation
    plt.subplot(1, 2, 1)
    plt.stem(list(cross_corr.keys()), list(cross_corr.values()), use_line_collection=True)
    plt.title("Cross-Correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")

    # Autocorrelation
    plt.subplot(1, 2, 2)
    plt.stem(list(autocorr.keys()), list(autocorr.values()), use_line_collection=True)
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")

    plt.tight_layout()
    plt.show()

# Calculate Correlations
pearson_corr = calculate_pearson_correlation(X, Y)
cross_corr = compute_cross_correlation(X, Y)
autocorr = compute_autocorrelation(X)
partial_corr = compute_partial_correlation(X, lag=1)

# Display Results
print(f"Pearson Correlation (lag=0): {pearson_corr}")
print("Cross-Correlation:", cross_corr)
print("Autocorrelation:", autocorr)
print(f"Partial Correlation (lag=1): {partial_corr}")

# Visualize Results
plot_correlations(X, Y, cross_corr, autocorr)
