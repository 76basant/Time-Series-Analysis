
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_cross_correlation(x, y, max_lag=None):
    """
    Compute cross-correlation for given lags.
    Args:
        x, y: Input arrays.
        max_lag: Maximum lag to compute. If None, max_lag = len(x)-1.
    Returns:
        Dictionary with lags and corresponding cross-correlation values.
    """
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

def compute_autocorrelation(x, max_lag=None):
    """
    Compute autocorrelation for given lags.
    Args:
        x: Input array.
        max_lag: Maximum lag to compute. If None, max_lag = len(x)-1.
    Returns:
        Dictionary with lags and corresponding autocorrelation values.
    """
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

# Example Data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# Pearson Correlation
pearson_corr, _ = pearsonr(X, Y)
print("Pearson Correlation (lag=0):", pearson_corr)

# Cross-Correlation
cross_corr = compute_cross_correlation(X, Y)
print("Cross-Correlation:", cross_corr)

# Autocorrelation
autocorr = compute_autocorrelation(X)
print("Autocorrelation:", autocorr)

# Visualization
def plot_correlations(X, Y, cross_corr, autocorr):
    lags_cross = list(cross_corr.keys())
    values_cross = list(cross_corr.values())
    lags_auto = list(autocorr.keys())
    values_auto = list(autocorr.values())

    plt.figure(figsize=(14, 6))

    # Plot X and Y
    plt.subplot(1, 3, 1)
    plt.plot(X, label='X', marker='o')
    plt.plot(Y, label='Y', marker='o')
    plt.title("Time Series X and Y")
    plt.legend()

    # Plot Cross-Correlation
    plt.subplot(1, 3, 2)
    plt.stem(lags_cross, values_cross)  # Removed 'use_line_collection' for compatibility
    plt.title("Cross-Correlation")
    plt.xlabel("Lag")
    plt.ylabel("Cross-Correlation Value")

    # Plot Autocorrelation
    plt.subplot(1, 3, 3)
    plt.stem(lags_auto, values_auto)  # Removed 'use_line_collection' for compatibility
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation Value")

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_correlations(X, Y, cross_corr, autocorr)

# User Interaction: Enter Lag
lag = int(input("Enter the lag value: "))
if lag in cross_corr:
    print(f"Cross-Correlation at lag {lag}: {cross_corr[lag]:.4f}")
else:
    print(f"Lag {lag} is out of range for Cross-Correlation.")

if lag > 0 and lag in autocorr:
    print(f"Autocorrelation at lag {lag}: {autocorr[lag]:.4f}")
elif lag == 0:
    print(f"Autocorrelation at lag 0 is always 1.0.")
else:
    print(f"Lag {lag} is out of range for Autocorrelation.")
