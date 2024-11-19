

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf
from sklearn.metrics import mean_squared_error

# Example time series data
np.random.seed(42)
data = np.cumsum(np.random.normal(size=100))  # Simulated random walk

# Print the length of the data (for debugging)
print(len(data))

# Define the range for lags
range_Me = 16

# Calculate PACF values for the given data
pacf_values = pacf(data, nlags=range_Me-1)

# Create a list of tuples (lag, PACF value)
pacf_lags_values = [(lag, pacf_values[lag]) for lag in range(range_Me)]

# Print the list of lags and their PACF values
print("Lags and corresponding PACF values:")
for lag, value in pacf_lags_values:
    print(f"Lag {lag}: PACF = {value:.4f}")

# Plot PACF to visually inspect the partial correlations
plt.figure(figsize=(10, 6))
plt.bar(range(range_Me), pacf_values, color='b')
plt.xlabel('Lag')
plt.ylabel('Partial Correlation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(True)
plt.show()

# Iterate over possible lags and compute AIC/BIC
best_aic = float('inf')
best_p = None

for lag in range(1, range_Me):  # Test lags from 1 to 10
    model = AutoReg(data, lags=lag, old_names=False)
    model_fit = model.fit()
    aic = model_fit.aic
    if aic < best_aic:
        best_aic = aic
        best_p = lag

print(f"Best lag (p) based on AIC: {best_p}")

# Split data into training and testing sets
train, test = data[:80], data[80:]

# Test different lags
errors = []
for lag in range(1, range_Me):
    model = AutoReg(train, lags=lag, old_names=False)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(data)-1)
    error = mean_squared_error(test, predictions)
    errors.append((lag, error))

# Find best lag based on minimum error
best_lag = min(errors, key=lambda x: x[1])
print(f"Best lag based on cross-validation: {best_lag[0]}")

# Plot cross-validation errors for each lag value
lags, mse_values = zip(*errors)
plt.figure(figsize=(10, 6))
plt.plot(lags, mse_values, marker='o', linestyle='-', color='b')
plt.xlabel('Lag')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Cross-validation MSE for Different Lag Values')
plt.grid(True)
plt.show()
