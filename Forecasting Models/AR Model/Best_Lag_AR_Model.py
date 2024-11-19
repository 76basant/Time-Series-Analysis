
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

# Define a threshold for significance (values near zero are insignificant)
threshold = 0.1  # You can adjust this threshold based on your data

# Find the lag where PACF first falls below the threshold
best_pacf_lag = None
for lag, value in pacf_lags_values:
    if abs(value) < threshold:
        best_pacf_lag = lag
        break

if best_pacf_lag is None:
    print("No significant cut-off found in PACF.")
else:
    print(f"The best lag based on PACF threshold of {threshold} is: {best_pacf_lag}")

# Plot PACF to visually inspect the partial correlations
plt.figure(figsize=(10, 6))
plt.bar(range(range_Me), pacf_values, color='b')
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.xlabel('Lag')
plt.ylabel('Partial Correlation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.legend()
plt.grid(True)
plt.show()

# Iterate over possible lags and compute AIC/BIC
best_aic = float('inf')
best_p = None

for lag in range(1, range_Me):  # Test lags from 1 to range_Me-1
    model = AutoReg(data, lags=lag, old_names=False)
    model_fit = model.fit()
    aic = model_fit.aic
    if aic < best_aic:
        best_aic = aic
        best_p = lag

print(f"Best lag (p) based on AIC: {best_p}")

# Split data into training and testing sets
train, test = data[:80], data[80:]

# Test different lags for cross-validation
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
