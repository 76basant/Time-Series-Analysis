
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

# Generate example time series data (you can replace this with your dataset)
np.random.seed(42)
data = np.cumsum(np.random.normal(size=100))  # Example series

# Check stationarity using Augmented Dickey-Fuller (ADF) test
result = adfuller(data)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Series is not stationary. Differencing may be required.")

# Fit AR model (e.g., AR(2))
model = AutoReg(data, lags=2, old_names=False)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Predict future values
predictions = model_fit.predict(start=len(data), end=len(data)+10)
print("Predictions:", predictions)
