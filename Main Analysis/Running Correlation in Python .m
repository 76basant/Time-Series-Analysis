import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example time series data (replace with your actual data)
time_series_1 = np.random.rand(100)  # Example first time series
time_series_2 = np.random.rand(100)  # Example second time series

# Convert the time series to pandas Series
ts1 = pd.Series(time_series_1)
ts2 = pd.Series(time_series_2)

# Define the window size for running correlation (e.g., 10 data points)
window_size = 10

# Compute running correlation using pandas rolling method
running_corr = ts1.rolling(window=window_size).corr(ts2)

# Drop the NaN values resulting from the initial window (as they don't have enough data points)
running_corr = running_corr.dropna()

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(running_corr, label='Running Correlation', color='black', lw=2)
plt.title(f'Running Correlation (Window Size = {window_size})')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.show()
