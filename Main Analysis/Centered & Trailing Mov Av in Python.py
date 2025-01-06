# Objective: 
# This script calculates both the centered and trailing moving averages of a time series dataset.

# Required Libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# --- Part 1: Basic Moving Average Calculations ---

# Generate a sequence of numbers from 10 to 100 with a step of 10.
data = list(range(10, 101, 10))
data_series = pd.Series(data)

# Define the window size (must be odd for centered moving average)
window = 3

# Compute the centered moving average
centered_MA = data_series.rolling(window=window, center=True).mean()

# Display results
print("Original Data:", data)
print("Centered Moving Average:", centered_MA.tolist())

# Compute the trailing moving average
trailing_MA = data_series.rolling(window=window, center=False).mean()

# Display results
print("Trailing Moving Average:", trailing_MA.tolist())

# --- Part 2: Moving Averages on Time Series Data ---

# Generate a synthetic time series dataset
np.random.seed(0)  # Ensure reproducibility
data = np.random.randn(100).cumsum()
dates = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

df = pd.DataFrame(data, index=dates, columns=['Value'])

# Define a window size
window = 5

# Calculate moving averages using the rolling() method
df['Trailing_MA'] = df['Value'].rolling(window=window).mean()
df['Centered_MA'] = df['Value'].rolling(window=window, center=True).mean()

# Visualize the original time series and moving averages
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Value'], label='Original Time Series', color='blue', linewidth=1)
plt.plot(df.index, df['Trailing_MA'], label=f'Trailing Moving Average (window={window})', color='orange', linewidth=2)
plt.plot(df.index, df['Centered_MA'], label=f'Centered Moving Average (window={window})', color='green', linewidth=2)

# Add labels, title, legend, and grid
plt.title('Trailing vs Centered Moving Average')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# --- Part 3: Manual Centered Moving Average Calculation ---

# Prompt user for window size
window = int(input("Please enter an odd window size: "))

# Validate window size
if window % 2 == 0:
    raise ValueError(f"Window size must be odd. Received: {window}")

# Initialize an empty list to store centered moving average values
centered_MA = []

# Compute the centered moving average manually
for i in range(0, len(data) - (window - 1)):
    avg = sum(data[i: i + window]) / window
    centered_MA.append(avg)

# Pad with NaN values to match original data length
num_nan = (window - 1) // 2
centered_MA_padded = [math.nan] * num_nan + centered_MA + [math.nan] * num_nan

# Display the results
print("Manually Calculated Centered Moving Average:", centered_MA_padded)
