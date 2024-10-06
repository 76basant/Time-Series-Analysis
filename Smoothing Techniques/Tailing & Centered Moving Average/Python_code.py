
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample time series data
np.random.seed(0)
data = np.random.randn(100).cumsum()  # Generate some random time series data
dates = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

# Create a DataFrame to hold the time series
df = pd.DataFrame(data, index=dates, columns=['Value'])

# Define a window size
window = 5

# Trailing Moving Average (SMA)
df['Trailing_MA'] = df['Value'].rolling(window=window).mean()

# Centered Moving Average
df['Centered_MA'] = df['Value'].rolling(window=window, center=True).mean()

# Plot the time series with both moving averages
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Value'], label='Original Time Series', color='blue', linewidth=1)
plt.plot(df.index, df['Trailing_MA'], label=f'Trailing Moving Average (window={window})', color='orange', linewidth=2)
plt.plot(df.index, df['Centered_MA'], label=f'Centered Moving Average (window={window})', color='green', linewidth=2)

# Add labels and legend
plt.title('Trailing vs Centered Moving Average')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()



##By using For loop

import pandas as pd
import numpy as np
import math 
# List of numbers from 10 to 100 with a step of 10
data = list(range(10, 101, 10))

# Convert the list to a pandas Series
data_series = pd.Series(data)

# Define the window size (must be odd for centered moving average)
window = 3

# Compute the centered moving average
centered_MA = data_series.rolling(window=window, center=True).mean()

# Display the original data and the centered moving average
print("Original Data:", data)
print("Centered Moving Average:", centered_MA.tolist())


import pandas as pd

# List of numbers from 10 to 100 with a step of 10
data = list(range(10, 101, 10))

# Convert the list to a pandas Series
data_series = pd.Series(data)

# Define the window size
window = 3

# Compute the trailing moving average
trailing_MA = data_series.rolling(window=window, center=False).mean()

# Display the original data and the trailing moving average
print("Original Data:", data)
print("Trailing Moving Average:", trailing_MA.tolist())


# List of numbers from 10 to 100 with a step of 10
data = list(range(10, 101, 10))

# Define the window size
window = int(input("please enter only odd numbers: "))
#window=3
# Initialize an empty list to store the centered moving average values
centered_MA = []

# Loop through the data to calculate the centered moving average
print(len(data))

print (list(range(0, len(data)-(window)+1)))
print(data[1: window+1])
if (window%2!=0):
      print("valid odd number")
      for i in range(0, len(data)-(window)+1):
        avg = sum(data[i : window+i])/ window
        centered_MA.append(avg)
       
else:
  print("you entered even value of window:  ",window)

# Print the result
print("Centered Moving Average:", centered_MA)

# Create the new list with NaN values at the beginning and end
num_nan=int((window-1)/2)

new_list = [math.nan] * num_nan + centered_MA + [math.nan] * num_nan

# Print the updated list
print(new_list)


