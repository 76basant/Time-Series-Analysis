
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Function to load and preprocess data
def load_data(file_path, date_columns, value_column, start_date):
    df = pd.read_excel(file_path, engine='openpyxl')
    try:
        # Ensure the date columns are integers
        df[date_columns] = df[date_columns].astype(int)
    except ValueError:
        raise ValueError("Ensure the date columns contain only integers.")
    
    # Create a datetime column
    df['Date'] = pd.to_datetime(df[date_columns].assign(DAY=1))
    start_date = pd.to_datetime(start_date)
    
    # Convert time to months since start_date
    df['Time'] = (df['Date'] - start_date) / pd.Timedelta(days=30.44)
    df['Signal'] = df[value_column]  # Rename signal column for consistency
    
    # Remove rows with missing or invalid signal values
    df = df.dropna(subset=['Signal'])
    
    return df[['Time', 'Signal']]

# Function to perform Lomb-Scargle analysis with dynamic frequency range
def lomb_scargle_analysis(time, signal, num_freqs=500, confidence_levels=(95, 99)):
    # Calculate the dynamic frequency range based on the length of the signal
    i = 1 / len(signal)  # Maximum of observed periodicities (min frequency)
    j = 1 / 2           # Minimum of observed periodicities (max frequency)

    # Generate the frequency array dynamically
    frequency = np.linspace(i, j, num_freqs)

    # Perform Lomb-Scargle analysis
    power = LombScargle(time, signal).power(frequency)
    periods = 1 / frequency

    # Find peaks in the power spectrum
    peaks, _ = find_peaks(power, height=np.percentile(power, confidence_levels[0]))
    peak_frequencies = frequency[peaks]
    peak_periods = 1 / peak_frequencies

    # Calculate confidence thresholds
    conf_thresholds = {level: np.percentile(power, level) for level in confidence_levels}

    return {
        'frequency': frequency,
        'power': power,
        'periods': periods,
        'peaks': peaks,
        'peak_frequencies': peak_frequencies,
        'peak_periods': peak_periods,
        'confidence_thresholds': conf_thresholds,
    }

# Function to plot Lomb-Scargle periodogram with only peaks
def plot_lomb_scargle(results, title="Lomb-Scargle Periodogram", xlabel="Frequency (1/month)", ylabel="Power"):
    plt.figure(figsize=(10, 6))

    # Interpolation for smoother plot
    cubic_interpolation_model = interp1d(results['frequency'], results['power'], kind="cubic")
    interpolated_freq = np.linspace(results['frequency'][0], results['frequency'][-1], 500)
    interpolated_power = cubic_interpolation_model(interpolated_freq)

    
    # Plot interpolated power spectrum
    plt.plot(interpolated_freq, interpolated_power, color='blue', label='Smoothed Power')

    # Confidence thresholds
    for level, threshold in results['confidence_thresholds'].items():
        plt.axhline(y=threshold, linestyle='--', label=f'{level}% Confidence')

    # Highlight peaks with vertical lines
    for freq, period in zip(results['peak_frequencies'], results['peak_periods']):
        plt.axvline(x=freq, color='green', linestyle=':', linewidth=1)
        plt.text(freq, max(results['power']) * 0.9, f'{period / 12.0:.2f} yr', color='red',
                 horizontalalignment='right', verticalalignment='center', fontsize=10, rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.ylim([0, max(results['power']) + 0.02])
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = r'file.xlsx'
    date_columns = ['Year', 'Month']
    value_column = 'parameter'  # Change this to the desired column
    start_date = '1986-01-01'

    # Load and preprocess data
    df = load_data(file_path, date_columns, value_column, start_date)

    # Perform Lomb-Scargle analysis with dynamic frequency range
    results = lomb_scargle_analysis(df['Time'].values, df['Signal'].values)

    # Plot results
    plot_lomb_scargle(results, title="Lomb-Scargle Periodogram of Signal")
