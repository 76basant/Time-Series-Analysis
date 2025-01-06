% Objective: This script calculates and visualizes trailing and centered moving averages of a time series dataset.

% Generate a sample time series dataset
rng(0); % Ensure reproducibility of random data generation
data = cumsum(randn(100, 1)); % Create cumulative sum of random numbers as time series data

% Define the window size for moving averages
window = 5;

% Compute the trailing moving average using the 'movmean' function
trailing_MA = movmean(data, window); % Default endpoint handling for trailing moving average

% Compute the centered moving average manually (compatible with older MATLAB versions)
half_window = floor(window / 2);
centered_MA = nan(size(data)); % Preallocate with NaN for centered moving average

for i = (1 + half_window):(length(data) - half_window)
    centered_MA(i) = mean(data((i - half_window):(i + half_window)));
end

% Plot the original time series and moving averages
figure;
plot(data, 'b-', 'LineWidth', 1); % Plot the original time series
hold on;
plot(trailing_MA, 'r-', 'LineWidth', 2); % Plot the trailing moving average
plot(centered_MA, 'g-', 'LineWidth', 2); % Plot the centered moving average

% Annotate the plot with titles, labels, and legend
title('Trailing vs Centered Moving Average');
xlabel('Time Index');
ylabel('Value');
legend('Original Time Series', sprintf('Trailing Moving Average (window = %d)', window), ...
       sprintf('Centered Moving Average (window = %d)', window), 'Location', 'best');
grid on;
hold off;
