% Generate a sample time series data
rng(0); % For reproducibility
data = cumsum(randn(100,1)); % Generate random time series data

% Define the window size
window = 5;

% Trailing Moving Average (SMA)
trailing_MA = movmean(data, window); % Default 'Endpoints' handling for trailing

% Centered Moving Average
centered_MA = movmean(data, window, 'Endpoints', 'discard', 'SamplePoints', 1:length(data));

% Plot the original time series and moving averages
figure;
plot(data, 'b-', 'LineWidth', 1); % Original time series
hold on;
plot(trailing_MA, 'r-', 'LineWidth', 2); % Trailing moving average
plot(centered_MA, 'g-', 'LineWidth', 2); % Centered moving average

% Add labels and legend
title('Trailing vs Centered Moving Average');
xlabel('Time');
ylabel('Value');
legend('Original Time Series', ['Trailing Moving Average (window = ', num2str(window), ')'], ...
    ['Centered Moving Average (window = ', num2str(window), ')']);
grid on;
hold off;
