%% Step 1: Data Preparation
% Example time series data (replace with actual yearly averages)
x = yearly_mean1;  % First time series (SSA_N)
y = yearly_mean2;  % Second time series (SSA_S)

% Ensure both time series have the same length
if length(x) ~= length(y)
    error('The two time series must have the same length.');
end

%% Step 2: Define Parameters for Running Correlation Calculation
% Calculate the length of the time series and define the window size
dataLength = length(x);
windowSize = round(dataLength / 3);  % Set window size as 1/3 of data length
halfWindowSize = floor(windowSize / 2);  % Half window size for centering the window

% Preallocate the running correlation vector to store NaN values initially
runningCorrelation = nan(dataLength, 1);

%% Step 3: Calculate Running Correlation with Centered Window
% Loop through the data points to compute the running correlation
for i = (halfWindowSize + 1):(dataLength - halfWindowSize)
    % Extract the current window of data for both time series
    xWindow = x(i - halfWindowSize : i + halfWindowSize);
    yWindow = y(i - halfWindowSize : i + halfWindowSize);
    
    % Compute the correlation coefficient for the current window
    runningCorrelation(i) = corr(xWindow, yWindow);
end

%% Step 4: Define the Time Axis for Plotting
% Define the time range from 1986 to 2021
startYear = 1986;
endYear = 2021;
timeAxis = (startYear:endYear)';  % Create the time axis corresponding to the years

% Adjust the time range for the window size
minTime = startYear + halfWindowSize;
maxTime = endYear - halfWindowSize;

%% Step 5: Interpolate for Smoothness
% Create a finer time axis for smooth interpolation
fineTimeAxis = linspace(minTime, maxTime, dataLength * 10);  % More data points for smoothness

% Interpolate the running correlation to get a smoother curve
smoothRunningCorrelation = interp1(timeAxis, runningCorrelation, fineTimeAxis, 'spline');

%% Step 6: Visualization of Results
% Plot the running correlation along with the smoothed curve
figure;
plot(timeAxis, runningCorrelation, '.k', 'MarkerSize', 15);  % Original running correlation as dots
hold on;
plot(fineTimeAxis, smoothRunningCorrelation, '-k', 'LineWidth', 2);  % Smooth curve
xlabel('Year');
ylabel('Running Correlation');
title('Centered Running Correlation between SSA_N and SSA_S (1986-2021)');
grid on;
