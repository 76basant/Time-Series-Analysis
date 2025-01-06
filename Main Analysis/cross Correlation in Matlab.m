
% Function to read data from Excel and calculate cross-correlation
function calculate_and_plot_cross_correlation(filename, column1, column2)

    % Read the data from the Excel file
    [numData, txtData, rawData] = xlsread(filename);

    % Extract the specified columns from the raw data
    x = rawData(:, column1); % First selected column (time series)
    y = rawData(:, column2); % Second selected column (time series)

    % Convert cell arrays to numeric arrays
    x = cell2mat(x); % Convert to numeric array
    y = cell2mat(y); % Convert to numeric array

    % Ensure both time series are of the same length
    if length(x) ~= length(y)
        error('The two time series must have the same length.');
    end

    % Calculate the data length and the time lag (1/3 of the data length)
    dataLength = length(x);
    maxLag = round(dataLength / 3);  % Define the maximum lag as 1/3 of the data length

    % Compute the cross-correlation with the specified lag
    [crossCorrValues, lags] = xcorr(x, y, maxLag, 'coeff');  % 'coeff' normalizes the correlation

    % Create a finer lags axis for smooth interpolation
    fineLagsAxis = linspace(min(lags), max(lags), dataLength * 10); % 10 times the original points for smoothness
    smoothCrossCorrelation = interp1(lags, crossCorrValues, fineLagsAxis, 'spline');

    % Plot the cross-correlation
    figure;
    stem(fineLagsAxis, smoothCrossCorrelation, 'filled');
    title('Cross-Correlation between Time Series');
    xlabel('Lag');
    ylabel('Cross-Correlation');
    grid on;
end

% Example of usage
filename = 'Aa.xlsx';  % Replace with your actual file name
column1 = 1;  % First column (time series 1)
column2 = 2;  % Second column (time series 2)

% Call the function to calculate and plot the cross-correlation
calculate_and_plot_cross_correlation(filename, column1, column2);
