% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Define the mean for the Gaussian distribution.
gaussianMean = 128;
% Define the variance for the Gaussian distribution.
gaussianVariance = 100;
% Define the array of gray levels from 0 to 255.
grayLevels = 0:1:255;

% Calculate the first part of the probability density function formula.
pdfPartOne = 1.0 ./ sqrt(2.0 .* pi .* gaussianVariance);
% Calculate the second part of the probability density function formula.
pdfPartTwo = ((grayLevels - gaussianMean).^2) ./ (2.0 .* gaussianVariance);
% Calculate the unnormalized probability density function values.
unnormalizedPdf = pdfPartOne .* exp(-pdfPartTwo);
% Normalize the probability density function so that it sums to one.
normalizedPdf = unnormalizedPdf ./ sum(unnormalizedPdf);

% Initialize the cumulative distribution function array.
cumulativeDistributionFunction = zeros(1, 256);
% Set the first value of the cumulative distribution function.
cumulativeDistributionFunction(1) = normalizedPdf(1);
% Loop through the remaining gray levels to calculate the cumulative sum.
for grayLevelIndex = 2:1:256
    % Calculate the cumulative sum for the current gray level.
    cumulativeDistributionFunction(grayLevelIndex) = cumulativeDistributionFunction(grayLevelIndex - 1) + normalizedPdf(grayLevelIndex);
    % End the cumulative sum loop.
end

% Initialize the simulated image matrix with zeros.
simulatedImage = zeros(256, 256);
% Loop through each row of the simulated image.
for rowIndex = 1:1:256
    % Loop through each column of the simulated image.
    for colIndex = 1:1:256
        % Generate a random value from a uniform distribution between 0 and 1.
        randomValue = rand;
        % Find the indices where the cumulative distribution function is less than or equal to the random value.
        matchIndices = find(cumulativeDistributionFunction <= randomValue);
        % Assign the gray level based on the number of matching indices.
        simulatedImage(rowIndex, colIndex) = length(matchIndices);
        % End the column loop.
    end
    % End the row loop.
end

% Open a new figure window to visualize the target PDF and CDF.
figure("Color", "w");

% Create the first subplot for the target PDF.
subplot(2, 1, 1);
% Plot the normalized probability density function.
plot(grayLevels, normalizedPdf, "b-", "LineWidth", 2);
% Add a title to the PDF subplot.
title("Target Probability Density Function (PDF)");
% Label the horizontal axis.
xlabel("Gray Level");
% Label the vertical axis.
ylabel("Probability");
% Add a grid for better readability.
grid on;

% Create the second subplot for the CDF.
subplot(2, 1, 2);
% Plot the cumulative distribution function.
plot(grayLevels, cumulativeDistributionFunction, "r-", "LineWidth", 2);
% Add a title to the CDF subplot.
title("Cumulative Distribution Function (CDF)");
% Label the horizontal axis.
xlabel("Gray Level");
% Label the vertical axis.
ylabel("Cumulative Probability");
% Add a grid for better readability.
grid on;

% Open a new figure window to visualize the simulated image and its histogram.
figure("Color", "w");

% Create the first subplot for the simulated image.
subplot(1, 2, 1);
% Display the simulated image with automatic scaling.
imshow(simulatedImage, []);
% Add a title to the simulated image subplot.
title("Simulated Image via Inverse Mapping");

% Create the second subplot for the histogram comparison.
subplot(1, 2, 2);
% Plot the normalized histogram of the simulated image pixels.
histogram(simulatedImage(:), "Normalization", "pdf", "FaceColor", [0.8, 0.8, 0.8], "EdgeColor", "none");
% Hold the plot to overlay the target PDF curve.
hold on;
% Plot the target PDF curve for comparison.
plot(grayLevels, normalizedPdf, "r-", "LineWidth", 2);
% Add a title to the histogram subplot.
title("Simulated Histogram vs. Target PDF");
% Label the horizontal axis.
xlabel("Gray Level");
% Label the vertical axis.
ylabel("Probability Density");
% Add a legend to identify the plotted elements.
legend("Simulated Histogram", "Target PDF", "Location", "best");
% Add a grid for better readability.
grid on;
% Release the hold on the plot.
hold off;