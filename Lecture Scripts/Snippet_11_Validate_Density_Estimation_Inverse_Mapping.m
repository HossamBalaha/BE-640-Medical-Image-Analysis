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

% Define the dimensionality for scalar gray levels.
dimensionality = 1.0;
% Define the bandwidth parameter controlling smoothness.
bandwidth = 0.5;
% Calculate the total number of samples in the simulated image.
numSamples = 256 * 256;
% Calculate the window volume.
windowVolume = (bandwidth .^ dimensionality);
% Calculate the Gaussian kernel normalization constant.
gaussianKernelConstant = 1.0 ./ sqrt(2.0 .* pi);

% Loop through each gray level to estimate its density.
for queryGrayLevel = 0:1:255
    % Initialize the sum of Parzen window contributions.
    parzenWindowSum = 0;

    % Loop through each row of the simulated image.
    for rowIndex = 1:1:256
        % Loop through each column of the simulated image.
        for colIndex = 1:1:256
            % Extract the gray level value of the current pixel.
            currentSample = simulatedImage(rowIndex, colIndex);
            % Calculate the squared distance scaled by bandwidth.
            scaledDistanceSquared = ((queryGrayLevel - currentSample) .^ 2) ./ (2.0 .* bandwidth .^ 2);
            % Add the Gaussian kernel contribution to the running sum.
            parzenWindowSum = parzenWindowSum + (1.0 ./ windowVolume) .* gaussianKernelConstant .* exp(-scaledDistanceSquared);
            % End the column loop.
        end
        % End the row loop.
    end

    % Store the normalized density estimate for the current gray level.
    estimatedDensityParzen(queryGrayLevel + 1) = parzenWindowSum ./ numSamples;
    % End the gray level loop.
end

% Normalize the estimated density so that it sums to one.
estimatedDensityParzen = estimatedDensityParzen ./ sum(estimatedDensityParzen);

% Plot the original target PDF and the Parzen Window estimate for comparison.
figure("Color", "w");
plot(grayLevels, normalizedPdf, "b-", "LineWidth", 2);
hold on;
plot(grayLevels, estimatedDensityParzen, "r--", "LineWidth", 2);
title("Validation: Target PDF vs. Parzen Window Estimate");
xlabel("Gray Level");
ylabel("Probability Density");
legend("Target PDF", "Parzen Window Estimate (h=0.5)", "Location", "best");
grid on;
hold off;