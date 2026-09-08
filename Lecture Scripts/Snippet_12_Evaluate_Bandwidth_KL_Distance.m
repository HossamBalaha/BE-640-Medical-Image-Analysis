% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Define the mean for the target Gaussian distribution.
gaussianMean = 128;
% Define the variance for the target Gaussian distribution.
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

% Define the array of bandwidth values to test.
bandwidthValues = [0.1, 0.5, 1.0, 5.0];
% Initialize an array to store the KL distances.
klDistances = zeros(1, length(bandwidthValues));
% Add a small epsilon to avoid log(0) during KL distance calculation.
epsilon = 1e-10;

% Open a new figure window with a white background.
figure("Color", "w");

% Loop through each bandwidth value to estimate density and calculate KL distance.
for bwIndex = 1:1:length(bandwidthValues)
    % Extract the current bandwidth.
    currentBandwidth = bandwidthValues(bwIndex);
    % Define the dimensionality for scalar gray levels.
    dimensionality = 1.0;
    % Calculate the total number of samples in the simulated image.
    numSamples = 256 * 256;
    % Calculate the window volume.
    windowVolume = (currentBandwidth .^ dimensionality);
    % Calculate the Gaussian kernel normalization constant.
    gaussianKernelConstant = 1.0 ./ sqrt(2.0 .* pi);

    % Initialize the estimated density array for the current bandwidth.
    estimatedDensityParzen = zeros(1, 256);

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
                scaledDistanceSquared = ((queryGrayLevel - currentSample) .^ 2) ./ (2.0 .* currentBandwidth .^ 2);
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

    % Calculate the Kullback-Leibler distance between the theoretical PDF and the estimated PDF.
    % We only calculate the sum where the theoretical PDF is greater than zero to avoid 0*log(0) = NaN.
    validIndices = normalizedPdf > 0;
    klDistances(bwIndex) = sum(normalizedPdf(validIndices) .* log(normalizedPdf(validIndices) ./ (estimatedDensityParzen(validIndices) + epsilon)));

    % Display the KL distance in the command window.
    fprintf("Bandwidth h = %.1f | KL Distance = %.4f\n", currentBandwidth, klDistances(bwIndex));

    % Create a subplot for the current bandwidth in a 2x2 grid.
    subplot(2, 2, bwIndex);
    % Plot the estimated density for the current bandwidth as a red dashed line.
    plot(grayLevels, estimatedDensityParzen, "r--", "LineWidth", 1.5);
    % Hold the subplot to overlay the target theoretical PDF.
    hold on;
    % Plot the theoretical Gaussian PDF as a solid black line.
    plot(grayLevels, normalizedPdf, "k-", "LineWidth", 2);
    % Add a title to the subplot including the bandwidth and KL distance.
    title(sprintf("h = %.1f (KL = %.4f)", currentBandwidth, klDistances(bwIndex)));
    % Label the horizontal axis.
    xlabel("Gray Level");
    % Label the vertical axis.
    ylabel("Probability Density");
    % Add a grid for better readability.
    grid on;
    % Release the hold on the subplot.
    hold off;
    % End the bandwidth loop.
end

% Add a main title to the entire figure window.
sgtitle("Validation: Theoretical Gaussian vs. Parzen Window Estimate");