% The file name is EstimateLungIntensityModels.m.
% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Read the lung image and convert it to double precision.
doubleImage = double(imread("LungCtSlice.tif"));
% Estimate the threshold that separates the two tissue classes.
thresholdValue = graythresh(uint8(doubleImage)) * 255;
% Gather the lung training samples below the threshold.
lungPixels = doubleImage(doubleImage < thresholdValue);
% Gather the chest training samples above the threshold.
chestPixels = doubleImage(doubleImage >= thresholdValue);

% Estimate the mean of the lung class.
lungMean = mean(lungPixels);
% Estimate the variance of the lung class.
lungVariance = var(lungPixels);
% Estimate the mean of the chest class.
chestMean = mean(chestPixels);
% Estimate the variance of the chest class.
chestVariance = var(chestPixels);

% Define the tissue names for the summary table.
tissueNames = ["Lung"; "Chest"];
% Calculate the pixel counts for both tissue classes.
pixelCounts = [numel(lungPixels); numel(chestPixels)];
% Combine the estimated means into a column vector.
means = [lungMean; chestMean];
% Combine the estimated variances into a column vector.
variances = [lungVariance; chestVariance];
% Create a table to summarize the estimated intensity model parameters.
statsTable = table(tissueNames, pixelCounts, means, variances, VariableNames=["Tissue", "PixelCount", "Mean", "Variance"]);
% Display the summary table in the command window.
disp(statsTable);

% Define the gray-level axis for the model curves.
grayLevels = 0:1:255;
% Compute the Gaussian probability density of the lung class.
lungPdf = (1 ./ sqrt(2 .* pi .* lungVariance)) .* exp(-((grayLevels - lungMean).^2) ./ (2 .* lungVariance));
% Compute the Gaussian probability density of the chest class.
chestPdf = (1 ./ sqrt(2 .* pi .* chestVariance)) .* exp(-((grayLevels - chestMean).^2) ./ (2 .* chestVariance));

% Open a new figure window for the model plot.
figure("Color", "w");
% Plot the lung Gaussian model in red.
plot(grayLevels, lungPdf, "r", "LineWidth", 2);
% Hold the plot to overlay the chest model.
hold on;
% Plot the chest Gaussian model in blue.
plot(grayLevels, chestPdf, "b", "LineWidth", 2);
% Add a legend for the two tissue models.
legend("Lung Tissue", "Chest Tissue");
% Add a descriptive title to the model plot.
title("Gaussian Intensity Models for Lung and Chest Tissues");
% Label the horizontal axis of the model plot.
xlabel("Gray Level");
% Label the vertical axis of the model plot.
ylabel("Probability Density");