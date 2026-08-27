% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Define the mean for the lung tissue.
lungMean = 89.139;
% Define the variance for the lung tissue.
lungVariance = 1399.8;
% Calculate the standard deviation for the lung tissue.
lungStd = sqrt(lungVariance);
% Define the target number of lung samples.
numLungSamples = 110690;

% Define the mean for the chest tissue.
chestMean = 202.38;
% Define the variance for the chest tissue.
chestVariance = 1010.8;
% Calculate the standard deviation for the chest tissue.
chestStd = sqrt(chestVariance);
% Define the target number of chest samples.
numChestSamples = 151450;

% Generate random lung samples from a Gaussian distribution.
lungSamples = lungMean + lungStd .* randn(numLungSamples, 1);
% Generate random chest samples from a Gaussian distribution.
chestSamples = chestMean + chestStd .* randn(numChestSamples, 1);

% Clip the lung samples to the valid gray level range of 0 to 255.
lungSamples = max(0, min(255, lungSamples));
% Clip the chest samples to the valid gray level range of 0 to 255.
chestSamples = max(0, min(255, chestSamples));

% Define the array of gray levels from 0 to 255.
grayLevels = 0:1:255;
% Define the bandwidth parameter for the Parzen window.
bandwidth = 1.0;

% Estimate the lung density using the Parzen Window function.
estimatedDensityLung = Snippet_06_Estimate_Parzen_Density(lungSamples, grayLevels, bandwidth);
% Estimate the chest density using the Parzen Window function.
estimatedDensityChest = Snippet_06_Estimate_Parzen_Density(chestSamples, grayLevels, bandwidth);

% Open a new figure window with a white background.
figure("Color", "w");

% Create the first subplot for the lung density.
subplot(1, 2, 1);
% Plot the estimated lung density.
plot(grayLevels, estimatedDensityLung, "r-", "LineWidth", 2);
% Add a title to the lung density subplot.
title("Lung Density (Parzen Window)");
% Label the horizontal axis of the lung density subplot.
xlabel("Gray Level");
% Label the vertical axis of the lung density subplot.
ylabel("Probability Density");
% Add a grid to the lung density subplot.
grid on;

% Create the second subplot for the chest density.
subplot(1, 2, 2);
% Plot the estimated chest density.
plot(grayLevels, estimatedDensityChest, "b-", "LineWidth", 2);
% Add a title to the chest density subplot.
title("Chest Density (Parzen Window)");
% Label the horizontal axis of the chest density subplot.
xlabel("Gray Level");
% Label the vertical axis of the chest density subplot.
ylabel("Probability Density");
% Add a grid to the chest density subplot.
grid on;