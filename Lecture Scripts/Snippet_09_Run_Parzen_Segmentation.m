% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Generate synthetic training data based on Lecture 02 parameters.
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

% Generate random lung samples from Gaussian distribution.
lungSamples = lungMean + lungStd .* randn(numLungSamples, 1);
% Generate random chest samples from Gaussian distribution.
chestSamples = chestMean + chestStd .* randn(numChestSamples, 1);

% Clip samples to valid gray level range 0-255.
lungSamples = max(0, min(255, lungSamples));
chestSamples = max(0, min(255, chestSamples));

% Read the test image and convert it to double precision.
testImage = double(imread("LungCtSlice.tif"));
% Read the expert mask and convert it to a logical ground truth.
groundTruthMask = imread("LungCtMask.tif") > 0;

% Define the bandwidth parameter for Parzen Window.
bandwidth = 5.0;

% Segment the test image using Parzen Window densities.
segmentationMask = Snippet_08_Segment_With_Parzen_Densities(testImage, lungSamples, chestSamples, bandwidth);

% Compute the intersection between the segmentation and the ground truth.
intersectionCount = sum(segmentationMask(:) & groundTruthMask(:));
% Compute the total number of foreground pixels in both masks.
totalForeground = sum(segmentationMask(:)) + sum(groundTruthMask(:));

% Check whether the denominator is positive before division.
if (totalForeground > 0)
    % Compute the Dice coefficient between the segmentation and the ground truth.
    diceCoefficient = (2.0 .* intersectionCount) ./ totalForeground;
    % Handle the degenerate case where both masks are empty.
else
    % Define the Dice coefficient as one when both masks are empty.
    diceCoefficient = 1.0;
    % End the conditional block.
end

% Display the Dice coefficient in the command window.
fprintf("Dice coefficient = %.4f\n", diceCoefficient);

% Open a new figure window for the visual comparison.
figure("Color", "w");
% Create the first subplot for the test image.
subplot(1, 3, 1);
% Display the test image as a grayscale image.
imagesc(testImage);
% Apply a grayscale colormap to the figure.
colormap(gray);
% Preserve the correct aspect ratio of the image.
axis image;
% Add a title to the test image subplot.
title("Test Image");

% Create the second subplot for the ground truth.
subplot(1, 3, 2);
% Display the expert ground truth mask.
imagesc(groundTruthMask);
% Preserve the correct aspect ratio of the mask.
axis image;
% Add a title to the ground truth subplot.
title("Ground Truth Mask");

% Create the third subplot for the automatic segmentation.
subplot(1, 3, 3);
% Display the model-based segmentation mask.
imagesc(segmentationMask);
% Preserve the correct aspect ratio of the mask.
axis image;
% Add a title to the segmentation subplot.
title("Model-Based Segmentation (Parzen Window)");

% Save the segmentation mask as a TIFF file.
imwrite(segmentationMask, "LungCtPredicted_Parzen.tif");