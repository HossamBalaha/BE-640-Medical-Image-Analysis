% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Read the test image and convert it to double precision.
testImage = double(imread("LungCtSlice.tif"));
% Read the expert mask and convert it to a logical ground truth.
groundTruthMask = imread("LungCtMask.tif") > 0;
% Initialize the model structure with the training factors.
modelParameters = struct();
% Store the lung mean estimated from the training data.
modelParameters.LungMean = 89.139;
% Store the lung variance estimated from the training data.
modelParameters.LungVariance = 1399.8;
% Store the chest mean estimated from the training data.
modelParameters.ChestMean = 202.38;
% Store the chest variance estimated from the training data.
modelParameters.ChestVariance = 1010.8;
% Define the lung pixel count.
lungCount = 110690;
% Define the chest pixel count.
chestCount = 151450;
% Store the lung prior computed from the training pixel counts.
modelParameters.LungPrior = lungCount ./ (lungCount + chestCount);
% Store the chest prior computed from the training pixel counts.
modelParameters.ChestPrior = chestCount ./ (lungCount + chestCount);
% Segment the test image using the estimated intensity models.
segmentationMask = Snippet_04_Segment_Image_With_Intensity_Models(testImage, modelParameters);
% Compute the intersection between the segmentation and the ground truth.
intersectionCount = sum(segmentationMask(:) & groundTruthMask(:));
% Compute the total number of foreground pixels in both masks.
totalForeground = sum(segmentationMask(:)) + sum(groundTruthMask(:));
% Check whether the denominator is positive before division.
if (totalForeground > 0)
    % Compute the Dice coefficient between the segmentation and the ground truth.
    diceCoefficient = (2 .* intersectionCount) ./ totalForeground;
    % Handle the degenerate case where both masks are empty.
else
    % Define the Dice coefficient as one when both masks are empty.
    diceCoefficient = 1;
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
title("Model-Based Segmentation");
% Export the predicted mask.
% Save the segmentation mask as a PNG file.
imwrite(segmentationMask, 'LungCtPredicted.tif');