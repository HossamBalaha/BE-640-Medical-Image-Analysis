% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Load the trained EM model parameters from the MAT file.
load("Snippet13_TrainedEMModelParameters.mat");
% Extract the Gaussian means from the structure.
gaussianMeans = modelParameters.GaussianMeans;
% Extract the Gaussian variances from the structure.
gaussianVariances = modelParameters.GaussianVariances;
% Extract the class priors from the structure.
classPriors = modelParameters.ClassPriors;

% Display the loaded model parameters.
fprintf("Loaded EM Model Parameters:\n");
fprintf("Class 1: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", gaussianMeans(1), gaussianVariances(1), classPriors(1));
fprintf("Class 2: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", gaussianMeans(2), gaussianVariances(2), classPriors(2));

% Load the held-out test image and convert it to double precision.
testImage = double(imread("LungCtSlice.tif"));
% Load the expert ground truth mask.
rawMask = imread("LungCtMask.tif");

% Check if the mask is an RGB image with three color channels.
if (size(rawMask, 3) == 3)
    % Extract the blue color channel from the mask.
    blueChannel = rawMask(:, :, 3);
    % Extract the red color channel from the mask.
    redChannel = rawMask(:, :, 1);
    % Extract the green color channel from the mask.
    greenChannel = rawMask(:, :, 2);
    % Create a logical mask for the blue regions representing the target class.
    groundTruthMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
else
    % Create a logical mask for the target class assuming white pixels.
    groundTruthMask = rawMask > 128;
    % End the mask type check.
end

% Get the dimensions of the test image.
[imageRows, imageCols] = size(testImage);
% Initialize the segmentation mask with false values.
segmentationMask = false(imageRows, imageCols);

% Loop through each row of the test image.
for rowIndex = 1:1:imageRows
    % Loop through each column of the test image.
    for colIndex = 1:1:imageCols
        % Extract the pixel intensity.
        pixelIntensity = testImage(rowIndex, colIndex);

        % Calculate the likelihood for Class 1.
        likelihoodClass1 = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(1))) .* exp(-((pixelIntensity - gaussianMeans(1)).^2) ./ (2.0 .* gaussianVariances(1)));
        % Calculate the likelihood for Class 2.
        likelihoodClass2 = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(2))) .* exp(-((pixelIntensity - gaussianMeans(2)).^2) ./ (2.0 .* gaussianVariances(2)));

        % Calculate the posterior score for Class 1.
        posteriorScore1 = likelihoodClass1 .* classPriors(1);
        % Calculate the posterior score for Class 2.
        posteriorScore2 = likelihoodClass2 .* classPriors(2);

        % Assign the pixel to Class 1 if its posterior score is higher.
        if (posteriorScore1 > posteriorScore2)
            % Set the segmentation mask pixel to true.
            segmentationMask(rowIndex, colIndex) = true;
            % End the class assignment check.
        end
        % End the column loop.
    end
    % End the row loop.
end

% Compute the intersection between the segmentation and the ground truth.
truePositives = sum(segmentationMask(:) & groundTruthMask(:));
% Compute the false positives.
falsePositives = sum(segmentationMask(:) & ~groundTruthMask(:));
% Compute the false negatives.
falseNegatives = sum(~segmentationMask(:) & groundTruthMask(:));

% Calculate the denominator for the Dice coefficient.
diceDenominator = (2.0 .* truePositives) + falsePositives + falseNegatives;

% Check if the denominator is positive.
if (diceDenominator > 0)
    % Calculate the Dice coefficient.
    diceCoefficient = (2.0 .* truePositives) ./ diceDenominator;
else
    % Set the Dice coefficient to 1.0 if both masks are empty.
    diceCoefficient = 1.0;
    % End the empty mask check.
end

% Display the evaluation metrics.
fprintf("\n=== EM Segmentation Evaluation ===\n");
fprintf("Dice Coefficient = %.4f\n", diceCoefficient);
fprintf("TP=%d, FP=%d, FN=%d\n", truePositives, falsePositives, falseNegatives);

% Open a new figure window for visualization.
figure("Color", "w");

% Create the first subplot for the test image.
subplot(1, 3, 1);
% Display the test image with automatic scaling.
imshow(testImage, []);
% Add a title to the test image.
title("Test Image");

% Create the second subplot for the ground truth.
subplot(1, 3, 2);
% Display the ground truth mask.
imshow(groundTruthMask);
% Add a title to the ground truth.
title("Ground Truth Mask");

% Create the third subplot for the EM segmentation.
subplot(1, 3, 3);
% Display the segmentation mask.
imshow(segmentationMask);
% Add a title to the segmentation mask including the Dice score.
title(sprintf("EM Segmentation\nDice=%.4f", diceCoefficient));
