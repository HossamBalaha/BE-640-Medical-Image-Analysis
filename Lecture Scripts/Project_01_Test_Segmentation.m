% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Load the frozen model parameters from the training phase.
load("Project01FrozenIntensityModel.mat");
% Extract the lung mean from the loaded structure.
lungMean = modelParameters.LungMean;
% Extract the lung variance from the loaded structure.
lungVariance = modelParameters.LungVariance;
% Extract the lung prior from the loaded structure.
lungPrior = modelParameters.LungPrior;
% Extract the chest mean from the loaded structure.
chestMean = modelParameters.ChestMean;
% Extract the chest variance from the loaded structure.
chestVariance = modelParameters.ChestVariance;
% Extract the chest prior from the loaded structure.
chestPrior = modelParameters.ChestPrior;

% Display the loaded model parameters.
fprintf("Loaded Model Parameters:\n");
fprintf("  Lung: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", lungMean, lungVariance, lungPrior);
fprintf("  Chest: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", chestMean, chestVariance, chestPrior);

% Define the testing directory for both images and masks.
testingImageDir = "./../Dataset/Sample1/test/images";
testingMaskDir = "./../Dataset/Sample1/test/masks";

% Get the list of testing image and mask files.
testingImageFiles = dir(fullfile(testingImageDir, "*.jpg"));
testingMaskFiles = dir(fullfile(testingMaskDir, "*.jpg"));

% Determine the number of test files available.
numTestFiles = length(testingImageFiles);

% Loop through each of the available test pairs.
for testIndex = 1:1:numTestFiles
    % Construct the full path to the current testing image.
    currentTestImagePath = fullfile(testingImageDir, testingImageFiles(testIndex).name);
    % Read the current test image.
    rawTestImage = imread(currentTestImagePath);

    % Check if the input image is an RGB image with three color channels.
    if (size(rawTestImage, 3) == 3)
        % Convert the RGB image to a grayscale image.
        testImage = double(rgb2gray(rawTestImage));
    else
        % Convert the grayscale image to double precision.
        testImage = double(rawTestImage);
    end

    % Construct the full path to the current mask.
    currentMaskPath = fullfile(testingMaskDir, testingMaskFiles(testIndex).name);
    % Read the current expert mask.
    rawMask = imread(currentMaskPath);

    % Check if the mask is an RGB image with three color channels.
    if (size(rawMask, 3) == 3)
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);

        % Create a logical mask for the blue regions representing the lungs.
        groundTruthMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
    else
        % Handle the case where the mask is a standard grayscale or binary image.
        groundTruthMask = rawMask > 128;
    end

    % Compute the Gaussian likelihood of the lung class for every pixel.
    lungLikelihood = (1.0 ./ sqrt(2.0 .* pi .* lungVariance)) .* exp(-((testImage - lungMean).^2) ./ (2.0 .* lungVariance));
    % Compute the Gaussian likelihood of the chest class for every pixel.
    chestLikelihood = (1.0 ./ sqrt(2.0 .* pi .* chestVariance)) .* exp(-((testImage - chestMean).^2) ./ (2.0 .* chestVariance));

    % Multiply the lung likelihood by the lung prior to obtain the posterior score.
    lungScore = lungLikelihood .* lungPrior;
    % Multiply the chest likelihood by the chest prior to obtain the posterior score.
    chestScore = chestLikelihood .* chestPrior;

    % Assign each pixel to the lung class when its lung score exceeds its chest score.
    segmentationMask = lungScore > chestScore;

    % Compute True Positives, False Positives, False Negatives, and True Negatives.
    truePositives = sum(segmentationMask(:) & groundTruthMask(:));
    falsePositives = sum(segmentationMask(:) & ~groundTruthMask(:));
    falseNegatives = sum(~segmentationMask(:) & groundTruthMask(:));
    trueNegatives = sum(~segmentationMask(:) & ~groundTruthMask(:));

    % Compute the Dice coefficient.
    diceCoefficient = (2.0 * truePositives) / (2.0 * truePositives + falsePositives + falseNegatives);
    % Compute the IoU (Jaccard Index).
    iouCoefficient = truePositives / (truePositives + falsePositives + falseNegatives);
    % Compute the Sensitivity (Recall).
    sensitivity = truePositives / (truePositives + falseNegatives);
    % Compute the Specificity.
    specificity = trueNegatives / (trueNegatives + falsePositives);

    % Display the evaluation metrics for the current test image.
    fprintf("\n=== Test Image %d Evaluation ===\n", testIndex);
    fprintf("Dice Coefficient = %.4f\n", diceCoefficient);
    fprintf("IoU (Jaccard)    = %.4f\n", iouCoefficient);
    fprintf("Sensitivity      = %.4f\n", sensitivity);
    fprintf("Specificity      = %.4f\n", specificity);
    fprintf("TP=%d, FP=%d, FN=%d, TN=%d\n", truePositives, falsePositives, falseNegatives, trueNegatives);

    % Open a new figure window for the current test image results.
    figure("Color", "w");

    % Create the first subplot for the test image.
    subplot(1, 3, 1);
    % Display the test image as a proper grayscale image.
    imshow(testImage, []);
    % Add a title to the test image subplot.
    title("Test Image");

    % Create the second subplot for the ground truth.
    subplot(1, 3, 2);
    % Display the expert ground truth mask in black and white.
    imshow(groundTruthMask);
    % Add a title to the ground truth subplot.
    title(sprintf("Ground Truth\nLung Pixels: %d", sum(groundTruthMask(:))));

    % Create the third subplot for the automatic segmentation.
    subplot(1, 3, 3);
    % Display the model-based segmentation mask in black and white.
    imshow(segmentationMask);
    % Add a title to the segmentation subplot including all metrics.
    title(sprintf("Segmentation\nDice=%.4f, IoU=%.4f\nSens=%.4f, Spec=%.4f", diceCoefficient, iouCoefficient, sensitivity, specificity));

    % Add a main title to the entire figure window.
    sgtitle(sprintf("Test Image %d Evaluation", testIndex));
end