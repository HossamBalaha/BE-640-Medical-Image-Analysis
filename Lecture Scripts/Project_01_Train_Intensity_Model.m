% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Initialize empty arrays to accumulate training samples.
allLungSamples = [];
allChestSamples = [];

% Define the training directory for both images and masks.
trainingImageDir = "./../Dataset/Sample1/train/images";
trainingMaskDir = "./../Dataset/Sample1/train/masks";

% Get the list of training image and mask files.
trainingImageFiles = dir(fullfile(trainingImageDir, "*.jpg"));
trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpg"));

% Open a new figure window to visualize the first five training pairs.
figure("Color", "w");

% Loop through the first five training pairs for visualization.
for visualizationIndex = 1:1:5
    % Construct the full path to the current training image.
    currentImagePath = fullfile(trainingImageDir, trainingImageFiles(visualizationIndex).name);
    % Read the current training image.
    currentImage = imread(currentImagePath);

    % Check if the input image is an RGB image with three color channels.
    if (size(currentImage, 3) == 3)
        % Convert the RGB image to a grayscale image for display.
        currentImage = rgb2gray(currentImage);
        % End the conditional block for image type.
    end

    % Construct the full path to the current mask.
    currentMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(visualizationIndex).name);
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
        lungMask = blueChannel > 128;
        % Handle the case where the mask is a standard grayscale or binary image.
    else
        % Create a logical mask for the lungs assuming white pixels.
        lungMask = rawMask > 128;
        % End the conditional block for mask type.
    end

    % Create the first row subplot for the training image.
    subplot(3, 5, visualizationIndex);
    % Display the training image.
    imagesc(currentImage);
    % Preserve the correct aspect ratio.
    axis image;
    % Add a title to the image subplot.
    title(sprintf("Image %d", visualizationIndex));

    % Create the second row subplot for the original colored mask.
    subplot(3, 5, visualizationIndex + 5);
    % Display the expert mask.
    imagesc(rawMask);
    % Preserve the correct aspect ratio.
    axis image;
    % Add a title to the original mask subplot.
    title(sprintf("Original Mask %d", visualizationIndex));

    % Create the third row subplot for the extracted B&W lung mask.
    subplot(3, 5, visualizationIndex + 10);
    % Display the extracted logical lung mask.
    imshow(lungMask);
    % Apply a grayscale colormap to the figure.
    colormap(gray);
    % Preserve the correct aspect ratio.
    axis image;
    % Add a title to the extracted mask subplot.
    title(sprintf("Extracted Lung Mask %d", visualizationIndex));
    % End the visualization loop.
end

% Add a main title to the entire figure window.
sgtitle("First Five Training Images, Original Masks, and Extracted Lung Masks");

% Loop through each of the 10 training pairs for model estimation.
for fileIndex = 1:1:10
    % Construct the full path to the current training image.
    currentImagePath = fullfile(trainingImageDir, trainingImageFiles(fileIndex).name);
    % Read the current training image and convert it to double precision.
    currentImage = double(imread(currentImagePath));

    % Check if the input image is an RGB image with three color channels.
    if (size(currentImage, 3) == 3)
        % Convert the RGB image to a grayscale image.
        currentImage = double(rgb2gray(uint8(currentImage)));
        % End the conditional block for image type.
    end

    % Construct the full path to the current mask.
    currentMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(fileIndex).name);
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

        % Stricter check to avoid JPG compression artifacts
        lungMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
        % Create a logical mask for the chest as the complement of the lung mask.
        chestMask = ~lungMask;
        % Handle the case where the mask is a standard grayscale or binary image.
    else
        % Create a logical mask for the lungs assuming white pixels.
        lungMask = rawMask > 128;
        % Create a logical mask for the chest as the complement of the lung mask.
        chestMask = ~lungMask;
        % End the conditional block for mask type.
    end

    % Extract the lung pixel intensities using the logical lung mask.
    lungPixels = currentImage(lungMask);
    % Extract the chest pixel intensities using the logical chest mask.
    chestPixels = currentImage(chestMask);

    % Append the current lung pixels to the global accumulation array.
    allLungSamples = [allLungSamples; lungPixels];
    % Append the current chest pixels to the global accumulation array.
    allChestSamples = [allChestSamples; chestPixels];
    % End the training loop.
end

% Calculate the mean and variance for the lung class.
lungMean = mean(allLungSamples);
lungVariance = var(allLungSamples);
% Calculate the mean and variance for the chest class.
chestMean = mean(allChestSamples);
chestVariance = var(allChestSamples);

% Calculate the total number of training pixels.
totalPixels = length(allLungSamples) + length(allChestSamples);
% Compute the prior probability for the lung class.
lungPrior = length(allLungSamples) / totalPixels;
% Compute the prior probability for the chest class.
chestPrior = length(allChestSamples) / totalPixels;

% Define the array of gray levels from 0 to 255 for visualization.
grayLevels = 0:1:255;
% Calculate the Gaussian probability density function (likelihood) for the lung class.
lungPdf = (1.0 ./ sqrt(2.0 .* pi .* lungVariance)) .* exp(-((grayLevels - lungMean).^2) ./ (2.0 .* lungVariance));
% Calculate the Gaussian probability density function (likelihood) for the chest class.
chestPdf = (1.0 ./ sqrt(2.0 .* pi .* chestVariance)) .* exp(-((grayLevels - chestMean).^2) ./ (2.0 .* chestVariance));

% Multiply the lung likelihood by the lung prior to obtain the posterior score.
lungPosterior = lungPdf .* lungPrior;
% Multiply the chest likelihood by the chest prior to obtain the posterior score.
chestPosterior = chestPdf .* chestPrior;

% Open a new figure window with a white background for the model plot.
figure("Color", "w");
% Plot the normalized histogram of the lung samples.
histogram(allLungSamples, grayLevels, "Normalization", "pdf", "FaceColor", [0.8, 0.8, 0.8], "EdgeColor", "none");
% Hold the plot to overlay the theoretical curves.
hold on;

% Plot the lung likelihood curve as a dashed red line.
plot(grayLevels, lungPdf, "r--", "LineWidth", 1.5);
% Plot the chest likelihood curve as a dashed blue line.
plot(grayLevels, chestPdf, "b--", "LineWidth", 1.5);

% Plot the lung posterior curve as a solid red line.
plot(grayLevels, lungPosterior, "r-", "LineWidth", 2);
% Plot the chest posterior curve as a solid blue line.
plot(grayLevels, chestPosterior, "b-", "LineWidth", 2);

% Add a descriptive title to the figure.
title("Trained Intensity Models: Likelihoods vs. Posteriors");
% Label the horizontal axis.
xlabel("Gray Level");
% Label the vertical axis.
ylabel("Probability Density");
% Add a legend to identify all plotted elements.
legend("Lung Samples Histogram", "Lung Likelihood", "Chest Likelihood", "Lung Posterior", "Chest Posterior", "Location", "northwest");
% Add a grid for better readability.
grid on;
% Release the hold on the plot.
hold off;

% Initialize the model parameters structure.
modelParameters = struct();
% Store the lung mean in the structure.
modelParameters.LungMean = lungMean;
% Store the lung variance in the structure.
modelParameters.LungVariance = lungVariance;
% Store the lung prior in the structure.
modelParameters.LungPrior = lungPrior;
% Store the chest mean in the structure.
modelParameters.ChestMean = chestMean;
% Store the chest variance in the structure.
modelParameters.ChestVariance = chestVariance;
% Store the chest prior in the structure.
modelParameters.ChestPrior = chestPrior;

% Save the frozen model to a MAT file for the testing phase.
save("Project01FrozenIntensityModel.mat", "modelParameters");

% Display the training results in the command window.
fprintf("Training Complete!\n");
fprintf("Lung: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", lungMean, lungVariance, lungPrior);
fprintf("Chest: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", chestMean, chestVariance, chestPrior);