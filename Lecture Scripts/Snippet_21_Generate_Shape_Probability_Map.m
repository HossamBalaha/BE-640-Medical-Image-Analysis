% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the number of training images to use for the shape model.
numTrainingImages = 10;

% --- Data Loading Paths ---
% Define the training masks directory path.
trainingMaskDir = "./../Dataset/Sample1/train/masks";

% Get the list of training mask files.
trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpg"));

% Check if mask files were found.
if (isempty(trainingMaskFiles))
    % Fallback to .jpeg if no .jpg files are found.
    trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpeg"));
    % End the fallback check.
end

% Get the actual number of mask files available.
numAvailableMasks = length(trainingMaskFiles);
% Determine the actual number of images to process based on availability.
actualNumImages = min(numTrainingImages, numAvailableMasks);

% Initialize the accumulator for the shape probability map as empty.
shapeProbabilityMap = [];

% --- Shape Probability Map Generation ---
% Loop through each training mask to accumulate the binary shapes.
for imgIndex = 1:1:actualNumImages
    % Construct the full path to the current training mask.
    currentMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(imgIndex).name);
    % Read the current expert mask.
    rawMask = imread(currentMaskPath);

    % Check if the mask is an RGB image with three color channels.
    if (size(rawMask, 3) == 3)
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Create a logical mask for the blue regions representing the lungs.
        binaryMask = double((blueChannel > 128) & (redChannel < 128) & (greenChannel < 128));
    else
        % Create a logical mask for the lungs assuming white pixels.
        binaryMask = double(rawMask > 128);
        % End the mask type check.
    end

    % Initialize the accumulator on the first iteration using the actual image dimensions.
    if (isempty(shapeProbabilityMap))
        % Get the dimensions of the current binary mask.
        [imageRows, imageCols] = size(binaryMask);
        % Initialize the accumulator with zeros.
        shapeProbabilityMap = zeros(imageRows, imageCols);
        % End the first iteration check.
    end

    % Add the current binary mask to the accumulator.
    shapeProbabilityMap = shapeProbabilityMap + binaryMask;
    % End the training mask loop.
end

% Calculate the final probability by dividing by the number of images.
shapeProbabilityMap = shapeProbabilityMap ./ actualNumImages;
% Add a small smoothing constant to avoid strict zeros in the probability map.
shapeProbabilityMap = shapeProbabilityMap + 0.00001;

% --- Visualization ---
% Open a new figure window to display the generated probability map.
figure("Color", "w");
% Display the shape probability map with automatic scaling.
imshow(shapeProbabilityMap, []);
% Add a colorbar to show the probability scale.
colorbar;
% Add a title to the probability map figure.
title("Generated Shape Probability Map (Lung)");

% Display a completion message in the command window.
fprintf("Successfully generated Shape Probability Map using %d training masks.\n", actualNumImages);