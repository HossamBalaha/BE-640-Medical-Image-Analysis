% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

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

% Use the first available mask as the reference for distance map computation.
referenceMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(1).name);
% Read the reference expert mask.
rawMask = imread(referenceMaskPath);

% --- Mask Binarization ---
% Check if the mask is an RGB image with three color channels.
if (size(rawMask, 3) == 3)
    % Extract the blue color channel from the mask.
    blueChannel = rawMask(:, :, 3);
    % Extract the red color channel from the mask.
    redChannel = rawMask(:, :, 1);
    % Extract the green color channel from the mask.
    greenChannel = rawMask(:, :, 2);
    % Create a logical mask for the blue regions representing the lungs.
    binaryMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
else
    % Create a logical mask for the lungs assuming white pixels.
    binaryMask = rawMask > 128;
    % End the mask type check.
end

% Convert the logical mask to double for mathematical operations.
binaryMask = double(binaryMask);
% Get the spatial dimensions of the binary mask.
[imageRows, imageCols] = size(binaryMask);

% --- Edge Detection ---
% Detect the edges of the object using the Prewitt operator.
edgeImage = edge(binaryMask, "prewitt");
% Find the row and column indices of all edge pixels.
[edgeRows, edgeCols, ~] = find(edgeImage == 1);

% --- Distance Map Initialization ---
% Initialize the distance map matrix with zeros.
distanceMap = zeros(imageRows, imageCols);

% --- Distance Calculation Loop ---
% Loop through each row of the image.
for rowIndex = 1:1:imageRows
    % Loop through each column of the image.
    for colIndex = 1:1:imageCols
        % Extract the current pixel intensity from the binary mask.
        currentPixel = binaryMask(rowIndex, colIndex);
        % Calculate the Euclidean distance from the current pixel to all edge pixels.
        distanceVector = sqrt((rowIndex - edgeRows).^2 + (colIndex - edgeCols).^2);
        % Find the minimum distance to the nearest edge.
        minDistance = min(distanceVector);

        % Check if the current pixel is part of the object (inside).
        if (currentPixel > 0)
            % Assign the negative minimum distance for interior pixels.
            distanceMap(rowIndex, colIndex) = -minDistance;
        else
            % Assign the positive minimum distance for exterior pixels.
            distanceMap(rowIndex, colIndex) = minDistance;
            % End the object check.
        end
        % End the column loop.
    end
    % End the row loop.
end

% --- Visualization ---
% Open a new figure window to display the final distance map.
figure("Color", "w");
% Display the computed distance map with automatic scaling.
imshow(distanceMap, []);
% Add a colorbar to show the distance scale.
colorbar;
% Add a title to the distance map figure.
title("Signed Euclidean Distance Map");

% Display a completion message in the command window.
fprintf("Successfully computed Signed Distance Map for the reference mask.\n");