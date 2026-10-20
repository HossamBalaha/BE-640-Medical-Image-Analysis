% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the number of training shapes to use for the ASM.
numTrainingShapes = 4;
% Define the initial guess for the shape weights.
initialWeights = [1.0, 1.0, 1.0, 1.0];

% --- Data Loading Paths ---
% Define the training masks directory path.
trainingMaskDir = "./../Dataset/Sample1/train/masks";
% Define the test image directory path.
testImageDir = "./../Dataset/Sample1/test/images";

% Get the list of training mask files.
trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpg"));
% Check if mask files were found.
if (isempty(trainingMaskFiles))
    % Fallback to .jpeg if no .jpg files are found.
    trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpeg"));
    % End the fallback check.
end

% Get the list of test image files.
testImageFiles = dir(fullfile(testImageDir, "*.jpg"));
% Check if test image files were found.
if (isempty(testImageFiles))
    % Fallback to .jpeg if no .jpg files are found.
    testImageFiles = dir(fullfile(testImageDir, "*.jpeg"));
    % End the fallback check.
end

% Use the first test image as the target for ASM fitting.
testImagePath = fullfile(testImageDir, testImageFiles(1).name);
% Read the test image and convert to double precision.
inputImage = double(imread(testImagePath));
% Check if the test image is RGB and convert to grayscale if necessary.
if (size(inputImage, 3) == 3)
    % Convert the RGB image to a grayscale image.
    inputImage = double(rgb2gray(uint8(inputImage)));
    % End the RGB check.
end

% --- Distance Map Generation for Training Shapes ---
% Initialize a cell array to store the distance maps.
distanceMaps = cell(1, numTrainingShapes);

% Loop through the first N training masks to generate distance maps.
for shapeIndex = 1:1:numTrainingShapes
    % Construct the full path to the current training mask.
    currentMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(shapeIndex).name);
    % Read the current expert mask.
    rawMask = imread(currentMaskPath);

    % Binarize the mask.
    if (size(rawMask, 3) == 3)
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Create a logical mask for the blue regions.
        binaryMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
    else
        % Create a logical mask assuming white pixels.
        binaryMask = rawMask > 128;
        % End the mask type check.
    end

    % Convert to double and detect edges.
    binaryMask = double(binaryMask);
    % Detect the edges of the object using the Prewitt operator.
    edgeImage = edge(binaryMask, "prewitt");
    % Find the row and column indices of all edge pixels.
    [edgeRows, edgeCols, ~] = find(edgeImage == 1);

    % Compute the signed distance map for the current shape.
    [imageRows, imageCols] = size(binaryMask);
    % Initialize the distance map matrix.
    distMap = zeros(imageRows, imageCols);
    % Loop through each row.
    for r = 1:1:imageRows
        % Loop through each column.
        for c = 1:1:imageCols
            % Calculate Euclidean distance to all edge pixels.
            distVector = sqrt((r - edgeRows).^2 + (c - edgeCols).^2);
            % Find the minimum distance.
            minDist = min(distVector);
            % Assign negative distance if inside, positive if outside.
            if (binaryMask(r, c) > 0)
                % Assign negative minimum distance.
                distMap(r, c) = -minDist;
            else
                % Assign positive minimum distance.
                distMap(r, c) = minDist;
                % End the inside/outside check.
            end
            % End the column loop.
        end
        % End the row loop.
    end
    % Store the computed distance map in the cell array.
    distanceMaps{shapeIndex} = distMap;
    % End the training shape loop.
end

% --- Average Shape Calculation ---
% Initialize the average distance map.
phiAvg = zeros(size(distanceMaps{1}));
% Loop through the distance maps to sum them.
for shapeIndex = 1:1:numTrainingShapes
    % Add the current distance map to the accumulator.
    phiAvg = phiAvg + distanceMaps{shapeIndex};
    % End the summation loop.
end
% Calculate the final average distance map.
phiAvg = phiAvg ./ numTrainingShapes;

% --- Define the Energy Function for fminsearch ---
% Define the nested function to calculate the energy for a given set of weights.
function energyValue = calculateEnergy(weights, phiAvg, distanceMaps, inputImage)
% Extract the individual weights from the input array.
w1 = weights(1);
w2 = weights(2);
w3 = weights(3);
w4 = weights(4);

% Generate the new shape using the linear combination formula.
phiNew = phiAvg + w1 .* (phiAvg - distanceMaps{1}) + w2 .* (phiAvg - distanceMaps{2}) + w3 .* (phiAvg - distanceMaps{3}) + w4 .* (phiAvg - distanceMaps{4});
% Threshold the new shape to create a binary mask.
binaryShape = double(phiNew <= 0);

% Get the spatial dimensions of the binary shape.
[rows, cols] = size(binaryShape);
% Initialize accumulators for inside and outside intensities.
sumInside = 0.0;
sumOutside = 0.0;
% Initialize counters for inside and outside pixels.
countInside = 0;
countOutside = 0;

% Loop through each row of the shape.
for r = 1:1:rows
    % Loop through each column of the shape.
    for c = 1:1:cols
        % Check if the current pixel is inside the shape.
        if (binaryShape(r, c) == 1)
            % Add the pixel intensity to the inside sum.
            sumInside = sumInside + inputImage(r, c);
            % Increment the inside pixel counter.
            countInside = countInside + 1;
        else
            % Add the pixel intensity to the outside sum.
            sumOutside = sumOutside + inputImage(r, c);
            % Increment the outside pixel counter.
            countOutside = countOutside + 1;
            % End the inside/outside check.
        end
        % End the column loop.
    end
    % End the row loop.
end

% Calculate the mean intensity inside the shape.
muInside = sumInside ./ countInside;
% Calculate the mean intensity outside the shape.
muOutside = sumOutside ./ countOutside;

% Calculate the energy as the inverse of the absolute contrast.
energyValue = 1.0 ./ abs(muInside - muOutside);
% End the nested function definition.
end

% --- Run Optimization ---
% Use fminsearch to find the optimal weights that minimize the energy function.
optimalWeights = fminsearch(@(w) calculateEnergy(w, phiAvg, distanceMaps, inputImage), initialWeights);

% --- Generate and Visualize Final Shape ---
% Extract the optimal weights.
w1Opt = optimalWeights(1);
w2Opt = optimalWeights(2);
w3Opt = optimalWeights(3);
w4Opt = optimalWeights(4);

% Generate the final optimized shape using the optimal weights.
finalShape = phiAvg + w1Opt .* (phiAvg - distanceMaps{1}) + w2Opt .* (phiAvg - distanceMaps{2}) + w3Opt .* (phiAvg - distanceMaps{3}) + w4Opt .* (phiAvg - distanceMaps{4});
% Threshold the final shape to create the final binary mask.
finalBinaryShape = double(finalShape <= 0);

% Detect the edges of the final binary shape.
objectBoundary = edge(finalBinaryShape, "prewitt");
% Find the row and column indices of the final boundary pixels.
[bndRows, bndCols, ~] = find(objectBoundary == 1);

% Open a new figure window to display the results.
figure("Color", "w");
% Display the test image.
imshow(inputImage, []);
% Hold the current plot to overlay the boundary.
hold on;
% Overlay the final boundary on the test image in red.
plot(bndCols, bndRows, ".r", "MarkerSize", 5);
% Add a title to the figure.
title("Final Optimized ASM Segmentation");
% Release the hold on the plot.
hold off;

% Display a completion message in the command window.
fprintf("Successfully optimized ASM weights and segmented the test image.\n");