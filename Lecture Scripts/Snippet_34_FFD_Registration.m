% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Data Loading ---
% Load the fixed reference image and convert to double precision.
fixedImageRGB = double(imread("FFD-1.jpg"));
% Extract the first channel to create a grayscale image.
fixedImage = fixedImageRGB(:, :, 1);

% Load the moving target image and convert to double precision.
movingImageRGB = double(imread("FFD-2.jpg"));
% Extract the first channel to create a grayscale image.
movingImage = movingImageRGB(:, :, 1);

% Ensure both images have the same dimensions.
[imageHeight, imageWidth] = size(fixedImage);
% Resize the moving image to match the fixed image if necessary.
movingImage = imresize(movingImage, [imageHeight, imageWidth]);

% --- FFD Parameter Definition ---
% Start with 32. If the registration fails to align small local features, 
% drop it to 16. If the optimization takes too long or the image looks 
% "tangled", increase it to 64.
% Define the spacing between control points.
controlPointSpacing = 32;
% Calculate the number of control points in the Y dimension.
numControlPointsY = floor(imageHeight / controlPointSpacing) + 3;
% Calculate the number of control points in the X dimension.
numControlPointsX = floor(imageWidth / controlPointSpacing) + 3;

% --- Optimization Setup ---
% Initialize the control point displacement field with a small random noise.
% IMPORTANT: We use randn * 5 instead of zeros. If we use zeros, fminsearch's 
% initial tiny steps (0.00075) will be rounded to 0 by the ApplyFFD function, 
% creating a "flat" cost landscape where the optimizer gets stuck immediately.
rng(42); % Set seed for reproducibility
initialControlPointsY = randn(numControlPointsY, numControlPointsX) * 5;
% Initialize the X component of the control point displacement field.
initialControlPointsX = randn(numControlPointsY, numControlPointsX) * 5;

% Flatten the 2D control point grids into a 1D vector for the optimizer.
% The vector format is [X_points(:); Y_points(:)].
initialVector = [initialControlPointsX(:); initialControlPointsY(:)];

% Set the optimization options to display iteration details.
% IMPORTANT: FFD has many variables (e.g., 200+). We must increase MaxFunEvals 
% significantly so the optimizer has enough attempts to find the minimum.
options = optimset("Display", "iter", "MaxFunEvals", 10000, "MaxIter", 10000);

% Run fminsearch to find the control point displacements that minimize the SSD.
[optimalVector, minSSD] = fminsearch(@(vec) CalculateFFD_Cost(vec, fixedImage, movingImage, controlPointSpacing, imageHeight, imageWidth), initialVector, options);

% Reshape the optimal 1D vector back into 2D control point grids.
optimalControlPointsX = reshape(optimalVector(1 : numControlPointsY * numControlPointsX), numControlPointsY, numControlPointsX);
% Reshape the Y displacements.
optimalControlPointsY = reshape(optimalVector(numControlPointsY * numControlPointsX + 1 : end), numControlPointsY, numControlPointsX);

% --- Apply Final FFD and Visualize ---
% Apply the optimal FFD to the moving image.
[registeredImage, displacementFieldX] = ApplyFFD(movingImage, optimalControlPointsX, optimalControlPointsY, controlPointSpacing);

% Open a new figure window.
figure("Color", "w");
% Display the fixed reference image.
subplot(2, 2, 1);
imshow(fixedImage, []);
title("Fixed Image (Reference)");
% Display the original moving image.
subplot(2, 2, 2);
imshow(movingImage, []);
title("Moving Image (Target)");
% Display the displacement field.
subplot(2, 2, 3);
imagesc(displacementFieldX);
colorbar;
title("Optimized Displacement Field (X)");
% Display the final registered image.
subplot(2, 2, 4);
imshow(registeredImage, []);
title(sprintf("Registered Image (SSD: %.2f)", minSSD));

% --- Local Function Definitions ---

% Define the function to calculate the Sum of Squared Differences cost.
function cost = CalculateFFD_Cost(vec, fixedImage, movingImage, controlPointSpacing, imageHeight, imageWidth)
% Get the number of control points.
numControlPointsY = floor(imageHeight / controlPointSpacing) + 3;
% Get the number of control points in X.
numControlPointsX = floor(imageWidth / controlPointSpacing) + 3;

% Reshape the 1D vector back into 2D grids.
controlPointsX = reshape(vec(1 : numControlPointsY * numControlPointsX), numControlPointsY, numControlPointsX);
% Reshape the Y displacements.
controlPointsY = reshape(vec(numControlPointsY * numControlPointsX + 1 : end), numControlPointsY, numControlPointsX);

% Warp the moving image using the current control points.
[warpedImage, ~] = ApplyFFD(movingImage, controlPointsX, controlPointsY, controlPointSpacing);

% Calculate the Sum of Squared Differences (SSD) between fixed and warped.
difference = fixedImage - warpedImage;
% Calculate the mean squared error as the cost.
cost = mean(difference(:).^2);
% End the function definition.
end

% Define the function to apply FFD warping.
function [warpedImage, displacementFieldX] = ApplyFFD(inputImage, controlPointsX, controlPointsY, controlPointSpacing)
% Get the dimensions of the input image.
[imageHeight, imageWidth] = size(inputImage);
% Initialize the warped image matrix.
warpedImage = zeros(imageHeight, imageWidth);
% Initialize the displacement field visualization matrix.
displacementFieldX = zeros(imageHeight, imageWidth);
% Get the number of control points.
numControlPointsY = size(controlPointsX, 1);
% Get the number of control points in X.
numControlPointsX = size(controlPointsX, 2);

% Loop through each row of the destination image.
for destRow = 1:1:imageHeight
    % Loop through each column of the destination image.
    for destCol = 1:1:imageWidth
        % Calculate the base control point index i.
        baseI = floor(destRow / controlPointSpacing);
        % Calculate the base control point index j.
        baseJ = floor(destCol / controlPointSpacing);
        % Calculate the fractional distance u.
        u = (destRow / controlPointSpacing) - floor(destRow / controlPointSpacing);
        % Calculate the fractional distance v.
        v = (destCol / controlPointSpacing) - floor(destCol / controlPointSpacing);

        % Clamp base indices to ensure they are within safe bounds.
        if (baseI < 1)
            baseI = 1;
            % End the lower bound check.
        end
        % Clamp the base J index to the lower bound.
        if (baseJ < 1)
            baseJ = 1;
            % End the lower bound check.
        end
        % Ensure baseI does not exceed the grid bounds.
        if (baseI > numControlPointsY - 3)
            baseI = numControlPointsY - 3;
            % End the upper bound check.
        end
        % Ensure baseJ does not exceed the grid bounds.
        if (baseJ > numControlPointsX - 3)
            baseJ = numControlPointsX - 3;
            % End the upper bound check.
        end

        % Calculate the B-spline basis weights for u.
        weightsU = CalculateBasisWeights(u);
        % Calculate the B-spline basis weights for v.
        weightsV = CalculateBasisWeights(v);

        % Initialize the total displacement for the current pixel.
        totalDispY = 0;
        % Initialize the total X displacement.
        totalDispX = 0;

        % Loop through the 4x4 neighborhood of control points.
        for l = 0:1:3
            % Loop through the second dimension of the neighborhood.
            for m = 0:1:3
                % Calculate the control point row index.
                cpRow = baseI + l;
                % Calculate the control point column index.
                cpCol = baseJ + m;
                % Calculate the combined weight for this control point.
                weight = weightsU(l + 1) * weightsV(m + 1);
                % Accumulate the Y displacement.
                totalDispY = totalDispY + weight * controlPointsY(cpRow, cpCol);
                % Accumulate the X displacement.
                totalDispX = totalDispX + weight * controlPointsX(cpRow, cpCol);
                % End the m loop.
            end
            % End the l loop.
        end

        % Store the displacement for visualization.
        displacementFieldX(destRow, destCol) = totalDispX;
        % Calculate the source row coordinate using inverse mapping.
        srcRow = round(destRow - totalDispY);
        % Calculate the source column coordinate.
        srcCol = round(destCol - totalDispX);

        % Clamp source coordinates to image boundaries.
        if (srcRow < 1)
            srcRow = 1;
            % End the lower bound check.
        end
        % Clamp the source row coordinate to the upper bound.
        if (srcRow > imageHeight)
            srcRow = imageHeight;
            % End the upper bound check.
        end
        % Clamp the source column coordinate to the lower bound.
        if (srcCol < 1)
            srcCol = 1;
            % End the lower bound check.
        end
        % Clamp the source column coordinate to the upper bound.
        if (srcCol > imageWidth)
            srcCol = imageWidth;
            % End the upper bound check.
        end

        % Pull the pixel intensity from the source image.
        warpedImage(destRow, destCol) = inputImage(srcRow, srcCol);
        % End the column loop.
    end
    % End the row loop.
end
% End the function definition.
end

% Define the function to calculate cubic B-spline basis weights.
function weights = CalculateBasisWeights(t)
% Initialize the weights array.
weights = zeros(1, 4);
% Calculate the first basis function B0.
weights(1) = (1 - t)^3 / 6;
% Calculate the second basis function B1.
weights(2) = (3 * t^3 - 6 * t^2 + 4) / 6;
% Calculate the third basis function B2.
weights(3) = (-3 * t^3 + 3 * t^2 + 3 * t + 1) / 6;
% Calculate the fourth basis function B3.
weights(4) = t^3 / 6;
% End the function definition.
end