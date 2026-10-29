% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the rotation angle in degrees.
thetaAngle = 30;
% Convert the angle to radians.
angleRadians = pi * thetaAngle / 180;
% Define the horizontal translation shift.
shiftX = 20;
% Define the vertical translation shift.
shiftY = 20;
% Define the horizontal scaling factor.
scaleX = 1.0;
% Define the vertical scaling factor.
scaleY = 1.0;

% --- Transformation Matrix Construction ---
% Initialize the 2x2 transformation matrix.
transformMatrix = zeros(2, 2);
% Assign the top-left element for scaled rotation.
transformMatrix(1, 1) = scaleX * cos(angleRadians);
% Assign the top-right element for scaled rotation.
transformMatrix(1, 2) = scaleY * sin(angleRadians);
% Assign the bottom-left element for scaled rotation.
transformMatrix(2, 1) = -scaleX * sin(angleRadians);
% Assign the bottom-right element for scaled rotation.
transformMatrix(2, 2) = scaleY * cos(angleRadians);

% Calculate the inverse of the transformation matrix for backward mapping.
invTransformMatrix = inv(transformMatrix);

% --- Data Loading ---
% Load the input image and convert to double precision.
inputImageRGB = double(imread("RegExample.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);

% --- Apply Inverse Mapping ---
% Call the local function to perform the inverse affine transformation.
transformedImage = InverseMappingFn(inputImage, shiftX, shiftY, invTransformMatrix);

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");
% Create the first subplot for the original image.
subplot(1, 2, 1);
% Display the original input image.
imshow(inputImage, []);
% Add a title to the first subplot.
title("Original Image");
% Create the second subplot for the transformed image.
subplot(1, 2, 2);
% Display the inverse mapped image.
imshow(transformedImage, []);
% Add a title to the second subplot.
title("Inverse Mapped Image (No Holes)");

% --- Local Function Definition ---
% Define the function to perform the inverse affine transformation.
function transformedImage = InverseMappingFn(inputImage, shiftX, shiftY, invTransformMatrix)
% Get the height and width of the input image.
[imageHeight, imageWidth] = size(inputImage);
% Initialize the new image matrix with zeros.
newImage = zeros(imageHeight, imageWidth);

% Loop through each row of the new destination image.
for newRow = 1:1:imageHeight
    % Loop through each column of the new destination image.
    for newCol = 1:1:imageWidth
        % Center the new row coordinate around the image origin.
        centerNewRow = newRow - imageHeight / 2 - shiftX;
        % Center the new column coordinate around the image origin.
        centerNewCol = newCol - imageWidth / 2 - shiftY;

        % Create a coordinate vector for the current destination pixel.
        destCoordVector = [centerNewRow, centerNewCol];
        % Apply the inverse transformation matrix to find the source coordinates.
        srcCoordVector = destCoordVector * invTransformMatrix;

        % Calculate the source row index and add the center offset.
        srcRow = round(srcCoordVector(1, 1) + imageHeight / 2);
        % Calculate the source column index and add the center offset.
        srcCol = round(srcCoordVector(1, 2) + imageWidth / 2);

        % Clamp the source row index to the lower bound.
        if (srcRow < 1)
            % Set the source row index to 1.
            srcRow = 1;
            % End the lower bound check.
        end
        % Clamp the source row index to the upper bound.
        if (srcRow > imageHeight)
            % Set the source row index to the image height.
            srcRow = imageHeight;
            % End the upper bound check.
        end
        % Clamp the source column index to the lower bound.
        if (srcCol < 1)
            % Set the source column index to 1.
            srcCol = 1;
            % End the lower bound check.
        end
        % Clamp the source column index to the upper bound.
        if (srcCol > imageWidth)
            % Set the source column index to the image width.
            srcCol = imageWidth;
            % End the upper bound check.
        end

        % Pull the pixel intensity from the source image to the destination.
        newImage(newRow, newCol) = inputImage(srcRow, srcCol);
        % End the column loop.
    end
    % End the row loop.
end

% Assign the new image to the output variable.
transformedImage = newImage;
% End the function definition.
end