% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the rotation angle in degrees.
thetaAngle = 45;
% Convert the angle to radians.
angleRadians = pi * thetaAngle / 180;
% Define the horizontal translation shift.
shiftX = 50;
% Define the vertical translation shift.
shiftY = 10;
% Define the horizontal scaling factor.
scaleX = 1.25;
% Define the vertical scaling factor.
scaleY = 1.25;

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

% --- Data Loading ---
% Load the input image and convert to double precision.
inputImageRGB = double(imread("RegExample.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);

% --- Apply Affine Transformation ---
% Call the local function to perform the affine transformation.
transformedImage = AffineTransformFn(inputImage, shiftX, shiftY, transformMatrix);

% --- Save the Transformed Image ---
% Convert the transformed image back to uint8 for saving.
transformedImageUint8 = uint8(transformedImage);
% Save the transformed image to disk for the next script.
imwrite(transformedImageUint8, "TransformedImage.jpg");

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
% Display the affine transformed image.
imshow(transformedImage, []);
% Add a title to the second subplot.
title("Affine Transformed Image");

% --- Local Function Definition ---
% Define the function to perform the affine transformation.
function transformedImage = AffineTransformFn(inputImage, shiftX, shiftY, transformMatrix)
% Get the height and width of the input image.
[imageHeight, imageWidth] = size(inputImage);
% Initialize the new image matrix with zeros.
newImage = zeros(imageHeight, imageWidth);

% Loop through each row of the input image.
for rowIndex = 1:1:imageHeight
    % Loop through each column of the input image.
    for colIndex = 1:1:imageWidth
        % Center the row coordinate around the image origin.
        centerRow = rowIndex - imageHeight / 2;
        % Center the column coordinate around the image origin.
        centerCol = colIndex - imageWidth / 2;

        % Create a coordinate vector for the current pixel.
        coordVector = [centerRow, centerCol];
        % Apply the transformation matrix to the coordinate vector.
        newCoordVector = round(coordVector * transformMatrix);

        % Calculate the new row index and add the shift and center offset.
        newRow = round(newCoordVector(1, 1) + imageHeight / 2 + shiftX);
        % Calculate the new column index and add the shift and center offset.
        newCol = round(newCoordVector(1, 2) + imageWidth / 2 + shiftY);

        % Clamp the new row index to the lower bound.
        if (newRow < 1)
            % Set the new row index to 1.
            newRow = 1;
            % End the lower bound check.
        end
        % Clamp the new row index to the upper bound.
        if (newRow > imageHeight)
            % Set the new row index to the image height.
            newRow = imageHeight;
            % End the upper bound check.
        end
        % Clamp the new column index to the lower bound.
        if (newCol < 1)
            % Set the new column index to 1.
            newCol = 1;
            % End the lower bound check.
        end
        % Clamp the new column index to the upper bound.
        if (newCol > imageWidth)
            % Set the new column index to the image width.
            newCol = imageWidth;
            % End the upper bound check.
        end

        % Map the original pixel intensity to the new transformed coordinates.
        newImage(newRow, newCol) = inputImage(rowIndex, colIndex);
        % End the column loop.
    end
    % End the row loop.
end

% Assign the new image to the output variable.
transformedImage = newImage;
% End the function definition.
end