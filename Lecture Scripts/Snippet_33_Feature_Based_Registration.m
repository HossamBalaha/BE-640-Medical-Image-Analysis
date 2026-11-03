% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Data Loading ---
% Load the fixed reference image and convert to double precision.
fixedImageRGB = double(imread("FixedReg.jpg"));
% Extract the first channel to create a grayscale image.
fixedImage = fixedImageRGB(:, :, 1);

% Load the moving target image and convert to double precision.
movingImageRGB = double(imread("MovingReg2.jpg"));
% Extract the first channel to create a grayscale image.
movingImage = movingImageRGB(:, :, 1);

% Ensure both images have the same dimensions.
[imageHeight, imageWidth] = size(fixedImage);
% Resize the moving image to match the fixed image if necessary.
movingImage = imresize(movingImage, [imageHeight, imageWidth]);

% --- Feature Template Extraction ---
% Define the size of the feature template window.
windowSize = 350;
% Define the center coordinates of the landmark in the fixed image.
centerRow = round(imageHeight / 2);
% Define the center column coordinate of the landmark.
centerCol = round(imageWidth / 2);

% Calculate the half window size for indexing.
halfWindow = floor(windowSize / 2);
% Extract the template window from the fixed image.
templateWindow = fixedImage(centerRow - halfWindow : centerRow + halfWindow, centerCol - halfWindow : centerCol + halfWindow);

% --- Rotation Search using Cross-Correlation ---
% Define the range of angles to search (e.g., -90 to 90 degrees).
angles = -90:1:90;
% Initialize variables to track the best match.
maxCorrelation = -inf;
% Initialize the best rotation angle.
bestAngle = 0;

% Loop through each candidate rotation angle.
for angle = angles
    % Rotate the template window by the current angle.
    % Use 'bilinear' interpolation and 'crop' to keep the same size.
    rotatedTemplate = imrotate(templateWindow, angle, "bilinear", "crop");

    % Extract the corresponding search window from the moving image (at the center).
    searchWindow = movingImage(centerRow - halfWindow : centerRow + halfWindow, centerCol - halfWindow : centerCol + halfWindow);

    % Calculate the mean of the rotated template.
    meanRotatedTemplate = mean(rotatedTemplate(:));
    % Calculate the mean of the search window.
    meanSearch = mean(searchWindow(:));

    % Center the windows by subtracting their means.
    centeredRotatedTemplate = rotatedTemplate - meanRotatedTemplate;
    % Center the search window.
    centeredSearch = searchWindow - meanSearch;

    % Calculate the numerator of the cross-correlation formula.
    numerator = sum(sum(centeredRotatedTemplate .* centeredSearch));
    % Calculate the denominator of the cross-correlation formula.
    denominator = sqrt(sum(sum(centeredRotatedTemplate.^2)) * sum(sum(centeredSearch.^2)));

    % Check if the denominator is greater than zero to avoid division by zero.
    if (denominator > 0)
        % Calculate the normalized cross-correlation score.
        correlationScore = numerator / denominator;

        % Check if the current score is the maximum found so far.
        if (correlationScore > maxCorrelation)
            % Update the maximum correlation score.
            maxCorrelation = correlationScore;
            % Update the best rotation angle.
            bestAngle = angle;
            % End the maximum score check.
        end
        % End the denominator check.
    end
    % End the angle loop.
end

% --- Apply the Best Rotation to the Moving Image ---
% Rotate the entire moving image by the negative of the best angle to align it.
registeredImage = imrotate(movingImage, -bestAngle, "bilinear", "crop");

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");

% Display the fixed image with the template window.
subplot(1, 3, 1);
% Show the fixed reference image.
imshow(fixedImage, []);
% Hold the plot to draw the template rectangle.
hold on;
% Draw a red rectangle around the template window in the fixed image.
rectangle("Position", [centerCol - halfWindow, centerRow - halfWindow, windowSize, windowSize], "EdgeColor", "r", "LineWidth", 2);
% Add a title to the first subplot.
title("Fixed Image (Template)");
% Release the hold on the plot.
hold off;

% Display the original moving image.
subplot(1, 3, 2);
% Show the moving target image.
imshow(movingImage, []);
% Hold the plot to draw the search rectangle.
hold on;
% Draw a green rectangle around the search window in the moving image.
rectangle("Position", [centerCol - halfWindow, centerRow - halfWindow, windowSize, windowSize], "EdgeColor", "g", "LineWidth", 2);
% Add a title to the second subplot.
title("Moving Image (Original)");
% Release the hold on the plot.
hold off;

% Display the registered moving image.
subplot(1, 3, 3);
% Show the registered image.
imshow(registeredImage, []);
% Add a title to the third subplot.
title(sprintf("Registered Image (Window Size: %d; Rotated: %.1f°)", windowSize, -bestAngle));

% Display the results in the command window.
fprintf("Feature-Based Rotation Registration completed.\n");
% Display the maximum correlation score.
fprintf("Maximum Cross-Correlation: %.4f\n", maxCorrelation);
% Display the best rotation angle.
fprintf("Best Rotation Angle: %.1f degrees\n", bestAngle);