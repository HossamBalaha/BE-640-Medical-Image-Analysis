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

% Load the registered moving image and convert to double precision.
registeredImageRGB = double(imread("RegisteredImage.jpg"));
% Extract the first channel to create a grayscale image.
registeredImage = registeredImageRGB(:, :, 1);

% --- Visualization Method 1: Edge Overlay ---
% Open a new figure window for the edge overlay.
figure("Color", "w");
% Display the fixed reference image.
imshow(fixedImage, []);
% Hold the current plot to overlay the edges.
hold on;
% Detect the edges of the registered image using the Prewitt operator.
edgeImage = edge(registeredImage, "prewitt");
% Find the row and column indices of the detected edge pixels.
[edgeRows, edgeCols, ~] = find(edgeImage == 1);
% Plot the edge pixels in red over the fixed image.
plot(edgeCols, edgeRows, ".r", "MarkerSize", 2);
% Add a title to the edge overlay figure.
title("Method 1: Edge Overlay Visualization");
% Release the hold on the plot.
hold off;

% --- Visualization Method 2: Checkerboard ---
% Open a new figure window for the checkerboard visualization.
figure("Color", "w");
% Get the dimensions of the images.
[imageHeight, imageWidth] = size(fixedImage);
% Define the size of each checkerboard square.
squareSize = 32;
% Initialize the checkerboard visualization image.
checkerboardImage = zeros(imageHeight, imageWidth);

% Loop through each row of the image.
for r = 1:1:imageHeight
    % Loop through each column of the image.
    for c = 1:1:imageWidth
        % Determine the checkerboard grid coordinates.
        gridR = floor((r - 1) / squareSize);
        gridC = floor((c - 1) / squareSize);

        % Check if the sum of grid coordinates is even (white square).
        if (mod(gridR + gridC, 2) == 0)
            % Assign the pixel from the fixed image.
            checkerboardImage(r, c) = fixedImage(r, c);
        else
            % Assign the pixel from the registered image.
            checkerboardImage(r, c) = registeredImage(r, c);
            % End the checkerboard parity check.
        end
        % End the column loop.
    end
    % End the row loop.
end

% Display the generated checkerboard image.
imshow(checkerboardImage, []);
% Add a title to the checkerboard figure.
title("Method 2: Checkerboard Visualization");