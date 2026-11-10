% The file name is Snippet_35_Active_Contour_Energy.m.
% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Hyperparameters ---
% Define the blur sigma parameter for Gaussian filtering.
blurSigma = 5;
% Define the number of points in the snake contour.
numPoints = 250;
% Define the internal force weight (elasticity/smoothing).
alpha = 0.1;
% Define the external force weight (edge attraction).
beta = 2.0;
% Define the balloon force weight (inflation).
balloonForce = 3.5;
% Define the number of iterations for the snake evolution.
numIterations = 1000;

% --- Data Loading ---
% Load the input image and convert to double precision.
inputImageRGB = double(imread("FixedReg.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);
% Get the dimensions of the input image.
[imageHeight, imageWidth] = size(inputImage);

% --- Create Gaussian Kernel Manually ---
% Define the kernel size (should be odd and approximately 6*sigma).
kernelSize = 2 * ceil(3 * blurSigma) + 1;
% Calculate the center of the kernel.
kernelCenter = (kernelSize - 1) / 2;
% Initialize the Gaussian kernel matrix.
gaussianKernel = zeros(kernelSize, kernelSize);

% Loop through each row of the kernel.
for i = 1:1:kernelSize
    % Loop through each column of the kernel.
    for j = 1:1:kernelSize
        % Calculate the squared distance from the center.
        distanceSquared = ((i - 1 - kernelCenter)^2 + (j - 1 - kernelCenter)^2);
        % Calculate the Gaussian value at this position.
        gaussianKernel(i, j) = exp(-distanceSquared / (2 * blurSigma^2));
        % End the column loop.
    end
    % End the row loop.
end

% Normalize the kernel so that it sums to 1.
gaussianKernel = gaussianKernel / sum(sum(gaussianKernel));

% Apply the Gaussian filter using convolution.
blurredImage = conv2(inputImage, gaussianKernel, "same");

% Calculate the gradient on the BLURRED image.
gradientX = conv2(blurredImage, [-1, 1], "same");
gradientY = conv2(blurredImage, [-1; 1], "same");

% Use positive gradient to pull snake toward bright regions (chest wall).
externalForceX = gradientX;
externalForceY = gradientY;

% --- Snake Initialization ---
% Define the center X coordinate (inside the left lung).
centerX = round(imageWidth / 3);
% Define the center Y coordinate.
centerY = round(imageHeight / 2);
% Define the initial radius.
initialRadius = 40;
% Initialize the X coordinates of the snake points.
snakeX = zeros(1, numPoints);
% Initialize the Y coordinates of the snake points.
snakeY = zeros(1, numPoints);

% Loop to initialize the snake as a circle.
for k = 1:1:numPoints
    % Calculate the angle for the current point.
    theta = 2 * pi * (k - 1) / numPoints;
    % Calculate the initial X coordinate.
    snakeX(k) = centerX + initialRadius * cos(theta);
    % Calculate the initial Y coordinate.
    snakeY(k) = centerY + initialRadius * sin(theta);
    % End the initialization loop.
end

% Store the initial snake coordinates for visualization.
initialSnakeX = snakeX;
% Store the initial Y coordinates.
initialSnakeY = snakeY;

% Initialize an array to store the total energy at each iteration.
totalEnergyHistory = zeros(1, numIterations);

% --- Snake Evolution Loop ---
for iter = 1:1:numIterations
    % Initialize the new snake coordinates.
    newSnakeX = zeros(1, numPoints);
    newSnakeY = zeros(1, numPoints);
    % Initialize the current iteration energy.
    currentEnergy = 0;

    % Calculate the centroid of the snake for the Balloon Force direction.
    centroidX = mean(snakeX);
    centroidY = mean(snakeY);

    % Loop through each point in the snake.
    for k = 1:1:numPoints
        % Get the previous and next point indices (wrap around).
        prevIdx = mod(k - 2, numPoints) + 1;
        nextIdx = mod(k, numPoints) + 1;

        % --- Internal Force (Laplacian Smoothing) ---
        % This force shrinks the snake and keeps it smooth.
        internalForceX = snakeX(prevIdx) - 2 * snakeX(k) + snakeX(nextIdx);
        internalForceY = snakeY(prevIdx) - 2 * snakeY(k) + snakeY(nextIdx);

        % --- Balloon Force (Inflation) ---
        % Calculate the outward normal vector (from centroid to point).
        dx = snakeX(k) - centroidX;
        dy = snakeY(k) - centroidY;
        dist = sqrt(dx^2 + dy^2);

        % Normalize to get the direction.
        if (dist > 0)
            balloonX = dx / dist;
            balloonY = dy / dist;
        else
            balloonX = 0;
            balloonY = 0;
        end

        % --- External Force (Edge Attraction) ---
        % Get the current point coordinates rounded to integers.
        currX = round(snakeX(k));
        currY = round(snakeY(k));

        % Clamp coordinates to image boundaries.
        if (currX < 1)
            currX = 1;
        end
        if (currX > imageWidth)
            currX = imageWidth;
        end
        if (currY < 1)
            currY = 1;
        end
        if (currY > imageHeight)
            currY = imageHeight;
        end

        % Extract the external force at the current point.
        % This pulls the snake toward the nearest high-gradient edge.
        extX = externalForceX(currY, currX);
        extY = externalForceY(currY, currX);

        % --- Update Coordinates ---
        % New Position = Old + Internal(Shrink) + Balloon(Expand) + External(Attract to Edge)
        newSnakeX(k) = snakeX(k) + alpha * internalForceX + balloonForce * balloonX + beta * extX;
        newSnakeY(k) = snakeY(k) + alpha * internalForceY + balloonForce * balloonY + beta * extY;

        % Calculate energy proxy for plotting.
        internalEnergy = abs(internalForceX) + abs(internalForceY);
        externalEnergy = abs(extX) + abs(extY);
        currentEnergy = currentEnergy + internalEnergy + externalEnergy;
        % End the point loop.
    end

    % Update the snake coordinates.
    snakeX = newSnakeX;
    snakeY = newSnakeY;
    % Store the energy history.
    totalEnergyHistory(iter) = currentEnergy;
    % End the iteration loop.
end

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");
% Display the BLURRED image.
subplot(1, 2, 1);
imshow(blurredImage, []);
hold on;
% Plot the initial snake in green (appended first point to close the loop).
plot([initialSnakeX, initialSnakeX(1)], [initialSnakeY, initialSnakeY(1)], "g--", "LineWidth", 2);
% Plot the final evolved snake in red (appended first point to close the loop).
plot([snakeX, snakeX(1)], [snakeY, snakeY(1)], "r-", "LineWidth", 2);
title("Snake Evolution (Blurred View)");
legend("Initial Snake", "Final Snake");
hold off;

% Display the original image with the snake overlay.
subplot(1, 2, 2);
imshow(inputImage, []);
hold on;
% Plot the initial snake in green (appended first point to close the loop).
plot([initialSnakeX, initialSnakeX(1)], [initialSnakeY, initialSnakeY(1)], "g--", "LineWidth", 2);
% Plot the final evolved snake in red (appended first point to close the loop).
plot([snakeX, snakeX(1)], [snakeY, snakeY(1)], "r-", "LineWidth", 2);
title("Snake on Original Image");
legend("Initial Snake", "Final Snake");
hold off;

% Open energy plot.
figure("Color", "w");
plot(1:numIterations, totalEnergyHistory, "b-", "LineWidth", 2);
title("Total Force Magnitude (Energy Proxy)");
xlabel("Iteration");
ylabel("Total Force");
grid on;