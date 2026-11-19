% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Hyperparameters (Matched to Segmentation Script) ---
% Define the diffusion parameter mu.
mu = 0.1;
% Define the gradient weighting parameter.
gradientWeight = 1 - mu;
% Define the time step for iterative convergence.
timeStep = 0.2;
% Define the number of iterations (Increased for better diffusion).
numIterations = 2000;

% --- Data Loading ---
% Load the input image and convert to double precision.
inputImageRGB = double(imread("FixedReg.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);
% Get the dimensions of the input image.
[imageHeight, imageWidth] = size(inputImage);

% --- Preprocessing (Added Gaussian Blur) ---
% Use a moderate blur to remove noise but keep edges.
blurSigma = 2.0;
kernelSize = 2 * ceil(3 * blurSigma) + 1;
kernelCenter = (kernelSize - 1) / 2;
gaussianKernel = zeros(kernelSize, kernelSize);
for i = 1:1:kernelSize
    for j = 1:1:kernelSize
        distanceSquared = ((i - 1 - kernelCenter)^2 + (j - 1 - kernelCenter)^2);
        gaussianKernel(i, j) = exp(-distanceSquared / (2 * blurSigma^2));
    end
end
gaussianKernel = gaussianKernel / sum(sum(gaussianKernel));
blurredImage = conv2(inputImage, gaussianKernel, "same");

% Normalize the image to the range [0, 1].
normalizedImage = blurredImage / max(blurredImage(:));

% --- Calculate Edge Map & Gradient (Added Sobel Operators) ---
% Use Sobel operators for stronger edge detection.
sobelX = [-1 0 1; -2 0 2; -1 0 1];
sobelY = sobelX';
% Calculate the gradient in the X direction.
gradientX = conv2(normalizedImage, sobelX, "same");
% Calculate the gradient in the Y direction.
gradientY = conv2(normalizedImage, sobelY, "same");
% Calculate the squared gradient magnitude.
gradientMagnitudeSquared = gradientX.^2 + gradientY.^2;

% --- Initialize the GVF Vector Field ---
% Initialize the X component of the GVF field to the raw gradient.
gvfU = gradientX;
% Initialize the Y component of the GVF field to the raw gradient.
gvfV = gradientY;

% --- Define the Laplacian Kernel for Diffusion ---
% Define the 3x3 discrete Laplacian operator.
laplacianKernel = [0, 1, 0; 1, -4, 1; 0, 1, 0];

% --- GVF Iterative Solver ---
fprintf("Computing GVF Field...\n");
% Loop through the iterations.
for iter = 1:1:numIterations
    % Calculate the Laplacian of the current U field.
    laplacianU = conv2(gvfU, laplacianKernel, "same");
    % Calculate the Laplacian of the current V field.
    laplacianV = conv2(gvfV, laplacianKernel, "same");

    % Calculate the attraction term for U.
    attractionU = gradientMagnitudeSquared .* (gvfU - gradientX);
    % Calculate the attraction term for V.
    attractionV = gradientMagnitudeSquared .* (gvfV - gradientY);

    % Update the U field using the Euler-Lagrange equation.
    gvfU = gvfU + timeStep * (mu * laplacianU - gradientWeight * attractionU);
    % Update the V field using the Euler-Lagrange equation.
    gvfV = gvfV + timeStep * (mu * laplacianV - gradientWeight * attractionV);

    % Print progress every 500 iterations.
    if (mod(iter, 500) == 0)
        fprintf("GVF Iteration %d/%d completed.\n", iter, numIterations);
    end
end
fprintf("GVF Field Computed.\n");

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");
% Display the normalized image.
imshow(normalizedImage, []);
hold on;

% Define a stride to make arrows readable (Full resolution is too dense).
stride = 15;
[rowGrid, colGrid] = meshgrid(1:stride:imageWidth, 1:stride:imageHeight);

% Extract downsampled fields for plotting.
gx_sub = gradientX(1:stride:end, 1:stride:end);
gy_sub = gradientY(1:stride:end, 1:stride:end);
gu_sub = gvfU(1:stride:end, 1:stride:end);
gv_sub = gvfV(1:stride:end, 1:stride:end);

% Plot the raw gradient arrows in red.
quiver(colGrid(:), rowGrid(:), gx_sub(:), gy_sub(:), 5, "r");
% Plot the GVF arrows in blue.
quiver(colGrid(:), rowGrid(:), gu_sub(:), gv_sub(:), 5, "b");

% Add a title to the figure.
title("Raw Gradient (Red) vs. GVF Field (Blue)");
% Release the hold on the plot.
hold off;