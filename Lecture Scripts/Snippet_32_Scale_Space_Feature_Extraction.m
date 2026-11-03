% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the size of the Gaussian kernel.
kernelSize = 256;
% Define the variance parameter (sigma squared) for the Gaussian.
variance = kernelSize;
% Calculate the amplitude normalization factor.
amplitude = 1.0 / sqrt(2.0 * pi * variance);
% Calculate the center of the kernel for the rows.
muI = kernelSize / 2.0;
% Calculate the center of the kernel for the columns.
muJ = kernelSize / 2.0;

% --- Generate 2D Gaussian Kernel ---
% Initialize the Gaussian kernel matrix.
gaussianKernel = zeros(kernelSize, kernelSize);

% Loop through each row of the kernel.
for i = 1:1:kernelSize
    % Loop through each column of the kernel.
    for j = 1:1:kernelSize
        % Calculate the squared distance from the center.
        distanceSquared = ((i - muI).^2 + (j - muJ).^2) / (2.0 * variance);
        % Calculate the Gaussian value at this position.
        gaussianKernel(i, j) = amplitude * exp(-distanceSquared);
        % End the column loop.
    end
    % End the row loop.
end

% Normalize the kernel so that it sums to 1.
gaussianKernel = gaussianKernel / sum(sum(gaussianKernel));

% --- Data Loading ---
% Load the input image for scale space feature extraction.
inputImageRGB = double(imread("RegExample.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);

% --- Apply Scale Space Transformation ---
% Apply 2D convolution with the Gaussian kernel.
convolvedImage = conv2(inputImage, gaussianKernel);
% Get the dimensions of the convolved image.
[convHeight, convWidth] = size(convolvedImage);
% Get the dimensions of the original image.
[origHeight, origWidth] = size(inputImage);

% Crop the center to match the original image dimensions.
startRow = floor((convHeight - origHeight) / 2.0) + 1;
endRow = startRow + origHeight - 1;
startCol = floor((convWidth - origWidth) / 2.0) + 1;
endCol = startCol + origWidth - 1;
scaleSpaceImage = convolvedImage(startRow:endRow, startCol:endCol);

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");
% Display the original image.
subplot(1, 3, 1);
imshow(inputImage, []);
title("Original Image");
% Display the Gaussian kernel.
subplot(1, 3, 2);
imshow(gaussianKernel, []);
title("2D Gaussian Kernel");
colorbar;
% Display the scale space feature image.
subplot(1, 3, 3);
imshow(scaleSpaceImage, []);
title("Scale Space Feature Extraction");

fprintf("Scale space feature extraction completed. Kernel size: %d x %d\n", kernelSize, kernelSize);