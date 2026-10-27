% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Data Loading ---
% Load the transformed image saved by the previous script.
inputImageRGB = double(imread("TransformedImage.jpg"));
% Extract the first channel to create a grayscale image.
inputImage = inputImageRGB(:, :, 1);

% --- Interpolation & Smoothing ---
% Define a 3x3 averaging window for interpolation.
window = (1.0 / 9.0) * ones(3, 3);
% Apply 2D convolution to smooth the image and fill mapping holes.
outputImage = conv2(inputImage, window);

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");
% Display the interpolated and smoothed output image.
imshow(outputImage, []);
% Add a title to the figure.
title("Interpolated and Smoothed Image");