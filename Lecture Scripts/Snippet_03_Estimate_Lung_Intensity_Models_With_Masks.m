% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Read the lung image and convert it to double precision.
doubleImage = double(imread("LungCtSlice.tif"));
% Read the single binary mask where white labels the lung and black labels the chest.
maskImage = double(imread("LungCtMask.tif"));
% Convert the white lung region to a logical array.
lungMask = maskImage > 0;
% Derive the black chest region as the logical complement of the lung mask.
chestMask = ~lungMask;
% Check whether the mask matches the size of the lung image.
if (~isequal(size(doubleImage), size(lungMask)))
    % Stop the script with a clear error message.
    error("The mask must have the same size as the lung image.");
% End the size-check conditional block.
end
% Gather the lung training samples inside the white mask region.
lungPixels = doubleImage(lungMask);
% Gather the chest training samples inside the black mask region.
chestPixels = doubleImage(chestMask);
% Estimate the mean of the lung class.
lungMean = mean(lungPixels);
% Estimate the variance of the lung class.
lungVariance = var(lungPixels);
% Estimate the mean of the chest class.
chestMean = mean(chestPixels);
% Estimate the variance of the chest class.
chestVariance = var(chestPixels);
% Summarize the estimated factors in a readable table.
factorTable = table(["Lung"; "Chest"], [numel(lungPixels); numel(chestPixels)], [lungMean; chestMean], [lungVariance; chestVariance], VariableNames=["Tissue", "PixelCount", "Mean", "Variance"]);
% Display the factor table in the command window.
disp(factorTable);
% Define the gray-level axis for the model curves.
grayLevels = 0:1:255;
% Compute the Gaussian probability density of the lung class.
lungPdf = (1 ./ sqrt(2 .* pi .* lungVariance)) .* exp(-((grayLevels - lungMean).^2) ./ (2 .* lungVariance));
% Compute the Gaussian probability density of the chest class.
chestPdf = (1 ./ sqrt(2 .* pi .* chestVariance)) .* exp(-((grayLevels - chestMean).^2) ./ (2 .* chestVariance));
% Open a new figure window to inspect the single mask and its complement.
figure("Color", "w");
% Create the first subplot for the original image.
subplot(1, 3, 1);
% Display the lung CT slice as a grayscale image.
imagesc(doubleImage);
% Apply a grayscale colormap to the figure.
colormap(gray);
% Preserve the correct aspect ratio of the image.
axis image;
% Add a title to the image subplot.
title("Lung CT Slice");
% Create the second subplot for the white lung region.
subplot(1, 3, 2);
% Display the white lung region of the binary mask.
imagesc(lungMask);
% Preserve the correct aspect ratio of the mask.
axis image;
% Add a title to the lung mask subplot.
title("Lung Mask (White, Class 1)");
% Create the third subplot for the black chest region.
subplot(1, 3, 3);
% Display the derived black chest region of the mask.
imagesc(chestMask);
% Preserve the correct aspect ratio of the mask.
axis image;
% Add a title to the chest mask subplot.
title("Chest Mask (Black, Class 2)");
% Open a new figure window for the model plot.
figure("Color", "w");
% Plot the lung Gaussian model in red.
plot(grayLevels, lungPdf, "r", "LineWidth", 2);
% Hold the plot to overlay the chest model.
hold on;
% Plot the chest Gaussian model in blue.
plot(grayLevels, chestPdf, "b", "LineWidth", 2);
% Add a legend for the two tissue models.
legend("Lung Tissue", "Chest Tissue");
% Add a descriptive title to the model plot.
title("Gaussian Intensity Models for Lung and Chest Tissues");
% Label the horizontal axis of the model plot.
xlabel("Gray Level");
% Label the vertical axis of the model plot.
ylabel("Probability Density");