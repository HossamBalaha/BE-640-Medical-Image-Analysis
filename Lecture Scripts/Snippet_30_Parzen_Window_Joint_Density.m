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

% Ensure both images have the same dimensions for joint histogram calculation.
[imageHeight, imageWidth] = size(fixedImage);
% Resize the moving image to match the fixed image if necessary.
movingImage = imresize(movingImage, [imageHeight, imageWidth]);

% --- Calculate the Real Joint Histogram ---
% Initialize the 256x256 joint histogram matrix.
jointHistogram = zeros(256, 256);

% Loop through each row of the images.
for r = 1:1:imageHeight
    % Loop through each column of the images.
    for c = 1:1:imageWidth
        % Get the intensity from the fixed image and clamp to 1-256.
        idxFixed = max(1, min(256, round(fixedImage(r, c)) + 1));
        % Get the intensity from the moving image and clamp to 1-256.
        idxMoving = max(1, min(256, round(movingImage(r, c)) + 1));
        % Increment the joint histogram bin.
        jointHistogram(idxFixed, idxMoving) = jointHistogram(idxFixed, idxMoving) + 1;
        % End the column loop.
    end
    % End the row loop.
end

% Normalize the joint histogram to sum to 1.
jointHistogram = jointHistogram / sum(sum(jointHistogram));

% --- Parzen Window Smoothing Parameters ---
% Define the Parzen window size (controls the smoothing radius).
windowSize = 10;
% Define the normalization constant for 2D Gaussian.
normalizationConstant = 1.0 / (2.0 * pi);
% Initialize the smoothed joint density matrix.
jointDensity = zeros(256, 256);

% --- Parzen Window Smoothing ---
% Loop through each possible intensity pair (q1, q2).
for q1 = 0:1:255
    % Loop through each second intensity value.
    for q2 = 0:1:255
        % Initialize the sum for Parzen window accumulation.
        sumParzen = 0;

        % Define the window boundaries (FIXED: ensure integer indices).
        halfWindow = round(windowSize / 3);
        startK1 = max(1, q1 - halfWindow);
        endK1 = min(256, q1 + halfWindow);
        startK2 = max(1, q2 - halfWindow);
        endK2 = min(256, q2 + halfWindow);

        % Loop through the neighborhood defined by the window.
        for k1 = startK1:1:endK1
            % Loop through the second dimension of the neighborhood.
            for k2 = startK2:1:endK2
                % Get the occurrence count from the real joint histogram.
                noOccurrence = jointHistogram(k1, k2);
                % Define the mean vector for the current position.
                mu = [q1; q2];
                % Define the query vector.
                q = [k1 - 1; k2 - 1];
                % Calculate the Mahalanobis distance (using identity covariance).
                part002 = (q - mu)' * inv([1, 0; 0, 1]) * (q - mu);
                % Accumulate the Parzen window contribution.
                sumParzen = sumParzen + noOccurrence * normalizationConstant * exp(-part002 / 2);
                % End the k2 loop.
            end
            % End the k1 loop.
        end

        % Assign the accumulated sum to the joint density matrix.
        jointDensity(q1 + 1, q2 + 1) = sumParzen;
        % End the q2 loop.
    end
    % End the q1 loop.
end

% Normalize the joint density to ensure it sums to 1.
jointDensity = jointDensity / sum(sum(jointDensity));

% --- Visualization ---
% Open a new figure window.
figure("Color", "w");

% Display the original real joint histogram.
subplot(1, 2, 1);
% Use imagesc to automatically scale the small probability values for visibility.
imagesc(jointHistogram);
% Add a colorbar to show the probability scale.
colorbar;
% Add a title.
title("Real Joint Histogram (Noisy/Sparse)");
% Set the colormap to jet for better contrast.
colormap jet;

% Display the smoothed joint density.
subplot(1, 2, 2);
% Use imagesc for the smoothed density.
imagesc(jointDensity);
% Add a colorbar.
colorbar;
% Add a title.
title("Smoothed Joint Density (Parzen Window)");
% Set the colormap to jet.
colormap jet;

% Display a completion message in the command window.
fprintf("Parzen Window smoothing completed on real data. Window size: %d\n", windowSize);