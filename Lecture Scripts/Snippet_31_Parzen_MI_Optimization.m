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

% --- Optimization Setup ---
% Define the initial guess for the rotation angle in degrees.
initialGuess = -60;
% Set the optimization options to display iteration details.
options = optimset("Display", "iter");

% Run fminsearch to find the angle that minimizes the negative MI.
[optimalAngle, minNegMI] = fminsearch(@(theta) CalculateNegMI(theta, fixedImage, movingImage), initialGuess, options);

% Display the results in the command window.
fprintf("Estimated Rotation Angle: %.4f degrees\n", optimalAngle);

% --- Final Visualization ---
% Convert the optimal angle from degrees to radians.
finalAngleRad = pi * optimalAngle / 180;
% Construct the final rotation matrix using the optimal angle.
finalRotMatrix = [cos(finalAngleRad), sin(finalAngleRad); -sin(finalAngleRad), cos(finalAngleRad)];
% Apply inverse mapping to generate the final registered image.
registeredImage = InverseMapFn(movingImage, 0, 0, inv(finalRotMatrix));

% Convert the registered image back to uint8 for saving.
registeredImageUint8 = uint8(registeredImage);
% Save the registered image to disk.
imwrite(registeredImageUint8, "RegisteredImage.jpg");

% Open a new figure window to display the registration results.
figure("Color", "w");
% Create the first subplot for the fixed reference image.
subplot(1, 3, 1);
% Display the fixed reference image.
imshow(fixedImage, []);
% Add a title to the first subplot.
title("Fixed Image (Reference)");
% Create the second subplot for the misaligned moving image.
subplot(1, 3, 2);
% Display the misaligned moving image.
imshow(movingImage, []);
% Add a title to the second subplot.
title("Misaligned Image (Target)");
% Create the third subplot for the final registered image.
subplot(1, 3, 3);
% Display the final registered image.
imshow(registeredImage, []);
% Add a title to the third subplot.
title(sprintf("Registered Image (Angle: %.2f°)", optimalAngle));

% --- Local Function Definitions ---

% Define the function to calculate the negative Mutual Information using Parzen smoothing.
function negMI = CalculateNegMI(theta, fixedImage, movingImage)
% Convert the angle from degrees to radians.
angleRad = pi * theta / 180;
% Construct the rotation matrix for the current guess.
rotMatrix = [cos(angleRad), sin(angleRad); -sin(angleRad), cos(angleRad)];

% Warp the moving image using inverse mapping.
warpedImage = InverseMapFn(movingImage, 0, 0, inv(rotMatrix));

% Get the dimensions of the images.
[imageHeight, imageWidth] = size(fixedImage);

% Initialize the joint histogram matrix (assuming 8-bit images, 0-255).
jointHist = zeros(256, 256);

% Loop through each row to build the joint histogram.
for r = 1:1:imageHeight
    % Loop through each column to build the joint histogram.
    for c = 1:1:imageWidth
        % Get the intensity from the fixed image.
        valFixed = fixedImage(r, c);
        % Get the intensity from the warped moving image.
        valMoving = warpedImage(r, c);

        % Clamp values to valid histogram indices (1 to 256).
        idxFixed = max(1, min(256, round(valFixed) + 1));
        % Clamp the moving index to valid histogram bounds.
        idxMoving = max(1, min(256, round(valMoving) + 1));

        % Increment the joint histogram bin.
        jointHist(idxFixed, idxMoving) = jointHist(idxFixed, idxMoving) + 1;
        % End the column loop.
    end
    % End the row loop.
end

% Convert the joint histogram to a joint probability distribution.
totalPixels = imageHeight * imageWidth;
% Calculate the raw joint probability matrix.
jointProb = jointHist / totalPixels;

% Apply the Parzen Window smoothing to create a continuous joint density.
jointDensity = ParzenWindowSmoothing(jointProb, 10);

% Calculate the marginal probabilities by summing across rows and columns.
probFixed = sum(jointDensity, 2);
% Calculate the marginal probability for the moving image.
probMoving = sum(jointDensity, 1);

% Calculate the marginal entropies H(X) and H(Y).
% Add a small epsilon to avoid log(0) errors.
epsilon = 1e-10;
% Calculate the entropy for the fixed image.
entropyFixed = -sum(probFixed .* log2(probFixed + epsilon));
% Calculate the entropy for the moving image.
entropyMoving = -sum(probMoving .* log2(probMoving + epsilon));

% Calculate the joint entropy H(X, Y) using the smoothed density.
jointEntropy = -sum(sum(jointDensity .* log2(jointDensity + epsilon)));

% Calculate the Mutual Information.
mi = entropyFixed + entropyMoving - jointEntropy;

% Return the negative MI because fminsearch minimizes the objective function.
negMI = -mi;
% End the function definition.
end

% Define the function to apply Parzen Window smoothing to the joint probability.
function jointDensity = ParzenWindowSmoothing(jointProb, windowSize)
% Initialize the smoothed joint density matrix.
jointDensity = zeros(256, 256);
% Define the normalization constant for the 2D Gaussian.
normalizationConstant = 1.0 / (2.0 * pi);

% Loop through each possible intensity pair (q1, q2).
for q1 = 0:1:255
    % Loop through each second intensity value.
    for q2 = 0:1:255
        % Initialize the sum for Parzen window accumulation.
        sumParzen = 0;

        % Define the integer window boundaries.
        halfWindow = round(windowSize / 3);
        startK1 = max(1, q1 - halfWindow);
        endK1 = min(256, q1 + halfWindow);
        startK2 = max(1, q2 - halfWindow);
        endK2 = min(256, q2 + halfWindow);

        % Loop through the neighborhood defined by the window.
        for k1 = startK1:1:endK1
            % Loop through the second dimension of the neighborhood.
            for k2 = startK2:1:endK2
                % Get the probability from the raw joint histogram.
                probOccurrence = jointProb(k1, k2);
                % Define the mean vector for the current center position.
                mu = [q1; q2];
                % Define the query vector for the neighbor.
                q = [k1 - 1; k2 - 1];
                % Calculate the Mahalanobis distance (using identity covariance).
                part002 = (q - mu)' * inv([1, 0; 0, 1]) * (q - mu);
                % Accumulate the Parzen window contribution.
                sumParzen = sumParzen + probOccurrence * normalizationConstant * exp(-part002 / 2);
                % End the k2 loop.
            end
            % End the k1 loop.
        end

        % Assign the accumulated smoothed sum to the joint density matrix.
        jointDensity(q1 + 1, q2 + 1) = sumParzen;
        % End the q2 loop.
    end
    % End the q1 loop.
end

% Normalize the joint density to ensure it sums to 1.
jointDensity = jointDensity / sum(sum(jointDensity));
% End the function definition.
end

% Define the function for inverse mapping.
function warpedImage = InverseMapFn(inputImage, shiftX, shiftY, invTransformMatrix)
% Get the height and width of the input image.
[imageHeight, imageWidth] = size(inputImage);
% Initialize the new image matrix with zeros.
newImage = zeros(imageHeight, imageWidth);

% Loop through each row of the destination image.
for r = 1:1:imageHeight
    % Loop through each column of the destination image.
    for c = 1:1:imageWidth
        % Center the destination row coordinate.
        centerR = r - imageHeight / 2 - shiftX;
        % Center the destination column coordinate.
        centerC = c - imageWidth / 2 - shiftY;
        % Create a destination coordinate vector.
        destVec = [centerR, centerC];
        % Apply the inverse transformation matrix to find the source coordinates.
        srcVec = destVec * invTransformMatrix;
        % Calculate the source row index.
        srcR = round(srcVec(1, 1) + imageHeight / 2);
        % Calculate the source column index.
        srcC = round(srcVec(1, 2) + imageWidth / 2);

        % Clamp the source row index to the image bounds.
        srcR = max(1, min(imageHeight, srcR));
        % Clamp the source column index to the image bounds.
        srcC = max(1, min(imageWidth, srcC));

        % Pull the pixel intensity from the source image to the destination.
        newImage(r, c) = inputImage(srcR, srcC);
        % End the column loop.
    end
    % End the row loop.
end
% Assign the new image to the output variable.
warpedImage = newImage;
% End the function definition.
end

% Define the function for forward mapping (used to create misalignment).
function warpedImage = ForwardMapFn(inputImage, shiftX, shiftY, transformMatrix)
% Get the height and width of the input image.
[imageHeight, imageWidth] = size(inputImage);
% Initialize the new image matrix with zeros.
newImage = zeros(imageHeight, imageWidth);

% Loop through each row of the input image.
for r = 1:1:imageHeight
    % Loop through each column of the input image.
    for c = 1:1:imageWidth
        % Center the row coordinate around the image origin.
        centerR = r - imageHeight / 2;
        % Center the column coordinate around the image origin.
        centerC = c - imageWidth / 2;
        % Create a source coordinate vector.
        srcVec = [centerR, centerC];
        % Apply the transformation matrix to the source vector.
        destVec = srcVec * transformMatrix;
        % Calculate the destination row index.
        destR = round(destVec(1, 1) + imageHeight / 2 + shiftX);
        % Calculate the destination column index.
        destC = round(destVec(1, 2) + imageWidth / 2 + shiftY);

        % Check if the destination coordinates are within image bounds.
        if (destR >= 1 && destR <= imageHeight && destC >= 1 && destC <= imageWidth)
            % Map the original pixel intensity to the new transformed coordinates.
            newImage(destR, destC) = inputImage(r, c);
            % End the bounds check.
        end
        % End the column loop.
    end
    % End the row loop.
end
% Assign the new image to the output variable.
warpedImage = newImage;
% End the function definition.
end