% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Load the real test CT image and convert to double precision.
testImage = double(imread("LungCtSlice.tif"));
% Load the expert ground truth mask.
rawMask = imread("LungCtMask.tif");

% Define the single-site clique potential parameter.
beta0 = -1;
% Define the horizontal pair clique potential parameter.
betaH = -1;
% Define the vertical pair clique potential parameter.
betaV = -1;
% Define the number of ICM iterations.
numIterations = 100;

% Check if the mask is an RGB image with three color channels.
if (size(rawMask, 3) == 3)
    % Extract the blue color channel from the mask.
    blueChannel = rawMask(:, :, 3);
    % Extract the red color channel from the mask.
    redChannel = rawMask(:, :, 1);
    % Extract the green color channel from the mask.
    greenChannel = rawMask(:, :, 2);
    % Create a logical mask for the blue regions representing the lungs.
    groundTruthMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
else
    % Create a logical mask for the lungs assuming white pixels.
    groundTruthMask = rawMask > 128;
    % End the mask type check.
end

% Create an initial noisy segmentation using a simple intensity threshold.
% Lung tissue is typically darker in CT scans, so we threshold below 128.
initialSegmentation = testImage < 128;

% Calculate the True Positives for the initial segmentation.
truePositivesInitial = sum(initialSegmentation(:) & groundTruthMask(:));
% Calculate the False Positives for the initial segmentation.
falsePositivesInitial = sum(initialSegmentation(:) & ~groundTruthMask(:));
% Calculate the False Negatives for the initial segmentation.
falseNegativesInitial = sum(~initialSegmentation(:) & groundTruthMask(:));

% Calculate the Dice coefficient for the initial noisy segmentation.
diceInitial = (2.0 .* truePositivesInitial) ./ (2.0 .* truePositivesInitial + falsePositivesInitial + falseNegativesInitial);

% Display the initial Dice coefficient in the command window.
fprintf("Initial Noisy Segmentation Dice = %.4f\n", diceInitial);

% Convert the logical initial segmentation to double for the ICM algorithm.
refinedImage = double(initialSegmentation);
% Get the dimensions of the image for the loops.
[imageRows, imageCols] = size(refinedImage);

% Start the Iterative Conditional Modes (ICM) refinement loop.
for iterationIndex = 1:1:numIterations
    % Loop through each row excluding the boundaries.
    for rowIndex = 2:1:imageRows - 1
        % Loop through each column excluding the boundaries.
        for colIndex = 2:1:imageCols - 1
            % Calculate the local energy if the pixel is assigned Label 0.
            energyLabel0 = GetEnergyValueICM(refinedImage, 0, rowIndex, colIndex, beta0, betaH, betaV);
            % Calculate the local energy if the pixel is assigned Label 1.
            energyLabel1 = GetEnergyValueICM(refinedImage, 1, rowIndex, colIndex, beta0, betaH, betaV);

            % ICM Decision: Assign the label that minimizes the local energy (maximizes probability).
            if (energyLabel0 < energyLabel1)
                % Assign Label 0 if it has lower energy.
                refinedImage(rowIndex, colIndex) = 0;
            else
                % Assign Label 1 if it has lower or equal energy.
                refinedImage(rowIndex, colIndex) = 1;
                % End the ICM label assignment check.
            end
            % End the column loop.
        end
        % End the row loop.
    end
    % Display the current iteration number.
    fprintf("Completed ICM iteration %d.\n", iterationIndex);
    % End the iteration loop.
end

% Convert the refined double image back to a logical mask.
refinedSegmentation = refinedImage == 1;

% Calculate the True Positives for the refined segmentation.
truePositivesRefined = sum(refinedSegmentation(:) & groundTruthMask(:));
% Calculate the False Positives for the refined segmentation.
falsePositivesRefined = sum(refinedSegmentation(:) & ~groundTruthMask(:));
% Calculate the False Negatives for the refined segmentation.
falseNegativesRefined = sum(~refinedSegmentation(:) & groundTruthMask(:));

% Calculate the Dice coefficient for the refined segmentation.
diceRefined = (2.0 .* truePositivesRefined) ./ (2.0 .* truePositivesRefined + falsePositivesRefined + falseNegativesRefined);

% Display the refined Dice coefficient in the command window.
fprintf("Refined ICM Segmentation Dice = %.4f\n", diceRefined);

% Open a new figure window to display the results.
figure("Color", "w");

% Create the first subplot for the original test image.
subplot(1, 4, 1);
% Display the original CT image.
imshow(testImage, []);
% Add a title to the original image subplot.
title("Original CT Image");

% Create the second subplot for the initial noisy segmentation.
subplot(1, 4, 2);
% Display the initial threshold-based segmentation.
imshow(initialSegmentation);
% Add a title including the initial Dice score.
title(sprintf("Initial Noisy Segmentation\nDice = %.4f", diceInitial));

% Create the third subplot for the refined ICM segmentation.
subplot(1, 4, 3);
% Display the refined ICM segmentation.
imshow(refinedSegmentation);
% Add a title including the refined Dice score.
title(sprintf("Refined ICM Segmentation\nDice = %.4f", diceRefined));

% Create the fourth subplot for the expert ground truth.
subplot(1, 4, 4);
% Display the expert ground truth mask.
imshow(groundTruthMask);
% Add a title to the ground truth subplot.
title("Expert Ground Truth Mask");

% --- Local Function Definition ---
% Define the function to calculate the local energy of a pixel configuration.
function energyValue = GetEnergyValueICM(currentImage, targetLabel, rowIndex, colIndex, beta0, betaH, betaV)
% Initialize the single-site energy component.
energySingle = 0;
% Check if the target label matches the current pixel label.
if (targetLabel == currentImage(rowIndex, colIndex))
    % Assign the single-site potential if they match.
    energySingle = beta0;
    % End the single-site match check.
end

% Initialize the horizontal pair energy component.
energyHorizontal = 0;
% Check the right neighbor for a label match.
if (targetLabel == currentImage(rowIndex, colIndex + 1))
    % Add the horizontal potential for the right neighbor.
    energyHorizontal = energyHorizontal + betaH;
    % End the right neighbor check.
end
% Check the left neighbor for a label match.
if (targetLabel == currentImage(rowIndex, colIndex - 1))
    % Add the horizontal potential for the left neighbor.
    energyHorizontal = energyHorizontal + betaH;
    % End the left neighbor check.
end

% Initialize the vertical pair energy component.
energyVertical = 0;
% Check the bottom neighbor for a label match.
if (targetLabel == currentImage(rowIndex + 1, colIndex))
    % Add the vertical potential for the bottom neighbor.
    energyVertical = energyVertical + betaV;
    % End the bottom neighbor check.
end
% Check the top neighbor for a label match.
if (targetLabel == currentImage(rowIndex - 1, colIndex))
    % Add the vertical potential for the top neighbor.
    energyVertical = energyVertical + betaV;
    % End the top neighbor check.
end

% Sum all energy components to get the total local energy.
energyValue = energySingle + energyHorizontal + energyVertical;
% End the function definition.
end