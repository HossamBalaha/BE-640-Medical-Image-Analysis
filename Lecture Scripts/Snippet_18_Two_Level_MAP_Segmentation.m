% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the number of Gaussian classes for the mixture model.
numClasses = 2;
% Define the number of EM iterations.
numEmIterations = 50;
% Initialize the EM means with reasonable starting guesses.
gaussianMeans = [100, 200];
% Initialize the EM variances with reasonable starting guesses.
gaussianVariances = [1000, 1000];
% Initialize the EM class priors.
classPriors = [0.5, 0.5];

% Define the single-site clique potential parameter.
beta0 = -2;
% Define the horizontal pair clique potential parameter.
betaH = -2;
% Define the vertical pair clique potential parameter.
betaV = -2;
% Define the number of MAP iterations.
numIterations = 100;

% --- Data Loading ---
% Load the real test CT image and convert to double precision.
testImage = double(imread("LungCtSlice.tif"));
% Load the expert ground truth mask.
rawMask = imread("LungCtMask.tif");

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

% --- Intensity Model Estimation (EM Algorithm) ---
% Flatten the image to run the EM algorithm for intensity model estimation.
imageVector = testImage(:);

% Run a fixed number of EM iterations to estimate the intensity model parameters.
for emIter = 1:1:numEmIterations
    % Initialize the responsibilities matrix for the E-step.
    responsibilities = zeros(length(imageVector), numClasses);
    % Loop through each pixel to calculate responsibilities.
    for px = 1:1:length(imageVector)
        % Loop through each class to calculate the weighted likelihood.
        for cl = 1:1:numClasses
            responsibilities(px, cl) = classPriors(cl) .* (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(cl))) .* exp(-((imageVector(px) - gaussianMeans(cl)).^2) ./ (2.0 .* gaussianVariances(cl)));
            % End the class loop.
        end
        % Normalize the responsibilities so they sum to one.
        responsibilities(px, :) = responsibilities(px, :) ./ sum(responsibilities(px, :));
        % End the pixel loop.
    end

    % Loop through each class to update parameters in the M-step.
    for cl = 1:1:numClasses
        % Calculate the effective number of pixels assigned to the current class.
        Nk = sum(responsibilities(:, cl));
        % Update the class prior probability.
        classPriors(cl) = Nk ./ length(imageVector);
        % Update the class mean using the weighted average.
        gaussianMeans(cl) = sum(responsibilities(:, cl) .* imageVector) ./ Nk;
        % Update the class variance using the weighted variance.
        gaussianVariances(cl) = sum(responsibilities(:, cl) .* ((imageVector - gaussianMeans(cl)).^2)) ./ Nk;
        % End the class parameter update loop.
    end
    % End the EM iteration loop.
end

% Define the array of gray levels from 0 to 255.
grayLevels = 0:1:255;
% Initialize the lung likelihood array for all gray levels.
pLung = zeros(1, 256);
% Initialize the chest likelihood array for all gray levels.
pChest = zeros(1, 256);

% Determine which estimated class corresponds to Lung (typically the lower mean).
if (gaussianMeans(1) < gaussianMeans(2))
    lungClassIdx = 1;
    chestClassIdx = 2;
else
    lungClassIdx = 2;
    chestClassIdx = 1;
    % End the class identification check.
end

% Loop through each gray level to calculate the intensity likelihoods.
for qIdx = 1:1:256
    % Extract the current gray level value.
    q = grayLevels(qIdx);
    % Calculate the Gaussian likelihood for the Lung class.
    pLung(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(lungClassIdx))) .* exp(-((q - gaussianMeans(lungClassIdx)).^2) ./ (2.0 .* gaussianVariances(lungClassIdx)));
    % Calculate the Gaussian likelihood for the Chest class.
    pChest(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(chestClassIdx))) .* exp(-((q - gaussianMeans(chestClassIdx)).^2) ./ (2.0 .* gaussianVariances(chestClassIdx)));
    % End the gray level loop.
end

% --- Initial Segmentation (Intensity Model Only) ---
% Create an initial noisy segmentation using the Bayes decision rule.
initialSegmentation = false(size(testImage));
% Loop through each row of the image for initial segmentation.
for rowIndex = 1:1:size(testImage, 1)
    % Loop through each column of the image for initial segmentation.
    for colIndex = 1:1:size(testImage, 2)
        % Extract the gray level of the current pixel.
        q = testImage(rowIndex, colIndex);
        % Map the gray level to the array index (1 to 256).
        qIdx = round(q) + 1;
        % Clamp the index to valid bounds.
        if (qIdx < 1)
            qIdx = 1;
            % End the lower bound check.
        end
        if (qIdx > 256)
            qIdx = 256;
            % End the upper bound check.
        end
        % Assign to Lung (true) if the lung posterior score is greater than the chest posterior score.
        if ((pLung(qIdx) .* classPriors(lungClassIdx)) > (pChest(qIdx) .* classPriors(chestClassIdx)))
            initialSegmentation(rowIndex, colIndex) = true;
            % End the Bayes decision check.
        end
        % End the column loop.
    end
    % End the row loop.
end

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

% --- MAP Refinement (Spatial + Intensity Models) ---
% Convert the logical initial segmentation to double for the MAP algorithm.
refinedImage = double(initialSegmentation);
% Get the dimensions of the image for the loops.
[imageRows, imageCols] = size(refinedImage);

% Start the Two-Level MAP Segmentation refinement loop.
for iterationIndex = 1:1:numIterations
    % Create a copy of the refined image for synchronous updating.
    refinedImageNew = refinedImage;
    % Loop through each row excluding the boundaries.
    for rowIndex = 2:1:imageRows - 1
        % Loop through each column excluding the boundaries.
        for colIndex = 2:1:imageCols - 1
            % Extract the gray level of the current pixel.
            q = testImage(rowIndex, colIndex);
            % Map the gray level to the array index.
            qIdx = round(q) + 1;
            % Clamp the index to valid bounds.
            if (qIdx < 1)
                qIdx = 1;
                % End the lower bound check.
            end
            if (qIdx > 256)
                qIdx = 256;
                % End the upper bound check.
            end

            % Extract the intensity likelihoods for the current pixel.
            pL = pLung(qIdx);
            pC = pChest(qIdx);

            % Calculate the local spatial energy if the pixel is assigned Label 0 (Chest).
            energyLabel0 = GetEnergyValueMAP(refinedImage, 0, rowIndex, colIndex, beta0, betaH, betaV);
            % Calculate the local spatial energy if the pixel is assigned Label 1 (Lung).
            energyLabel1 = GetEnergyValueMAP(refinedImage, 1, rowIndex, colIndex, beta0, betaH, betaV);

            % Convert spatial energies to spatial probabilities using the Gibbs distribution.
            probLabel0 = exp(-energyLabel0) ./ (exp(-energyLabel0) + exp(-energyLabel1));
            probLabel1 = exp(-energyLabel1) ./ (exp(-energyLabel0) + exp(-energyLabel1));

            % Calculate the joint MAP scores for both classes (Intensity × Spatial).
            scoreChest = pC .* probLabel0;
            scoreLung = pL .* probLabel1;

            % MAP Decision: Assign the label that maximizes the joint score.
            if (scoreLung > scoreChest)
                % Assign Label 1 (Lung) if it has a higher joint score.
                refinedImageNew(rowIndex, colIndex) = 1;
            else
                % Assign Label 0 (Chest) if it has a higher or equal joint score.
                refinedImageNew(rowIndex, colIndex) = 0;
                % End the MAP label assignment check.
            end
            % End the column loop.
        end
        % End the row loop.
    end
    % Update the refined image with the new labels for the next iteration.
    refinedImage = refinedImageNew;
    % Display the current iteration number.
    fprintf("Completed MAP iteration %d.\n", iterationIndex);
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
fprintf("Refined MAP Segmentation Dice = %.4f\n", diceRefined);

% --- Visualization ---
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
% Display the initial Bayes-based segmentation.
imshow(initialSegmentation);
% Add a title including the initial Dice score.
title(sprintf("Initial Noisy Segmentation\nDice = %.4f", diceInitial));

% Create the third subplot for the refined MAP segmentation.
subplot(1, 4, 3);
% Display the refined MAP segmentation.
imshow(refinedSegmentation);
% Add a title including the refined Dice score.
title(sprintf("Refined MAP Segmentation\nDice = %.4f", diceRefined));

% Create the fourth subplot for the expert ground truth.
subplot(1, 4, 4);
% Display the expert ground truth mask.
imshow(groundTruthMask);
% Add a title to the ground truth subplot.
title("Expert Ground Truth Mask");

% --- Local Function Definition ---
% Define the function to calculate the local spatial energy of a pixel configuration.
function energyValue = GetEnergyValueMAP(currentImage, targetLabel, rowIndex, colIndex, beta0, betaH, betaV)
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

% Sum all energy components to get the total local spatial energy.
energyValue = energySingle + energyHorizontal + energyVertical;
% End the function definition.
end