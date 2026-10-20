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
numEmIterations = 100;
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
numMapIterations = 100;

% --- Data Loading Paths ---
% Define the test image directory path.
testImageDir = "./../Dataset/Sample1/test/images";
% Define the test mask directory path.
testMaskDir = "./../Dataset/Sample1/test/masks";
% Define the training mask directory path.
trainingMaskDir = "./../Dataset/Sample1/train/masks";

% Get the list of test image files.
testImageFiles = dir(fullfile(testImageDir, "*.jpg"));
% Check if test image files were found.
if (isempty(testImageFiles))
    % Fallback to .jpeg if no .jpg files are found.
    testImageFiles = dir(fullfile(testImageDir, "*.jpeg"));
    % End the fallback check.
end

% Get the list of test mask files.
testMaskFiles = dir(fullfile(testMaskDir, "*.jpg"));
% Check if test mask files were found.
if (isempty(testMaskFiles))
    % Fallback to .jpeg if no .jpg files are found.
    testMaskFiles = dir(fullfile(testMaskDir, "*.jpeg"));
    % End the fallback check.
end

% Get the list of training mask files for shape model generation.
trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpg"));
% Check if training mask files were found.
if (isempty(trainingMaskFiles))
    % Fallback to .jpeg if no .jpg files are found.
    trainingMaskFiles = dir(fullfile(trainingMaskDir, "*.jpeg"));
    % End the fallback check.
end

% --- Shape Probability Map Generation ---
% Get the actual number of training masks available.
numTrainingMasks = length(trainingMaskFiles);
% Initialize the accumulator for the shape probability map as empty.
shapeProbabilityMap = [];

% Loop through each training mask to accumulate the binary shapes.
for imgIndex = 1:1:numTrainingMasks
    % Construct the full path to the current training mask.
    currentMaskPath = fullfile(trainingMaskDir, trainingMaskFiles(imgIndex).name);
    % Read the current expert mask.
    rawMask = imread(currentMaskPath);
    % Check if the mask is an RGB image.
    if (size(rawMask, 3) == 3)
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Create a logical mask for the blue regions.
        binaryMask = double((blueChannel > 128) & (redChannel < 128) & (greenChannel < 128));
    else
        % Create a logical mask assuming white pixels.
        binaryMask = double(rawMask > 128);
        % End the mask type check.
    end

    % Initialize the accumulator on the first iteration using the actual image dimensions.
    if (isempty(shapeProbabilityMap))
        % Get the dimensions of the current binary mask.
        [imageRows, imageCols] = size(binaryMask);
        % Initialize the accumulator with zeros.
        shapeProbabilityMap = zeros(imageRows, imageCols);
        % End the first iteration check.
    end

    % Add the current binary mask to the accumulator.
    shapeProbabilityMap = shapeProbabilityMap + binaryMask;
    % End the training mask loop.
end

% Calculate the final probability map.
shapeProbabilityMap = shapeProbabilityMap ./ numTrainingMasks;

% --- Testing and Evaluation Loop ---
% Get the number of test files available.
numTestFiles = length(testImageFiles);

% Loop through each test image.
for fileIndex = 1:1:numTestFiles
    % Construct the full path to the current test image.
    currentImagePath = fullfile(testImageDir, testImageFiles(fileIndex).name);
    % Read the current test image and convert to double precision.
    testImage = double(imread(currentImagePath));
    % Check if the input image is an RGB image.
    if (size(testImage, 3) == 3)
        % Convert the RGB image to a grayscale image.
        testImage = double(rgb2gray(uint8(testImage)));
        % End the RGB check.
    end

    % Construct the full path to the current test mask.
    currentMaskPath = fullfile(testMaskDir, testMaskFiles(fileIndex).name);
    % Read the expert ground truth mask.
    rawMask = imread(currentMaskPath);
    % Check if the mask is an RGB image.
    if (size(rawMask, 3) == 3)
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Create a logical mask for the blue regions.
        groundTruthMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
    else
        % Create a logical mask assuming white pixels.
        groundTruthMask = rawMask > 128;
        % End the mask type check.
    end

    % --- Intensity Model Estimation (EM Algorithm) ---
    % Flatten the image to run the EM algorithm.
    imageVector = testImage(:);
    % Run a fixed number of EM iterations.
    for emIter = 1:1:numEmIterations
        % Initialize the responsibilities matrix for the E-step.
        responsibilities = zeros(length(imageVector), numClasses);
        % Loop through each pixel to calculate responsibilities.
        for px = 1:1:length(imageVector)
            % Loop through each class to calculate the weighted likelihood.
            for cl = 1:1:numClasses
                % Calculate the Gaussian likelihood.
                responsibilities(px, cl) = classPriors(cl) .* (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(cl))) .* exp(-((imageVector(px) - gaussianMeans(cl)).^2) ./ (2.0 .* gaussianVariances(cl)));
                % End the class loop.
            end
            % Normalize the responsibilities.
            responsibilities(px, :) = responsibilities(px, :) ./ sum(responsibilities(px, :));
            % End the pixel loop.
        end
        % Loop through each class to update parameters in the M-step.
        for cl = 1:1:numClasses
            % Calculate the effective number of pixels.
            Nk = sum(responsibilities(:, cl));
            % Update the class prior probability.
            classPriors(cl) = Nk ./ length(imageVector);
            % Update the class mean.
            gaussianMeans(cl) = sum(responsibilities(:, cl) .* imageVector) ./ Nk;
            % Update the class variance.
            gaussianVariances(cl) = sum(responsibilities(:, cl) .* ((imageVector - gaussianMeans(cl)).^2)) ./ Nk;
            % End the class update loop.
        end
        % End the EM iteration loop.
    end

    % Define the array of gray levels.
    grayLevels = 0:1:255;
    % Initialize the lung likelihood array.
    pLung = zeros(1, 256);
    % Initialize the chest likelihood array.
    pChest = zeros(1, 256);
    % Determine which class is Lung based on the mean.
    if (gaussianMeans(1) < gaussianMeans(2))
        lungClassIdx = 1;
        chestClassIdx = 2;
    else
        lungClassIdx = 2;
        chestClassIdx = 1;
        % End the class identification check.
    end
    % Loop through each gray level to calculate likelihoods.
    for qIdx = 1:1:256
        % Extract the current gray level value.
        q = grayLevels(qIdx);
        % Calculate the Gaussian likelihood for the Lung class.
        pLung(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(lungClassIdx))) .* exp(-((q - gaussianMeans(lungClassIdx)).^2) ./ (2.0 .* gaussianVariances(lungClassIdx)));
        % Calculate the Gaussian likelihood for the Chest class.
        pChest(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(chestClassIdx))) .* exp(-((q - gaussianMeans(chestClassIdx)).^2) ./ (2.0 .* gaussianVariances(chestClassIdx)));
        % End the gray level loop.
    end

    % --- Initial Segmentation (Intensity Only) ---
    % Get image dimensions.
    [imageRows, imageCols] = size(testImage);
    % Initialize the initial segmentation mask.
    initialSegmentation = false(imageRows, imageCols);
    % Loop through each row for initial segmentation.
    for rowIndex = 1:1:imageRows
        % Loop through each column.
        for colIndex = 1:1:imageCols
            % Extract the gray level.
            q = testImage(rowIndex, colIndex);
            % Map to array index.
            qIdx = round(q) + 1;
            % Clamp index to lower bound.
            if (qIdx < 1)
                qIdx = 1;
                % End the lower bound check.
            end
            % Clamp index to upper bound.
            if (qIdx > 256)
                qIdx = 256;
                % End the upper bound check.
            end
            % Apply Bayes decision rule.
            if ((pLung(qIdx) .* classPriors(lungClassIdx)) > (pChest(qIdx) .* classPriors(chestClassIdx)))
                % Assign the pixel to the lung class.
                initialSegmentation(rowIndex, colIndex) = true;
                % End the Bayes decision check.
            end
            % End the column loop.
        end
        % End the row loop.
    end

    % --- Shape-Aware MAP Refinement Loop ---
    % Convert to double for the MAP algorithm.
    refinedImage = double(initialSegmentation);
    % Start the MAP refinement loop.
    for iterationIndex = 1:1:numMapIterations
        % Create a copy for synchronous updating.
        refinedImageNew = refinedImage;
        % Loop through rows excluding boundaries.
        for rowIndex = 2:1:imageRows - 1
            % Loop through columns excluding boundaries.
            for colIndex = 2:1:imageCols - 1
                % Extract the gray level.
                q = testImage(rowIndex, colIndex);
                % Map to array index.
                qIdx = round(q) + 1;
                % Clamp index to lower bound.
                if (qIdx < 1)
                    qIdx = 1;
                    % End the lower bound check.
                end
                % Clamp index to upper bound.
                if (qIdx > 256)
                    qIdx = 256;
                    % End the upper bound check.
                end

                % Extract intensity likelihoods.
                pL = pLung(qIdx);
                pC = pChest(qIdx);

                % Calculate spatial energy for Label 0 (Chest).
                energyLabel0 = GetEnergyValueMAP(refinedImage, 0, rowIndex, colIndex, beta0, betaH, betaV);
                % Calculate spatial energy for Label 1 (Lung).
                energyLabel1 = GetEnergyValueMAP(refinedImage, 1, rowIndex, colIndex, beta0, betaH, betaV);

                % Convert spatial energies to probabilities.
                probLabel0 = exp(-energyLabel0) ./ (exp(-energyLabel0) + exp(-energyLabel1));
                probLabel1 = exp(-energyLabel1) ./ (exp(-energyLabel0) + exp(-energyLabel1));

                % Extract the shape probabilities for the current pixel.
                pShapeOrgan = shapeProbabilityMap(rowIndex, colIndex);
                pShapeBackground = 1.0 - shapeProbabilityMap(rowIndex, colIndex);

                % Calculate joint scores (Intensity * Spatial * Shape).
                scoreChest = pC .* probLabel0 .* pShapeBackground;
                scoreLung = pL .* probLabel1 .* pShapeOrgan;

                % Shape-Aware MAP Decision.
                if (scoreLung > scoreChest)
                    % Assign Label 1 (Lung).
                    refinedImageNew(rowIndex, colIndex) = 1;
                else
                    % Assign Label 0 (Chest).
                    refinedImageNew(rowIndex, colIndex) = 0;
                    % End the MAP label assignment check.
                end
                % End the column loop.
            end
            % End the row loop.
        end
        % Update the refined image.
        refinedImage = refinedImageNew;
        % End the MAP iteration loop.
    end

    % Convert back to logical mask.
    refinedSegmentation = refinedImage == 1;

    % --- Evaluation Metrics ---
    % Calculate True Positives.
    truePositives = sum(refinedSegmentation(:) & groundTruthMask(:));
    % Calculate False Positives.
    falsePositives = sum(refinedSegmentation(:) & ~groundTruthMask(:));
    % Calculate False Negatives.
    falseNegatives = sum(~refinedSegmentation(:) & groundTruthMask(:));
    % Calculate Dice coefficient.
    dice = (2.0 .* truePositives) ./ (2.0 .* truePositives + falsePositives + falseNegatives);

    % Display results.
    fprintf("=== Test Image %d Evaluation ===\n", fileIndex);
    % Display Dice.
    fprintf("Shape-Aware MAP Dice = %.4f\n", dice);

    % --- Visualization ---
    % Open a new figure window.
    figure("Color", "w");
    % Create the first subplot.
    subplot(2, 3, 1);
    % Display the original test image.
    imshow(testImage, []);
    % Add a title.
    title("Original Test Image");
    % Create the second subplot.
    subplot(2, 3, 2);
    % Display the initial segmentation.
    imshow(initialSegmentation);
    % Add a title.
    title("Initial Noisy Segmentation");
    % Create the third subplot.
    subplot(2, 3, 3);
    % Display the refined segmentation.
    imshow(refinedSegmentation);
    % Add a title.
    title(sprintf("Refined Shape-Aware MAP\nDice = %.4f", dice));
    % Create the fourth subplot.
    subplot(2, 3, 4);
    % Display the shape probability map.
    imshow(shapeProbabilityMap, []);
    % Add a colorbar.
    colorbar;
    % Add a title.
    title("Shape Probability Map");
    % Create the fifth subplot.
    subplot(2, 3, 5);
    % Display the expert ground truth.
    imshow(groundTruthMask);
    % Add a title.
    title("Expert Ground Truth");
    % End the test image loop.
end

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