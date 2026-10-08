% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the maximum number of images to test.
numImagesToTest = 3;
% Define the boolean flag to use static, manually defined MGRF parameters.
useStaticParams = false;
% Define the static MGRF parameters (used if useStaticParams is true).
staticBeta0 = -1;
staticBetaH = -1;
staticBetaV = -1;
% Define the number of MAP refinement iterations.
numMapIterations = 100;

% --- Data Loading Paths ---
% Define the base dataset path.
datasetPath = "./../Dataset/Sample2";

% Check if the base dataset path exists; if not, stop execution.
if (~exist(datasetPath, "dir"))
    % Throw an error if the dataset path is missing.
    error("Dataset path does not exist: %s.\nPlease check the path and ensure the dataset is downloaded.", datasetPath);
    % End the path existence check.
end

% Define the subdirectories for testing images and masks.
testImageDir = fullfile(datasetPath, "test", "images");
testMaskDir = fullfile(datasetPath, "test", "masks");
% Define the output directory for saving the generated figures.
outputDir = fullfile(datasetPath, "test", "outputs");

% Check if testing subdirectories exist; if not, stop execution.
if (~exist(testImageDir, "dir"))
    % Throw an error if the test images directory is missing.
    error("Testing images directory does not exist. Please check the dataset structure.");
    % End the test images directory check.
end
if (~exist(testMaskDir, "dir"))
    % Throw an error if the test masks directory is missing.
    error("Testing masks directory does not exist. Please check the dataset structure.");
    % End the test masks directory check.
end

% Create the output directory if it does not exist.
if (~exist(outputDir, "dir"))
    % Create the missing output directory.
    mkdir(outputDir);
    % End the output directory existence check.
end

% Get all testing image files (supports .jpg, .jpeg, and .png).
allTestImageFiles = dir(fullfile(testImageDir, "*.jpg"));
if (isempty(allTestImageFiles))
    % Fallback to .jpeg if no .jpg files are found.
    allTestImageFiles = dir(fullfile(testImageDir, "*.jpeg"));
    % End the .jpeg fallback check.
end
if (isempty(allTestImageFiles))
    % Fallback to .png if no .jpeg files are found.
    allTestImageFiles = dir(fullfile(testImageDir, "*.png"));
    % End the .png fallback check.
end

% ROBUST CHECK: Ensure files were actually found.
if (isempty(allTestImageFiles))
    % Throw an error if no image files are found in the directory.
    error("No image files (.jpg, .jpeg, or .png) found in directory:\n%s", testImageDir);
    % End the empty files check.
end

% Sort the files to ensure consistent ordering.
allTestImageNames = sort({allTestImageFiles.name});
numTestFiles = length(allTestImageNames);
% Display the number of testing files found.
fprintf("Found %d total testing image files.\n", numTestFiles);

% Generate corresponding mask names for the testing set.
testMaskNames = cell(1, numTestFiles);
for i = 1:1:numTestFiles
    % Extract the current image name.
    imgName = allTestImageNames{i};
    % Find the last underscore to insert "_mask" before the slice number.
    lastUnderscore = find(imgName == '_', 1, 'last');
    if (~isempty(lastUnderscore))
        % Construct the mask name by injecting "_mask".
        testMaskNames{i} = [imgName(1:lastUnderscore - 1), '_mask', imgName(lastUnderscore:end)];
    else
        % Fallback if no underscore is found.
        testMaskNames{i} = imgName;
        % End the underscore check.
    end
    % End the mask name generation loop.
end

% --- Load the Frozen Model ---
% Load the trained model parameters.
load("Project02FrozenMAPModel.mat");
% Extract intensity parameters.
lungMean = modelParameters.LungMean;
lungVariance = modelParameters.LungVariance;
lungPrior = modelParameters.LungPrior;
chestMean = modelParameters.ChestMean;
chestVariance = modelParameters.ChestVariance;
chestPrior = modelParameters.ChestPrior;

% Determine the final beta parameters based on the static flag.
if (useStaticParams)
    % Use the manually defined static parameters.
    beta0 = staticBeta0;
    betaH = staticBetaH;
    betaV = staticBetaV;
    % Display a message indicating static parameters are being used.
    fprintf("Testing with static MGRF parameters: Beta0=%.4f, BetaH=%.4f, BetaV=%.4f\n", beta0, betaH, betaV);
else
    % Extract the estimated spatial parameters from the loaded model.
    beta0 = modelParameters.Beta0;
    betaH = modelParameters.BetaH;
    betaV = modelParameters.BetaV;
    % Display a message indicating estimated parameters are being used.
    fprintf("Testing with estimated MGRF parameters: Beta0=%.4f, BetaH=%.4f, BetaV=%.4f\n", beta0, betaH, betaV);
end

% Define the array of gray levels.
grayLevels = 0:1:255;
% Initialize the lung likelihood array.
pLung = zeros(1, 256);
% Initialize the chest likelihood array.
pChest = zeros(1, 256);

% Loop through each gray level to calculate likelihoods.
for qIdx = 1:1:256
    % Extract the current gray level value.
    q = grayLevels(qIdx);
    % Calculate the Gaussian likelihood for the Lung class.
    pLung(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* lungVariance)) .* exp(-((q - lungMean).^2) ./ (2.0 .* lungVariance));
    % Calculate the Gaussian likelihood for the Chest class.
    pChest(qIdx) = (1.0 ./ sqrt(2.0 .* pi .* chestVariance)) .* exp(-((q - chestMean).^2) ./ (2.0 .* chestVariance));
    % End the gray level loop.
end

% Determine the actual number of files to process based on the limit.
actualNumFiles = min(numImagesToTest, numTestFiles);

% --- Testing and Evaluation Loop ---
% Loop through the specified number of test images.
for fileIndex = 1:1:actualNumFiles
    % Construct the full path to the current test image.
    currentImagePath = fullfile(testImageDir, allTestImageNames{fileIndex});
    % Read the current test image and convert to double precision.
    testImage = double(imread(currentImagePath));

    % Check if the input image is an RGB image with three color channels.
    if (size(testImage, 3) == 3)
        % Convert the RGB image to a grayscale image.
        testImage = double(rgb2gray(uint8(testImage)));
        % End the conditional block for image type.
    end

    % Construct the full path to the current mask.
    currentMaskPath = fullfile(testMaskDir, testMaskNames{fileIndex});
    % Read the expert ground truth mask.
    rawMask = imread(currentMaskPath);

    % Check if the mask is an RGB image with three color channels.
    if (size(rawMask, 3) == 3)
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);

        % Create a logical mask for the blue regions representing the lungs.
        groundTruthMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
        % End the conditional block for mask type.
    else
        % Create a logical mask for the lungs assuming white pixels.
        groundTruthMask = rawMask > 128;
        % End the conditional block for mask type.
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
            % Apply Bayes decision rule using likelihoods to avoid prior fallback.
            if (q > 20 && pLung(qIdx) > pChest(qIdx))
                % Assign the pixel to the lung class.
                initialSegmentation(rowIndex, colIndex) = true;
                % End the Bayes decision check.
            end
            % End the column loop.
        end
        % End the row loop.
    end

    % --- MAP Refinement Loop ---
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
                % Extract chest likelihood.
                pC = pChest(qIdx);

                % Calculate spatial energy for Label 0 (Chest).
                energyLabel0 = GetEnergyValueMAP(refinedImage, 0, rowIndex, colIndex, beta0, betaH, betaV);
                % Calculate spatial energy for Label 1 (Lung).
                energyLabel1 = GetEnergyValueMAP(refinedImage, 1, rowIndex, colIndex, beta0, betaH, betaV);

                % Convert spatial energies to probabilities.
                probLabel0 = exp(-energyLabel0) ./ (exp(-energyLabel0) + exp(-energyLabel1));
                % Calculate probability for Label 1.
                probLabel1 = exp(-energyLabel1) ./ (exp(-energyLabel0) + exp(-energyLabel1));

                % Calculate joint MAP scores.
                scoreChest = pC .* probLabel0;
                % Calculate lung score.
                scoreLung = pL .* probLabel1;

                % MAP Decision.
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
    % Calculate True Negatives.
    trueNegatives = sum(~refinedSegmentation(:) & ~groundTruthMask(:));

    % Calculate Dice coefficient.
    dice = (2.0 .* truePositives) ./ (2.0 .* truePositives + falsePositives + falseNegatives);
    % Calculate IoU.
    iou = truePositives ./ (truePositives + falsePositives + falseNegatives);
    % Calculate Sensitivity.
    sensitivity = truePositives ./ (truePositives + falseNegatives);
    % Calculate Specificity.
    specificity = trueNegatives ./ (trueNegatives + falsePositives);

    % Display results.
    fprintf("=== Test Image %d Evaluation ===\n", fileIndex);
    % Display Dice.
    fprintf("Dice Coefficient = %.4f\n", dice);
    % Display IoU.
    fprintf("IoU (Jaccard)    = %.4f\n", iou);
    % Display Sensitivity.
    fprintf("Sensitivity      = %.4f\n", sensitivity);
    % Display Specificity.
    fprintf("Specificity      = %.4f\n", specificity);

    % --- Visualization and Saving ---
    % Open a new figure window with a white background.
    figure("Color", "w");
    % Create a tiled layout with one row and four columns to tighten the spacing.
    currentTiledLayout = tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");
    % Create the first tile for the original test image.
    nexttile;
    % Display the original test image.
    imshow(testImage, []);
    % Add a title to the first tile.
    title("Original Test Image");
    % Create the second tile for the initial segmentation.
    nexttile;
    % Display the initial segmentation.
    imshow(initialSegmentation);
    % Add a title to the second tile.
    title("Initial Noisy Segmentation");
    % Create the third tile for the refined segmentation.
    nexttile;
    % Display the refined segmentation.
    imshow(refinedSegmentation);
    % Add a title to the third tile.
    title(sprintf("Refined MAP Segmentation\nDice = %.4f", dice));
    % Create the fourth tile for the expert ground truth.
    nexttile;
    % Display the expert ground truth.
    imshow(groundTruthMask);
    % Add a title to the fourth tile.
    title("Expert Ground Truth");

    % Define the file name for the saved figure.
    figureFileName = sprintf("TestResult_%d_Dice_%.4f.png", fileIndex, dice);
    % Construct the full path for the saved figure.
    fullFigurePath = fullfile(outputDir, figureFileName);
    % Save the current figure to the specified path.
    saveas(gcf, fullFigurePath);
    % Display a message indicating the figure was saved.
    fprintf("Saved figure to: %s\n", fullFigurePath);
    % End the test image loop.
end

% --- Local Function Definition ---
% Define the function to calculate the local spatial energy of a pixel configuration.
function energyValue = GetEnergyValueMAP(currentImage, targetLabel, rowIndex, colIndex, beta0, betaH, betaV)
% Initialize the single-site energy component based only on the target label.
if (targetLabel == 1)
    % Assign the positive single-site potential for label 1.
    energySingle = beta0;
else
    % Assign the negative single-site potential for label 0.
    energySingle = -beta0;
    % End the target label check.
end

% Initialize the horizontal pair energy component.
energyHorizontal = 0;
% Check the right neighbor for a label match.
if (targetLabel == currentImage(rowIndex, colIndex + 1))
    % Add the horizontal potential for the right neighbor.
    energyHorizontal = energyHorizontal + betaH;
else
    % Subtract the horizontal potential if they differ.
    energyHorizontal = energyHorizontal - betaH;
    % End the right neighbor check.
end
% Check the left neighbor for a label match.
if (targetLabel == currentImage(rowIndex, colIndex - 1))
    % Add the horizontal potential for the left neighbor.
    energyHorizontal = energyHorizontal + betaH;
else
    % Subtract the horizontal potential if they differ.
    energyHorizontal = energyHorizontal - betaH;
    % End the left neighbor check.
end

% Initialize the vertical pair energy component.
energyVertical = 0;
% Check the bottom neighbor for a label match.
if (targetLabel == currentImage(rowIndex + 1, colIndex))
    % Add the vertical potential for the bottom neighbor.
    energyVertical = energyVertical + betaV;
else
    % Subtract the vertical potential if they differ.
    energyVertical = energyVertical - betaV;
    % End the bottom neighbor check.
end
% Check the top neighbor for a label match.
if (targetLabel == currentImage(rowIndex - 1, colIndex))
    % Add the vertical potential for the top neighbor.
    energyVertical = energyVertical + betaV;
else
    % Subtract the vertical potential if they differ.
    energyVertical = energyVertical - betaV;
    % End the top neighbor check.
end

% Sum all energy components to get the total local spatial energy.
energyValue = energySingle + energyHorizontal + energyVertical;
% End the function definition.
end