% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the boolean flag to use static, manually defined MGRF parameters.
useStaticParams = false;
% Define the static MGRF parameters (used if useStaticParams is true).
staticBeta0 = -1;
staticBetaH = -1;
staticBetaV = -1;
% Define the smoothing constant to avoid log(0) errors in MGRF estimation.
smoothingConstant = 1.0;
% Define the boolean flag to enable boundary counting with zero padding.
includeBoundaries = true;

% --- Data Loading Paths ---
% Define the base dataset path.
datasetPath = "./../Dataset/Sample2";

% Check if the base dataset path exists; if not, stop execution.
if (~exist(datasetPath, "dir"))
    % Throw an error if the dataset path is missing.
    error("Dataset path does not exist: %s.\nPlease check the path and ensure the dataset is downloaded.", datasetPath);
    % End the path existence check.
end

% Define the subdirectories for training images and masks.
trainImageDir = fullfile(datasetPath, "train", "images");
trainMaskDir = fullfile(datasetPath, "train", "masks");

% Check if training subdirectories exist; if not, stop execution.
if (~exist(trainImageDir, "dir"))
    % Throw an error if the training images directory is missing.
    error("Training images directory does not exist. Please check the dataset structure.");
    % End the training images directory check.
end
if (~exist(trainMaskDir, "dir"))
    % Throw an error if the training masks directory is missing.
    error("Training masks directory does not exist. Please check the dataset structure.");
    % End the training masks directory check.
end

% Get all training image files (supports .jpg, .jpeg, and .png).
allImageFiles = dir(fullfile(trainImageDir, "*.jpg"));
if (isempty(allImageFiles))
    % Fallback to .jpeg if no .jpg files are found.
    allImageFiles = dir(fullfile(trainImageDir, "*.jpeg"));
    % End the .jpeg fallback check.
end
if (isempty(allImageFiles))
    % Fallback to .png if no .jpeg files are found.
    allImageFiles = dir(fullfile(trainImageDir, "*.png"));
    % End the .png fallback check.
end

% ROBUST CHECK: Ensure files were actually found.
if (isempty(allImageFiles))
    % Throw an error if no image files are found in the directory.
    error("No image files (.jpg, .jpeg, or .png) found in directory:\n%s\n\nPossible fixes:\n1. Ensure the .zip files have been fully unzipped.\n2. Check if the folder name has extra spaces or different capitalization.\n3. Verify the files are not nested in another subfolder.", trainImageDir);
    % End the empty files check.
end

% Sort the files to ensure consistent ordering.
allImageNames = sort({allImageFiles.name});
numTrainFiles = length(allImageNames);
% Display the number of training files found.
fprintf("Found %d total training image files.\n", numTrainFiles);

% Generate corresponding mask names for the training set.
trainMaskNames = cell(1, numTrainFiles);
for i = 1:1:numTrainFiles
    % Extract the current image name.
    imgName = allImageNames{i};
    % Find the last underscore to insert "_mask" before the slice number.
    lastUnderscore = find(imgName == '_', 1, 'last');
    if (~isempty(lastUnderscore))
        % Construct the mask name by injecting "_mask".
        trainMaskNames{i} = [imgName(1:lastUnderscore - 1), '_mask', imgName(lastUnderscore:end)];
    else
        % Fallback if no underscore is found.
        trainMaskNames{i} = imgName;
        % End the underscore check.
    end
    % End the mask name generation loop.
end

% Initialize empty arrays to accumulate training samples for the intensity model.
allLungSamples = [];
allChestSamples = [];

% Initialize the count matrix for neighborhood configurations for the spatial model.
countMatrix = zeros(3, 3, 2);

% --- Phase 1: Intensity Model & Spatial Model Estimation ---
% Loop through each of the N training pairs for model estimation.
for fileIndex = 1:1:numTrainFiles
    % Construct the full path to the current training image.
    currentImagePath = fullfile(trainImageDir, allImageNames{fileIndex});
    % Read the current training image and convert it to double precision.
    currentImage = double(imread(currentImagePath));

    % Check if the input image is an RGB image with three color channels.
    if (size(currentImage, 3) == 3)
        % Convert the RGB image to a grayscale image.
        currentImage = double(rgb2gray(uint8(currentImage)));
        % End the conditional block for image type.
    end

    % Construct the full path to the current mask.
    currentMaskPath = fullfile(trainMaskDir, trainMaskNames{fileIndex});
    % Read the current expert mask.
    rawMask = imread(currentMaskPath);

    % Check if the mask is an RGB image with three color channels.
    if (size(rawMask, 3) == 3)
        % Extract the red color channel from the mask.
        redChannel = rawMask(:, :, 1);
        % Extract the green color channel from the mask.
        greenChannel = rawMask(:, :, 2);
        % Extract the blue color channel from the mask.
        blueChannel = rawMask(:, :, 3);

        % Stricter check to avoid JPG compression artifacts (matching Project 1).
        lungMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
        % Create a logical mask for the chest as the complement of the lung mask.
        chestMask = ~lungMask;
        % End the conditional block for mask type.
    else
        % Create a logical mask for the lungs assuming white pixels.
        lungMask = rawMask > 128;
        % Create a logical mask for the chest as the complement of the lung mask.
        chestMask = ~lungMask;
        % End the conditional block for mask type.
    end

    % Extract the lung pixel intensities using the logical lung mask.
    lungPixels = currentImage(lungMask);
    % Extract the chest pixel intensities using the logical chest mask.
    chestPixels = currentImage(chestMask);

    % Append the current lung pixels to the global accumulation array.
    allLungSamples = [allLungSamples; lungPixels];
    % Append the current chest pixels to the global accumulation array.
    allChestSamples = [allChestSamples; chestPixels];

    % --- MGRF Parameter Estimation (Spatial Model) ---
    % Convert the logical lung mask to double for mathematical operations.
    binaryMask = double(lungMask);
    % Get the dimensions of the training mask.
    [imageRows, imageCols] = size(binaryMask);

    % Check if boundary inclusion is enabled for the spatial model.
    if (includeBoundaries)
        % Pad the mask with zeros on all sides by one pixel.
        paddedMask = padarray(binaryMask, [1, 1], 0, "both");
        % Set the starting row index for the padded mask.
        rowStart = 2;
        % Set the ending row index for the padded mask.
        rowEnd = imageRows + 1;
        % Set the starting column index for the padded mask.
        colStart = 2;
        % Set the ending column index for the padded mask.
        colEnd = imageCols + 1;
    else
        % Assign the original mask without padding.
        paddedMask = binaryMask;
        % Set the starting row index to exclude boundaries.
        rowStart = 2;
        % Set the ending row index to exclude boundaries.
        rowEnd = imageRows - 1;
        % Set the starting column index to exclude boundaries.
        colStart = 2;
        % Set the ending column index to exclude boundaries.
        colEnd = imageCols - 1;
        % End the boundary flag check.
    end

    % Loop through the rows based on the boundary flag settings.
    for rowIndex = rowStart:1:rowEnd
        % Loop through the columns based on the boundary flag settings.
        for colIndex = colStart:1:colEnd
            % Extract the current pixel label from the working mask.
            currentLabel = paddedMask(rowIndex, colIndex);
            % Calculate the sum of horizontal neighbors.
            sumHorizontal = paddedMask(rowIndex, colIndex - 1) + paddedMask(rowIndex, colIndex + 1);
            % Calculate the sum of vertical neighbors.
            sumVertical = paddedMask(rowIndex - 1, colIndex) + paddedMask(rowIndex + 1, colIndex);
            % Increment the count for this specific configuration.
            countMatrix(sumHorizontal + 1, sumVertical + 1, currentLabel + 1) = countMatrix(sumHorizontal + 1, sumVertical + 1, currentLabel + 1) + 1;
            % End the column loop.
        end
        % End the row loop.
    end
    % End the training loop.
end

% --- Intensity Model Parameter Calculation ---
% Calculate the mean and variance for the lung class.
lungMean = mean(allLungSamples);
lungVariance = var(allLungSamples);
% Calculate the mean and variance for the chest class.
chestMean = mean(allChestSamples);
chestVariance = var(allChestSamples);

% Calculate the total number of training pixels.
totalPixels = length(allLungSamples) + length(allChestSamples);
% Compute the prior probability for the lung class.
lungPrior = length(allLungSamples) / totalPixels;
% Compute the prior probability for the chest class.
chestPrior = length(allChestSamples) / totalPixels;

% --- Spatial Model Parameter Calculation (Derin & Elliott Least Squares) ---
% Check if static parameters should be used.
if (~useStaticParams)
    % Initialize the design matrix A with nine rows and three columns.
    matrixA = zeros(9, 3);
    % Initialize the observation vector b with nine rows.
    vectorB = zeros(9, 1);
    % Initialize the row index for the matrices.
    rowIndexMatrix = 1;

    % Loop through all possible horizontal sums.
    for sumH = 0:1:2
        % Loop through all possible vertical sums.
        for sumV = 0:1:2
            % Extract the count for the configuration where the center pixel is 1.
            countY1 = countMatrix(sumH + 1, sumV + 1, 2);
            % Extract the count for the configuration where the center pixel is 0.
            countY0 = countMatrix(sumH + 1, sumV + 1, 1);
            % Calculate the log-ratio with smoothing to avoid log(0).
            logRatio = log10((countY1 + smoothingConstant) ./ (countY0 + smoothingConstant));

            % Calculate Derin & Elliott energy difference coefficients.
            coeffBeta0 = 2.0;
            % Calculate the horizontal coefficient based on the sum of horizontal neighbors.
            coeffBetaH = 4.0 * (sumH - 1);
            % Calculate the vertical coefficient based on the sum of vertical neighbors.
            coeffBetaV = 4.0 * (sumV - 1);

            % Assign the row in matrix A using the Derin & Elliott coefficients.
            matrixA(rowIndexMatrix, :) = [coeffBeta0, coeffBetaH, coeffBetaV];
            % Assign the corresponding log-ratio value in vector b.
            vectorB(rowIndexMatrix) = logRatio;
            % Increment the row index for the next configuration.
            rowIndexMatrix = rowIndexMatrix + 1;
            % End the vertical sum loop.
        end
        % End the horizontal sum loop.
    end

    % Calculate the estimated beta parameters using the normal equation.
    estimatedBetas = (matrixA' * matrixA) \ (matrixA' * (-vectorB));
    % Extract the single-site estimated parameter from the result vector.
    estimatedBeta0 = estimatedBetas(1);
    % Extract the horizontal estimated parameter from the result vector.
    estimatedBetaH = estimatedBetas(2);
    % Extract the vertical estimated parameter from the result vector.
    estimatedBetaV = estimatedBetas(3);
    
    % Assign the estimated parameters to the final variables.
    finalBeta0 = estimatedBeta0;
    finalBetaH = estimatedBetaH;
    finalBetaV = estimatedBetaV;
    % Display a message indicating estimated parameters are being used.
    fprintf("Using estimated MGRF parameters: Beta0=%.4f, BetaH=%.4f, BetaV=%.4f\n", finalBeta0, finalBetaH, finalBetaV);
else
    % Assign the static parameters to the final variables.
    finalBeta0 = staticBeta0;
    finalBetaH = staticBetaH;
    finalBetaV = staticBetaV;
    % Display a message indicating static parameters are being used.
    fprintf("Using static MGRF parameters: Beta0=%.4f, BetaH=%.4f, BetaV=%.4f\n", finalBeta0, finalBetaH, finalBetaV);
end

% --- Save the Model ---
% Initialize the model parameters structure.
modelParameters = struct();
% Store the lung mean in the structure.
modelParameters.LungMean = lungMean;
% Store the lung variance in the structure.
modelParameters.LungVariance = lungVariance;
% Store the lung prior in the structure.
modelParameters.LungPrior = lungPrior;
% Store the chest mean in the structure.
modelParameters.ChestMean = chestMean;
% Store the chest variance in the structure.
modelParameters.ChestVariance = chestVariance;
% Store the chest prior in the structure.
modelParameters.ChestPrior = chestPrior;
% Store the spatial single-site parameter.
modelParameters.Beta0 = finalBeta0;
% Store the spatial horizontal parameter.
modelParameters.BetaH = finalBetaH;
% Store the spatial vertical parameter.
modelParameters.BetaV = finalBetaV;

% Save the frozen model to a MAT file for the testing phase.
save("Project02FrozenMAPModel.mat", "modelParameters");

% Display the training results in the command window.
fprintf("Training Complete!\n");
fprintf("Lung: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", lungMean, lungVariance, lungPrior);
fprintf("Chest: Mean=%.4f, Variance=%.4f, Prior=%.4f\n", chestMean, chestVariance, chestPrior);
fprintf("MGRF: Beta0=%.4f, BetaH=%.4f, BetaV=%.4f\n", finalBeta0, finalBetaH, finalBetaV);