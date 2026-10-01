% The file name is Snippet_20_MGRF_Parameter_Estimation_DerinElliott.m.
% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% --- Parameter Definition ---
% Define the smoothing constant to avoid log(0) errors.
smoothingConstant = 1.0;
% Define the boolean flag to enable boundary counting with zero padding.
includeBoundaries = true;
% Define the boolean flag to control detailed command window output.
isVerbose = true;

% --- Data Loading ---
% % Load the expert ground truth mask to act as the training data.
% rawMask = imread("LungCtMask.tif");
% 
% % Check if the mask is an RGB image with three color channels.
% if (size(rawMask, 3) == 3)
%     % Extract the blue color channel from the mask.
%     blueChannel = rawMask(:, :, 3);
%     % Extract the red color channel from the mask.
%     redChannel = rawMask(:, :, 1);
%     % Extract the green color channel from the mask.
%     greenChannel = rawMask(:, :, 2);
%     % Create a logical mask for the blue regions representing the lungs.
%     trainingMask = (blueChannel > 128) & (redChannel < 128) & (greenChannel < 128);
% else
%     % Create a logical mask for the lungs assuming white pixels.
%     trainingMask = rawMask > 128;
%     % End the mask type check.
% end

% (Optional) Load a small example matrix for testing.
trainingMask = [1, 1, 0; 1, 1, 0; 0, 0, 0];

% Convert the logical mask to double for mathematical operations.
trainingMask = double(trainingMask);
% Get the dimensions of the training mask.
[imageRows, imageCols] = size(trainingMask);

% --- Boundary Handling ---
% Check if boundary inclusion is enabled.
if (includeBoundaries)
    % Pad the mask with zeros on all sides by one pixel.
    paddedMask = padarray(trainingMask, [1, 1], 0, "both");
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
    paddedMask = trainingMask;
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

% Initialize the count matrix for neighborhood configurations.
% The dimensions are 3 x 3 x 2 for (sumHorizontal, sumVertical, currentLabel).
countMatrix = zeros(3, 3, 2);

% --- Clique Counting ---
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

% --- Print Clique Counting Results ---
% Check if verbose output is enabled.
if (isVerbose)
    % Display a header for the clique counting results.
    fprintf("Clique Counting Results:\n");
    % Display the counts for configurations where the center pixel is Label 1.
    fprintf("Counts for Label 1:\n");
    % Print the 3x3 matrix for Label 1.
    disp(countMatrix(:, :, 2));
    % Display the counts for configurations where the center pixel is Label 0.
    fprintf("Counts for Label 0:\n");
    % Print the 3x3 matrix for Label 0.
    disp(countMatrix(:, :, 1));
    % End the verbose check.
end

% --- Building the Linear System (A * beta = b) ---
% Initialize the design matrix A with nine rows and three columns.
matrixA = zeros(9, 3);
% Initialize the observation vector b with nine rows.
vectorB = zeros(9, 1);
% Initialize the row index for the matrices.
rowIndexMatrix = 1;

% Check if verbose output is enabled for the linear system steps.
if (isVerbose)
    % Display a header for the linear system building steps.
    fprintf("Building Linear System (A * beta = b) Step-by-Step:\n");
    % End the verbose check.
end

% Loop through all possible horizontal sums (k = 0, 1, 2).
for sumH = 0:1:2
    % Loop through all possible vertical sums (m = 0, 1, 2).
    for sumV = 0:1:2
        % Extract the count for the configuration where the center pixel is 1.
        countY1 = countMatrix(sumH + 1, sumV + 1, 2);
        % Extract the count for the configuration where the center pixel is 0.
        countY0 = countMatrix(sumH + 1, sumV + 1, 1);
        % Calculate the log-ratio with smoothing to avoid log(0).
        logRatio = log10((countY1 + smoothingConstant) ./ (countY0 + smoothingConstant));

        % --- Mathematical Derivation of Derin & Elliott Coefficients ---
        % 1. Single-Site Potential (G1): G1(1) = beta0, G1(0) = -beta0.
        % 2. Pair-Site Potential (G2): G2 = beta (same label), G2 = -beta (different label).
        %
        % Energy if center pixel is 1 (U(1)):
        % Horizontal: k matches, (2-k) mismatches => k*betaH - (2-k)*betaH = 2*(k-1)*betaH
        % Vertical: m matches, (2-m) mismatches => 2*(m-1)*betaV
        % U(1) = beta0 + 2*(k-1)*betaH + 2*(m-1)*betaV
        %
        % Energy if center pixel is 0 (U(0)):
        % Horizontal: (2-k) matches, k mismatches => (2-k)*betaH - k*betaH = -2*(k-1)*betaH
        % Vertical: -2*(m-1)*betaV
        % U(0) = -beta0 - 2*(k-1)*betaH - 2*(m-1)*betaV
        %
        % Energy Difference (Delta U = U(1) - U(0)):
        % Delta U = [beta0 + 2*(k-1)*betaH + 2*(m-1)*betaV] - [-beta0 - 2*(k-1)*betaH - 2*(m-1)*betaV]
        % Delta U = 2*beta0 + 4*(k-1)*betaH + 4*(m-1)*betaV
        % -------------------------------------------------

        % Calculate Derin & Elliott energy difference coefficients.
        % Delta U = U(1) - U(0) = 2*beta0 + 4*(k-1)*betaH + 4*(m-1)*betaV.
        coeffBeta0 = 2.0;
        coeffBetaH = 4.0 * (sumH - 1);
        coeffBetaV = 4.0 * (sumV - 1);

        % Assign the row in matrix A using the Derin & Elliott coefficients.
        matrixA(rowIndexMatrix, :) = [coeffBeta0, coeffBetaH, coeffBetaV];
        % Assign the corresponding log-ratio value in vector b.
        vectorB(rowIndexMatrix) = logRatio;

        % Check if verbose output is enabled.
        if (isVerbose)
            % Print the details of the current step.
            fprintf("Step %d: sumH=%d, sumV=%d | CountY1=%d, CountY0=%d | LogRatio=%.4f | Row A: [%.1f, %.1f, %.1f] | b: %.4f\n", rowIndexMatrix, sumH, sumV, countY1, countY0, logRatio, coeffBeta0, coeffBetaH, coeffBetaV, logRatio);
            % End the verbose print check.
        end

        % Increment the row index for the next configuration.
        rowIndexMatrix = rowIndexMatrix + 1;
        % End the vertical sum loop.
    end
    % End the horizontal sum loop.
end

% --- Solving the Least Squares Problem ---
% Calculate the estimated beta parameters using the normal equation.
% Note: The negative sign is applied because log(P(1)/P(0)) is proportional to -Delta U.
estimatedBetas = (matrixA' * matrixA) \ (matrixA' * (-vectorB));
% Extract the single-site estimated parameter from the result vector.
estimatedBeta0 = estimatedBetas(1);
% Extract the horizontal estimated parameter from the result vector.
estimatedBetaH = estimatedBetas(2);
% Extract the vertical estimated parameter from the result vector.
estimatedBetaV = estimatedBetas(3);

% Display the estimated parameters in the command window.
fprintf("Estimated MGRF Parameters (Derin & Elliott Least Squares):\n");
% Display the single-site parameter.
fprintf("  beta0 (Single-site) = %.4f\n", estimatedBeta0);
% Display the horizontal parameter.
fprintf("  betaH (Horizontal)  = %.4f\n", estimatedBetaH);
% Display the vertical parameter.
fprintf("  betaV (Vertical)    = %.4f\n", estimatedBetaV);

% --- Visualization of the Linear System ---
% Open a new figure window to display the configurations and counts.
figure("Color", "w");
% Create the first subplot for the count matrix visualization.
subplot(1, 2, 1);
% Display the count matrix for y_i = 1 as an image.
imagesc(countMatrix(:, :, 2));
% Add a colorbar to show the count scale.
colorbar;
% Add a title to the first subplot.
title("Counts of Configurations where y_i = 1");
% Label the horizontal axis.
xlabel("Vertical Sum (S_V)");
% Label the vertical axis.
ylabel("Horizontal Sum (S_H)");
% Create the second subplot for the count matrix visualization.
subplot(1, 2, 2);
% Display the count matrix for y_i = 0 as an image.
imagesc(countMatrix(:, :, 1));
% Add a colorbar to show the count scale.
colorbar;
% Add a title to the second subplot.
title("Counts of Configurations where y_i = 0");
% Label the horizontal axis.
xlabel("Vertical Sum (S_V)");
% Label the vertical axis.
ylabel("Horizontal Sum (S_H)");