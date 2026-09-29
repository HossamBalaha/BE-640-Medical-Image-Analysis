% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Define the image size.
imageSize = 128;
% Define the number of discrete labels.
numberOfLabels = 2;
% Initialize the image with random labels using element-wise multiplication.
initialImage = (numberOfLabels - 1) .* rand(imageSize);
% Round to get integer labels.
initialImage = round(initialImage);
% Copy the initial image to the final image variable.
finalImage = initialImage;

% Define the single-site clique potential parameter.
beta0 = -1;
% Define the horizontal pair clique potential parameter.
betaH = -1;
% Define the vertical pair clique potential parameter.
betaV = -1;
% Define the number of Metropolis iterations.
numIterations = 20;

% Run the Metropolis Sampler for multiple iterations.
for iterationIndex = 1:1:numIterations
    % Loop through each pixel excluding the boundaries.
    for rowIndex = 2:1:imageSize - 1
        % Loop through each column excluding the boundaries.
        for colIndex = 2:1:imageSize - 1
            % Extract the current label of the pixel.
            currentLabel = finalImage(rowIndex, colIndex);
            % Calculate the local energy for the current label.
            energyCurrent = GetEnergyValueMetropolis(finalImage, currentLabel, rowIndex, colIndex, beta0, betaH, betaV);

            % Propose a new random label.
            proposedLabel = round((numberOfLabels - 1) .* rand);
            % Calculate the local energy for the proposed label.
            energyProposed = GetEnergyValueMetropolis(finalImage, proposedLabel, rowIndex, colIndex, beta0, betaH, betaV);

            % Calculate the unnormalized probabilities for the current and proposed labels.
            probCurrentUnnormalized = exp(-energyCurrent);
            probProposedUnnormalized = exp(-energyProposed);
            % Calculate the normalized probabilities.
            probCurrent = probCurrentUnnormalized ./ (probCurrentUnnormalized + probProposedUnnormalized);
            probProposed = probProposedUnnormalized ./ (probCurrentUnnormalized + probProposedUnnormalized);

            % Generate a random value for the acceptance criteria.
            randomValue = rand;
            % Check if the proposed probability is greater than or equal to the current probability and the random value is less than or equal to the proposed probability.
            if (probProposed >= probCurrent) && (randomValue <= probProposed)
                % Update the pixel in the final image with the proposed label.
                finalImage(rowIndex, colIndex) = proposedLabel;
                % End the acceptance conditional check.
            end
            % End the column loop.
        end
        % End the row loop.
    end
    % Display the current iteration number to the command window.
    fprintf("Completed Metropolis iteration %d.\n", iterationIndex);
    % End the iteration loop.
end

% Open a new figure window to display the before and after results.
figure("Color", "w");

% Create the first subplot for the initial image.
subplot(1, 2, 1);
% Display the initial simulated image.
imshow(initialImage, []);
% Add a title to the first subplot.
title("Initial Image (Random Labels)");

% Create the second subplot for the final image.
subplot(1, 2, 2);
% Display the final simulated image.
imshow(finalImage, []);
% Add a title to the second subplot.
title("Final Image after Metropolis Sampling");

% --- Local Function Definition ---
% Define the function to calculate the local energy of a pixel configuration.
function energyValue = GetEnergyValueMetropolis(currentImage, targetLabel, rowIndex, colIndex, beta0, betaH, betaV)
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