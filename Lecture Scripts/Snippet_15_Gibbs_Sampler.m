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
% Initialize the image with random labels.
initialImage = (numberOfLabels - 1) .* rand(imageSize);
% Round to get integer labels.
initialImage = round(initialImage);
% Copy to final image.
finalImage = initialImage;
% Set the number of iterations.
noIterations = 100;

% Define clique potential parameters.
beta0 = -1;
betaH = -1;
betaV = -1;

% Run Gibbs Sampler for multiple iterations.
for iterationIndex = 1:1:noIterations
    % Loop through each pixel (excluding boundaries).
    for rowIndex = 2:1:imageSize - 1
        % Loop through each column.
        for colIndex = 2:1:imageSize - 1
            % Initialize the energy array for the current pixel.
            energyArray = zeros(1, numberOfLabels);
            % Initialize the probability array for the current pixel.
            probabilityArray = zeros(1, numberOfLabels);

            % Loop through each possible label.
            for labelIndex = 1:1:numberOfLabels
                % Calculate the label value.
                currentLabel = labelIndex - 1;
                % Calculate the energy for this label.
                energyValue = GetEnergyValue(finalImage, currentLabel, rowIndex, colIndex, beta0, betaH, betaV);
                % Store the energy.
                energyArray(labelIndex) = energyValue;
                % End the label loop.
            end

            % Calculate the unnormalized probabilities using the Gibbs distribution.
            probabilityArray = exp(-energyArray);
            % Normalize the probabilities so they sum to one.
            probabilityArray = probabilityArray ./ sum(probabilityArray);

            % Generate a random value to sample from the distribution.
            randomValue = rand;
            % Initialize the cumulative sum for sampling.
            cumulativeSum = 0;
            % Initialize the selected label index.
            selectedLabelIndex = 1;

            % Loop through the probabilities to find the selected label.
            for labelIndex = 1:1:numberOfLabels
                % Add the current probability to the cumulative sum.
                cumulativeSum = cumulativeSum + probabilityArray(labelIndex);
                % Check if the random value falls within the current cumulative probability.
                if (randomValue <= cumulativeSum)
                    % Store the selected label index.
                    selectedLabelIndex = labelIndex;
                    % Break out of the loop once a label is selected.
                    break;
                    % End the selection check.
                end
                % End the probability sampling loop.
            end

            % Update the pixel in the final image with the selected label.
            finalImage(rowIndex, colIndex) = selectedLabelIndex - 1;
            % End the column loop.
        end
        % End the row loop.
    end
    % Display the current iteration number.
    fprintf("Completed iteration %d.\n", iterationIndex);
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
title("Final Image after Gibbs Sampling");

% --- Local Function Definition ---
% Define the function to calculate the energy of a pixel configuration.
function energyValue = GetEnergyValue(currentImage, targetLabel, rowIndex, colIndex, beta0, betaH, betaV)
% Initialize the single-site energy component.
energySingle = 0;
% Check if the target label matches the current pixel label.
if (targetLabel == currentImage(rowIndex, colIndex))
    % Assign the single-site potential if they match.
    energySingle = beta0;
    % End the match check.
end

% Initialize the horizontal pair energy component.
energyHorizontal = 0;
% Check the right neighbor.
if (targetLabel == currentImage(rowIndex, colIndex + 1))
    % Add the horizontal potential for the right neighbor.
    energyHorizontal = energyHorizontal + betaH;
    % End the right neighbor check.
end
% Check the left neighbor.
if (targetLabel == currentImage(rowIndex, colIndex - 1))
    % Add the horizontal potential for the left neighbor.
    energyHorizontal = energyHorizontal + betaH;
    % End the left neighbor check.
end

% Initialize the vertical pair energy component.
energyVertical = 0;
% Check the bottom neighbor.
if (targetLabel == currentImage(rowIndex + 1, colIndex))
    % Add the vertical potential for the bottom neighbor.
    energyVertical = energyVertical + betaV;
    % End the bottom neighbor check.
end
% Check the top neighbor.
if (targetLabel == currentImage(rowIndex - 1, colIndex))
    % Add the vertical potential for the top neighbor.
    energyVertical = energyVertical + betaV;
    % End the top neighbor check.
end

% Sum all energy components to get the total energy.
energyValue = energySingle + energyHorizontal + energyVertical;
% End the function definition.
end