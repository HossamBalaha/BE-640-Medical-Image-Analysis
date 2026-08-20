% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Define the example image as a two-dimensional matrix.
imageMatrix = [0, 1, 1, 0; 0, 2, 2, 3; 0, 1, 1, 3; 0, 2, 3, 3];
% Define the vector of gray levels present in the example image.
grayLevels = 0:1:3;

% Initialize a row vector to store the manual histogram counts.
manualCounts = zeros(1, numel(grayLevels));
% Loop over the index of each gray level in the example image.
for grayLevelIndex = 1:1:numel(grayLevels)
    % Find the row and column indices of pixels equal to the current gray level.
    [rowIndices, columnIndices, flag] = find(imageMatrix == grayLevels(grayLevelIndex));
    % Count the number of pixels found for the current gray level.
    pixelCount = numel(rowIndices);
    % Store the count in the manual histogram vector.
    manualCounts(grayLevelIndex) = pixelCount;
    % End the gray-level loop.
end
% Display a completion message for the manual method.
disp("Manual histogram computation is complete.");
% Display the final manual histogram counts.
disp(manualCounts);

% Compute the reference counts using the built-in integer-bin histogram function.
builtinCounts = histcounts(imageMatrix(:), "BinMethod", "integers");

% Open a new figure window with a white background for the enhanced bar chart.
figure("Color", "w");
% Draw a colored bar chart of the manual histogram counts.
barHandle = bar( ...
    grayLevels, manualCounts, 0.6, "FaceColor", [0.20, 0.47, 0.80], ...
    "EdgeColor", [0.10, 0.10, 0.10], "LineWidth", 1.2 ...
    );
% Restrict the horizontal axis ticks to the integer gray levels.
xticks(grayLevels);
% Extend the vertical axis above the tallest bar to leave room for labels.
ylim([0, max(manualCounts) + 1]);
% Restrict the vertical axis ticks to integer counts.
yticks(0:1:max(manualCounts) + 1);
% Enable a light grid to improve readability.
grid on;
% Loop over each bar to add a numeric label above it.
for barIndex = 1:1:numel(manualCounts)
    % Place a bold label slightly above the top of the current bar.
    text(grayLevels(barIndex), manualCounts(barIndex) + 0.15, string(manualCounts(barIndex)), "HorizontalAlignment", "center", "FontSize", 16, "FontWeight", "bold");
    % End the bar-labeling loop.
end
% Add a descriptive title to the bar chart.
title("Histogram of the Example Image", "FontSize", 18);
% Label the horizontal axis of the bar chart.
xlabel("Gray Level", "FontSize", 16);
% Label the vertical axis of the bar chart.
ylabel("Count", "FontSize", 16);

% Display the manual counts as a readable table.
disp("Counts using the manual find method:");
disp(table(grayLevels.', manualCounts.', VariableNames=["GrayLevel", "Count"]));
% Display the built-in counts as a readable table.
disp("Counts using the built-in histogram function:");
disp(table(grayLevels.', builtinCounts.', VariableNames=["GrayLevel", "Count"]));

% Define the expected histogram counts for the example image.
expectedCounts = [5, 4, 3, 4];
% Verify that both methods match the expected counts.
if (isequal(manualCounts, expectedCounts) && isequal(builtinCounts, expectedCounts))
    % Confirm agreement among all histogram methods.
    disp("All histogram methods produce the expected counts.");
    % Handle the mismatch case.
else
    % Warn the user that a mismatch occurred.
    disp("Warning: the histogram counts do not match the expected values.");
    % End the verification conditional block.
end