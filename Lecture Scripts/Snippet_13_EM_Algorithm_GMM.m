% Clear the command window.
clc;
% Close all open figure windows.
close all;
% Clear all variables from the workspace.
clear all;

% Load the input grayscale image and convert it to double precision.
inputImage = double(imread("LungCtSlice.tif"));
% Flatten the 2D image into a 1D column vector for processing.
imageVector = inputImage(:);
% Calculate the total number of pixels in the image.
numPixels = length(imageVector);

% Define the number of Gaussian classes for the mixture model.
numClasses = 2;
% Define the maximum number of EM iterations.
maxIterations = 500;
% Define the convergence tolerance for the log-likelihood.
tolerance = 1.0;

% Initialize the Gaussian means evenly across the intensity range.
gaussianMeans = linspace(min(imageVector), max(imageVector), numClasses);
% Initialize the Gaussian variances to the global image variance.
gaussianVariances = ones(1, numClasses) .* var(imageVector);
% Initialize the class prior probabilities to be equal.
classPriors = ones(1, numClasses) ./ numClasses;

% Initialize the responsibilities matrix to zeros.
responsibilities = zeros(numPixels, numClasses);
% Initialize the previous log-likelihood to negative infinity.
previousLogLikelihood = -inf;
% Initialize the array to store log-likelihood history for plotting.
logLikelihoodHistory = zeros(1, maxIterations);

% Start the Expectation-Maximization iterative loop.
for currentIteration = 1:1:maxIterations
    % --- E-Step: Calculate Responsibilities ---
    % Loop through each pixel in the image.
    for pixelIndex = 1:1:numPixels
        % Extract the intensity value of the current pixel.
        currentPixelValue = imageVector(pixelIndex);
        % Initialize the weighted likelihoods for the current pixel.
        weightedLikelihoods = zeros(1, numClasses);

        % Loop through each Gaussian class.
        for classIndex = 1:1:numClasses
            % Calculate the Gaussian likelihood for the current pixel and class.
            likelihood = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(classIndex))) .* exp(-((currentPixelValue - gaussianMeans(classIndex)).^2) ./ (2.0 .* gaussianVariances(classIndex)));
            % Multiply the likelihood by the class prior to get the weighted likelihood.
            weightedLikelihoods(classIndex) = likelihood .* classPriors(classIndex);
            % End the class loop.
        end

        % Normalize the weighted likelihoods to get the posterior responsibilities.
        responsibilities(pixelIndex, :) = weightedLikelihoods ./ sum(weightedLikelihoods);
        % End the pixel loop.
    end

    % --- M-Step: Update Parameters ---
    % Loop through each Gaussian class to update its parameters.
    for classIndex = 1:1:numClasses
        % Extract the responsibilities for the current class.
        currentClassResponsibilities = responsibilities(:, classIndex);
        % Calculate the effective number of pixels assigned to the current class.
        effectivePixelCount = sum(currentClassResponsibilities);

        % Update the class prior probability.
        classPriors(classIndex) = effectivePixelCount ./ numPixels;
        % Update the class mean using the weighted average.
        gaussianMeans(classIndex) = sum(currentClassResponsibilities .* imageVector) ./ effectivePixelCount;
        % Update the class variance using the weighted variance.
        gaussianVariances(classIndex) = sum(currentClassResponsibilities .* ((imageVector - gaussianMeans(classIndex)).^2)) ./ effectivePixelCount;
        % End the class parameter update loop.
    end

    % --- Convergence Check ---
    % Initialize the current log-likelihood.
    currentLogLikelihood = 0;
    % Loop through each pixel to calculate the log-likelihood.
    for pixelIndex = 1:1:numPixels
        % Extract the intensity value of the current pixel.
        currentPixelValue = imageVector(pixelIndex);
        % Initialize the mixture probability for the current pixel.
        mixtureProbability = 0;

        % Loop through each Gaussian class.
        for classIndex = 1:1:numClasses
            % Calculate the Gaussian likelihood for the current pixel and class.
            likelihood = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(classIndex))) .* exp(-((currentPixelValue - gaussianMeans(classIndex)).^2) ./ (2.0 .* gaussianVariances(classIndex)));
            % Add the weighted likelihood to the mixture probability.
            mixtureProbability = mixtureProbability + classPriors(classIndex) .* likelihood;
            % End the class loop.
        end

        % Add the log of the mixture probability to the total log-likelihood.
        currentLogLikelihood = currentLogLikelihood + log(mixtureProbability + 1e-10);
        % End the log-likelihood pixel loop.
    end

    % Store the log-likelihood for the current iteration.
    logLikelihoodHistory(currentIteration) = currentLogLikelihood;
    % Display the log-likelihood for the current iteration.
    fprintf("Iteration %d: Log-Likelihood = %.4f\n", currentIteration, currentLogLikelihood);

    % Check if the change in log-likelihood is below the tolerance.
    if (currentIteration > 1)
        % Calculate the absolute change in log-likelihood.
        logLikelihoodChange = abs(currentLogLikelihood - previousLogLikelihood);
        % Check if the change is smaller than the tolerance.
        if (logLikelihoodChange < tolerance)
            % Display the convergence message.
            fprintf("EM Algorithm converged after %d iterations.\n", currentIteration);
            % Break out of the EM loop.
            break;
            % End the convergence check.
        end
        % End the tolerance check.
    end

    % Update the previous log-likelihood for the next iteration.
    previousLogLikelihood = currentLogLikelihood;
    % End the EM iteration loop.
end

% Initialize the model parameters structure.
modelParameters = struct();
% Store the Gaussian means in the structure.
modelParameters.GaussianMeans = gaussianMeans;
% Store the Gaussian variances in the structure.
modelParameters.GaussianVariances = gaussianVariances;
% Store the class priors in the structure.
modelParameters.ClassPriors = classPriors;

% Save the trained model parameters to a MAT file.
save("Snippet13_TrainedEMModelParameters.mat", "modelParameters");

% --- Visualization Section ---

% Open a new figure window for the convergence plot.
figure("Color", "w");
% Plot the log-likelihood history up to the final iteration.
plot(1:currentIteration, logLikelihoodHistory(1:currentIteration), "b-", "LineWidth", 2);
% Add a title to the convergence plot.
title("EM Algorithm Convergence: Log-Likelihood vs. Iteration");
% Label the horizontal axis.
xlabel("Iteration");
% Label the vertical axis.
ylabel("Log-Likelihood");
% Add a grid for better readability.
grid on;

% Open a new figure window for the histogram and segmentation.
figure("Color", "w");

% Create the first subplot for the histogram and fitted Gaussians.
subplot(1, 2, 1);
% Plot the normalized histogram of the image pixels.
histogram(imageVector, 0:1:255, "Normalization", "pdf", "FaceColor", [0.8, 0.8, 0.8], "EdgeColor", "none");
% Hold the plot to overlay the Gaussian curves.
hold on;

% Define the query gray levels for plotting the curves.
queryGrayLevels = 0:1:255;
% Loop through each class to plot its fitted Gaussian curve.
for classIndex = 1:1:numClasses
    % Calculate the Gaussian PDF for the current class parameters.
    fittedCurve = (classPriors(classIndex) ./ sqrt(2.0 .* pi .* gaussianVariances(classIndex))) .* exp(-((queryGrayLevels - gaussianMeans(classIndex)).^2) ./ (2.0 .* gaussianVariances(classIndex)));
    % Plot the fitted curve with a distinct color.
    plot(queryGrayLevels, fittedCurve, "LineWidth", 2, "DisplayName", sprintf("Class %d (\\mu=%.1f)", classIndex, gaussianMeans(classIndex)));
    % End the class plotting loop.
end

% Add a title to the histogram subplot.
title("Image Histogram and Fitted GMM");
% Label the horizontal axis.
xlabel("Gray Level");
% Label the vertical axis.
ylabel("Probability Density");
% Add a legend to identify the curves.
legend("Location", "best");
% Add a grid for better readability.
grid on;
% Release the hold on the plot.
hold off;

% Create the second subplot for the final segmentation mask.
subplot(1, 2, 2);
% Initialize the segmentation mask with false values.
segmentationMask = false(size(inputImage));

% Loop through each pixel to apply the Bayes decision rule.
for pixelIndex = 1:1:numPixels
    % Extract the intensity value of the current pixel.
    currentPixelValue = imageVector(pixelIndex);
    % Initialize the posterior scores for the current pixel.
    posteriorScores = zeros(1, numClasses);

    % Loop through each Gaussian class.
    for classIndex = 1:1:numClasses
        % Calculate the Gaussian likelihood for the current pixel and class.
        likelihood = (1.0 ./ sqrt(2.0 .* pi .* gaussianVariances(classIndex))) .* exp(-((currentPixelValue - gaussianMeans(classIndex)).^2) ./ (2.0 .* gaussianVariances(classIndex)));
        % Calculate the posterior score by multiplying by the prior.
        posteriorScores(classIndex) = likelihood .* classPriors(classIndex);
        % End the class loop.
    end

    % Find the class with the maximum posterior score.
    [~, bestClassIndex] = max(posteriorScores);
    % Assign the pixel to the best class (Class 2 is true, Class 1 is false).
    if (bestClassIndex == 1)
        segmentationMask(pixelIndex) = true;
        % End the class assignment check.
    end
    % End the pixel loop.
end

% Display the final segmentation mask.
imshow(segmentationMask);
% Add a title to the segmentation subplot.
title("Final EM-Based Segmentation");