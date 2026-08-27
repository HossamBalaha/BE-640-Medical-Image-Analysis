% Define a function that estimates density using Parzen Window with Gaussian kernel.
function estimatedDensity = Snippet_06_Estimate_Parzen_Density(trainingSamples, queryPoints, bandwidth)
% Get the number of training samples.
numSamples = length(trainingSamples);
% Get the number of query points.
numQueryPoints = length(queryPoints);
% Initialize the density estimate array.
estimatedDensity = zeros(1, numQueryPoints);
% Calculate the Gaussian kernel normalization constant.
kernelConstant = 1.0 / (bandwidth * sqrt(2.0 * pi));

% Loop through each query point to estimate its density.
for queryIndex = 1:1:numQueryPoints
    % Extract the current query point value.
    currentQuery = queryPoints(queryIndex);
    % Initialize the sum of kernel contributions.
    kernelSum = 0;

    % Loop through each training sample.
    for sampleIndex = 1:1:numSamples
        % Extract the current training sample.
        currentSample = trainingSamples(sampleIndex);
        % Calculate the squared distance scaled by bandwidth.
        scaledDistanceSquared = ((currentQuery - currentSample) ^ 2) / (2.0 * bandwidth ^ 2);
        % Add the kernel contribution to the sum.
        kernelSum = kernelSum + exp(-scaledDistanceSquared);
        % End the training sample loop.
    end

    % Store the normalized density estimate for the current query point.
    estimatedDensity(queryIndex) = (1.0 / numSamples) * kernelConstant * kernelSum;
    % End the query point loop.
end

% Normalize the density so that it sums to 1.
estimatedDensity = estimatedDensity / sum(estimatedDensity);
% End the function definition.
end