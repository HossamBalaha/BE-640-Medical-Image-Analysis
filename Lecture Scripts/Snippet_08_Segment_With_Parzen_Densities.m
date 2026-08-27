% Define a function that segments a test image using Parzen Window densities.
function segmentationMask = Snippet_08_Segment_With_Parzen_Densities(testImage, lungSamples, chestSamples, bandwidth)
% Define the array of gray levels from 0 to 255.
grayLevels = 0:1:255;

% Estimate the lung density using Parzen Window.
pLung = Snippet_06_Estimate_Parzen_Density(lungSamples, grayLevels, bandwidth);
% Estimate the chest density using Parzen Window.
pChest = Snippet_06_Estimate_Parzen_Density(chestSamples, grayLevels, bandwidth);

% Calculate the number of lung training samples.
numLungSamples = length(lungSamples);
% Calculate the number of chest training samples.
numChestSamples = length(chestSamples);
% Calculate the total number of training samples.
totalSamples = numLungSamples + numChestSamples;

% Compute the prior probability for the lung class.
lungPrior = numLungSamples / totalSamples;
% Compute the prior probability for the chest class.
chestPrior = numChestSamples / totalSamples;

% Multiply the lung density by its prior to get the posterior score.
lungScore = pLung * lungPrior;
% Multiply the chest density by its prior to get the posterior score.
chestScore = pChest * chestPrior;

% Get the dimensions of the test image.
[imageHeight, imageWidth] = size(testImage);
% Initialize the segmentation mask.
segmentationMask = false(imageHeight, imageWidth);

% Loop through each row of the image.
for rowIndex = 1:1:imageHeight
    % Loop through each column of the image.
    for colIndex = 1:1:imageWidth
        % Extract the gray level of the current pixel.
        pixelGrayLevel = testImage(rowIndex, colIndex);
        % Check if the lung score is greater than or equal to the chest score.
        if (lungScore(pixelGrayLevel + 1) >= chestScore(pixelGrayLevel + 1))
            % Assign the pixel to the lung class.
            segmentationMask(rowIndex, colIndex) = true;
            % End the conditional block.
        end
        % End the column loop.
    end
    % End the row loop.
end
% End the function definition.
end