% Define a function that segments a test image using the estimated Gaussian intensity models.
function segmentationMask = Snippet_04_Segment_Image_With_Intensity_Models(testImage, modelParameters)
% Extract the lung mean from the model structure.
lungMean = modelParameters.LungMean;
% Extract the lung variance from the model structure.
lungVariance = modelParameters.LungVariance;
% Extract the lung prior probability from the model structure.
lungPrior = modelParameters.LungPrior;
% Extract the chest mean from the model structure.
chestMean = modelParameters.ChestMean;
% Extract the chest variance from the model structure.
chestVariance = modelParameters.ChestVariance;
% Extract the chest prior probability from the model structure.
chestPrior = modelParameters.ChestPrior;
% Compute the Gaussian likelihood of the lung class for every pixel.
lungLikelihood = (1 ./ sqrt(2 .* pi .* lungVariance)) .* exp(-((testImage - lungMean).^2) ./ (2 .* lungVariance));
% Compute the Gaussian likelihood of the chest class for every pixel.
chestLikelihood = (1 ./ sqrt(2 .* pi .* chestVariance)) .* exp(-((testImage - chestMean).^2) ./ (2 .* chestVariance));
% Multiply the lung likelihood by the lung prior to obtain the posterior score.
lungScore = lungLikelihood .* lungPrior;
% Multiply the chest likelihood by the chest prior to obtain the posterior score.
chestScore = chestLikelihood .* chestPrior;
% Assign each pixel to the lung class when its lung score exceeds its chest score.
segmentationMask = lungScore > chestScore;
% End the function definition.
end