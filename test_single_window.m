
clc;
clear;
close all;

%% 1. Load the trained ONNX model
modelFile = 'best_model.onnx';

if ~isfile(modelFile)
    [file, path] = uigetfile('*.onnx', 'Select best_model.onnx');

    if isequal(file, 0)
        error('No ONNX model selected.');
    end

    modelFile = fullfile(path, file);
end

net = importNetworkFromONNX(modelFile);

fprintf('ONNX model loaded successfully.\n');

%% 2. Define activity names and raw label mapping
LABEL_NAMES = {
    'Walking'
    'Running'
    'Shuffling'
    'Stairs (ascending)'
    'Stairs (descending)'
    'Standing'
    'Sitting'
    'Lying'
    'Cycling (sit)'
    'Cycling (stand)'
    'Cycling (sit, inactive)'
    'Cycling (stand, inactive)'
};

RAW_LABELS = [1 2 3 4 5 6 7 8 13 14 130 140];

%% 3. Normalization parameters from Python training
MU = [-0.885417, 0.001669, -0.147612, ...
      -0.507975, 0.059537, 0.532519];

SIGMA = [0.295474, 0.225258, 0.338422, ...
         0.499131, 0.308348, 0.587967];

%% 4. Select the dataset
[file, path] = uigetfile('*.csv', ...
    'Select the S029.csv dataset');

if isequal(file, 0)
    error('No dataset selected.');
end

data = readtable(fullfile(path, file));

fprintf('Dataset loaded: %s\n', file);

%% 5. Extract a 128-sample window
startIdx = 1001;
sequenceLength = 128;
endIdx = startIdx + sequenceLength - 1;

if height(data) < endIdx
    error('Dataset must contain at least 1128 rows.');
end

windowData = data(startIdx:endIdx, :);

%% 6. Extract the six sensor channels
featureCols = {
    'back_x'
    'back_y'
    'back_z'
    'thigh_x'
    'thigh_y'
    'thigh_z'
};

if ~all(ismember(featureCols, data.Properties.VariableNames))
    error('One or more required sensor columns are missing.');
end

rawFeatures = windowData{:, featureCols};

if any(~isfinite(rawFeatures), 'all')
    error('Sensor data contains NaN or Inf values.');
end

%% 7. Determine the ground-truth activity
% Center sample: Python index 64 = MATLAB index 65
rawLabel = windowData.label(65);

labelIdx = find(RAW_LABELS == rawLabel, 1);

if isempty(labelIdx)
    trueClassIdx = -1;
    trueActivity = 'Unknown';
else
    trueClassIdx = labelIdx;
    trueActivity = LABEL_NAMES{labelIdx};
end

%% 8. Normalize sensor features
normalizedData = (rawFeatures - MU) ./ SIGMA;

%% 9. Prepare the model input
% normalizedData is [128 x 6].
% Transpose to [6 x 128], then arrange as [C x B x T].
X = reshape(single(normalizedData.'), [6 1 128]);

fprintf('\nInput dimensions [C B T]:\n');
disp(size(X));

dlX = dlarray(X, 'CBT');

%% 10. Run model inference
Y = predict(net, dlX);

rawScores = double(gather(extractdata(Y)));
rawScores = squeeze(rawScores);
rawScores = rawScores(:);

if numel(rawScores) ~= 12
    error('Expected 12 output scores; received %d.', ...
        numel(rawScores));
end

fprintf('Raw network output:\n');
disp(rawScores);

%% 11. Convert logits to probabilities
% This assumes the ONNX model outputs logits, not probabilities.
shiftedScores = rawScores - max(rawScores);
probabilities = exp(shiftedScores) ./ sum(exp(shiftedScores));

[confidence, predictedIdx] = max(probabilities);
confidence = confidence * 100;

predictedActivity = LABEL_NAMES{predictedIdx};

%% 12. Display inference results
fprintf('\n====== IMU Activity Recognition ======\n');
fprintf('Dataset            : %s\n', file);
fprintf('True Activity      : %s (raw label: %g)\n', ...
    trueActivity, rawLabel);
fprintf('Predicted Activity : %s\n', predictedActivity);
fprintf('Confidence         : %.2f%%\n', confidence);
fprintf('======================================\n');

if predictedIdx == trueClassIdx
    fprintf('Prediction matches ground truth.\n');
else
    fprintf('Prediction does not match ground truth.\n');
end

%% 13. Display probabilities for all activities
resultTable = table( ...
    LABEL_NAMES, probabilities * 100, ...
    'VariableNames', {'Activity', 'ProbabilityPercent'});

disp(resultTable);

%% 14. Plot the class probabilities
figure('Name', 'HAR Activity Prediction');

bar(probabilities * 100);

xticks(1:12);
xticklabels(LABEL_NAMES);
xtickangle(45);

ylabel('Probability (%)');
title(sprintf('Predicted Activity: %s (%.2f%%)', ...
    predictedActivity, confidence));

grid on;
