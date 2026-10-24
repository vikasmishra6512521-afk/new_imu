
clc;
clear;
close all;

%% 1. Activity names and label mapping
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

% Maps raw dataset labels to MATLAB class indices (1-based)
RAW_LABELS = [1 2 3 4 5 6 7 8 13 14 130 140];
CLASS_INDICES = 1:12;

%% 2. Normalization parameters from training
MEAN = [-0.885417, 0.001669, -0.147612, ...
        -0.507975, 0.059537, 0.532519];

STD = [0.295474, 0.225258, 0.338422, ...
       0.499131, 0.308348, 0.587967];

%% 3. Load the trained model
% Convert best_model.pth to best_model.onnx first.
modelFile = 'best_model.onnx';

if ~isfile(modelFile)
    error(['Model file not found: %s\n' ...
           'Convert best_model.pth to ONNX first.'], modelFile);
end

net = importNetworkFromONNX(modelFile);

%% 4. Load the test dataset
testFile = fullfile('dataset', 'harth', 'S029.csv');

if ~isfile(testFile)
    error('Dataset not found: %s', testFile);
end

fprintf('Reading sample from %s...\n', testFile);

data = readtable(testFile);

%% 5. Select a window of 128 samples
startIdx = 1001;  % MATLAB uses 1-based indexing
sequenceLength = 128;

endIdx = startIdx + sequenceLength - 1;

if height(data) < endIdx
    error('Dataset does not contain enough rows for this window.');
end

windowData = data(startIdx:endIdx, :);

%% 6. Extract the six IMU features
featureCols = {
    'back_x'
    'back_y'
    'back_z'
    'thigh_x'
    'thigh_y'
    'thigh_z'
};

rawFeatures = windowData{:, featureCols};

% Check for missing or invalid sensor values
if any(~isfinite(rawFeatures), 'all')
    error('Selected sensor window contains NaN or Inf values.');
end

%% 7. Get the ground-truth label from the center
rawLabel = windowData.label(65);

mapIdx = find(RAW_LABELS == rawLabel, 1);

if isempty(mapIdx)
    trueClassIdx = -1;
    trueClassName = 'Unknown';
else
    trueClassIdx = CLASS_INDICES(mapIdx);
    trueClassName = LABEL_NAMES{trueClassIdx};
end

%% 8. Normalize the features
normFeatures = (rawFeatures - MEAN) ./ STD;

%% 9. Prepare the model input
% This assumes the exported ONNX model accepts
% input in the shape [batch, sequence, channels].
%
% Verify the ONNX model's input shape and format.
inputData = reshape(single(normFeatures), [1, 128, 6]);

% For a dlnetwork supporting batch-time-channel format:
dlX = dlarray(inputData, 'BTC');

%% 10. Run inference
scores = predict(net, dlX);

% Convert output to a regular numeric array
scores = gather(extractdata(scores));
scores = double(squeeze(scores));
scores = scores(:);

if numel(scores) ~= 12
    error(['Expected 12 output scores, but received %d. ' ...
           'Check the imported model output shape.'], numel(scores));
end

% Convert logits to probabilities using softmax
scores = scores - max(scores);
probabilities = exp(scores) ./ sum(exp(scores));

[confidence, predictedIdx] = max(probabilities);
confidence = confidence * 100;

predictedClassName = LABEL_NAMES{predictedIdx};

%% 11. Display the results
fprintf('\n========== Inference Result ==========\n');

fprintf('True Activity      : %s (Raw Label: %g)\n', ...
        trueClassName, rawLabel);

fprintf('Predicted Activity : %s (%.2f%% confidence)\n', ...
        predictedClassName, confidence);

fprintf('======================================\n');

if predictedIdx == trueClassIdx
    fprintf('Success! Prediction matches the true label.\n');
else
    fprintf('Prediction does not match.\n');
end
