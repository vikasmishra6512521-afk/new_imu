
clc;
clear;
close all;

%% 1. Load the corrected ONNX model
modelFile = '/MATLAB Drive/new imu/best_model_corrected.onnx';

if ~isfile(modelFile)
    error('Model not found at: %s', modelFile);
end

net = importNetworkFromONNX(modelFile);

disp('Corrected ONNX model loaded successfully.');

%% 2. Select the dataset
[file, path] = uigetfile('*.csv', 'Select S029.csv');

if isequal(file, 0)
    error('No dataset selected.');
end

data = readtable(fullfile(path, file));

fprintf('Dataset loaded: %s\n', file);

%% 3. Select the 128-sample window
startIdx = 1001;
sequenceLength = 128;
endIdx = startIdx + sequenceLength - 1;

if height(data) < endIdx
    error('Dataset must contain at least 1128 rows.');
end

windowData = data(startIdx:endIdx, :);

%% 4. Extract the six sensor channels
featureCols = {'back_x', 'back_y', 'back_z', ...
               'thigh_x', 'thigh_y', 'thigh_z'};

if ~all(ismember(featureCols, data.Properties.VariableNames))
    error('Required IMU columns are missing.');
end

rawFeatures = windowData{:, featureCols};

if any(~isfinite(rawFeatures), 'all')
    error('Sensor data contains NaN or Inf.');
end

%% 5. Normalize using training-set parameters
MU = [-0.885417, 0.001669, -0.147612, ...
      -0.507975, 0.059537, 0.532519];

SIGMA = [0.295474, 0.225258, 0.338422, ...
         0.499131, 0.308348, 0.587967];

normalizedData = (rawFeatures - MU) ./ SIGMA;

%% 6. Get the ground-truth activity
RAW_LABELS = [1 2 3 4 5 6 7 8 13 14 130 140];

names = {'Walking'; 'Running'; 'Shuffling'; ...
    'Stairs (ascending)'; 'Stairs (descending)'; ...
    'Standing'; 'Sitting'; 'Lying'; ...
    'Cycling (sit)'; 'Cycling (stand)'; ...
    'Cycling (sit, inactive)'; ...
    'Cycling (stand, inactive)'};

rawLabel = windowData.label(65);

labelIdx = find(RAW_LABELS == rawLabel, 1);

if isempty(labelIdx)
    trueActivity = 'Unknown';
else
    trueActivity = names{labelIdx};
end

%% 7. Prepare input for MATLAB inference
% Channels x Batch x Time
X = reshape(single(normalizedData.'), [6 1 128]);

fprintf('\nInput dimensions [C B T]:\n');
disp(size(X));

dlX = dlarray(X, 'CBT');

%% 8. Run inference
Y = predict(net, dlX);

rawScores = double(gather(extractdata(Y)));
rawScores = rawScores(:);

if numel(rawScores) ~= 12
    error('Expected 12 output scores; received %d.', ...
          numel(rawScores));
end

%% 9. Display raw scores
fprintf('\nMATLAB raw output:\n');
disp(rawScores);

%% 10. Calculate probabilities
% Assumes the network outputs logits.
shiftedScores = rawScores - max(rawScores);
probabilities = exp(shiftedScores);
probabilities = probabilities ./ sum(probabilities);

[confidence, predictedIdx] = max(probabilities);

predictedActivity = names{predictedIdx};

%% 11. Display results
fprintf('\n====== IMU Activity Recognition ======\n');
fprintf('Dataset            : %s\n', file);
fprintf('True Activity      : %s (raw label: %g)\n', ...
        trueActivity, rawLabel);
fprintf('Predicted Activity : %s\n', predictedActivity);
fprintf('Confidence         : %.2f%%\n', confidence * 100);
fprintf('======================================\n');

if predictedIdx == labelIdx
    fprintf('Prediction matches ground truth.\n');
else
    fprintf('Prediction does not match ground truth.\n');
end

%% 12. Display all class probabilities
resultTable = table(names, probabilities * 100, ...
    'VariableNames', {'Activity', 'ProbabilityPercent'});

disp(resultTable);

%% 13. Plot activity probabilities
figure('Name', 'HAR Activity Prediction');

bar(probabilities * 100);

xticks(1:12);
xticklabels(names);
xtickangle(45);

ylabel('Probability (%)');
title(sprintf('Predicted Activity: %s (%.2f%%)', ...
    predictedActivity, confidence * 100));

grid on;

% Diagnostic: inspect network output before softmax
disp("----- NETWORK OUTPUT DIAGNOSTIC -----");

disp("Input dimensions:");
disp(size(X));

disp("Raw network output:");
disp(Y);

disp("Output dimensions:");
disp(size(Y));

disp("Output format:");
disp(dims(Y));
