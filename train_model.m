%% HARTH Motion Classification using 1D-CNN + BiLSTM
% Setup directory and parameters
clear; clc; close all;

dataDir = fullfile(pwd, 'harth dataset');
if ~exist(dataDir, 'dir')
    error('Directory "%s" not found. Ensure the dataset folder is located inside your current folder.', dataDir);
end

% HARTH sampling frequency is 50 Hz
Fs = 50; 
windowSec = 2.5; 
windowSize = round(Fs * windowSec); % 125 samples per window
stepSize = round(windowSize * 0.5);  % 50% overlap

% Activity mapping based on HARTH dataset documentation
labelMap = containers.Map(...
    [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140], ...
    {'Walking', 'Running', 'Shuffling', 'Stairs (ascending)', ...
     'Stairs (descending)', 'Standing', 'Sitting', 'Lying', ...
     'Cycling (sit)', 'Cycling (stand)', 'Cycling (sit, inactive)', 'Cycling (stand, inactive)'} ...
);

csvFiles = dir(fullfile(dataDir, '*.csv'));
if isempty(csvFiles)
    error('No CSV files found in %s', dataDir);
end

fprintf('Found %d subject files. Extracting sliding windows...\n', numel(csvFiles));

%% Data Ingestion and Window Segmentation
X = {};
Y = [];

for i = 1:numel(csvFiles)
    filePath = fullfile(csvFiles(i).folder, csvFiles(i).name);
    fprintf('Processing: %s (%d/%d)\n', csvFiles(i).name, i, numel(csvFiles));
    
    opts = detectImportOptions(filePath);
    opts.VariableNamingRule = 'preserve';
    rawTable = readtable(filePath, opts);
    
    % Required channels: 3-axis back acceleration and 3-axis thigh acceleration
    backAcc = [rawTable.back_x, rawTable.back_y, rawTable.back_z];
    thighAcc = [rawTable.thigh_x, rawTable.thigh_y, rawTable.thigh_z];
    signals = [backAcc, thighAcc]'; % Dimensions: [6 x N]
    labels = rawTable.label;
    
    numSamples = size(signals, 2);
    
    for startIdx = 1:stepSize:(numSamples - windowSize + 1)
        endIdx = startIdx + windowSize - 1;
        
        windowSignals = signals(:, startIdx:endIdx);
        windowLabels = labels(startIdx:endIdx);
        
        % Majority voting for window label
        modeLabel = mode(windowLabels);
        
        % Exclude unlabeled segments (code 0) or unmapped keys
        if isKey(labelMap, modeLabel)
            X{end+1, 1} = single(windowSignals); %#ok<AGROW>
            Y = [Y; string(labelMap(modeLabel))]; %#ok<AGROW>
        end
    end
end

Y = categorical(Y);
fprintf('\nTotal windowed segments generated: %d\n', numel(X));

%% Partitioning: Stratified Train / Test Split
cv = cvpartition(Y, 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx);
YTrain = Y(trainIdx);

XTest = X(testIdx);
YTest = Y(testIdx);

numFeatures = size(XTrain{1}, 1); % 6 channels
numClasses = numel(categories(YTrain));

%% Deep Learning Architecture (1D-CNN + BiLSTM)
layers = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore', 'Name', 'input')
    
    % Spatial/local temporal feature extraction
    convolution1dLayer(5, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')
    
    convolution1dLayer(5, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'pool2')
    
    % Sequential dependencies across motions
    bilstmLayer(100, 'OutputMode', 'last', 'Name', 'bilstm')
    dropoutLayer(0.4, 'Name', 'drop')
    
    % Classification output
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Training Configuration
miniBatchSize = 128;
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 8, ...
    'LearnRateDropFactor', 0.2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto');

%% Train Network
fprintf('\nTraining 1D-CNN + BiLSTM model...\n');
trainedNet = trainNetwork(XTrain, YTrain, layers, options);

%% Evaluation & Model Persistence
YPred = classify(trainedNet, XTest, 'MiniBatchSize', miniBatchSize);
testAccuracy = mean(YPred == YTest);
fprintf('\nTest Set Accuracy: %.2f%%\n', testAccuracy * 100);

figure('Name', 'Activity Recognition Confusion Matrix');
confusionchart(YTest, YPred, 'Normalization', 'row-normalized', 'Title', 'HARTH Classification Performance');

save('harth_trained_model.mat', 'trainedNet', 'labelMap', 'windowSize', 'Fs');
fprintf('Model saved as "harth_trained_model.mat" successfully.\n');