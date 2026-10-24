import torch
import torch.nn as nn

class HAR1DCNN(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) model for Human Activity Recognition (HAR)
    using IMU sensor data (e.g., Accelerometer and Gyroscope signals).
    
    This model extracts temporal features using 1D convolutions and classifies them
    into activity categories.
    """
    def __init__(self, in_channels=9, num_classes=6, sequence_length=128):
        """
        Args:
            in_channels (int): Number of input sensor channels/axes (default: 9 for 3-axis accel, gyro, total accel).
            num_classes (int): Number of target activity classes (default: 6).
            sequence_length (int): Length of the input signal window (default: 128).
        """
        super(HAR1DCNN, self).__init__()
        
        # Feature Extraction (Convolutional Layers)
        # Using padding='same' (via padding=1 for kernel_size=3) to keep sequence length consistent
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        
        # Max pooling reduces the sequence length by half (128 -> 64)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size of flattened features:
        # conv outputs 64 channels. After maxpool, sequence_length is halved.
        self.flat_features = 64 * (sequence_length // 2)
        
        # Classification (Fully Connected Layers)
        self.fc1 = nn.Linear(self.flat_features, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor. 
                       Expected shape: (batch_size, in_channels, sequence_length)
                       If input shape is (batch_size, sequence_length, in_channels),
                       it will be automatically permuted.
        """
        # Automatically handle input format (batch_size, sequence_length, channels) 
        # by permuting to PyTorch standard (batch_size, channels, sequence_length)
        if x.shape[1] != self.conv1.in_channels and x.shape[2] == self.conv1.in_channels:
            x = x.permute(0, 2, 1)
            
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        
        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Quick sanity check/validation of the architecture
    model = HAR1DCNN(in_channels=9, num_classes=6, sequence_length=128)
    print("PyTorch Model Architecture:")
    print(model)
    
    # Generate dummy batch: (batch_size=16, channels=9, sequence_length=128)
    dummy_input = torch.randn(16, 9, 128)
    dummy_output = model(dummy_input)
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Output Shape: {dummy_output.shape} (batch_size, num_classes)")

    %% Compare predictions using the corrected model

    % Prepare the input in the verified channel-first layout
    X = single(normalizedData.');
               X = reshape(X, [6 1 128]);

               dlX = dlarray(X, 'CBT');

               % Run MATLAB inference
               Y = predict(net, dlX);

               matlabScores = double(gather(extractdata(Y)));
               matlabScores = matlabScores(:);

               fprintf('\nMATLAB raw scores:\n');
               disp(matlabScores);

               % Display the predicted class
               names = {'Walking'; 'Running'; 'Shuffling'; ...
                        'Stairs (ascending)'; 'Stairs (descending)'; ...
                        'Standing'; 'Sitting'; 'Lying'; ...
                        'Cycling (sit)'; 'Cycling (stand)'; ...
                        'Cycling (sit, inactive)'; ...
                        'Cycling (stand, inactive)'};

               [~, idx] = max(matlabScores);

               fprintf('MATLAB prediction: %s\n', names{idx});
