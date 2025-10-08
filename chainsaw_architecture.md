# Chainsaw Detection System Architecture Documentation

## Overview

This document describes the architecture of the chainsaw detection system implemented in the AI8X training framework. The system is designed to detect and classify chainsaws and other machinery in forest environments using audio processing. The models are optimized for deployment on Analog Devices' MAX78000/MAX78002 microcontrollers.

## System Architecture

The chainsaw detection system employs a two-stage architecture:

1. **Anomaly Detection Stage**: Binary classifier that distinguishes between normal forest sounds and anomalous sounds (chainsaws, vehicles, machinery)
2. **Classification Stage**: Multi-class classifier that identifies the specific type of anomalous sound (chainsaw vs vehicles/machinery)

## Model Architecture

### Anomaly Detection Model

The Anomaly Detection Model is a binary classifier that identifies whether an audio segment contains anomalous sounds versus normal forest sounds.

#### Architecture Details:

- **Input**: Audio segments with shape (batch_size, time_steps=62, features=13)
  - Features: 13 MFCC coefficients extracted from 1-second audio clips
  - Time steps: 62 time steps after feature extraction
- **Output**: Binary classification (normal vs abnormal) with shape (batch_size, 2)

#### Convolutional Feature Extraction:

1. **First Conv Block**:
   - AI8X Custom: `ai8x.FusedMaxPoolConv1dReLU`
     - Input channels: 13
     - Output channels: 32
     - Kernel size: 3
     - Padding: 1
     - Pool size: 2
     - Pool stride: 2
   - PyTorch Equivalent: `nn.Sequential(nn.Conv1d(13, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2))`

2. **Second Conv Block**:
   - AI8X Custom: `ai8x.FusedMaxPoolConv1dReLU`
     - Input channels: 32
     - Output channels: 64
     - Kernel size: 3
     - Padding: 1
     - Pool size: 2
     - Pool stride: 2
   - PyTorch Equivalent: `nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2))`

3. **Third Conv Block**:
   - AI8X Custom: `ai8x.FusedMaxPoolConv1dReLU`
     - Input channels: 64
     - Output channels: 128
     - Kernel size: 3
     - Padding: 1
     - Pool size: 2
     - Pool stride: 2
   - PyTorch Equivalent: `nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2))`

#### Fully Connected Layers:

After convolutional layers, the feature map is flattened and passed through:

1. **Dropout Layer**:
   - AI8X Custom: Standard PyTorch `nn.Dropout` with probability 0.3

2. **First Linear Layer**:
   - AI8X Custom: `ai8x.FusedLinearReLU`
     - Input features: 128 * 7 = 896 (channels * time_steps_after_pools)
     - Output features: 128
   - PyTorch Equivalent: `nn.Sequential(nn.Linear(896, 128), nn.ReLU())`

3. **Second Linear Layer**:
   - AI8X Custom: `ai8x.Linear`
     - Input features: 128
     - Output features: 2 (binary classification)
   - PyTorch Equivalent: `nn.Linear(128, 2)`

#### Forward Method:
- Input: Tensor of shape (batch_size, time_steps, features)
- Reshapes input from (batch, time, features) to (batch, features, time)
- Applies convolutional layers
- Flattens for fully connected layers
- Applies classifier
- Returns logits of shape (batch_size, num_classes)

#### Additional Features:
- Field-adjustable detection threshold with default value of 0.5
- `predict_anomaly` method for direct predictions with probability threshold

### Classification Model

The Classification Model is a multi-class classifier that distinguishes between different types of anomalous sounds (chainsaw vs vehicles/machinery).

#### Architecture Details:

- **Input**: Audio segments with shape (batch_size, time_steps=62, features=13)
  - Features: 13 MFCC coefficients extracted from 1-second audio clips
  - Time steps: 62 time steps after feature extraction
- **Output**: Multi-class classification (chainsaw vs vehicles/machinery) with shape (batch_size, 2)

The Classification Model shares the same architecture as the Anomaly Detection Model:
- Identical convolutional feature extraction layers
- Identical fully connected classification layers
- Input/output shapes optimized for the AI8X framework

### Key AI8X Framework Components

The models utilize several specialized AI8X components optimized for MAX78000/MAX78002 microcontrollers:

1. **ai8x.FusedMaxPoolConv1dReLU**: Fused operation combining max pooling, 1D convolution, and ReLU activation
2. **ai8x.FusedLinearReLU**: Fused operation combining linear transformation and ReLU activation
3. **ai8x.Linear**: Linear transformation optimized for the target hardware

### Model Configuration

From `chainsaw_config.yaml`:

- **Audio Processing**:
  - Sample Rate: 32000 Hz
  - Segment Length: 1.0 seconds
  - Feature Type: MFCC
  - Number of MFCC coefficients: 13
  - Target Length: 62 (time steps for 1-second audio)

- **Model Settings**:
  - Fully Connected Channels: 128
  - Dropout Probability: 0.3
  - Target Length: 62

- **Training Settings**:
  - Batch Size: 32
  - Number of Epochs: 50
  - Learning Rate: 0.001
  - Weight Decay: 0.0001

- **AI8X Settings**:
  - Device ID: 85 (MAX78000)
  - Simulation Mode: False
  - Bias Usage: False

- **Performance Targets**:
  - Minimum Accuracy: 85%
  - Maximum Inference Time: 100ms
  - Memory Limit: 256KB

## Data Pipeline

### Feature Extraction
The system uses MFCC (Mel-Frequency Cepstral Coefficients) as the primary audio feature:
- Number of MFCC coefficients: 13
- Frame length: 1024 samples (FFT window)
- Hop length: 512 samples
- Target time steps: 62 (for 1-second audio at 32kHz)

### Dataset Structure
The input dataset is organized as follows:
- `data/normal/`: Contains normal forest sounds (animals, calm forest, etc.)
- `data/anomaly/chainsaw/`: Contains chainsaw audio samples
- `data/anomaly/machinery/`: Contains other machinery/vehicle audio samples

### Data Augmentation
The system implements several augmentation techniques:
- Time stretching (0.9x and 1.1x speeds)
- Noise injection at different levels
- Pitch shifting (±1, ±2 semitones)

### Class Balancing
The dataset generator implements a hierarchical balancing approach:
1. Equal number of samples between 'normal' and 'abnormal' classes
2. Within 'abnormal', equal number of samples between 'chainsaw' and 'machinery' subclasses

## Deployment Considerations

### Hardware Optimization
- Models are specifically optimized for MAX78000/MAX78002 microcontrollers
- Utilizes AI8X framework for efficient inference on edge devices
- Memory usage constrained to 256KB
- Inference time target of under 100ms

### Field Deployment Features
- Adjustable detection threshold for sensitivity control
- Noise robustness tested through simulation
- Real-time processing capability

## Performance Metrics

The system tracks the following metrics:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC AUC (Area Under the Curve)
- False Positive Rate
- False Negative Rate
- True Positive Rate and True Negative Rate

## Model Usage Examples

### Creating Model Instances
```python
# Get anomaly detection model
anomaly_model = get_anomaly_model(
    num_classes=2,
    dim=(62, 13),
    fc_channels=128
)

# Get classification model
classification_model = get_classification_model(
    num_classes=2,
    dim=(62, 13),
    fc_channels=128
)
```

### Forward Pass
Both models accept input tensors of shape (batch_size, time_steps, features) and return output tensors of shape (batch_size, num_classes).

## Conclusion

This architecture provides an efficient and effective solution for chainsaw detection in forest monitoring applications. The two-stage approach allows for accurate anomaly detection followed by specific classification, while the AI8X optimizations ensure efficient deployment on edge devices. The system is designed to be robust in noisy forest environments while maintaining real-time processing capabilities.