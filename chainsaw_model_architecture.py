"""
Model Architecture for Chainsaw Detection System

This module implements two models:
1. Anomaly Detection Model: Binary classifier (normal vs abnormal sounds)
2. Classification Model: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)

Both models are optimized for MAX78000 memory and processing constraints
and are compatible with the AI8X framework.
"""
import torch
import torch.nn as nn
import ai8x
from typing import Tuple, Optional
import numpy as np


class AnomalyDetectionNet(nn.Module):
    """
    Anomaly Detection Model: Binary classifier (normal vs abnormal sounds)
    
    This model detects whether an audio segment contains anomalous sounds
    (chainsaws, vehicles, etc.) versus normal forest sounds.
    """
    def __init__(self, 
                 num_channels: int = 13,  # Number of MFCC features
                 feature_length: int = 62,  # Time steps after feature extraction
                 num_classes: int = 2,  # normal vs abnormal
                 dim: Tuple[int, int] = (62, 13),  # (time_steps, features)
                 fc_channels: int = 128,
                 dropout_prob: float = 0.3,
                 bias: bool = False, 
                 **kwargs):
        """
        Initialize the Anomaly Detection Model.
        
        Args:
            num_channels: Number of input feature channels (e.g., MFCC coefficients)
            feature_length: Length of the time dimension in features
            num_classes: Number of output classes (2 for binary classification)
            dim: Input dimensions as (height, width) or (time, features)
            fc_channels: Number of channels in fully connected layers
            dropout_prob: Dropout probability for regularization
            bias: Whether to use bias in layers
            **kwargs: Additional arguments passed to AI8X layers
        """
        super(AnomalyDetectionNet, self).__init__()
        
        # Store dimensions
        self.num_channels = num_channels
        self.feature_length = feature_length
        self.num_classes = num_classes
        self.dim = dim
        self.fc_channels = fc_channels
        
        # Convolutional feature extraction layers
        # Input: (batch, channels, height, width) -> (batch, 13, 62, 1) after reshaping
        # For 1D time series, we can use 1D convolutions or 2D with one spatial dimension
        
        self.conv_layers = nn.Sequential(
            # First conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
            
            # Second conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
            
            # Third conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
        )
        
        # Calculate the size after convolutions
        # Input: 62 time steps
        # After first pool: 62/2 = 31
        # After second pool: 31/2 = 15 (floor)
        # After third pool: 15/2 = 7 (floor)
        self.conv_output_size = 128 * 7  # channels * time_steps_after_pools
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            ai8x.FusedLinearReLU(
                in_features=self.conv_output_size,
                out_features=fc_channels,
                bias=bias,
                **kwargs
            ),
            nn.Dropout(dropout_prob),
            ai8x.Linear(
                in_features=fc_channels,
                out_features=num_classes,
                bias=bias,
                **kwargs
            )
        )
        
        # Field-adjustable threshold parameter
        self.register_buffer('detection_threshold', torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the anomaly detection model.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input if necessary: (batch, time, features) -> (batch, features, time, 1)
        if len(x.shape) == 3:
            # For 1D convolutions, we can keep it as (batch, features, time)
            x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        elif len(x.shape) == 4:
            # Already in (batch, channels, height, width) format
            pass
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def set_detection_threshold(self, threshold: float):
        """
        Set the detection threshold for anomaly classification.
        
        Args:
            threshold: New threshold value between 0 and 1
        """
        self.detection_threshold.fill_(threshold)
    
    def predict_anomaly(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict if input is anomalous and return probability.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            anomaly_probs = probabilities[:, 1]  # Probability of anomalous class
            predictions = (anomaly_probs > self.detection_threshold).long()
            
        return predictions, anomaly_probs


class ClassificationNet(nn.Module):
    """
    Classification Model: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)
    
    This model classifies abnormal sounds into specific categories.
    """
    def __init__(self, 
                 num_channels: int = 13,  # Number of MFCC features
                 feature_length: int = 62,  # Time steps after feature extraction
                 num_classes: int = 2,  # chainsaw vs vehicles
                 dim: Tuple[int, int] = (62, 13),  # (time_steps, features)
                 fc_channels: int = 128,
                 dropout_prob: float = 0.3,
                 bias: bool = False, 
                 **kwargs):
        """
        Initialize the Classification Model.
        
        Args:
            num_channels: Number of input feature channels (e.g., MFCC coefficients)
            feature_length: Length of the time dimension in features
            num_classes: Number of output classes (2 for chainsaw vs vehicles)
            dim: Input dimensions as (height, width) or (time, features)
            fc_channels: Number of channels in fully connected layers
            dropout_prob: Dropout probability for regularization
            bias: Whether to use bias in layers
            **kwargs: Additional arguments passed to AI8X layers
        """
        super(ClassificationNet, self).__init__()
        
        # Store dimensions
        self.num_channels = num_channels
        self.feature_length = feature_length
        self.num_classes = num_classes
        self.dim = dim
        self.fc_channels = fc_channels
        
        # Convolutional feature extraction layers
        self.conv_layers = nn.Sequential(
            # First conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
            
            # Second conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
            
            # Third conv block
            ai8x.FusedMaxPoolConv1dReLU(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                pool_size=2,
                pool_stride=2,
                bias=bias,
                **kwargs
            ),
        )
        
        # Calculate the size after convolutions
        self.conv_output_size = 128 * 7  # channels * time_steps_after_pools
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            ai8x.FusedLinearReLU(
                in_features=self.conv_output_size,
                out_features=fc_channels,
                bias=bias,
                **kwargs
            ),
            nn.Dropout(dropout_prob),
            ai8x.Linear(
                in_features=fc_channels,
                out_features=num_classes,
                bias=bias,
                **kwargs
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classification model.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input if necessary
        if len(x.shape) == 3:
            x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


def get_anomaly_model(num_classes: int = 2, dim: Tuple[int, int] = (62, 13), 
                     fc_channels: int = 128, pretrained: bool = False, 
                     **kwargs) -> AnomalyDetectionNet:
    """
    Get an instance of the anomaly detection model.
    
    Args:
        num_classes: Number of output classes (default 2: normal vs abnormal)
        dim: Input dimensions
        fc_channels: Number of channels in fully connected layers
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to the model
        
    Returns:
        AnomalyDetectionNet instance
    """
    model = AnomalyDetectionNet(
        num_classes=num_classes,
        dim=dim,
        fc_channels=fc_channels,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights if available
        pass
    
    return model


def get_classification_model(num_classes: int = 2, dim: Tuple[int, int] = (62, 13), 
                            fc_channels: int = 128, pretrained: bool = False, 
                            **kwargs) -> ClassificationNet:
    """
    Get an instance of the classification model.
    
    Args:
        num_classes: Number of output classes (default 2: chainsaw vs vehicles)
        dim: Input dimensions
        fc_channels: Number of channels in fully connected layers
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to the model
        
    Returns:
        ClassificationNet instance
    """
    model = ClassificationNet(
        num_classes=num_classes,
        dim=dim,
        fc_channels=fc_channels,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights if available
        pass
    
    return model


# Model definitions for compatibility with AI8X training framework
models = [
    {
        'name': 'chainsaw_anomaly_detector',
        'model': get_anomaly_model,
        'dim': (62, 13),  # Expected input dimensions (time_steps, features)
        'out_shape': (2,),  # Output shape for binary classification
    },
    {
        'name': 'chainsaw_classifier',
        'model': get_classification_model,
        'dim': (62, 13),  # Expected input dimensions (time_steps, features)
        'out_shape': (2,),  # Output shape for binary classification (chainsaw vs vehicles)
    }
]


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Test anomaly detection model
    print("Testing Anomaly Detection Model...")
    anomaly_model = AnomalyDetectionNet(num_channels=13, feature_length=62)
    
    # Create dummy input (batch_size=4, time_steps=62, features=13)
    dummy_input = torch.randn(4, 62, 13)
    output = anomaly_model(dummy_input)
    print(f"Anomaly model output shape: {output.shape}")
    
    # Test with detection threshold
    predictions, probs = anomaly_model.predict_anomaly(dummy_input)
    print(f"Predictions shape: {predictions.shape}, Probabilities shape: {probs.shape}")
    
    # Test classification model
    print("\nTesting Classification Model...")
    classification_model = ClassificationNet(num_channels=13, feature_length=62)
    output = classification_model(dummy_input)
    print(f"Classification model output shape: {output.shape}")
    
    print("\nModel architectures created successfully!")