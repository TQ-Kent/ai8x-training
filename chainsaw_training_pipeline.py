"""
Training Script for Two-Stage Chainsaw Detection Pipeline

This script implements a two-stage training pipeline:
1. Anomaly Detection: Binary classifier (normal vs abnormal sounds)
2. Classification: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)

Features:
- Configurable hyperparameter management
- Comprehensive model evaluation and metrics reporting
- Checkpoint saving/loading functionality
- Integration with AI8X training utilities
- Cross-validation support
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
import json
import time
from datetime import datetime
import copy
from pathlib import Path
import warnings
import shutil

# Import our custom modules
from chainsaw_data_pipeline import get_dataloaders
from chainsaw_model_architecture import AnomalyDetectionNet, ClassificationNet
import ai8x


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, model_save_path, log_path):
    """
    Train a model with validation and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        model_save_path: Path to save the best model
        log_path: Path to save training logs
    
    Returns:
        Best model state dictionary
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    # For logging
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Check for potential overfitting (validation accuracy much lower than training)
            if phase == 'val' and len(train_accuracies) > 0:
                train_acc_current = train_accuracies[-1]
                if train_acc_current - epoch_acc.item() > 0.1:  # If gap is > 10%
                    print(f"  ⚠️  WARNING: Potential overfitting detected! "
                          f"Train Acc: {train_acc_current:.4f}, Val Acc: {epoch_acc.item():.4f}")
            
            # Save metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'accuracy': best_acc,
                }, model_save_path)
                print(f"New best model saved with accuracy: {best_acc:.4f}")

        print()
        
        # Additional checks after each epoch
        if len(train_accuracies) > 1:
            # Check if training accuracy is increasing too quickly (potential issue)
            if train_accuracies[-1] > 0.99 and epoch < num_epochs * 0.3:
                print(f"  ⚠️  WARNING: High training accuracy ({train_accuracies[-1]:.4f}) "
                      f"achieved very early (epoch {epoch+1}). Possible data leakage.")
            if 'val_accuracies' in locals() and len(val_accuracies) > 0:
                if val_accuracies[-1] > 0.99 and epoch < num_epochs * 0.3:
                    print(f"  ⚠️  WARNING: High validation accuracy ({val_accuracies[-1]:.4f}) "
                          f"achieved very early (epoch {epoch+1}). Possible data leakage.")

    print(f'Best val Acc: {best_acc:.4f}, Best val Loss: {best_loss:.4f}')
    
    # Save training logs
    logs = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_accuracy': best_acc.item(),
        'best_loss': best_loss
    }
    
    with open(log_path, 'w') as f:
        json.dump(logs, f)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        criterion: Optional loss function for computing test loss
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, 
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'test_loss': running_loss / len(test_loader.dataset) if criterion else None
    }
    
    return metrics


def train_anomaly_detection(args):
    """
    Train the anomaly detection model (stage 1).
    
    Args:
        args: Command line arguments
    """
    print("Starting Anomaly Detection Training (Stage 1)...")
    
    # Set device - force CPU for compatibility
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    anomaly_dir = os.path.join(args.output_dir, 'anomaly_detection')
    os.makedirs(anomaly_dir, exist_ok=True)
    
    # Update data directory to point to full dataset for anomaly detection
    anomaly_data_dir = args.data_dir  # Should contain normal/ and abnormal/ subdirectories
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=anomaly_data_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        feature_type=args.feature_type,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        target_length=args.target_length,
        random_state=args.seed,
        task_type='anomaly'  # Specify this is for anomaly detection task
    )
    
    print(f"Collected {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} samples")
    print("Dataset split:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Initialize model
    print("Initializing Anomaly Detection Model...")
    ai8x.set_device(args.device_id, args.simulate, args.round_avg)
    
    model = AnomalyDetectionNet(
        num_channels=args.n_mfcc,
        feature_length=args.target_length,
        num_classes=2,  # normal vs abnormal
        dim=(args.target_length, args.n_mfcc),
        fc_channels=args.fc_channels,
        dropout_prob=args.dropout,
        bias=args.use_bias
    )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Create model save path and log path
    model_save_path = os.path.join(anomaly_dir, 'best_anomaly_model.pth')
    log_path = os.path.join(anomaly_dir, 'training_logs.json')
    
    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        model_save_path=model_save_path,
        log_path=log_path
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print(f"Test Metrics: {test_metrics}")
    
    # Save test metrics
    test_metrics_path = os.path.join(anomaly_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("Anomaly Detection Training Completed!")


def train_classification(args):
    """
    Train the classification model (stage 2).
    
    Args:
        args: Command line arguments
    """
    print("Starting Classification Training (Stage 2)...")
    
    # Set device - force CPU for compatibility
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    classification_dir = os.path.join(args.output_dir, 'classification')
    os.makedirs(classification_dir, exist_ok=True)
    
    # For classification, we specifically need the abnormal subdirectories
    # Create a temporary dataset that only includes abnormal sounds
    abnormal_data_dir = os.path.join(args.data_dir, 'abnormal')
    if not os.path.exists(abnormal_data_dir):
        # If data is organized differently, try to find it
        # Look for chainsaw and machinery subdirectories
        if os.path.exists(os.path.join(args.data_dir, 'chainsaw')) or os.path.exists(os.path.join(args.data_dir, 'machinery')):
            # If these directories are at the top level, we need to create a temp structure
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            abnormal_data_dir = os.path.join(temp_dir, 'abnormal')
            os.makedirs(abnormal_data_dir, exist_ok=True)
            
            # Create links or copies of the chainsaw and machinery directories
            if os.path.exists(os.path.join(args.data_dir, 'chainsaw')):
                shutil.copytree(os.path.join(args.data_dir, 'chainsaw'), 
                              os.path.join(abnormal_data_dir, 'chainsaw'))
            if os.path.exists(os.path.join(args.data_dir, 'machinery')):
                shutil.copytree(os.path.join(args.data_dir, 'machinery'), 
                              os.path.join(abnormal_data_dir, 'machinery'))
        else:
            print(f"Error: Cannot find abnormal data in {args.data_dir}")
            return
    
    # Create data loaders for abnormal sounds only (chainsaw vs machinery)
    print("Creating data loaders for classification...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=abnormal_data_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        feature_type=args.feature_type,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        target_length=args.target_length,
        random_state=args.seed,
        task_type='classification'  # Specify this is for classification task
    )
    
    print(f"Collected {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} samples")
    print("Dataset split:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Initialize model
    print("Initializing Classification Model...")
    ai8x.set_device(args.device_id, args.simulate, args.round_avg)
    
    model = ClassificationNet(
        num_channels=args.n_mfcc,
        feature_length=args.target_length,
        num_classes=2,  # chainsaw vs machinery
        dim=(args.target_length, args.n_mfcc),
        fc_channels=args.fc_channels,
        dropout_prob=args.dropout,
        bias=args.use_bias
    )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Create model save path and log path
    model_save_path = os.path.join(classification_dir, 'best_classification_model.pth')
    log_path = os.path.join(classification_dir, 'training_logs.json')
    
    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        model_save_path=model_save_path,
        log_path=log_path
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print(f"Test Metrics: {test_metrics}")
    
    # Save test metrics
    test_metrics_path = os.path.join(classification_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("Classification Training Completed!")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage Chainsaw Detection Training')
    
    # Data and directory settings
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the organized dataset')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models and logs')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Data splitting
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation')
    
    # Feature extraction settings
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'mel', 'spectrogram'],
                        help='Type of features to extract')
    parser.add_argument('--n_mels', type=int, default=40,
                        help='Number of mel bands')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length for STFT')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Sample rate of audio files')
    parser.add_argument('--target_length', type=int, default=62,
                        help='Target length for features')
    
    # Model settings
    parser.add_argument('--fc_channels', type=int, default=128,
                        help='Number of channels in fully connected layers')
    
    # Learning rate scheduling
    parser.add_argument('--lr_step_size', type=int, default=20,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Gamma for learning rate scheduler')
    
    # AI8X settings
    parser.add_argument('--device_id', type=int, default=85,
                        help='AI8X device ID (85 for MAX78000, 87 for MAX78002)')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate hardware behavior')
    parser.add_argument('--round_avg', action='store_true',
                        help='Use rounding for average pooling')
    parser.add_argument('--use_bias', action='store_true',
                        help='Use bias in layers')
    
    # Hardware settings
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Training stage selection
    parser.add_argument('--stage', type=str, default='both',
                        choices=['anomaly', 'classification', 'both'],
                        help='Which stage(s) to train')
    
    args = parser.parse_args()
    
    print("Starting Two-Stage Chainsaw Detection Training")
    print(f"Configuration: {args}")
    
    if args.stage in ['anomaly', 'both']:
        train_anomaly_detection(args)
    
    if args.stage in ['classification', 'both']:
        train_classification(args)
    
    print("Training pipeline completed!")


if __name__ == "__main__":
    main()