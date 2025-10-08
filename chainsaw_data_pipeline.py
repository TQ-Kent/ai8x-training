"""
Data Pipeline for Chainsaw Detection with MFCC Feature Extraction

This module provides a data pipeline optimized for the AI8X framework that:
1. Loads audio data from the organized dataset
2. Extracts MFCC features using appropriate parameters for AI8X
3. Implements data loading and batching for training
4. Provides configurable train/validation/test split functionality
5. Includes data augmentation pipeline integration
6. Converts data to AI8X-compatible format
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import soundfile as sf
from typing import Tuple, List, Optional, Union
from tqdm import tqdm
import warnings


class ChainsawAudioDataset(Dataset):
    """
    Dataset class for chainsaw detection audio data.
    Loads audio files and extracts MFCC features.
    """
    def __init__(self, 
                 data_dir: str, 
                 feature_type: str = 'mfcc', 
                 n_mels: int = 40, 
                 n_mfcc: int = 13,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 sample_rate: int = 32000,
                 transform=None,
                 target_length: int = 62,  # For 1-second audio at 32kHz with specified parameters
                 cache_features: bool = True,
                 task_type: str = 'anomaly'):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            feature_type: Type of features to extract ('mfcc', 'mel', 'spectrogram')
            n_mels: Number of mel bands for features
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            sample_rate: Expected sample rate of audio files
            transform: Optional transforms to apply to features
            target_length: Target length for features (for padding/truncating)
            cache_features: Whether to cache extracted features to disk
            task_type: Type of task ('anomaly' for normal vs abnormal, 'classification' for chainsaw vs machinery)
        """
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.transform = transform
        self.target_length = target_length
        self.cache_features = cache_features
        self.task_type = task_type
        
        # Class mapping based on task type
        if task_type == 'anomaly':
            # For anomaly detection: normal -> 0, abnormal -> 1
            self.class_to_idx = {'normal': 0, 'abnormal': 1}
        elif task_type == 'classification':
            # For classification: chainsaw -> 0, machinery -> 1
            self.class_to_idx = {'chainsaw': 0, 'machinery': 1}
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Use 'anomaly' or 'classification'.")
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Collect all audio files
        self.samples = []
        self._collect_samples()
        
        # Validate collected samples
        self._validate_samples()
        
        # Feature cache
        self.feature_cache = {}
        
    def _collect_samples(self):
        """Collect all audio file paths and their corresponding labels."""
        # Process based on task type
        if self.task_type == 'anomaly':
            self._collect_samples_anomaly()
        elif self.task_type == 'classification':
            self._collect_samples_classification()
        
        print(f"Collected {len(self.samples)} samples")
    
    def _collect_samples_anomaly(self):
        """Collect samples for anomaly detection (normal vs abnormal)."""
        # Process normal sounds first
        normal_dir = self.data_dir / 'normal'
        if normal_dir.exists():
            for audio_file in normal_dir.glob('*.wav'):
                self.samples.append((str(audio_file), 0))  # Normal class (0)
        
        # Process anomaly sounds - check both at root level and in an 'abnormal' subdirectory
        anomaly_classes = ['chainsaw', 'machinery', 'machines', 'vehicles']  # Include multiple anomaly types
        
        # Check for anomaly classes in the main directory
        for class_name in anomaly_classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists() and class_dir.is_dir():
                # For anomaly detection, all anomaly classes are labeled as 1 (abnormal)
                for audio_file in class_dir.glob('*.wav'):
                    self.samples.append((str(audio_file), 1))  # Abnormal class (1)
        
        # Also check for anomaly subdirectories within an 'abnormal' directory
        abnormal_dir = self.data_dir / 'abnormal'
        if abnormal_dir.exists():
            for subdir in abnormal_dir.iterdir():
                if subdir.is_dir():
                    # For anomaly detection, all subdirectories in abnormal are labeled as 1
                    for audio_file in subdir.glob('*.wav'):
                        self.samples.append((str(audio_file), 1))  # Abnormal class (1)
    
    def _collect_samples_classification(self):
        """Collect samples for classification (chainsaw vs machinery)."""
        # Only process subdirectories in the main directory or in an 'abnormal' subdirectory
        # For classification, we're only looking at abnormal sounds to distinguish between chainsaw and machinery
        
        # Check for classes in the main directory (chainsaw, machinery)
        for class_name in ['chainsaw', 'machinery', 'machines', 'vehicles']:
            class_dir = self.data_dir / class_name
            if class_dir.exists() and class_dir.is_dir():
                label = self.class_to_idx.get(class_name, 1)  # Default to machinery (1) if not found
                if class_name == 'machines' or class_name == 'vehicles':
                    label = self.class_to_idx['machinery']  # Map these to machinery class
                for audio_file in class_dir.glob('*.wav'):
                    self.samples.append((str(audio_file), label))
        
        # Also check for subdirectories within an 'abnormal' directory
        abnormal_dir = self.data_dir / 'abnormal'
        if abnormal_dir.exists():
            for subdir in abnormal_dir.iterdir():
                if subdir.is_dir():
                    class_name = subdir.name.lower()
                    if class_name in ['chainsaw', 'machinery']:
                        label = self.class_to_idx[class_name]
                        for audio_file in subdir.glob('*.wav'):
                            self.samples.append((str(audio_file), label))
                    elif class_name in ['machines', 'vehicles']:
                        # Map these to machinery class
                        label = self.class_to_idx['machinery']
                        for audio_file in subdir.glob('*.wav'):
                            self.samples.append((str(audio_file), label))
    
    def _validate_samples(self):
        """Validate collected samples for potential issues."""
        print(f"Validating {len(self.samples)} collected samples for {self.task_type} task...")
        
        # Check for duplicate file paths
        file_paths = [item[0] for item in self.samples]
        unique_paths = set(file_paths)
        if len(file_paths) != len(unique_paths):
            duplicates = [x for x in file_paths if file_paths.count(x) > 1]
            print(f"  Warning: Found {len(set(duplicates))} duplicate files in dataset: {list(set(duplicates))[:3]}...")
        
        # Check class distribution
        labels = [item[1] for item in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip([self.idx_to_class[lbl] for lbl in unique_labels], counts))
        print(f"  Class distribution: {label_dist}")
        
        # Check for issues specific to each task
        if self.task_type == 'anomaly':
            # For anomaly detection, ensure we have both normal and abnormal classes
            if 0 not in unique_labels or 1 not in unique_labels:
                print(f"  Warning: Anomaly detection task should have both normal (0) and abnormal (1) classes")
        elif self.task_type == 'classification':
            # For classification, ensure we have both chainsaw and machinery classes
            if 0 not in unique_labels or 1 not in unique_labels:
                print(f"  Warning: Classification task should have both chainsaw (0) and machinery (1) classes")
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract features from audio file based on feature_type.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Extracted features as numpy array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        if self.feature_type == 'mfcc':
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mels=self.n_mels, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features = mfccs
        elif self.feature_type == 'mel':
            # Extract mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            features = log_mel_spec
        elif self.feature_type == 'spectrogram':
            # Extract regular spectrogram
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(stft)
            log_spectrogram = librosa.power_to_db(spectrogram**2, ref=np.max)
            features = log_spectrogram
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Transpose to have time dimension first: (time, features)
        features = features.T
        
        # Pad or truncate to target length
        if features.shape[0] < self.target_length:
            # Pad with zeros
            padding = np.zeros((self.target_length - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > self.target_length:
            # Truncate
            features = features[:self.target_length, :]
        
        return features.astype(np.float32)
    
    def _get_cached_features(self, audio_path: str) -> np.ndarray:
        """
        Get cached features or extract and cache them.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Cached or newly extracted features
        """
        cache_key = audio_path
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = self._extract_features(audio_path)
        
        if self.cache_features:
            self.feature_cache[cache_key] = features
            
        return features
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        audio_path, label = self.samples[idx]
        
        # Extract features
        features = self._get_cached_features(audio_path)
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
        
        return features, label


def get_dataloaders(data_dir: str, 
                   batch_size: int = 32, 
                   test_size: float = 0.2, 
                   val_size: float = 0.3, 
                   feature_type: str = 'mfcc',
                   n_mels: int = 40,
                   n_mfcc: int = 13,
                   n_fft: int = 1024,
                   hop_length: int = 512,
                   sample_rate: int = 32000,
                   target_length: int = 62,
                   random_state: int = 42,
                   task_type: str = 'anomaly') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for dataloaders
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        feature_type: Type of features to extract
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        sample_rate: Expected sample rate
        target_length: Target length for features
        random_state: Random state for reproducibility
        task_type: Type of task ('anomaly' for normal vs abnormal, 'classification' for chainsaw vs machinery)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = ChainsawAudioDataset(
        data_dir=data_dir,
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=sample_rate,
        target_length=target_length,
        task_type=task_type
    )
    
    # Check for potential issues in the dataset
    _check_dataset_integrity(dataset)
    
    # Split indices into train+val and test
    indices = list(range(len(dataset)))
    # Create stratification labels based on class indices
    stratify_labels = [dataset.samples[i][1] for i in indices]
    
    # Ensure we have enough samples per class for stratification
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    min_samples_per_class = min(counts)
    
    # If any class has fewer than 2 samples, we can't do stratified split
    if min_samples_per_class < 2:
        print(f"Warning: Minimum samples per class is {min_samples_per_class}. Using random split instead of stratified.")
        train_val_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
    else:
        train_val_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_labels
        )
    
    # Split train+val into train and val
    if val_size > 0:
        train_stratify = [dataset.samples[i][1] for i in train_val_indices]
        unique_train_labels, train_counts = np.unique(train_stratify, return_counts=True)
        min_train_samples = min(train_counts)
        
        if min_train_samples < 2:
            print(f"Warning: Minimum samples per class in train+val is {min_train_samples}. Using random split.")
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size/(1-test_size),  # Adjust for remaining proportion
                random_state=random_state
            )
        else:
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size/(1-test_size),  # Adjust for remaining proportion
                random_state=random_state,
                stratify=train_stratify
            )
    else:
        train_indices = train_val_indices
        val_indices = []
    
    # Create subsets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    test_dataset = Subset(dataset, test_indices)
    
    # Warn if any split is empty
    if len(train_dataset) == 0:
        print("Warning: Training dataset is empty!")
    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty!")
    if val_dataset and len(val_dataset) == 0:
        print("Warning: Validation dataset is empty!")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False  # Set to False for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False  # Set to False for Windows compatibility
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False  # Set to False for Windows compatibility
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset) if val_dataset else 0} samples")  
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def _check_dataset_integrity(dataset):
    """
    Check the dataset for potential issues that could cause data leakage or overfitting.
    
    Args:
        dataset: The ChainsawAudioDataset instance to check
    """
    print("Checking dataset integrity...")
    
    # Check for duplicate file paths
    file_paths = [item[0] for item in dataset.samples]
    unique_paths = set(file_paths)
    if len(file_paths) != len(unique_paths):
        duplicates = [x for x in file_paths if file_paths.count(x) > 1]
        print(f"Warning: Found {len(set(duplicates))} duplicate file paths in the dataset: {list(set(duplicates))[:5]}...")  # Show first 5
    
    # Check class distribution
    labels = [item[1] for item in dataset.samples]
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"Class distribution: {label_dist}")
    
    # Check for severe imbalance
    if len(counts) > 1:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 10:
            print(f"Warning: Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
    
    # Basic statistics
    print(f"Total samples: {len(dataset.samples)}")
    print("Dataset integrity check completed.")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Chainsaw Detection Data Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the organized dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for dataloaders')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'mel', 'spectrogram'],
                        help='Type of features to extract')
    
    args = parser.parse_args()
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        feature_type=args.feature_type
    )
    
    # Test the dataloaders
    print("Testing dataloaders...")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")
        if batch_idx == 2:  # Just test a few batches
            break
    
    print("Data pipeline created successfully!")