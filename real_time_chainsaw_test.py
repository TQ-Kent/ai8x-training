"""
Real-time Chainsaw Detection Testing Script

This script implements a real-time testing system for the two-stage chainsaw detection pipeline:
1. Anomaly Detection: Binary classifier (normal vs abnormal sounds)
2. Classification: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)

The script can process audio from a microphone in real-time and display detection results.
"""
import os
import sys
import torch
import numpy as np
import librosa
import time
import threading
import queue
from datetime import datetime
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import sounddevice as sd
    import wavio  # For recording audio to file
    SOUNDDEVICE_AVAILABLE = True
    WAVIO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Required packages not installed. Install with 'pip install sounddevice wavio'")
    print("For file-based testing, use --file option instead.")
    SOUNDDEVICE_AVAILABLE = False
    WAVIO_AVAILABLE = False

from chainsaw_model_architecture import AnomalyDetectionNet, ClassificationNet
import ai8x

class RealTimeChainsawDetector:
    """
    Real-time chainsaw detection system using two trained models in sequence.
    """
    
    def __init__(self, 
                 anomaly_model_path: str,
                 classification_model_path: str,
                 sample_rate: int = 32000,
                 segment_length: float = 1.0,  # Changed back to 1.0 second
                 n_mfcc: int = 13,
                 n_mels: int = 40,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 target_length: int = 62,  # Updated back to 62 for 1s at 32kHz
                 anomaly_threshold: float = 0.8,
                 record_audio: bool = False,
                 output_dir: str = "./recordings"):
        """
        Initialize the real-time detector.
        
        Args:
            anomaly_model_path: Path to the trained anomaly detection model
            classification_model_path: Path to the trained classification model
            sample_rate: Audio sample rate
            segment_length: Length of audio segments to process (seconds)
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            target_length: Target length for features
            anomaly_threshold: Threshold for anomaly classification
            record_audio: Whether to record audio for verification
            output_dir: Directory to save recorded audio files
        """
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.anomaly_threshold = anomaly_threshold
        self.record_audio = record_audio
        self.output_dir = output_dir
        
        # Calculate parameters
        self.audio_segment_samples = int(self.sample_rate * self.segment_length)
        
        # Configure AI8X device
        ai8x.set_device(85, False, False)  # MAX78000, no simulation, no round avg
        
        # Load models
        self.device = torch.device("cpu")  # Use CPU for real-time processing
        self.load_models(anomaly_model_path, classification_model_path)
        
        # Audio buffer for real-time processing
        self.audio_buffer = np.zeros(self.audio_segment_samples)
        self.full_audio_buffer = []  # For recording full audio stream
        self.buffer_lock = threading.Lock()
        self.new_segment_available = threading.Event()
        
        # Recording settings
        if self.record_audio:
            os.makedirs(self.output_dir, exist_ok=True)
            self.recording_counter = 0
        
        # Statistics
        self.total_segments = 0
        self.anomaly_count = 0
        self.chainsaw_count = 0
        self.vehicles_count = 0
        
        # Confidence tracking
        self.last_anomaly_confidence = 0.0
        self.last_classification_confidence = 0.0
        self.last_classification_result = "unknown"
        self.last_processed_time = time.time()  # Track when last segment was processed
        
    def load_models(self, anomaly_model_path: str, classification_model_path: str):
        """
        Load both trained models.
        
        Args:
            anomaly_model_path: Path to the anomaly detection model
            classification_model_path: Path to the classification model
        """
        print("Loading anomaly detection model...")
        # Initialize anomaly detection model
        self.anomaly_model = AnomalyDetectionNet(
            num_channels=self.n_mfcc,
            feature_length=self.target_length,
            num_classes=2
        )
        
        # Load trained weights
        anomaly_checkpoint = torch.load(anomaly_model_path, map_location=self.device)
        if 'model_state_dict' in anomaly_checkpoint:
            self.anomaly_model.load_state_dict(anomaly_checkpoint['model_state_dict'])
        else:
            self.anomaly_model.load_state_dict(anomaly_checkpoint)
        
        self.anomaly_model = self.anomaly_model.to(self.device)
        self.anomaly_model.eval()
        print("Anomaly detection model loaded successfully!")
        
        print("Loading classification model...")
        # Initialize classification model
        self.classification_model = ClassificationNet(
            num_channels=self.n_mfcc,
            feature_length=self.target_length,
            num_classes=2
        )
        
        # Load trained weights
        class_checkpoint = torch.load(classification_model_path, map_location=self.device)
        if 'model_state_dict' in class_checkpoint:
            self.classification_model.load_state_dict(class_checkpoint['model_state_dict'])
        else:
            self.classification_model.load_state_dict(class_checkpoint)
        
        self.classification_model = self.classification_model.to(self.device)
        self.classification_model.eval()
        print("Classification model loaded successfully!")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio segment.
        
        Args:
            audio: Audio segment as numpy array
            
        Returns:
            Extracted features as numpy array
        """
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Transpose to have time dimension first: (time, features)
        features = mfccs.T
        
        # Pad or truncate to target length
        if features.shape[0] < self.target_length:
            # Pad with zeros
            padding = np.zeros((self.target_length - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > self.target_length:
            # Truncate
            features = features[:self.target_length, :]
        
        return features.astype(np.float32)
    
    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio segment for model input.
        
        Args:
            audio: Audio segment as numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        return features_tensor
    
    def detect_anomaly(self, features: torch.Tensor) -> tuple:
        """
        Detect if the audio segment contains an anomaly.
        
        Args:
            features: Preprocessed features tensor
            
        Returns:
            Tuple of (is_anomaly: bool, probability: float)
        """
        with torch.no_grad():
            # Move features to device
            features = features.to(self.device)
            
            # Get anomaly detection output
            anomaly_output = self.anomaly_model(features)
            anomaly_probs = torch.softmax(anomaly_output, dim=1)
            
            # Get probability of anomaly (class 1)
            anomaly_prob = anomaly_probs[0, 1].item()
            
            # Determine if it's an anomaly based on threshold
            is_anomaly = anomaly_prob > self.anomaly_threshold
            
            return is_anomaly, anomaly_prob
    
    def classify_anomaly(self, features: torch.Tensor) -> tuple:
        """
        Classify the type of anomaly (chainsaw vs vehicles).
        
        Args:
            features: Preprocessed features tensor
            
        Returns:
            Tuple of (class_label: str, probability: float)
        """
        with torch.no_grad():
            # Move features to device
            features = features.to(self.device)
            
            # Get classification output
            class_output = self.classification_model(features)
            class_probs = torch.softmax(class_output, dim=1)
            
            # Get probabilities for each class
            chainsaw_prob = class_probs[0, 0].item()  # Assuming chainsaw is class 0
            vehicle_prob = class_probs[0, 1].item()   # Assuming vehicle is class 1
            
            # Determine the class with higher probability
            if chainsaw_prob > vehicle_prob:
                return "chainsaw", chainsaw_prob
            else:
                return "vehicles", vehicle_prob
    
    def process_audio_segment(self, audio_segment: np.ndarray) -> dict:
        """
        Process a single audio segment through the two-stage pipeline.
        
        Args:
            audio_segment: Audio segment as numpy array
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Preprocess audio
        features = self.preprocess_audio(audio_segment)
        
        # Stage 1: Anomaly Detection
        is_anomaly, anomaly_prob = self.detect_anomaly(features)
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'anomaly_detected': is_anomaly,
            'anomaly_probability': anomaly_prob,
            'inference_time_ms': (time.time() - start_time) * 1000
        }
        
        if is_anomaly:
            # Stage 2: Classification
            start_time = time.time()
            class_label, class_prob = self.classify_anomaly(features)
            classification_time_ms = (time.time() - start_time) * 1000
            
            result['classification'] = class_label
            result['classification_probability'] = class_prob
            result['classification_time_ms'] = classification_time_ms
            result['inference_time_ms'] += classification_time_ms
            
            # Update statistics
            with self.buffer_lock:
                self.anomaly_count += 1
                if class_label == "chainsaw":
                    self.chainsaw_count += 1
                else:
                    self.vehicles_count += 1
        else:
            result['classification'] = None
            result['classification_probability'] = None
            result['classification_time_ms'] = 0.0
        
        # Update statistics
        with self.buffer_lock:
            self.total_segments += 1
            # Update last known confidence values
            self.last_anomaly_confidence = result['anomaly_probability']
            if result['classification_probability'] is not None:
                self.last_classification_confidence = result['classification_probability']
                self.last_classification_result = result['classification']
            self.last_processed_time = time.time()
        
        return result
    
    def print_results(self, result: dict):
        """
        Print the detection results in a formatted way.
        
        Args:
            result: Results dictionary from process_audio_segment
        """
        anomaly_prob_percent = result['anomaly_probability'] * 100
        is_anomaly = result['anomaly_detected']
        
        # Format the main detection result
        if is_anomaly:
            class_label = result['classification']
            class_prob_percent = result['classification_probability'] * 100
            main_result = f"🚨 ANOMALY: {class_label.upper()} ({class_prob_percent:.1f}%)"
        else:
            main_result = f"✅ NORMAL ({anomaly_prob_percent:.1f}%)"
        
        # Get and print current statistics
        stats = self.get_statistics()
        stats_str = f"[S:{stats['total_segments_processed']}, A:{stats['anomalies_detected']}, C:{stats['chainsaws_detected']}, V:{stats['vehicles_detected']}]"
        
        # Format timing info
        total_time = result['inference_time_ms']
        timing_str = f"[{total_time:.1f}ms]"
        
        # Print all information in a compact format
        print(f"[{result['timestamp'][:19]}] {main_result} {stats_str} {timing_str}")
    
    def save_audio_segment(self, audio_segment, classification, timestamp):
        """
        Save an audio segment to a file.
        
        Args:
            audio_segment: The audio data to save
            classification: Classification result (chainsaw, vehicles, etc.)
            timestamp: Timestamp for the filename
        """
        if not WAVIO_AVAILABLE:
            print("wavio not available, cannot save audio recordings")
            return
            
        # Clean up timestamp for filename
        clean_timestamp = timestamp.replace(" ", "_").replace(":", "-").replace(".", "-")
        
        # Create filename with classification and timestamp
        filename = f"anomaly_{classification}_{clean_timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save audio segment
        try:
            wavio.write(filepath, audio_segment, self.sample_rate, sampwidth=3)
            print(f"  [RECORDED] Audio segment saved: {filename}")
        except Exception as e:
            print(f"  [ERROR] Could not save audio segment: {e}")
    
    def save_full_recording(self):
        """
        Save the full audio recording to a file.
        """
        if not WAVIO_AVAILABLE or not self.record_audio:
            return
            
        if len(self.full_audio_buffer) == 0:
            print("No audio was recorded")
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"full_recording_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to numpy array
        full_audio = np.array(self.full_audio_buffer, dtype=np.float32)
        
        # Save the full recording
        try:
            wavio.write(filepath, full_audio, self.sample_rate, sampwidth=3)
            print(f"Full recording saved: {filename}")
        except Exception as e:
            print(f"Could not save full recording: {e}")
    
    def get_statistics(self) -> dict:
        """
        Get current statistics about detections.
        
        Returns:
            Dictionary with statistics
        """
        with self.buffer_lock:
            stats = {
                'total_segments_processed': self.total_segments,
                'anomalies_detected': self.anomaly_count,
                'chainsaws_detected': self.chainsaw_count,
                'vehicles_detected': self.vehicles_count,
                'anomaly_rate_percent': (self.anomaly_count / max(1, self.total_segments)) * 100,
                'chainsaw_rate_percent': (self.chainsaw_count / max(1, self.anomaly_count)) * 100 if self.anomaly_count > 0 else 0
            }
        return stats
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio input.
        """
        # Convert audio data to the same format as training
        audio_data = indata[:, 0].copy()  # Take mono channel
        
        # Add to full audio buffer for recording if needed
        if self.record_audio:
            with self.buffer_lock:
                self.full_audio_buffer.extend(audio_data.copy())
        
        # Add to processing buffer
        with self.buffer_lock:
            # Shift buffer left and add new data
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
            self.audio_buffer[-len(audio_data):] = audio_data
        
        # Check if we have sufficient non-zero data in the buffer to process
        # Using a more lenient threshold to ensure segments get processed
        nonzero_count = np.count_nonzero(self.audio_buffer)
        if nonzero_count >= len(self.audio_buffer) * 0.1:  # At least 10% of buffer has data
            # Process the segment
            result = self.process_audio_segment(self.audio_buffer.copy())
            self.print_results(result)
            
            # If recording and anomaly detected, save the segment
            if self.record_audio and result['anomaly_detected']:
                self.save_audio_segment(self.audio_buffer.copy(), result['classification'], result['timestamp'])
    
    def test_from_file(self, audio_file_path: str):
        """
        Test the models on an audio file.
        
        Args:
            audio_file_path: Path to the audio file to test
        """
        print(f"Testing from file: {audio_file_path}")
        
        # Load audio file
        audio, sr = librosa.load(audio_file_path, sr=self.sample_rate)
        
        print(f"Audio file loaded: {len(audio)/self.sample_rate:.2f}s at {sr}Hz")
        
        # Process in segments
        num_segments = int(len(audio) / self.audio_segment_samples)
        
        for i in range(num_segments):
            start_idx = i * self.audio_segment_samples
            end_idx = start_idx + self.audio_segment_samples
            audio_segment = audio[start_idx:end_idx]
            
            result = self.process_audio_segment(audio_segment)
            self.print_results(result)
            
            # Calculate progress
            progress = (i + 1) / num_segments * 100
            print(f"Progress: {progress:.1f}%")
        
        # Print final statistics
        stats = self.get_statistics()
        print("\nFinal Statistics:")
        print(f"  Total segments processed: {stats['total_segments_processed']}")
        print(f"  Anomalies detected: {stats['anomalies_detected']} ({stats['anomaly_rate_percent']:.1f}%)")
        print(f"  Chainsaws detected: {stats['chainsaws_detected']}")
        print(f"  Vehicles detected: {stats['vehicles_detected']}")
        if stats['anomalies_detected'] > 0:
            print(f"  Chainsaw classification rate: {stats['chainsaw_rate_percent']:.1f}%")
    
    def start_real_time_detection(self, device_id=None):
        """
        Start real-time detection using microphone input.
        
        Args:
            device_id: ID of the audio input device (None for default)
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("Error: sounddevice module not available. Install with 'pip install sounddevice'")
            return
            
        print("Starting real-time detection...")
        print(f"Using device: {device_id or 'default'}")
        print(f"Sample rate: {self.sample_rate}Hz, Segment length: {self.segment_length}s")
        if self.record_audio:
            print(f"Recording enabled: Detected anomalies will be saved to '{self.output_dir}'")
        print("Press Ctrl+C to stop...")
        
        try:
            # List available devices to help user
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  {i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
            print()
            
            # Start audio stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.audio_segment_samples,
                device=device_id
            ):
                last_stats_time = time.time()
                stats_interval = 5.0  # Print stats every 5 seconds instead of based on segment count
                while True:
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    # Print statistics periodically based on time, not segment count
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        stats = self.get_statistics()
                        stats_str = f"[S:{stats['total_segments_processed']}, A:{stats['anomalies_detected']}, C:{stats['chainsaws_detected']}, V:{stats['vehicles_detected']}]"
                        
                        print(f"\n[PERIODIC-{datetime.now().strftime('%H:%M:%S')}] "
                              f"Last: {self.last_classification_result.upper() if self.last_classification_result != 'unknown' else 'NO_DATA'} "
                              f"({self.last_anomaly_confidence:.1f}%) {stats_str}")
                        last_stats_time = current_time
        
        except KeyboardInterrupt:
            print("\nStopping real-time detection...")
            
            # Save full recording if enabled
            if self.record_audio:
                self.save_full_recording()
            
            # Print final statistics
            stats = self.get_statistics()
            print("\nFinal Statistics:")
            print(f"  Total segments processed: {stats['total_segments_processed']}")
            print(f"  Anomalies detected: {stats['anomalies_detected']} ({stats['anomaly_rate_percent']:.1f}%)")
            print(f"  Chainsaws detected: {stats['chainsaws_detected']}")
            print(f"  Vehicles detected: {stats['vehicles_detected']}")
            if stats['anomalies_detected'] > 0:
                print(f"  Chainsaw classification rate: {stats['chainsaw_rate_percent']:.1f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Chainsaw Detection Testing')
    
    # Model paths
    parser.add_argument('--anomaly_model', type=str, 
                        default='./trained_models/anomaly_detection/best_anomaly_model.pth',
                        help='Path to the trained anomaly detection model')
    parser.add_argument('--classification_model', type=str,
                        default='./trained_models/classification/best_classification_model.pth',
                        help='Path to the trained classification model')
    
    # Audio settings
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Audio sample rate')
    parser.add_argument('--segment_length', type=float, default=1,
                        help='Length of audio segments to process in seconds')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5,
                        help='Threshold for anomaly classification')
    
    # Feature extraction settings
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--target_length', type=int, default=62,
                        help='Target length for features')
    
    # Mode of operation
    parser.add_argument('--file', type=str, default=None,
                        help='Path to audio file for file-based testing (instead of real-time)')
    parser.add_argument('--device_id', type=int, default=None,
                        help='Audio input device ID (for real-time mode)')
    parser.add_argument('--record', action='store_true',
                        help='Enable audio recording of detected anomalies')
    parser.add_argument('--output_dir', type=str, default='./recordings',
                        help='Directory to save recorded audio files')
    
    args = parser.parse_args()
    
    print("Initializing Real-time Chainsaw Detection System...")
    
    # Check if models exist
    if not os.path.exists(args.anomaly_model):
        print(f"Error: Anomaly model not found at {args.anomaly_model}")
        return
    
    if not os.path.exists(args.classification_model):
        print(f"Error: Classification model not found at {args.classification_model}")
        return
    
    # Create detector
    detector = RealTimeChainsawDetector(
        anomaly_model_path=args.anomaly_model,
        classification_model_path=args.classification_model,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        n_mfcc=args.n_mfcc,
        target_length=args.target_length,
        anomaly_threshold=args.anomaly_threshold,
        record_audio=args.record,
        output_dir=args.output_dir
    )
    
    # Run in appropriate mode
    if args.file:
        # File-based testing
        if not os.path.exists(args.file):
            print(f"Error: Audio file not found at {args.file}")
            return
        
        detector.test_from_file(args.file)
    else:
        # Real-time testing
        detector.start_real_time_detection(device_id=args.device_id)


if __name__ == "__main__":
    main()