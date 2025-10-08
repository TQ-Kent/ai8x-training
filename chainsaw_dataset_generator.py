"""
Dataset Generator for Chainsaw Detection in Forest Monitoring

This script processes audio files organized in the following structure:
- data/normal/ with subclasses: animals, calm_forest, etc. (forest sounds)
- data/anomaly/ with subclasses: chainsaw, vehicles

Output organized dataset:
- Normalized to 32kHz, 1-second samples
- Two main classes: `normal/` and `abnormal/`
- `abnormal/` contains: `chainsaw/` and `machinery/` subfolders
- `normal/` contains all forest sounds directly (no subfolders)
- Generate maximum possible samples through segmentation and augmentation
"""
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from tqdm import tqdm
import random


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = 32000) -> np.ndarray:
    """
    Resample audio data to target sample rate.
    
    Args:
        audio_data: Audio signal as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default 32000)
    
    Returns:
        Resampled audio data
    """
    if orig_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    return audio_data


def segment_audio(audio_data: np.ndarray, sample_rate: int = 32000, 
                  segment_length: float = 1.0, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Segment audio into fixed-length clips with overlap.
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        segment_length: Length of each segment in seconds (default 1.0)
        overlap: Overlap ratio between segments (default 0.5)
    
    Returns:
        List of segmented audio clips
    """
    segment_samples = int(sample_rate * segment_length)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start < len(audio_data):
        end = min(start + segment_samples, len(audio_data))
        segment = audio_data[start:end]
        
        # Pad if segment is shorter than required length
        if len(segment) < segment_samples:
            padding = segment_samples - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant', constant_values=0)
        
        segments.append(segment)
        start += hop_samples
        
        # Prevent infinite loops when hop size is too small
        if hop_samples == 0:
            break
    
    return segments


def apply_augmentation(audio: np.ndarray, sample_rate: int = 32000) -> List[np.ndarray]:
    """
    Apply various audio augmentations to increase dataset diversity.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of the audio
    
    Returns:
        List of augmented audio signals (original + augmented versions)
    """
    augmented_versions = [audio]  # Include original
    
    # Time stretching (speed perturbation)
    try:
        # Slow down (0.9x speed)
        stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
        if len(stretched_slow) > len(audio):
            stretched_slow = stretched_slow[:len(audio)]
        else:
            stretched_slow = np.pad(stretched_slow, (0, len(audio) - len(stretched_slow)), mode='constant')
        augmented_versions.append(stretched_slow)
        
        # Speed up (1.1x speed)
        stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
        if len(stretched_fast) > len(audio):
            stretched_fast = stretched_fast[:len(audio)]
        else:
            stretched_fast = np.pad(stretched_fast, (0, len(audio) - len(stretched_fast)), mode='constant')
        augmented_versions.append(stretched_fast)
    except:
        # If stretching fails, just add the original
        pass
    
    # Add noise
    for noise_factor in [0.001, 0.005]:
        noise = np.random.normal(0, noise_factor, audio.shape[0]).astype(audio.dtype)
        noisy_audio = audio + noise
        augmented_versions.append(noisy_audio)
    
    # Pitch shifting (with bounds checking)
    try:
        shift_semitones = random.choice([-2, -1, 1, 2])
        pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=shift_semitones)
        if len(pitch_shifted) > len(audio):
            pitch_shifted = pitch_shifted[:len(audio)]
        else:
            pitch_shifted = np.pad(pitch_shifted, (0, len(audio) - len(pitch_shifted)), mode='constant')
        augmented_versions.append(pitch_shifted)
    except:
        # If pitch shifting fails, just continue
        pass
    
    return augmented_versions


def balance_dataset(class_segments: dict) -> dict:
    """
    Balance the dataset with the following hierarchy:
    1. Equal number of samples between 'normal' and 'abnormal' (anomaly) classes
    2. Within 'abnormal', equal number of samples between 'chainsaw' and 'machinery'
    
    Args:
        class_segments: Dictionary mapping class names to lists of file paths
                       Expected keys: 'normal', 'chainsaw', 'machinery'
    
    Returns:
        Balanced dictionary with samples distributed according to the specified hierarchy
    """
    # Calculate the current number of samples in each class
    class_counts = {cls: len(segments) for cls, segments in class_segments.items()}
    print(f"Before balancing: {class_counts}")
    
    # Calculate normal count and anomaly count (chainsaw + machinery)
    normal_count = class_counts.get('normal', 0)
    chainsaw_count = class_counts.get('chainsaw', 0)
    machinery_count = class_counts.get('machinery', 0)
    anomaly_count = chainsaw_count + machinery_count
    
    print(f"Normal samples: {normal_count}, Anomaly samples: {anomaly_count} "
          f"(chainsaw: {chainsaw_count}, machinery: {machinery_count})")
    
    # Determine target counts
    # Total anomaly samples should equal normal samples
    target_normal_count = normal_count
    target_anomaly_total = normal_count  # Equal to normal samples
    target_anomaly_per_subclass = target_anomaly_total // 2  # Half for each anomaly subclass
    
    # Handle odd numbers
    if target_anomaly_total % 2 == 1:
        # If odd, assign one extra to one of the subclasses
        target_chainsaw_count = target_anomaly_per_subclass + 1
        target_machinery_count = target_anomaly_per_subclass
    else:
        target_chainsaw_count = target_anomaly_per_subclass
        target_machinery_count = target_anomaly_per_subclass
    
    print(f"Target distribution - Normal: {target_normal_count}, "
          f"Chainsaw: {target_chainsaw_count}, Machinery: {target_machinery_count}")
    
    balanced_class_segments = {}
    
    # Balance normal class
    current_normal = class_segments.get('normal', [])
    current_normal_count = len(current_normal)
    
    if current_normal_count == target_normal_count:
        balanced_class_segments['normal'] = current_normal
    elif current_normal_count < target_normal_count:
        # Need to upsample normal
        needed = target_normal_count - current_normal_count
        additional_samples = [random.choice(current_normal) for _ in range(needed)]
        balanced_class_segments['normal'] = current_normal + additional_samples
    else:
        # Need to downsample normal
        balanced_class_segments['normal'] = random.sample(current_normal, target_normal_count)
    
    print(f"Normal class: {current_normal_count} -> {len(balanced_class_segments['normal'])} samples")
    
    # Balance chainsaw class
    current_chainsaw = class_segments.get('chainsaw', [])
    current_chainsaw_count = len(current_chainsaw)
    
    if current_chainsaw_count == target_chainsaw_count:
        balanced_class_segments['chainsaw'] = current_chainsaw
    elif current_chainsaw_count < target_chainsaw_count:
        # Need to upsample chainsaw
        needed = target_chainsaw_count - current_chainsaw_count
        additional_samples = [random.choice(current_chainsaw) for _ in range(needed)]
        balanced_class_segments['chainsaw'] = current_chainsaw + additional_samples
    else:
        # Need to downsample chainsaw
        balanced_class_segments['chainsaw'] = random.sample(current_chainsaw, target_chainsaw_count)
    
    print(f"Chainsaw class: {current_chainsaw_count} -> {len(balanced_class_segments['chainsaw'])} samples")
    
    # Balance machinery class
    current_machinery = class_segments.get('machinery', [])
    current_machinery_count = len(current_machinery)
    
    if current_machinery_count == target_machinery_count:
        balanced_class_segments['machinery'] = current_machinery
    elif current_machinery_count < target_machinery_count:
        # Need to upsample machinery
        needed = target_machinery_count - current_machinery_count
        additional_samples = [random.choice(current_machinery) for _ in range(needed)]
        balanced_class_segments['machinery'] = current_machinery + additional_samples
    else:
        # Need to downsample machinery
        balanced_class_segments['machinery'] = random.sample(current_machinery, target_machinery_count)
    
    print(f"Machinery class: {current_machinery_count} -> {len(balanced_class_segments['machinery'])} samples")
    
    # Verify the final balance
    final_counts = {cls: len(segments) for cls, segments in balanced_class_segments.items()}
    final_normal_count = final_counts.get('normal', 0)
    final_chainsaw_count = final_counts.get('chainsaw', 0)
    final_machinery_count = final_counts.get('machinery', 0)
    final_anomaly_count = final_chainsaw_count + final_machinery_count
    
    print(f"After balancing: Normal={final_normal_count}, Anomaly_total={final_anomaly_count} "
          f"(chainsaw={final_chainsaw_count}, machinery={final_machinery_count})")
    
    # Check if the requirements are met
    if final_normal_count == final_anomaly_count and final_chainsaw_count == final_machinery_count:
        print("[SUCCESS] Balancing requirements satisfied!")
    else:
        print("[INFO] Balancing requirements may not be fully satisfied due to initial data distribution.")
    
    return balanced_class_segments


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[np.ndarray], Optional[int]]:
    """
    Validate audio file and load it if valid.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (is_valid, audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = sf.read(file_path, dtype='float32')
        # Basic validation
        if len(audio_data) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return False, None, None
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            if audio_data.shape[1] == 1:
                audio_data = audio_data.flatten()
            else:
                # Take the first channel or average all channels
                audio_data = np.mean(audio_data, axis=1)
        
        return True, audio_data, sample_rate
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return False, None, None


def process_audio_files(input_dir: str, output_dir: str, 
                      target_sr: int = 32000, segment_length: float = 1.0, 
                      overlap: float = 0.5, augmentation: bool = True) -> dict:
    """
    Process audio files from input directory to output directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_sr: Target sample rate (default 32000)
        segment_length: Length of each segment in seconds (default 1.0)
        overlap: Overlap ratio between segments (default 0.5)
        augmentation: Whether to apply augmentation (default True)
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'processed_files': 0,
        'segments_generated': 0,
        'skipped_files': 0,
        'errors': []
    }
    
    print(f"Processing audio files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all audio files
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    for file_path in tqdm(audio_files, desc="Processing files"):
        try:
            # Validate and load audio file
            is_valid, audio_data, sample_rate = validate_audio_file(file_path)
            if not is_valid:
                stats['skipped_files'] += 1
                stats['errors'].append(f"Invalid file: {file_path}")
                continue
            
            # Resample to target rate
            audio_data = resample_audio(audio_data, sample_rate, target_sr)
            
            # Segment the audio
            segments = segment_audio(audio_data, target_sr, segment_length, overlap)
            
            # Process each segment
            for idx, segment in enumerate(segments):
                # Apply augmentation if enabled
                if augmentation:
                    augmented_versions = apply_augmentation(segment, target_sr)
                else:
                    augmented_versions = [segment]
                
                # Save each version
                file_stem = Path(file_path).stem
                for aug_idx, aug_segment in enumerate(augmented_versions):
                    # Generate unique filename
                    output_filename = f"{file_stem}_seg{idx:03d}_aug{aug_idx:02d}.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Write the audio file
                    sf.write(output_path, aug_segment, target_sr)
                    stats['segments_generated'] += 1
            
            stats['processed_files'] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            stats['errors'].append(f"Error processing {file_path}: {str(e)}")
            stats['skipped_files'] += 1
    
    return stats


def organize_dataset_structure(base_input_dir: str, output_dir: str, 
                              target_sr: int = 32000, augmentation: bool = True,
                              balance_classes: bool = True):
    """
    Organize audio dataset from source structure to target structure for chainsaw detection.
    
    Source structure:
    - data/normal/ with subclasses: animals, calm_forest, etc.
    - data/anomaly/ with subclasses: chainsaw, vehicles
    
    Target structure:
    - normal/ (all forest sounds)
    - abnormal/chainsaw/ (chainsaws)
    - abnormal/machinery/ (vehicles and other machinery)
    """
    print("Starting dataset organization...")
    
    # Create output directories
    normal_output = os.path.join(output_dir, 'normal')
    chainsaw_output = os.path.join(output_dir, 'abnormal', 'chainsaw')
    machinery_output = os.path.join(output_dir, 'abnormal', 'machinery')
    
    os.makedirs(normal_output, exist_ok=True)
    os.makedirs(chainsaw_output, exist_ok=True)
    os.makedirs(machinery_output, exist_ok=True)
    
    # Dictionary to store segment lists for each class
    class_segments = {
        'normal': [],
        'chainsaw': [],
        'machinery': []
    }
    
    # Process normal sounds
    normal_input = os.path.join(base_input_dir, 'normal')
    if os.path.exists(normal_input):
        print("Processing normal sounds...")
        # Instead of directly writing to output, collect the segment paths
        normal_temp_dir = os.path.join(output_dir, '_temp_normal')
        if os.path.exists(normal_temp_dir):
            shutil.rmtree(normal_temp_dir)
        os.makedirs(normal_temp_dir, exist_ok=True)
        
        normal_stats = process_audio_files(
            normal_input, 
            normal_temp_dir, 
            target_sr=target_sr, 
            augmentation=augmentation
        )
        class_segments['normal'] = [os.path.join(normal_temp_dir, f) for f in os.listdir(normal_temp_dir)]
        print(f"Normal sounds - Files: {normal_stats['processed_files']}, Segments: {normal_stats['segments_generated']}")

    # Process anomaly sounds
    anomaly_input = os.path.join(base_input_dir, 'anomaly')
    if os.path.exists(anomaly_input):
        print("Processing anomaly sounds...")
        
        # Process each subdirectory in anomaly folder
        for sub_dir in os.listdir(anomaly_input):
            sub_dir_path = os.path.join(anomaly_input, sub_dir)
            if os.path.isdir(sub_dir_path):
                if sub_dir.lower() == 'chainsaw':
                    # Process chainsaw sounds
                    chainsaw_temp_dir = os.path.join(output_dir, '_temp_chainsaw')
                    if os.path.exists(chainsaw_temp_dir):
                        shutil.rmtree(chainsaw_temp_dir)
                    os.makedirs(chainsaw_temp_dir, exist_ok=True)
                    
                    chainsaw_stats = process_audio_files(
                        sub_dir_path, 
                        chainsaw_temp_dir, 
                        target_sr=target_sr, 
                        augmentation=augmentation
                    )
                    class_segments['chainsaw'] = [os.path.join(chainsaw_temp_dir, f) for f in os.listdir(chainsaw_temp_dir)]
                    print(f"Chainsaw sounds - Files: {chainsaw_stats['processed_files']}, Segments: {chainsaw_stats['segments_generated']}")
                elif sub_dir.lower() in ['machinery', 'vehicles', 'machines']:  # Include machinery/machines/vehicles
                    # Process machinery sounds (vehicles and other machinery)
                    machinery_temp_dir = os.path.join(output_dir, '_temp_machinery')
                    if os.path.exists(machinery_temp_dir):
                        shutil.rmtree(machinery_temp_dir)
                    os.makedirs(machinery_temp_dir, exist_ok=True)
                    
                    mach_stats = process_audio_files(
                        sub_dir_path, 
                        machinery_temp_dir, 
                        target_sr=target_sr, 
                        augmentation=augmentation
                    )
                    class_segments['machinery'] = [os.path.join(machinery_temp_dir, f) for f in os.listdir(machinery_temp_dir)]
                    print(f"Machinery ({sub_dir}) sounds - Files: {mach_stats['processed_files']}, Segments: {mach_stats['segments_generated']}")
                else:
                    # Process any other anomaly subdirectories as machinery
                    machinery_temp_dir = os.path.join(output_dir, '_temp_machinery')
                    if os.path.exists(machinery_temp_dir):
                        shutil.rmtree(machinery_temp_dir)
                    os.makedirs(machinery_temp_dir, exist_ok=True)
                    
                    mach_stats = process_audio_files(
                        sub_dir_path, 
                        machinery_temp_dir, 
                        target_sr=target_sr, 
                        augmentation=augmentation
                    )
                    # Extend existing machinery list if already populated
                    new_mach_segments = [os.path.join(machinery_temp_dir, f) for f in os.listdir(machinery_temp_dir)]
                    class_segments['machinery'].extend(new_mach_segments)
                    print(f"Machinery ({sub_dir}) sounds - Files: {mach_stats['processed_files']}, Segments: {mach_stats['segments_generated']}")

    # Apply balancing if requested
    if balance_classes:
        print("Applying class balancing...")
        class_segments = balance_dataset(class_segments)
    
    # Move balanced data to final output directories
    # Move normal segments
    for i, segment_path in enumerate(class_segments['normal']):
        filename = f"normal_balanced_{i:05d}.wav"
        final_path = os.path.join(normal_output, filename)
        shutil.copy2(segment_path, final_path)
    
    # Move chainsaw segments
    for i, segment_path in enumerate(class_segments['chainsaw']):
        filename = f"chainsaw_balanced_{i:05d}.wav"
        final_path = os.path.join(chainsaw_output, filename)
        shutil.copy2(segment_path, final_path)
    
    # Move machinery segments
    for i, segment_path in enumerate(class_segments['machinery']):
        filename = f"machinery_balanced_{i:05d}.wav"
        final_path = os.path.join(machinery_output, filename)
        shutil.copy2(segment_path, final_path)
    
    # Clean up temporary directories
    temp_dirs = [os.path.join(output_dir, '_temp_normal'), 
                 os.path.join(output_dir, '_temp_chainsaw'),
                 os.path.join(output_dir, '_temp_machinery')]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Print summary
    print("\nDataset organization completed!")
    print(f"Normal sounds: {len(class_segments['normal'])} segments")
    print(f"Abnormal - Chainsaw: {len(class_segments['chainsaw'])} segments")
    print(f"Abnormal - Machinery: {len(class_segments['machinery'])} segments")


def main():
    parser = argparse.ArgumentParser(description='Dataset Generator for Chainsaw Detection')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with normal/ and anomaly/ subdirectories')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Target sample rate (default: 32000)')
    parser.add_argument('--segment_length', type=float, default=1.0,
                        help='Length of each segment in seconds (default: 1.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio between segments (default: 0.5)')
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable audio augmentation')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false',
                        help='Disable audio augmentation')
    parser.add_argument('--balance_classes', action='store_true',
                        help='Balance classes by upsampling minority classes (default: True)')
    parser.add_argument('--no-balance_classes', dest='balance_classes', action='store_false',
                        help='Do not balance classes (default: False)')
    
    parser.set_defaults(augmentation=True, balance_classes=True)
    
    args = parser.parse_args()
    
    print("Starting Chainsaw Detection Dataset Generator")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Segment length: {args.segment_length} seconds")
    print(f"Overlap: {args.overlap}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Balance classes: {args.balance_classes}")
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    try:
        organize_dataset_structure(
            args.input_dir, 
            args.output_dir, 
            target_sr=args.sample_rate,
            augmentation=args.augmentation,
            balance_classes=args.balance_classes
        )
        
        # Print final statistics
        total_normal = len(os.listdir(os.path.join(args.output_dir, 'normal'))) if os.path.exists(os.path.join(args.output_dir, 'normal')) else 0
        total_chainsaw = len(os.listdir(os.path.join(args.output_dir, 'abnormal', 'chainsaw'))) if os.path.exists(os.path.join(args.output_dir, 'abnormal', 'chainsaw')) else 0
        total_machinery = len(os.listdir(os.path.join(args.output_dir, 'abnormal', 'machinery'))) if os.path.exists(os.path.join(args.output_dir, 'abnormal', 'machinery')) else 0
        
        print(f"\nFinal Dataset Statistics:")
        print(f"Normal: {total_normal} samples")
        print(f"Abnormal > Chainsaw: {total_chainsaw} samples")
        print(f"Abnormal > Machinery: {total_machinery} samples")
        print(f"Total: {total_normal + total_chainsaw + total_machinery} samples")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()