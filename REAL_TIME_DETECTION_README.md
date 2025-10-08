# Real-time Chainsaw Detection System

This system implements a real-time chainsaw detection application using two pre-trained models in a two-stage pipeline:

1. **Anomaly Detection**: Binary classifier (normal vs abnormal sounds)
2. **Classification**: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)

## Features

- Real-time audio processing from microphone input
- Two-stage detection pipeline for improved accuracy
- Configurable detection threshold
- Performance statistics tracking
- File-based testing capability

## Prerequisites

- Python 3.8+
- PyTorch
- librosa
- sounddevice (for real-time microphone input)
- numpy, scipy

Install required packages with:
```bash
pip install sounddevice
```

## Usage

### Real-time Detection from Microphone

```bash
python real_time_chainsaw_test.py
```

This will start real-time detection using your default microphone. The system will process audio in 1-second segments and display results in real-time.

### File-based Testing

```bash
python real_time_chainsaw_test.py --file path/to/audio/file.wav
```

Use this to test the system with pre-recorded audio files.

### Custom Model Paths

```bash
python real_time_chainsaw_test.py \
    --anomaly_model /path/to/anomaly/model.pth \
    --classification_model /path/to/classification/model.pth \
    --file test_audio.wav
```

### Adjusting Sensitivity

```bash
python real_time_chainsaw_test.py --file test_audio.wav --anomaly_threshold 0.7
```

Lower threshold values make the system more sensitive (detect more anomalies), while higher values make it more selective.

### Using Specific Audio Device

```bash
python real_time_chainsaw_test.py --device_id 1
```

Use `--device_id` to specify a particular audio input device. The system will show available devices when starting.

## Output Format

The system provides real-time output in the following format:

```
[2025-01-01 12:00:00.000] ANOMALY DETECTED (Confidence: 95.2%) -> CHAINSAW (Confidence: 87.3%)
    Inference time: 15.2ms + 8.3ms = 23.5ms
```

- Timestamp of detection
- Anomaly detection result with confidence percentage
- Classification result (if anomaly detected) with confidence percentage
- Inference times for both stages

## Models

The system expects two trained models in the following locations by default:

- `./trained_models/anomaly_detection/best_anomaly_model.pth` - Anomaly detection model
- `./trained_models/classification/best_classification_model.pth` - Classification model

## Audio Format

The system expects audio at 32kHz sample rate. Input audio is automatically resampled if needed.

## Performance Notes

- Processing time is typically under 100ms per audio segment
- The system is optimized for real-time performance
- Detection accuracy may vary based on audio quality and environmental conditions

## Troubleshooting

If you encounter issues with audio input:
1. Make sure you have a working microphone connected
2. Check that your system allows Python applications to access the microphone
3. Verify that the audio device is not being used by another application
4. Make sure the `sounddevice` package is installed correctly: `pip install sounddevice`