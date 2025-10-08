# Chainsaw Detection System for Illegal Logging Monitoring

## Overview
This system provides a comprehensive solution for detecting chainsaws in forest environments to combat illegal logging. The solution uses a two-stage machine learning approach:
1. Anomaly Detection: Binary classifier (normal vs abnormal sounds)
2. Classification: Multi-class classifier (chainsaw vs vehicles for abnormal sounds)

The system is optimized for deployment on the MAX78000 microcontroller using the AI8X training framework.

## Features
- Audio processing at 32kHz sample rate
- MFCC feature extraction optimized for AI8X framework
- Two-stage detection pipeline for improved accuracy
- Field-adjustable parameters for environmental adaptation
- Real-time processing capability
- Memory and power optimized for edge deployment

## System Architecture

### 1. Dataset Generator
- Processes audio files organized in `data/normal/` and `data/anomaly/` structure
- Converts all files to 32kHz sample rate
- Segments into 1-second clips with overlap
- Applies audio data augmentation techniques
- Outputs organized dataset with consistent format

### 2. Data Pipeline
- MFCC feature extraction optimized for AI8X framework
- Efficient data loading and batching for training
- Configurable train/validation/test split functionality
- Data augmentation pipeline integration
- AI8X-compatible data format conversion

### 3. Model Architecture
- **Anomaly Detection Model**: Binary classifier optimized for MAX78000 constraints
- **Classification Model**: Multi-class classifier for abnormal sound categorization
- Both models support field-adjustable parameters

### 4. Training Pipeline
- Two-stage training implementation
- Configurable hyperparameter management
- Comprehensive model evaluation and metrics reporting
- Checkpoint saving/loading functionality

### 5. Evaluation Tools
- Performance metrics calculation
- Confusion matrix and detailed reports
- Threshold optimization for field deployment
- Noise robustness testing

### 6. Deployment Configuration
- Model conversion to C code for MAX78000
- Field-adjustable parameter definitions
- Memory and processing optimization settings

## Requirements

### Hardware
- Target: MAX78000 microcontroller
- Development: Compatible system for training (see requirements.txt)

### Software
- Python 3.8+ (recommended 3.11)
- PyTorch 2.0+
- Additional dependencies listed in `chainsaw_requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/ai8x-training.git
cd ai8x-training
```

2. Install dependencies:
```bash
pip install -r chainsaw_requirements.txt
```

## Usage

### 1. Dataset Preparation
Organize your audio data as follows:
```
data/
├── normal/
│   ├── animals/
│   ├── calm_forest/
│   └── ...
└── anomaly/
    ├── chainsaw/
    ├── machinery/  (or 'machines' or 'vehicles')
    └── ...
```

Alternative structure (if organizing at root level):
```
data/
├── normal/
├── chainsaw/
├── machinery/  (or 'machines' or 'vehicles')
└── ...
```

### 2. Generate Dataset
```bash
python chainsaw_dataset_generator.py \
    --input_dir ./data \
    --output_dir ./dataset \
    --sample_rate 32000 \
    --segment_length 1.0 \
    --overlap 0.5 \
    --augmentation
```

### 3. Train Models
```bash
python chainsaw_training_pipeline.py \
    --data_dir ./dataset \
    --output_dir ./trained_models \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --stage both
```

### 4. Evaluate Models
```bash
python chainsaw_evaluation.py \
    --anomaly_model_path ./trained_models/anomaly_detection/best_anomaly_model.pth \
    --classification_model_path ./trained_models/classification/best_classification_model.pth \
    --data_dir ./dataset \
    --output_dir ./evaluation_results \
    --threshold 0.5
```

### 5. Generate Deployment Files
```bash
python chainsaw_deployment_config.py
```

## Configuration

The system can be configured using `chainsaw_config.yaml`:
- Audio processing parameters
- Model architecture settings
- Training hyperparameters
- Performance targets
- Field-adjustable parameters

## Performance Targets
- Accuracy: >85% for both anomaly detection and classification
- Real-time processing capability (continuous operation)
- Optimized for MAX78000 memory and processing constraints
- Robust performance in forest environment conditions

## Deployment

1. Convert trained models to C code using AI8X synthesis tools
2. Integrate with MAX78000 MSDK application
3. Configure field parameters using generated configuration files
4. Deploy to field monitoring device

## File Structure
```
chainsaw_detection/
├── chainsaw_dataset_generator.py    # Dataset generation
├── chainsaw_data_pipeline.py        # Data loading pipeline
├── chainsaw_model_architecture.py   # Model definitions  
├── chainsaw_training_pipeline.py    # Training implementation
├── chainsaw_evaluation.py           # Evaluation tools
├── chainsaw_deployment_config.py    # Deployment tools
├── chainsaw_requirements.txt        # Dependencies
├── chainsaw_config.yaml             # Configuration
└── README.md                       # This file
```

## Contributing
Contributions to improve the chainsaw detection system are welcome. Please follow the standard fork-and-pull request workflow.

## License
This project is part of the AI8X training repository and follows its licensing terms.

## Support
For support, please refer to the main AI8X training repository documentation or create an issue in the repository.