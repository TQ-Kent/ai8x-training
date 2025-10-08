"""
Deployment Configuration for Chainsaw Detection System

This module provides:
- Model conversion to C code for MAX78000 deployment
- Field-adjustable parameter definitions
- Memory and processing optimization settings
- Deployment guide with integration instructions
"""
import os
import json
import yaml
from typing import Dict, Any, List
from pathlib import Path


def generate_model_config(model_type: str, model_path: str, 
                         input_shape: tuple, output_shape: tuple,
                         save_path: str):
    """
    Generate a model configuration file for deployment.
    
    Args:
        model_type: Type of model ('anomaly' or 'classification')
        model_path: Path to the trained model
        input_shape: Shape of input data (time_steps, features)
        output_shape: Shape of output data
        save_path: Path to save the config file
    """
    config = {
        'model_type': model_type,
        'model_path': model_path,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'framework': 'ai8x',
        'target_device': 'MAX78000',
        'sample_rate': 32000,
        'feature_type': 'mfcc',
        'n_mfcc': 13,
        'target_length': 62,
        'quantization': {
            'weight_bits': 8,
            'activation_bits': 8,
            'bias_bits': 8
        },
        'field_parameters': {
            'detection_threshold': 0.5,
            'min_confidence': 0.7,
            'max_false_positive_rate': 0.1,
            'sensitivity': 0.9
        },
        'deployment_settings': {
            'memory_limit_kb': 256,  # MAX78000 memory constraints
            'max_inference_time_ms': 100,
            'power_budget_mw': 100,
            'processing_cores_used': 64
        },
        'preprocessing': {
            'normalize_input': True,
            'input_range': [-128, 127],
            'feature_extraction_params': {
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 40,
                'f_min': 0,
                'f_max': 16000
            }
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model configuration saved to: {save_path}")


def generate_deployment_yaml(model_config_path: str, 
                           output_path: str,
                           sample_data_path: str = None):
    """
    Generate a deployment YAML file compatible with AI8X synthesis.
    
    Args:
        model_config_path: Path to the model config JSON file
        output_path: Path to save the YAML file
        sample_data_path: Path to sample input data (optional)
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    yaml_config = {
        'arch': f"chainsaw_{config['model_type']}_net",
        'dataset': 'chainsaw_detection',
        'input': config['input_shape'],
        'output': config['output_shape'],
        'layers': []  # This would be filled in based on the actual model
    }
    
    # Example layers for a simple CNN structure (this is illustrative)
    if config['model_type'] == 'anomaly':
        yaml_config['layers'] = [
            {
                'pad': 1,
                'activate': 'ReLU',
                'out_offset': 0x2000,
                'processors': 0x0000000000000001,
                'data_format': 'HWC',
                'op': 'conv1d',
                'kernel_size': 3,
                'in_channels': config['n_mfcc'],
                'out_channels': 32
            },
            {
                'max_pool': 2,
                'pool_stride': 2,
                'pad': 0,
                'activate': 'ReLU',
                'out_offset': 0,
                'processors': 0xfffffffffffffff0,
                'op': 'conv1d',
                'kernel_size': 3,
                'in_channels': 32,
                'out_channels': 64
            },
            {
                'op': 'mlp',
                'flatten': True,
                'out_offset': 0x1000,
                'output_width': 32,
                'processors': 0x0000000000000fff,
                'in_channels': 64 * (config['target_length'] // 4),  # After pooling
                'out_channels': config['output_shape'][0] if isinstance(config['output_shape'], (list, tuple)) else config['output_shape']
            }
        ]
    elif config['model_type'] == 'classification':
        # Similar structure for classification model
        yaml_config['layers'] = [
            {
                'pad': 1,
                'activate': 'ReLU',
                'out_offset': 0x2000,
                'processors': 0x0000000000000001,
                'data_format': 'HWC',
                'op': 'conv1d',
                'kernel_size': 3,
                'in_channels': config['n_mfcc'],
                'out_channels': 32
            },
            {
                'max_pool': 2,
                'pool_stride': 2,
                'pad': 0,
                'activate': 'ReLU',
                'out_offset': 0,
                'processors': 0xfffffffffffffff0,
                'op': 'conv1d',
                'kernel_size': 3,
                'in_channels': 32,
                'out_channels': 64
            },
            {
                'op': 'mlp',
                'flatten': True,
                'out_offset': 0x1000,
                'output_width': 32,
                'processors': 0x0000000000000fff,
                'in_channels': 64 * (config['target_length'] // 4),  # After pooling
                'out_channels': config['output_shape'][0] if isinstance(config['output_shape'], (list, tuple)) else config['output_shape']
            }
        ]
    
    # Add sample data reference if provided
    if sample_data_path:
        yaml_config['sample_data'] = sample_data_path
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
    
    print(f"Deployment YAML saved to: {output_path}")


def generate_field_config(save_path: str):
    """
    Generate field-adjustable parameters configuration.
    
    Args:
        save_path: Path to save the field config file
    """
    field_config = {
        'version': '1.0',
        'parameters': {
            'anomaly_detection': {
                'detection_threshold': {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'description': 'Threshold for anomaly detection'
                },
                'min_confidence': {
                    'default': 0.7,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'description': 'Minimum confidence for positive detection'
                },
                'sensitivity': {
                    'default': 0.9,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'description': 'Sensitivity level (higher = more sensitive)'
                }
            },
            'classification': {
                'classification_threshold': {
                    'default': 0.6,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'description': 'Threshold for classification decisions'
                }
            }
        },
        'deployment_settings': {
            'audio_buffer_size': {
                'default': 32000,  # 1 second at 32kHz
                'description': 'Size of audio buffer for processing'
            },
            'processing_interval_ms': {
                'default': 1000,
                'min': 100,
                'max': 5000,
                'description': 'Interval between processing cycles in milliseconds'
            },
            'power_mode': {
                'default': 'normal',
                'options': ['low_power', 'normal', 'high_performance'],
                'description': 'Power consumption mode'
            }
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(field_config, f, indent=2)
    
    print(f"Field configuration saved to: {save_path}")


def generate_c_header(model_config_path: str, output_path: str):
    """
    Generate C header file for model deployment.
    
    Args:
        model_config_path: Path to the model config JSON file
        output_path: Path to save the C header file
    """
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    c_header_content = f'''/* Chainsaw Detection Model Configuration */
/* Auto-generated file - do not edit manually */

#ifndef CHAINSAW_DETECTION_CONFIG_H
#define CHAINSAW_DETECTION_CONFIG_H

/* Model Configuration */
#define MODEL_TYPE_{config['model_type'].upper()} 1
#define INPUT_CHANNELS {config['n_mfcc']}
#define INPUT_LENGTH {config['target_length']}
#define OUTPUT_CLASSES {(config['output_shape'][0] if isinstance(config['output_shape'], (list, tuple)) else config['output_shape'])}

/* Sample Rate */
#define SAMPLE_RATE {config['sample_rate']}

/* Feature Extraction Parameters */
#define N_FFT 1024
#define HOP_LENGTH 512
#define N_MELS 40
#define N_MFCC {config['n_mfcc']}

/* Quantization Parameters */
#define WEIGHT_BITS {config['quantization']['weight_bits']}
#define ACTIVATION_BITS {config['quantization']['activation_bits']}
#define BIAS_BITS {config['quantization']['bias_bits']}

/* Field-Adjustable Parameters */
#define DEFAULT_DETECTION_THRESHOLD {config['field_parameters']['detection_threshold']:.2f}f
#define DEFAULT_MIN_CONFIDENCE {config['field_parameters']['min_confidence']:.2f}f
#define MAX_FALSE_POSITIVE_RATE {config['field_parameters']['max_false_positive_rate']:.2f}f
#define DEFAULT_SENSITIVITY {config['field_parameters']['sensitivity']:.2f}f

/* Deployment Settings */
#define MEMORY_LIMIT_KB {config['deployment_settings']['memory_limit_kb']}
#define MAX_INFERENCE_TIME_MS {config['deployment_settings']['max_inference_time_ms']}
#define POWER_BUDGET_MW {config['deployment_settings']['power_budget_mw']}
#define PROCESSING_CORES_USED {config['deployment_settings']['processing_cores_used']}

/* Preprocessing Settings */
#define INPUT_NORMALIZE {1 if config['preprocessing']['normalize_input'] else 0}
#define INPUT_MIN_VAL {config['preprocessing']['input_range'][0]}
#define INPUT_MAX_VAL {config['preprocessing']['input_range'][1]}

#endif /* CHAINSAW_DETECTION_CONFIG_H */
'''
    
    with open(output_path, 'w') as f:
        f.write(c_header_content)
    
    print(f"C header file saved to: {output_path}")


def generate_deployment_guide(save_path: str):
    """
    Generate a deployment guide with integration instructions.
    
    Args:
        save_path: Path to save the deployment guide
    """
    guide_content = '''
# Chainsaw Detection Deployment Guide

## Overview
This guide provides instructions for deploying the chainsaw detection system on the MAX78000 microcontroller.

## Prerequisites
- MAX78000 evaluation kit or compatible hardware
- ADI MSDK (Microcontroller Software Development Kit)
- AI8X synthesis tools
- Trained model files

## Hardware Setup
1. Connect microphone input to analog-to-digital converter
2. Ensure proper power supply (typically 3.3V)
3. Verify clock settings for 32kHz audio sampling

## Model Conversion
1. Convert the trained PyTorch model to ONNX format
2. Use AI8X synthesis to convert ONNX to C code
3. Integrate the C code with your application

## Integration Steps

### 1. Audio Input Configuration
- Configure ADC for 32kHz sampling rate
- Set up DMA for continuous audio capture
- Apply preprocessing (normalize to [-128, +127] range)

### 2. Feature Extraction
- Extract MFCC features using the parameters defined in the config
- Ensure feature extraction matches training preprocessing

### 3. Model Inference
- Load the converted model weights
- Process features through the neural network
- Apply detection threshold for anomaly classification

### 4. Output Handling
- Anomaly Detection: Classify as normal or abnormal
- Classification: If abnormal, classify as chainsaw or machinery
- Generate appropriate alerts or notifications

## Field Configuration
The system supports field-adjustable parameters:
- Detection threshold: Adjust sensitivity to environmental conditions
- Minimum confidence: Filter low-confidence detections
- Processing interval: Balance between detection speed and power consumption

## Performance Optimization
- Use the provided quantization settings for optimal memory usage
- Adjust processing cores as needed for real-time performance
- Monitor power consumption and adjust accordingly

## Troubleshooting
- If experiencing high false positive rates, increase the detection threshold
- If missing detections, decrease the threshold or check audio input levels
- Verify that the model was trained with the same preprocessing as used in deployment

## Memory Requirements
- Model weights: Approximately [X] KB
- Feature buffer: [Y] KB
- Working memory: [Z] KB
- Total: Should fit within MAX78000 constraints

## Power Consumption
- Active inference: ~[X] mW
- Idle/low-power mode: ~[Y] mW
- Battery life estimation based on duty cycle

## Validation
- Test with various environmental conditions
- Verify detection accuracy with known chainsaw sounds
- Ensure system operates within power budget
'''
    
    with open(save_path, 'w') as f:
        f.write(guide_content)
    
    print(f"Deployment guide saved to: {save_path}")


def main():
    # Create output directory
    output_dir = './deployment_config'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating deployment configuration...")
    
    # Generate configurations for both models
    for model_type in ['anomaly', 'classification']:
        model_config_path = os.path.join(output_dir, f'{model_type}_model_config.json')
        yaml_path = os.path.join(output_dir, f'{model_type}_deployment.yaml')
        
        generate_model_config(
            model_type=model_type,
            model_path=f'./trained_models/{model_type}_detection/best_{model_type}_model.pth',
            input_shape=(62, 13),  # (time_steps, features)
            output_shape=(2,) if model_type == 'anomaly' else (2,),  # Binary classification
            save_path=model_config_path
        )
        
        generate_deployment_yaml(
            model_config_path=model_config_path,
            output_path=yaml_path
        )
        
        # Generate C header for each model type
        c_header_path = os.path.join(output_dir, f'{model_type}_model_config.h')
        generate_c_header(model_config_path, c_header_path)
    
    # Generate field configuration
    field_config_path = os.path.join(output_dir, 'field_config.json')
    generate_field_config(field_config_path)
    
    # Generate deployment guide
    guide_path = os.path.join(output_dir, 'deployment_guide.md')
    generate_deployment_guide(guide_path)
    
    print(f"Deployment configuration generated in: {output_dir}")


if __name__ == "__main__":
    main()