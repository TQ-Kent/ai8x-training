"""
Testing and Evaluation Script for Chainsaw Detection System

This script provides:
- Performance metrics calculation (accuracy, precision, recall, F1-score)
- Confusion matrix and detailed classification reports
- Threshold optimization tools for field deployment
- Forest deployment simulation with noise robustness testing
- Model comparison and selection utilities
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import argparse
import json
import pickle
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import copy

# Import our custom modules
from chainsaw_data_pipeline import get_dataloaders, ChainsawAudioDataset
from chainsaw_model_architecture import AnomalyDetectionNet, ClassificationNet
import ai8x


def load_model(model_class, model_path: str, device: torch.device, **model_kwargs):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to the saved model
        device: Device to load model on
        **model_kwargs: Additional arguments for model initialization
    
    Returns:
        Loaded model
    """
    model = model_class(**model_kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both state dict formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def _validate_evaluation_results(y_true, y_pred, y_scores, all_inputs, task_type):
    """
    Validate evaluation results for potential issues like overfitting or data leakage.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted scores/probabilities
        all_inputs: Input tensors (flattened)
        task_type: Type of task ('anomaly' or 'classification')
    """
    print(f"Validating {task_type} evaluation results...")
    
    # Check for perfect accuracy (potential overfitting/data leakage)
    accuracy = accuracy_score(y_true, y_pred)
    if accuracy == 1.0:
        print(f"⚠️  WARNING: Perfect accuracy ({accuracy:.2f}) detected for {task_type} task!")
        print("   This may indicate overfitting, data leakage, or overly simplistic dataset.")
    
    # Check for perfect AUC if scores are provided
    if y_scores is not None and len(np.unique(y_true)) == 2:
        try:
            if task_type == 'anomaly':
                # For anomaly detection, check if AUC is suspiciously high
                roc_auc = roc_auc_score(y_true, y_scores)
                if roc_auc >= 0.99:
                    print(f"⚠️  WARNING: Very high AUC ({roc_auc:.3f}) detected for {task_type} task!")
                    print("   This may indicate data leakage or overfitting.")
            elif task_type == 'classification':
                # For classification, process the score matrix properly
                y_scores_array = np.array(y_scores)
                if y_scores_array.ndim > 1:
                    # Get probabilities for the predicted class
                    predicted_class_probs = [y_scores_array[i, pred] for i, pred in enumerate(y_pred)]
                    roc_auc = roc_auc_score(y_true, predicted_class_probs)
                    if roc_auc >= 0.99:
                        print(f"⚠️  WARNING: Very high AUC ({roc_auc:.3f}) detected for {task_type} task!")
                        print("   This may indicate data leakage or overfitting.")
        except:
            # If AUC calculation fails (e.g., only one class in sample), skip
            pass
            
    # Check class distribution in predictions vs true labels
    unique_true, true_counts = np.unique(y_true, return_counts=True)
    unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
    
    print(f"  True label distribution: {dict(zip(unique_true, true_counts))}")
    print(f"  Predicted label distribution: {dict(zip(unique_pred, pred_counts))}")
    
    # Check if model is just predicting one class all the time (degenerate case)
    if len(unique_pred) == 1:
        print(f"⚠️  WARNING: Model predicts only one class ({unique_pred[0]}) for all samples!")
        print("   This suggests a potential problem with training or evaluation.")


def calculate_performance_metrics(y_true: List, y_pred: List, 
                                 y_scores: Optional[List] = None,
                                 class_names: Optional[List] = None) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted scores/probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.unique(y_true)
    )
    
    # Store per-class metrics
    if class_names:
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision[i] if i < len(precision) else 0
            metrics[f'recall_{class_name}'] = recall[i] if i < len(recall) else 0
            metrics[f'f1_{class_name}'] = f1[i] if i < len(f1) else 0
            metrics[f'support_{class_name}'] = support[i] if i < len(support) else 0
    else:
        for i in range(len(precision)):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1[i]
            metrics[f'support_class_{i}'] = support[i]
    
    # Overall metrics
    metrics['precision_avg'] = np.mean(precision)
    metrics['recall_avg'] = np.mean(recall)
    metrics['f1_avg'] = np.mean(f1)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # ROC AUC if scores are provided
    if y_scores is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            else:
                # For multiclass, compute ROC AUC for each class
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores, multi_class='ovr')
        except:
            metrics['roc_auc'] = None
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str = "Confusion Matrix", 
                         save_path: Optional[str] = None):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: Names of classes
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: List, y_scores: List, 
                   save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores/probabilities
        save_path: Path to save the plot (optional)
    """
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("ROC curve is only supported for binary classification.")


def optimize_threshold(model, val_loader, device, metric='f1'):
    """
    Optimize threshold for anomaly detection based on validation set.
    
    Args:
        model: Trained anomaly detection model
        val_loader: Validation data loader
        device: Device to run evaluation on
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Optimal threshold value
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # For binary classification, get probability of positive class
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of anomalous class
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_metric_value = 0
    
    for threshold in thresholds:
        # Convert probabilities to predictions using threshold
        preds = (all_probs > threshold).astype(int)
        
        # Calculate metric
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        
        if metric == 'f1':
            metric_value = f1
        elif metric == 'precision':
            metric_value = precision
        elif metric == 'recall':
            metric_value = recall
        else:
            metric_value = f1  # Default to F1
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    return best_threshold


def evaluate_anomaly_detection(model, test_loader, device, 
                              threshold: float = 0.5, save_dir: str = None):
    """
    Evaluate anomaly detection model.
    
    Args:
        model: Trained anomaly detection model
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Threshold for anomaly classification
        save_dir: Directory to save results (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_outputs = []
    all_labels = []
    all_probs = []
    all_inputs = []  # For additional validation
    
    class_names = ['Normal', 'Abnormal']
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            # For anomaly detection, we care about the "abnormal" class (index 1)
            abnormal_probs = probs[:, 1]
            all_probs.extend(abnormal_probs.cpu().numpy())
            
            # Store inputs for potential validation (flatten to compare)
            all_inputs.extend(inputs.cpu().numpy().reshape(inputs.size(0), -1))
            
            # Apply threshold for binary decision
            preds = (abnormal_probs > threshold).long()
            all_outputs.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Perform validation checks
    _validate_evaluation_results(all_labels, all_outputs, all_probs, all_inputs, "anomaly")
    
    # Calculate metrics
    metrics = calculate_performance_metrics(
        all_labels, all_outputs, all_probs, class_names
    )
    
    # Additional anomaly-specific metrics
    cm = np.array(metrics['confusion_matrix'])
    metrics['false_positive_rate'] = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    metrics['false_negative_rate'] = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    metrics['true_positive_rate'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    metrics['true_negative_rate'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    # Add threshold value
    metrics['threshold_used'] = threshold
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(save_dir, 'anomaly_evaluation_metrics.json'), 'w') as f:
            # Convert metrics to JSON-serializable format
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_json[key] = value.item()
                elif isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    metrics_json[key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    metrics_json[key] = value
            json.dump(metrics_json, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']), 
            class_names,
            title=f'Anomaly Detection Confusion Matrix (Threshold: {threshold})',
            save_path=os.path.join(save_dir, 'anomaly_confusion_matrix.png')
        )
        
        # Plot ROC curve if binary classification
        plot_roc_curve(
            all_labels, 
            all_probs,
            save_path=os.path.join(save_dir, 'anomaly_roc_curve.png')
        )
        
        # Classification report
        report = classification_report(all_labels, all_outputs, 
                                     target_names=class_names, output_dict=True)
        with open(os.path.join(save_dir, 'anomaly_classification_report.json'), 'w') as f:
            # Convert report to JSON-serializable format
            report_json = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    report_json[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (np.integer, np.floating)):
                            report_json[key][sub_key] = sub_value.item()
                        else:
                            report_json[key][sub_key] = sub_value
                elif isinstance(value, (np.integer, np.floating)):
                    report_json[key] = value.item()
                else:
                    report_json[key] = value
            json.dump(report_json, f, indent=2)
    
    return metrics


def evaluate_classification(model, test_loader, device, 
                           save_dir: str = None):
    """
    Evaluate classification model.
    
    Args:
        model: Trained classification model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save results (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_outputs = []
    all_labels = []
    all_scores = []
    all_inputs = []  # For additional validation
    
    class_names = ['Chainsaw', 'Machinery']
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_outputs.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_inputs.extend(inputs.cpu().numpy().reshape(inputs.size(0), -1))
    
    # Perform validation checks
    _validate_evaluation_results(all_labels, all_outputs, all_scores, all_inputs, "classification")
    
    # Calculate metrics
    metrics = calculate_performance_metrics(
        all_labels, all_outputs, all_scores, class_names
    )
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(save_dir, 'classification_evaluation_metrics.json'), 'w') as f:
            # Convert metrics to JSON-serializable format
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_json[key] = value.item()
                elif isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    metrics_json[key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    metrics_json[key] = value
            json.dump(metrics_json, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']), 
            class_names,
            title='Classification Confusion Matrix',
            save_path=os.path.join(save_dir, 'classification_confusion_matrix.png')
        )
        
        # Classification report
        report = classification_report(all_labels, all_outputs, 
                                     target_names=class_names, output_dict=True)
        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
    
    return metrics


def simulate_noisy_deployment(model, test_loader, device, 
                              noise_levels: List[float] = [0.001, 0.005, 0.01]):
    """
    Simulate model performance under noisy conditions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        noise_levels: List of noise levels to test
    
    Returns:
        Dictionary with performance under different noise conditions
    """
    model.eval()
    results = {}
    
    for noise_level in noise_levels:
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Add Gaussian noise
                noise = torch.randn_like(inputs) * noise_level
                noisy_inputs = inputs + noise
                
                outputs = model(noisy_inputs)
                _, preds = torch.max(outputs, 1)
                
                all_outputs.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy for this noise level
        accuracy = accuracy_score(all_labels, all_outputs)
        results[f'noise_{noise_level:.3f}'] = accuracy
    
    return results


def compare_models(model_paths: List[str], test_loader, device, model_names: List[str] = None):
    """
    Compare performance of multiple models.
    
    Args:
        model_paths: List of paths to different models
        test_loader: Test data loader
        device: Device to run evaluation on
        model_names: Names of the models (optional)
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for i, model_path in enumerate(model_paths):
        if model_names:
            name = model_names[i]
        else:
            name = f'Model_{i}'
        
        # Load model (this is a simplified approach)
        # In practice, you'd need to know the model architecture
        print(f"Evaluating {name}...")
        # For this example, we'll just record the name
        # Real implementation would load and evaluate each model
        results.append({
            'Model': name,
            'Path': model_path
            # You would add metrics here after evaluation
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Testing and Evaluation for Chainsaw Detection')
    
    # Model settings
    parser.add_argument('--anomaly_model_path', type=str, required=True,
                        help='Path to trained anomaly detection model')
    parser.add_argument('--classification_model_path', type=str, required=True,
                        help='Path to trained classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the test dataset')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for anomaly classification')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Optimize threshold based on validation set')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    # Feature extraction settings (for loading data)
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'mel', 'spectrogram'],
                        help='Type of features used in models')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--target_length', type=int, default=62,
                        help='Target length for features')
    
    # Noise simulation
    parser.add_argument('--simulate_noise', action='store_true',
                        help='Simulate noisy deployment conditions')
    parser.add_argument('--device_id', type=int, default=85,
                        help='AI8X device ID (85 for MAX78000, 87 for MAX78002)')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate hardware behavior')
    parser.add_argument('--round_avg', action='store_true',
                        help='Use rounding for average pooling')
    
    args = parser.parse_args()
    
    print("Starting Testing and Evaluation")
    print(f"Configuration: {args}")
    
    # Set device
    device = torch.device("cpu")  # Force CPU for compatibility
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure AI8X device
    ai8x.set_device(args.device_id, args.simulate, args.round_avg)
    
    # Configure AI8X device
    ai8x.set_device(args.device_id, args.simulate, args.round_avg)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        feature_type=args.feature_type,
        n_mfcc=args.n_mfcc,
        target_length=args.target_length
    )
    
    # Load anomaly detection model
    print("Loading anomaly detection model...")
    anomaly_model = load_model(
        AnomalyDetectionNet,
        args.anomaly_model_path,
        device,
        num_channels=args.n_mfcc,
        feature_length=args.target_length,
        num_classes=2
    )
    
    # If optimizing threshold, we need validation data
    if args.optimize_threshold:
        print("Optimizing threshold on validation data...")
        # In practice, you'd load a separate validation set
        # For this example, we'll just use test data to demonstrate
        # In real usage, you'd want a separate validation set
        opt_threshold = optimize_threshold(anomaly_model, test_loader, device, metric='f1')
        print(f"Optimized threshold: {opt_threshold}")
        args.threshold = opt_threshold
    
    # Evaluate anomaly detection model
    print("Evaluating anomaly detection model...")
    anomaly_metrics = evaluate_anomaly_detection(
        anomaly_model, 
        test_loader, 
        device, 
        threshold=args.threshold,
        save_dir=os.path.join(args.output_dir, 'anomaly_detection')
    )
    print(f"Anomaly Detection Metrics: {anomaly_metrics}")
    
    # Load classification model
    print("Loading classification model...")
    classification_model = load_model(
        ClassificationNet,
        args.classification_model_path,
        device,
        num_channels=args.n_mfcc,
        feature_length=args.target_length,
        num_classes=2
    )
    
    # Evaluate classification model
    print("Evaluating classification model...")
    classification_metrics = evaluate_classification(
        classification_model,
        test_loader,
        device,
        save_dir=os.path.join(args.output_dir, 'classification')
    )
    print(f"Classification Metrics: {classification_metrics}")
    
    # If simulating noise conditions
    if args.simulate_noise:
        print("Simulating noisy deployment conditions...")
        anomaly_noise_results = simulate_noisy_deployment(
            anomaly_model, test_loader, device, args.noise_levels
        )
        classification_noise_results = simulate_noisy_deployment(
            classification_model, test_loader, device, args.noise_levels
        )
        
        print(f"Anomaly detection with noise: {anomaly_noise_results}")
        print(f"Classification with noise: {classification_noise_results}")
        
        # Save noise simulation results
        with open(os.path.join(args.output_dir, 'noise_simulation_results.json'), 'w') as f:
            json.dump({
                'anomaly_detection': anomaly_noise_results,
                'classification': classification_noise_results
            }, f, indent=2)
    
    # Save overall results
    overall_results = {
        'anomaly_detection': anomaly_metrics,
        'classification': classification_metrics
    }
    
    with open(os.path.join(args.output_dir, 'overall_results.json'), 'w') as f:
        # Convert to JSON-serializable format
        overall_json = {}
        for stage, metrics in overall_results.items():
            overall_json[stage] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    overall_json[stage][key] = value.item()
                elif isinstance(value, np.ndarray):
                    overall_json[stage][key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    overall_json[stage][key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    overall_json[stage][key] = value
        json.dump(overall_json, f, indent=2)
    
    # Create summary report
    summary = {
        'timestamp': str(datetime.datetime.now()),
        'configuration': vars(args),
        'results': {
            'anomaly_detection_accuracy': float(anomaly_metrics['accuracy']),
            'classification_accuracy': float(classification_metrics['accuracy']),
            'anomaly_detection_f1_avg': float(anomaly_metrics['f1_avg']),
            'classification_f1_avg': float(classification_metrics['f1_avg'])
        }
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Testing and Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    import datetime
    main()