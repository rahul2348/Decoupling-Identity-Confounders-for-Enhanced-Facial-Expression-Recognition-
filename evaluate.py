#!/usr/bin/env python3
"""
Evaluation script for DICE-FER model
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import logging

from dice_fer import DICEFER, load_model
from datasets import create_data_loaders, get_dataset_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate DICE-FER model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='ckplus',
                       choices=['ckplus', 'rafdb', 'affectnet', 'oulucasia'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset directory')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_path', type=str, default='evaluation_results',
                       help='Path to save evaluation results')
    
    return parser.parse_args()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy)
    plt.xlabel('Expression Classes')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and return detailed results"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    expression_features = []
    identity_features = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model.forward(data)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs['expression_logits'], dim=1)
            predictions = torch.argmax(outputs['expression_logits'], dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            # Store features for analysis
            expression_features.append(outputs['expression_features'].cpu().numpy())
            identity_features.append(outputs['identity_features'].cpu().numpy())
    
    # Concatenate all results
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    expression_features = np.concatenate(expression_features, axis=0)
    identity_features = np.concatenate(identity_features, axis=0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Generate classification report
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'expression_features': expression_features,
        'identity_features': identity_features,
        'classification_report': report
    }

def analyze_feature_disentanglement(expression_features, identity_features, targets, save_path):
    """Analyze feature disentanglement"""
    # Compute correlation between expression and identity features
    correlation_matrix = np.corrcoef(expression_features.T, identity_features.T)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Matrix (Expression vs Identity)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute feature variance explained by class labels
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # PCA on expression features
    pca_exp = PCA(n_components=min(10, expression_features.shape[1]))
    pca_exp.fit(expression_features)
    exp_variance_ratio = pca_exp.explained_variance_ratio_
    
    # LDA on expression features
    lda_exp = LinearDiscriminantAnalysis()
    lda_exp.fit(expression_features, targets)
    exp_lda_score = lda_exp.score(expression_features, targets)
    
    # PCA on identity features
    pca_id = PCA(n_components=min(10, identity_features.shape[1]))
    pca_id.fit(identity_features)
    id_variance_ratio = pca_id.explained_variance_ratio_
    
    # Plot variance explained
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(np.cumsum(exp_variance_ratio), 'b-', label='Expression Features')
    ax1.plot(np.cumsum(id_variance_ratio), 'r-', label='Identity Features')
    ax1.set_title('Cumulative Variance Explained')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Variance Ratio')
    ax1.legend()
    ax1.grid(True)
    
    # Plot LDA score
    ax2.bar(['Expression Features'], [exp_lda_score], color='blue', alpha=0.7)
    ax2.set_title('LDA Classification Score')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'expression_variance_ratio': exp_variance_ratio,
        'identity_variance_ratio': id_variance_ratio,
        'expression_lda_score': exp_lda_score
    }

def save_results(results, save_path):
    """Save evaluation results"""
    # Save classification report
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_df.to_csv(os.path.join(save_path, 'classification_report.csv'))
    
    # Save detailed results
    detailed_results = {
        'accuracy': results['accuracy'],
        'classification_report': results['classification_report'],
        'feature_analysis': results.get('feature_analysis', {})
    }
    
    import json
    with open(os.path.join(save_path, 'evaluation_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': results['targets'],
        'predicted_label': results['predictions'],
        'confidence': np.max(results['probabilities'], axis=1)
    })
    predictions_df.to_csv(os.path.join(save_path, 'predictions.csv'), index=False)

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get dataset information
    dataset_info = get_dataset_info(args.dataset)
    class_names = dataset_info.get('expressions', [f'class_{i}' for i in range(7)])
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Class names: {class_names}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, test_loader = create_data_loaders(
            dataset_name=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers
        )
        logger.info(f"Test samples: {len(test_loader.dataset)}")  # type: ignore
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return
    
    # Create model and load checkpoint
    logger.info("Loading model...")
    model = DICEFER(
        num_classes=len(class_names),
        feature_dim=64
    )
    
    try:
        model = load_model(args.checkpoint, model, args.device)
        logger.info(f"Model loaded from {args.checkpoint}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, args.device, class_names)
    
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(results['targets'], results['predictions'], class_names, args.save_path)
    
    # Per-class accuracy
    plot_class_accuracy(results['targets'], results['predictions'], class_names, args.save_path)
    
    # Feature analysis
    feature_analysis = analyze_feature_disentanglement(
        results['expression_features'], 
        results['identity_features'], 
        results['targets'], 
        args.save_path
    )
    results['feature_analysis'] = feature_analysis
    
    # Save results
    save_results(results, args.save_path)
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {args.save_path}")
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Expression LDA Score: {feature_analysis['expression_lda_score']:.4f}")
    
    # Print per-class results
    print("\nPer-class Results:")
    for class_name in class_names:
        if class_name in results['classification_report']:
            precision = results['classification_report'][class_name]['precision']
            recall = results['classification_report'][class_name]['recall']
            f1 = results['classification_report'][class_name]['f1-score']
            print(f"{class_name:15s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    main() 