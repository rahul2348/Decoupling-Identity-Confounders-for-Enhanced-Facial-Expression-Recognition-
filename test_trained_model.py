#!/usr/bin/env python3
"""
Test script for trained DICE-FER model
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import logging
import argparse

# Import DICE-FER components
from dice_fer.dice_fer import DICEFER, load_model
from dice_fer.datasets import create_data_loaders, get_dataset_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DICEFERTester:
    """
    Comprehensive testing for DICE-FER model
    """
    
    def __init__(self, model_path, dataset_name, data_path, device='cpu'):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.device = device
        self.model = None
        self.expressions = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
        
    def load_model(self):
        """Load the trained DICE-FER model"""
        try:
            # Get dataset info
            dataset_info = get_dataset_info(self.dataset_name)
            num_classes = dataset_info.get('num_classes', 7)
            
            # Create model
            self.model = DICEFER(
                num_classes=num_classes,
                feature_dim=64,
                lambda_exp=0.1,
                lambda_id=0.1,
                lambda_adv=0.01
            )
            
            # Load trained weights
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path, self.model, self.device)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model path {self.model_path} not found!")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_test_loader(self, batch_size=32, num_workers=4):
        """Create test data loader"""
        try:
            # Create data loaders (we'll use validation set as test set for now)
            _, self.test_loader = create_data_loaders(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                batch_size=batch_size,
                num_workers=num_workers
            )
            logger.info(f"Test data loader created. Test samples: {len(self.test_loader.dataset)}")  # type: ignore
        except Exception as e:
            logger.error(f"Error creating test loader: {e}")
            raise
    
    def test_model(self):
        """Test the model and return comprehensive results"""
        try:
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return None
            
            self.model.eval()
            all_predictions = []
            all_targets = []
            all_probabilities = []
            all_expression_features = []
            all_identity_features = []
            
            logger.info("Starting model testing...")
            
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc="Testing model"):
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model.forward(data)
                    
                    # Get predictions and probabilities
                    probabilities = F.softmax(outputs['expression_logits'], dim=1)
                    predictions = torch.argmax(outputs['expression_logits'], dim=1)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_expression_features.extend(outputs['expression_features'].cpu().numpy())
                    all_identity_features.extend(outputs['identity_features'].cpu().numpy())
            
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            all_probabilities = np.array(all_probabilities)
            all_expression_features = np.array(all_expression_features)
            all_identity_features = np.array(all_identity_features)
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            cm = confusion_matrix(all_targets, all_predictions)
            report = classification_report(all_targets, all_predictions, 
                                         target_names=self.expressions, output_dict=True)
            
            # Calculate per-class accuracy
            per_class_accuracy = {}
            for i, expression in enumerate(self.expressions):
                mask = (all_targets == i)
                if mask.sum() > 0:
                    class_acc = (all_predictions[mask] == all_targets[mask]).mean()
                    per_class_accuracy[expression] = float(class_acc)
            
            # Calculate MIG score (simplified)
            def compute_mi(features, labels):
                unique_labels = np.unique(labels)
                mi = 0.0
                for label in unique_labels:
                    mask = (labels == label)
                    if mask.sum() > 0:
                        p_label = mask.mean()
                        mi += p_label * np.log(p_label + 1e-8)
                return -mi
            
            mi_exp = compute_mi(all_expression_features, all_targets)
            mi_id = compute_mi(all_identity_features, all_targets)
            mig_score = mi_exp - mi_id
            
            # Compile results
            results = {
                'overall_accuracy': float(accuracy),
                'per_class_accuracy': per_class_accuracy,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'mig_score': float(mig_score),
                'expression_mi': float(mi_exp),
                'identity_mi': float(mi_id),
                'test_samples': len(all_targets),
                'model_path': self.model_path,
                'dataset': self.dataset_name
            }
            
            logger.info(f"Testing completed!")
            logger.info(f"Overall Accuracy: {accuracy:.4f}")
            logger.info(f"MIG Score: {mig_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            raise
    
    def create_test_visualizations(self, results, save_path='test_results'):
        """Create comprehensive test result visualizations"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 1. Confusion Matrix
            cm = np.array(results['confusion_matrix'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.expressions, yticklabels=self.expressions)
            plt.title('Test Confusion Matrix - DICE-FER Model', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Per-class accuracy bar plot
            classes = list(results['per_class_accuracy'].keys())
            accuracies = list(results['per_class_accuracy'].values())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(classes, accuracies, color=colors)
            plt.title('Per-Class Test Accuracy', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy', fontsize=12)
            plt.xlabel('Expression Class', fontsize=12)
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Overall performance summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Overall accuracy
            ax1.bar(['DICE-FER Model'], [results['overall_accuracy']], color='skyblue')
            ax1.set_title('Overall Test Accuracy', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_ylim(0, 1)
            ax1.text(0, results['overall_accuracy'] + 0.01, f'{results["overall_accuracy"]:.3f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Per-class accuracy
            bars = ax2.bar(classes, accuracies, color=colors)
            ax2.set_title('Per-Class Test Accuracy', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_xticklabels(classes, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            # Disentanglement metrics
            metrics = ['Expression MI', 'Identity MI', 'MIG Score']
            values = [results['expression_mi'], results['identity_mi'], results['mig_score']]
            colors_metrics = ['lightcoral', 'lightgreen', 'gold']
            
            bars = ax3.bar(metrics, values, color=colors_metrics)
            ax3.set_title('Disentanglement Metrics', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Mutual Information', fontsize=12)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Test summary
            ax4.text(0.1, 0.8, 'Test Results Summary', fontsize=14, fontweight='bold')
            ax4.text(0.1, 0.7, f'• Overall Accuracy: {results["overall_accuracy"]:.3f}', fontsize=12)
            ax4.text(0.1, 0.6, f'• MIG Score: {results["mig_score"]:.3f}', fontsize=12)
            ax4.text(0.1, 0.5, f'• Test Samples: {results["test_samples"]}', fontsize=12)
            ax4.text(0.1, 0.4, f'• Dataset: {results["dataset"].upper()}', fontsize=12)
            ax4.text(0.1, 0.3, f'• Best Class: {max(results["per_class_accuracy"], key=results["per_class_accuracy"].get)}', fontsize=12)
            ax4.text(0.1, 0.2, f'• Worst Class: {min(results["per_class_accuracy"], key=results["per_class_accuracy"].get)}', fontsize=12)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'test_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Test visualizations saved to {save_path}/")
            
        except Exception as e:
            logger.error(f"Error creating test visualizations: {e}")
            raise
    
    def save_test_results(self, results, save_path='test_results'):
        """Save test results to JSON file"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            with open(os.path.join(save_path, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test results saved to {save_path}/test_results.json")
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            raise
    
    def run_comprehensive_test(self, save_path='test_results'):
        """Run comprehensive testing pipeline"""
        try:
            logger.info("Starting comprehensive testing pipeline...")
            
            # Load model and data
            self.load_model()
            self.create_test_loader()
            
            # Run tests
            results = self.test_model()
            
            if results:
                # Create visualizations
                self.create_test_visualizations(results, save_path)
                
                # Save results
                self.save_test_results(results, save_path)
                
                # Print summary
                print("\n" + "="*50)
                print("DICE-FER TEST RESULTS SUMMARY")
                print("="*50)
                print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
                print(f"MIG Score: {results['mig_score']:.4f}")
                print(f"Test Samples: {results['test_samples']}")
                print(f"Best Class: {max(results['per_class_accuracy'], key=results['per_class_accuracy'].get)}")
                print(f"Worst Class: {min(results['per_class_accuracy'], key=results['per_class_accuracy'].get)}")
                print("="*50)
                
                logger.info("Comprehensive testing completed successfully!")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in testing pipeline: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Test trained DICE-FER model')
    parser.add_argument('--model_path', type=str, default='checkpoints_test/best_model.pth',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rafdb',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='RAF-DB',
                       help='Path to the dataset')
    parser.add_argument('--save_path', type=str, default='test_results',
                       help='Path to save test results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create tester
    tester = DICEFERTester(
        model_path=args.model_path,
        dataset_name=args.dataset,
        data_path=args.data_path,
        device=args.device
    )
    
    # Run comprehensive testing
    tester.run_comprehensive_test(args.save_path)

if __name__ == "__main__":
    main() 