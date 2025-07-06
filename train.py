#!/usr/bin/env python3
"""
Training script for DICE-FER model
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch.nn.functional as F

from dice_fer import DICEFER, load_model
from datasets import create_data_loaders, get_dataset_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DICE-FER model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ckplus',
                       choices=['ckplus', 'rafdb', 'affectnet', 'oulucasia'],
                       help='Dataset to use for training')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset directory')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of expression classes (auto-detected if not specified)')
    parser.add_argument('--feature_dim', type=int, default=64,
                       help='Dimension of feature representations')
    parser.add_argument('--lambda_exp', type=float, default=1.0,
                       help='Weight for expression mutual information loss')
    parser.add_argument('--lambda_id', type=float, default=1.0,
                       help='Weight for identity mutual information loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                       help='Weight for adversarial mutual information loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_path', type=str, default='checkpoints',
                       help='Path to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # CPU optimization settings
    torch.set_num_threads(8)  # Use multiple CPU threads
    torch.set_num_interop_threads(4)  # Inter-op parallelism

def plot_training_curves(train_losses, val_accuracies, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get dataset information
    dataset_info = get_dataset_info(args.dataset)
    if args.num_classes is None:
        args.num_classes = dataset_info.get('num_classes', 7)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Expressions: {dataset_info.get('expressions', [])}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_name=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers
        )
        
        # Ensure datasets have __len__ method before calling len()
        if hasattr(train_loader.dataset, '__len__'):
            logger.info(f"Train samples: {len(train_loader.dataset)}")  # type: ignore
        else:
            logger.info("Train samples: Unknown (dataset does not support len())")
            
        if hasattr(val_loader.dataset, '__len__'):
            logger.info(f"Validation samples: {len(val_loader.dataset)}")  # type: ignore
        else:
            logger.info("Validation samples: Unknown (dataset does not support len())")
            
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return
    
    # Create model
    logger.info("Creating DICE-FER model...")
    model = DICEFER(
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        lambda_exp=args.lambda_exp,
        lambda_id=args.lambda_id,
        lambda_adv=args.lambda_adv
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        model = load_model(args.resume, model, args.device)
        checkpoint = torch.load(args.resume, map_location=args.device)
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_path, 'tensorboard'))
    
    # Train the model
    logger.info("Starting training...")
    try:
        train_losses, val_accuracies = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            save_path=args.save_path
        )
        
        # Plot training curves
        plot_training_curves(train_losses, val_accuracies, args.save_path)
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        final_val_acc = model.evaluate_model(val_loader, args.device)
        mig_score = model.compute_mig_score(val_loader, args.device)
        
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        logger.info(f"Final MIG Score: {mig_score:.4f}")
        
        # Save final results
        results = {
            'final_val_accuracy': final_val_acc,
            'final_mig_score': mig_score,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
        
        import json
        with open(os.path.join(args.save_path, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create feature visualizations
        logger.info("Creating feature visualizations...")
        model.visualize_features(val_loader, args.device, args.save_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    finally:
        writer.close()

if __name__ == "__main__":
    main() 