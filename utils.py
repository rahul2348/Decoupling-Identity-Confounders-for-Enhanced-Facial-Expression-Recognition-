"""
Utility functions for DICE-FER implementation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
import os
import json
import logging
import mediapipe as mp
from typing import Optional

logger = logging.getLogger(__name__)

# Initialize MediaPipe face detection with error handling
try:
    mp_face_detection = mp.solutions.face_detection  # type: ignore
    mp_drawing = mp.solutions.drawing_utils  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except AttributeError as e:
    logger.warning(f"MediaPipe face detection not available: {e}")
    mp_face_detection = None
    mp_drawing = None
    MEDIAPIPE_AVAILABLE = False

def compute_mutual_information_estimate(x, y, estimator_network, n_samples=16):
    """
    Compute mutual information estimate using Donsker-Varadhan representation
    
    Args:
        x: First random variable
        y: Second random variable  
        estimator_network: Neural network for MI estimation
        n_samples: Number of samples for estimation
    
    Returns:
        MI estimate
    """
    batch_size = x.size(0)
    
    # Positive samples (joint distribution)
    pos_samples = estimator_network(x, y)
    
    # Negative samples (product of marginals)
    # Shuffle y to create negative samples
    y_shuffled = y[torch.randperm(batch_size)]
    neg_samples = estimator_network(x, y_shuffled)
    
    # Compute mutual information estimate
    mi_estimate = torch.mean(pos_samples) - torch.log(torch.mean(torch.exp(neg_samples)))
    
    return mi_estimate

def compute_mig_score(expression_features, identity_features, labels):
    """
    Compute Mutual Information Gap (MIG) score for disentanglement evaluation
    
    Args:
        expression_features: Expression feature representations
        identity_features: Identity feature representations
        labels: Ground truth labels
    
    Returns:
        MIG score
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # Compute mutual information between features and labels
    mi_exp = mutual_info_regression(expression_features, labels)
    mi_id = mutual_info_regression(identity_features, labels)
    
    # Sort mutual information values
    mi_exp_sorted = np.sort(mi_exp)[::-1]
    mi_id_sorted = np.sort(mi_id)[::-1]
    
    # Compute MIG score (difference between top-1 and top-2 MI)
    if len(mi_exp_sorted) >= 2:
        mig_exp = (mi_exp_sorted[0] - mi_exp_sorted[1]) / mi_exp_sorted[0]
    else:
        mig_exp = 0.0
    
    if len(mi_id_sorted) >= 2:
        mig_id = (mi_id_sorted[0] - mi_id_sorted[1]) / mi_id_sorted[0]
    else:
        mig_id = 0.0
    
    # Overall MIG score
    mig_score = (mig_exp + mig_id) / 2.0
    
    return mig_score

def visualize_feature_space(expression_features, identity_features, labels, 
                           class_names, save_path='visualizations'):
    """
    Visualize the learned feature space using t-SNE
    
    Args:
        expression_features: Expression feature representations
        identity_features: Identity feature representations
        labels: Ground truth labels
        class_names: Names of expression classes
        save_path: Path to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Expression features visualization
    exp_tsne = tsne.fit_transform(expression_features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(exp_tsne[:, 0], exp_tsne[:, 1], c=labels, cmap='tab10')
    plt.title('Expression Features (t-SNE)')
    plt.colorbar(scatter)
    
    # Identity features visualization
    id_tsne = tsne.fit_transform(identity_features)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(id_tsne[:, 0], id_tsne[:, 1], c=labels, cmap='tab10')
    plt.title('Identity Features (t-SNE)')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature visualizations saved to {save_path}")

def analyze_feature_correlations(expression_features, identity_features, save_path='analysis'):
    """
    Analyze correlations between expression and identity features
    
    Args:
        expression_features: Expression feature representations
        identity_features: Identity feature representations
        save_path: Path to save analysis results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(expression_features.T, identity_features.T)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Matrix (Expression vs Identity)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_correlation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute average correlation
    exp_dim = expression_features.shape[1]
    id_dim = identity_features.shape[1]
    
    # Correlation between expression and identity features
    cross_correlation = correlation_matrix[:exp_dim, exp_dim:]
    avg_cross_correlation = np.mean(np.abs(cross_correlation))
    
    logger.info(f"Average cross-correlation: {avg_cross_correlation:.4f}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'avg_cross_correlation': avg_cross_correlation
    }

def compute_feature_statistics(expression_features, identity_features, labels):
    """
    Compute statistical measures for feature analysis
    
    Args:
        expression_features: Expression feature representations
        identity_features: Identity feature representations
        labels: Ground truth labels
    
    Returns:
        Dictionary containing various statistics
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    
    # PCA analysis
    pca_exp = PCA()
    pca_exp.fit(expression_features)
    exp_variance_ratio = pca_exp.explained_variance_ratio_
    
    pca_id = PCA()
    pca_id.fit(identity_features)
    id_variance_ratio = pca_id.explained_variance_ratio_
    
    # LDA analysis
    lda_exp = LinearDiscriminantAnalysis()
    lda_exp.fit(expression_features, labels)
    exp_lda_score = lda_exp.score(expression_features, labels)
    
    lda_id = LinearDiscriminantAnalysis()
    lda_id.fit(identity_features, labels)
    id_lda_score = lda_id.score(identity_features, labels)
    
    # Feature variance
    exp_variance = np.var(expression_features, axis=0)
    id_variance = np.var(identity_features, axis=0)
    
    # Feature sparsity
    exp_sparsity = np.mean(expression_features == 0)
    id_sparsity = np.mean(identity_features == 0)
    
    return {
        'expression_variance_ratio': exp_variance_ratio,
        'identity_variance_ratio': id_variance_ratio,
        'expression_lda_score': exp_lda_score,
        'identity_lda_score': id_lda_score,
        'expression_variance': exp_variance,
        'identity_variance': id_variance,
        'expression_sparsity': exp_sparsity,
        'identity_sparsity': id_sparsity
    }

def save_model_config(model, save_path):
    """
    Save model configuration to JSON file
    
    Args:
        model: DICE-FER model
        save_path: Path to save configuration
    """
    config = {
        'num_classes': model.num_classes,
        'feature_dim': model.feature_dim,
        'lambda_exp': model.lambda_exp,
        'lambda_id': model.lambda_id,
        'lambda_adv': model.lambda_adv
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model configuration saved to {save_path}")

def load_model_config(config_path):
    """
    Load model configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def create_model_summary(model, input_size=(3, 224, 224)):
    """
    Create a summary of model architecture and parameters
    
    Args:
        model: DICE-FER model
        input_size: Input tensor size
    
    Returns:
        Model summary string
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    trainable_params = count_parameters(model)
    
    summary = f"""
DICE-FER Model Summary
======================
Input Size: {input_size}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}

Architecture:
- Expression Encoder: ResNet-18 + projection layer
- Identity Encoder: ResNet-18 + projection layer
- Expression Classifier: Linear layer
- Statistical Networks: 3 MLPs for MI estimation

Loss Weights:
- λ_exp (Expression MI): {model.lambda_exp}
- λ_id (Identity MI): {model.lambda_id}
- λ_adv (Adversarial MI): {model.lambda_adv}

Feature Dimensions:
- Expression Features: {model.feature_dim}
- Identity Features: {model.feature_dim}
- Number of Classes: {model.num_classes}
"""
    
    return summary

def preprocess_face_image(image, target_size=224):
    """
    Preprocess face image for model input
    
    Args:
        image: Input face image (numpy array)
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image tensor
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def detect_face_mediapipe(image_path: str, confidence: float = 0.5) -> Optional[np.ndarray]:
    """
    Detect and extract face using MediaPipe.
    
    Args:
        image_path: Path to the image file
        confidence: Detection confidence threshold
        
    Returns:
        Cropped face image or None if no face detected
    """
    if not MEDIAPIPE_AVAILABLE or mp_face_detection is None:
        logger.warning("MediaPipe face detection not available, falling back to OpenCV")
        return detect_face_opencv(image_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=confidence) as face_detection:
        
        # Detect faces
        results = face_detection.process(image_rgb)
        
        if results.detections:
            # Get the first detected face
            detection = results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Crop face
            face_crop = image[y:y+height, x:x+width]
            
            if face_crop.size > 0:
                return face_crop
        
        print(f"No face detected in {image_path}")
        return None

def detect_face_opencv(image_path: str) -> Optional[np.ndarray]:
    """
    Detect and extract face using OpenCV's Haar cascade.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Cropped face image or None if no face detected
    """
    try:
        # Load the pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Crop face
            face_crop = image[y:y+h, x:x+w]
            return face_crop
        
        print(f"No face detected in {image_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error in OpenCV face detection: {e}")
        return None

def detect_face(image_path: str, method: str = 'mediapipe') -> Optional[np.ndarray]:
    """
    Detect and extract face using specified method.
    
    Args:
        image_path: Path to the image file
        method: 'mediapipe' or 'opencv'
        
    Returns:
        Cropped face image or None if no face detected
    """
    if method == 'mediapipe':
        if MEDIAPIPE_AVAILABLE:
            return detect_face_mediapipe(image_path)
        else:
            logger.warning("MediaPipe not available, falling back to OpenCV")
            return detect_face_opencv(image_path)
    elif method == 'opencv':
        return detect_face_opencv(image_path)
    else:
        raise ValueError(f"Unknown face detection method: {method}")

def compute_expression_confidence(predictions, temperature=1.0):
    """
    Compute confidence scores for expression predictions
    
    Args:
        predictions: Model predictions (logits)
        temperature: Temperature for softmax scaling
    
    Returns:
        Confidence scores and predicted classes
    """
    # Apply temperature scaling
    scaled_logits = predictions / temperature
    
    # Compute softmax probabilities
    probabilities = torch.softmax(scaled_logits, dim=1)
    
    # Get predicted classes and confidence
    confidence, predicted = torch.max(probabilities, dim=1)
    
    return confidence, predicted, probabilities

def create_attention_visualization(image, expression_features, identity_features, save_path):
    """
    Create attention visualization for feature analysis
    
    Args:
        image: Input image
        expression_features: Expression feature representations
        identity_features: Identity feature representations
        save_path: Path to save visualization
    """
    # This is a simplified attention visualization
    # In practice, you would implement more sophisticated attention mechanisms
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Expression features heatmap
    exp_heatmap = np.mean(np.abs(expression_features), axis=0)
    exp_heatmap = exp_heatmap.reshape(8, 8)  # Reshape to 8x8 for visualization
    axes[0, 1].imshow(exp_heatmap, cmap='hot')
    axes[0, 1].set_title('Expression Features Heatmap')
    axes[0, 1].axis('off')
    
    # Identity features heatmap
    id_heatmap = np.mean(np.abs(identity_features), axis=0)
    id_heatmap = id_heatmap.reshape(8, 8)  # Reshape to 8x8 for visualization
    axes[1, 0].imshow(id_heatmap, cmap='cool')
    axes[1, 0].set_title('Identity Features Heatmap')
    axes[1, 0].axis('off')
    
    # Feature correlation
    correlation = np.corrcoef(expression_features.flatten(), identity_features.flatten())[0, 1]
    axes[1, 1].text(0.5, 0.5, f'Feature Correlation: {correlation:.3f}', 
                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Feature Disentanglement')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_model_outputs(model, test_loader, device):
    """
    Validate model outputs and check for potential issues
    
    Args:
        model: DICE-FER model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Validation results dictionary
    """
    model.eval()
    issues = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 5:  # Check first 5 batches
                break
                
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # Check for NaN values
            if torch.isnan(outputs['expression_logits']).any():
                issues.append(f"NaN values in expression logits at batch {batch_idx}")
            
            if torch.isnan(outputs['expression_features']).any():
                issues.append(f"NaN values in expression features at batch {batch_idx}")
            
            if torch.isnan(outputs['identity_features']).any():
                issues.append(f"NaN values in identity features at batch {batch_idx}")
            
            # Check for infinite values
            if torch.isinf(outputs['expression_logits']).any():
                issues.append(f"Infinite values in expression logits at batch {batch_idx}")
            
            # Check feature ranges
            exp_features = outputs['expression_features']
            id_features = outputs['identity_features']
            
            if torch.abs(exp_features).max() > 100:
                issues.append(f"Large expression feature values at batch {batch_idx}")
            
            if torch.abs(id_features).max() > 100:
                issues.append(f"Large identity feature values at batch {batch_idx}")
    
    return {
        'issues_found': len(issues),
        'issues': issues,
        'status': 'OK' if len(issues) == 0 else 'WARNING'
    } 