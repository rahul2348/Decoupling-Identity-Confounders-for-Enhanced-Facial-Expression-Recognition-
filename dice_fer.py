import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalNetwork(nn.Module):
    """
    Statistical network for mutual information estimation using Donsker-Varadhan representation
    """
    def __init__(self, input_dim, hidden_dim=512):
        super(StatisticalNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for stability
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, y):
        # Concatenate x and y for mutual information estimation
        combined = torch.cat([x, y], dim=1)
        return self.net(combined)

class ExpressionEncoder(nn.Module):
    """
    Expression encoder based on ResNet-18
    """
    def __init__(self, feature_dim=64):
        super(ExpressionEncoder, self).__init__()
        # Load pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add projection layer to get expression features
        self.projection = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        expression_features = self.projection(features)
        return expression_features

class IdentityEncoder(nn.Module):
    """
    Identity encoder based on ResNet-18
    """
    def __init__(self, feature_dim=64):
        super(IdentityEncoder, self).__init__()
        # Load pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add projection layer to get identity features
        self.projection = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        identity_features = self.projection(features)
        return identity_features

class ExpressionClassifier(nn.Module):
    """
    Expression classifier
    """
    def __init__(self, feature_dim, num_classes):
        super(ExpressionClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, expression_features):
        return self.classifier(expression_features)

class DICEFER(nn.Module):
    """
    DICE-FER: Decoupling Identity Confounders for Enhanced Facial Expression Recognition
    """
    def __init__(self, num_classes=7, feature_dim=64, lambda_exp=0.1, lambda_id=0.1, lambda_adv=0.01):
        super(DICEFER, self).__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_exp = lambda_exp
        self.lambda_id = lambda_id
        self.lambda_adv = lambda_adv
        
        # Encoders
        self.expression_encoder = ExpressionEncoder(feature_dim)
        self.identity_encoder = IdentityEncoder(feature_dim)
        
        # Classifier
        self.expression_classifier = ExpressionClassifier(feature_dim, num_classes)
        
        # Statistical networks for mutual information estimation
        self.mi_net_exp = StatisticalNetwork(feature_dim * 2)  # image + expression features
        self.mi_net_id = StatisticalNetwork(feature_dim * 2)   # image + identity features
        self.mi_net_adv = StatisticalNetwork(feature_dim * 2)  # expression + identity features
        
        # Image feature extractor for mutual information computation
        self.image_encoder = ExpressionEncoder(feature_dim)
        
    def forward(self, x):
        # Extract features
        expression_features = self.expression_encoder(x)
        identity_features = self.identity_encoder(x)
        
        # Classify expressions
        expression_logits = self.expression_classifier(expression_features)
        
        return {
            'expression_features': expression_features,
            'identity_features': identity_features,
            'expression_logits': expression_logits
        }
    
    def compute_mutual_information(self, x, y, mi_net, n_samples=16):
        """
        Compute mutual information using Donsker-Varadhan representation
        """
        batch_size = x.size(0)
        
        # Positive samples (joint distribution)
        pos_samples = mi_net(x, y)
        
        # Negative samples (product of marginals)
        # Shuffle y to create negative samples
        y_shuffled = y[torch.randperm(batch_size)]
        neg_samples = mi_net(x, y_shuffled)
        
        # Compute mutual information estimate
        mi_estimate = torch.mean(pos_samples) - torch.log(torch.mean(torch.exp(neg_samples)))
        
        return mi_estimate
    
    def compute_losses(self, x, y_true, stage='expression'):
        """
        Compute all losses for training
        """
        outputs = self.forward(x)
        expression_features = outputs['expression_features']
        identity_features = outputs['identity_features']
        expression_logits = outputs['expression_logits']
        
        # Classification loss
        cls_loss = F.cross_entropy(expression_logits, y_true)
        
        # Extract image features for mutual information computation
        image_features = self.image_encoder(x)
        
        # Initialize losses with safe defaults
        mi_exp_loss = torch.tensor(0.0, device=x.device)
        mi_id_loss = torch.tensor(0.0, device=x.device)
        mi_adv_loss = torch.tensor(0.0, device=x.device)
        mi_exp = torch.tensor(0.0, device=x.device)
        mi_id = torch.tensor(0.0, device=x.device)
        mi_adv = torch.tensor(0.0, device=x.device)
        
        try:
            if stage == 'expression':
                # Stage 1: Learn expression representations
                # Maximize mutual information between image and expression features
                mi_exp = self.compute_mutual_information(image_features, expression_features, self.mi_net_exp)
                mi_exp_loss = -self.lambda_exp * mi_exp
                
                # Minimize mutual information between expression and identity features
                mi_adv = self.compute_mutual_information(expression_features, identity_features, self.mi_net_adv)
                mi_adv_loss = self.lambda_adv * mi_adv
                
                # Clip losses to prevent NaN
                mi_exp_loss = torch.clamp(mi_exp_loss, -10.0, 10.0)
                mi_adv_loss = torch.clamp(mi_adv_loss, -10.0, 10.0)
                
                total_loss = cls_loss + mi_exp_loss + mi_adv_loss
                
                return {
                    'total_loss': total_loss,
                    'cls_loss': cls_loss,
                    'mi_exp_loss': mi_exp_loss,
                    'mi_adv_loss': mi_adv_loss,
                    'mi_exp': mi_exp,
                    'mi_adv': mi_adv
                }
            
            elif stage == 'identity':
                # Stage 2: Learn identity representations
                # Maximize mutual information between image and identity features
                mi_id = self.compute_mutual_information(image_features, identity_features, self.mi_net_id)
                mi_id_loss = -self.lambda_id * mi_id
                
                # Minimize mutual information between expression and identity features
                mi_adv = self.compute_mutual_information(expression_features, identity_features, self.mi_net_adv)
                mi_adv_loss = self.lambda_adv * mi_adv
                
                # Clip losses to prevent NaN
                mi_id_loss = torch.clamp(mi_id_loss, -10.0, 10.0)
                mi_adv_loss = torch.clamp(mi_adv_loss, -10.0, 10.0)
                
                total_loss = cls_loss + mi_id_loss + mi_adv_loss
                
                return {
                    'total_loss': total_loss,
                    'cls_loss': cls_loss,
                    'mi_id_loss': mi_id_loss,
                    'mi_adv_loss': mi_adv_loss,
                    'mi_id': mi_id,
                    'mi_adv': mi_adv
                }
        except Exception as e:
            # If mutual information computation fails, fall back to classification loss only
            logger.warning(f"Mutual information computation failed: {e}. Using classification loss only.")
            return {
                'total_loss': cls_loss,
                'cls_loss': cls_loss,
                'mi_exp_loss': mi_exp_loss,
                'mi_id_loss': mi_id_loss,
                'mi_adv_loss': mi_adv_loss,
                'mi_exp': mi_exp,
                'mi_id': mi_id,
                'mi_adv': mi_adv
            }
        
        # Invalid stage - return classification loss only
        assert stage in ['expression', 'identity'], f"Invalid stage: {stage}. Must be 'expression' or 'identity'"
        return {
            'total_loss': cls_loss,
            'cls_loss': cls_loss
        }
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=1e-4, 
                   device='cuda', save_path='checkpoints'):
        """
        Train the DICE-FER model
        """
        self.to(device)
        
        # Optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Start with classification-only training for stability
                if epoch < 5:
                    # First 5 epochs: classification only
                    outputs = self.forward(data)
                    cls_loss = F.cross_entropy(outputs['expression_logits'], target)
                    losses = {
                        'total_loss': cls_loss,
                        'cls_loss': cls_loss,
                        'mi_exp_loss': torch.tensor(0.0, device=device),
                        'mi_adv_loss': torch.tensor(0.0, device=device),
                        'mi_exp': torch.tensor(0.0, device=device),
                        'mi_adv': torch.tensor(0.0, device=device)
                    }
                else:
                    # After 5 epochs: full DICE-FER training
                    if epoch < epochs // 2:
                        # Stage 1: Learn expression representations
                        losses = self.compute_losses(data, target, stage='expression')
                    else:
                        # Stage 2: Learn identity representations
                        losses = self.compute_losses(data, target, stage='identity')
                
                # Ensure losses is not None
                assert losses is not None, "compute_losses returned None"
                
                losses['total_loss'].backward()
                
                # Add gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += losses['total_loss'].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'CLS': f"{losses['cls_loss'].item():.4f}"
                })
            
            scheduler.step()
            
            # Validation
            val_acc = self.evaluate_model(val_loader, device)
            val_accuracies.append(val_acc)
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_path, 'best_model.pth'))
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        return train_losses, val_accuracies
    
    def evaluate_model(self, test_loader, device='cuda'):
        """
        Evaluate the model on test data
        """
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.forward(data)
                predictions = torch.argmax(outputs['expression_logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        return accuracy
    
    def compute_mig_score(self, test_loader, device='cuda'):
        """
        Compute Mutual Information Gap (MIG) score for disentanglement evaluation
        """
        self.eval()
        expression_features = []
        identity_features = []
        labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = self.forward(data)
                
                expression_features.append(outputs['expression_features'].cpu())
                identity_features.append(outputs['identity_features'].cpu())
                labels.append(target.cpu())
        
        expression_features = torch.cat(expression_features, dim=0)
        identity_features = torch.cat(identity_features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Compute mutual information between features and labels
        mi_exp = self._compute_mi_with_labels(expression_features, labels)
        mi_id = self._compute_mi_with_labels(identity_features, labels)
        
        # MIG score: difference between top-1 and top-2 mutual information
        mig_score = mi_exp - mi_id  # Simplified MIG computation
        
        return mig_score
    
    def _compute_mi_with_labels(self, features, labels):
        """
        Compute mutual information between features and labels
        """
        # Simplified mutual information computation
        # In practice, you would use a more sophisticated method
        unique_labels = torch.unique(labels)
        mi = 0.0
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                p_label = mask.float().mean()
                mi += p_label * torch.log(p_label + 1e-8)
        
        return -mi
    
    def visualize_features(self, test_loader, device='cuda', save_path='visualizations'):
        """
        Visualize the learned features using t-SNE
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            self.eval()
            expression_features = []
            identity_features = []
            labels = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = self.forward(data)
                    
                    expression_features.append(outputs['expression_features'].cpu())
                    identity_features.append(outputs['identity_features'].cpu())
                    labels.append(target.cpu())
            
            expression_features = torch.cat(expression_features, dim=0).numpy()
            identity_features = torch.cat(identity_features, dim=0).numpy()
            labels = torch.cat(labels, dim=0).numpy()
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            exp_tsne = tsne.fit_transform(expression_features)
            id_tsne = tsne.fit_transform(identity_features)
            
            # Create visualizations
            os.makedirs(save_path, exist_ok=True)
            
            # Expression features visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(exp_tsne[:, 0], exp_tsne[:, 1], c=labels, cmap='tab10')
            plt.title('Expression Features (t-SNE)')
            plt.colorbar(scatter)
            
            # Identity features visualization
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(id_tsne[:, 0], id_tsne[:, 1], c=labels, cmap='tab10')
            plt.title('Identity Features (t-SNE)')
            plt.colorbar(scatter)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'feature_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature visualizations saved to {save_path}")
            
        except ImportError:
            logger.warning("sklearn not available. Skipping feature visualization.")

def load_model(checkpoint_path, model, device='cuda'):
    """
    Load a trained model from checkpoint
    """
    # Handle CPU-only machines
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model 