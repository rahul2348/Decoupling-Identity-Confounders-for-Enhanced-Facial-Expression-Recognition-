import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List, Any, Union

class FERDataset(Dataset):
    """
    Base class for Facial Expression Recognition datasets
    """
    def __init__(self, data_path: str, transform: Optional[Any] = None, mode: str = 'train') -> None:
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.samples: List[str] = []
        self.labels: List[int] = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, label

class CKPlusDataset(FERDataset):
    """
    CK+ (Extended Cohn-Kanade) Dataset
    """
    def __init__(self, data_path, transform=None, mode='train'):
        super().__init__(data_path, transform, mode)
        
        # CK+ has 7 expressions: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
        self.expression_map = {
            'neutral': 0, 'anger': 1, 'contempt': 2, 'disgust': 3,
            'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load CK+ dataset"""
        # This is a simplified loading mechanism
        # In practice, you would parse the CK+ directory structure
        subjects = os.listdir(self.data_path)
        
        for subject in subjects:
            subject_path = os.path.join(self.data_path, subject)
            if os.path.isdir(subject_path):
                sequences = os.listdir(subject_path)
                for sequence in sequences:
                    sequence_path = os.path.join(subject_path, sequence)
                    if os.path.isdir(sequence_path):
                        # Look for emotion label file
                        emotion_file = os.path.join(sequence_path, 'emotion.txt')
                        if os.path.exists(emotion_file):
                            with open(emotion_file, 'r') as f:
                                emotion_label = int(f.read().strip())
                            
                            # Get the last frame (peak expression)
                            images = [f for f in os.listdir(sequence_path) if f.endswith('.png')]
                            if images:
                                last_image = sorted(images)[-1]
                                image_path = os.path.join(sequence_path, last_image)
                                self.samples.append(image_path)
                                self.labels.append(emotion_label)

class RAFDBDataset(FERDataset):
    """
    RAF-DB (Real-world Affective Faces Database) Dataset
    """
    def __init__(self, data_path, transform=None, mode='train'):
        super().__init__(data_path, transform, mode)
        
        # RAF-DB has 7 expressions: 1=surprise, 2=fear, 3=disgust, 4=happiness, 5=sadness, 6=anger, 7=neutral
        self.expression_map = {
            '1': 0,  # surprise
            '2': 1,  # fear  
            '3': 2,  # disgust
            '4': 3,  # happiness
            '5': 4,  # sadness
            '6': 5,  # anger
            '7': 6   # neutral
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load RAF-DB dataset"""
        # Check if we have train/test split directories
        train_dir = os.path.join(self.data_path, 'train')
        test_dir = os.path.join(self.data_path, 'test')
        
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            # Use the provided train/test split
            if self.mode == 'train':
                base_dir = train_dir
            else:
                base_dir = test_dir
        else:
            # Fallback to the original annotation file approach
            annotation_file = os.path.join(self.data_path, 'list_patition_label.txt')
            
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label = int(parts[1]) - 1  # Convert to 0-based indexing
                        
                        image_path = os.path.join(self.data_path, 'aligned', image_name)
                        if os.path.exists(image_path):
                            self.samples.append(image_path)
                            self.labels.append(label)
                return
        
        # Load from directory structure
        for expression_dir in os.listdir(base_dir):
            if expression_dir in self.expression_map:
                expression_path = os.path.join(base_dir, expression_dir)
                if os.path.isdir(expression_path):
                    label = self.expression_map[expression_dir]
                    
                    # Get all images in this expression directory
                    for image_file in os.listdir(expression_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(expression_path, image_file)
                            self.samples.append(image_path)
                            self.labels.append(label)

class AffectNetDataset(FERDataset):
    """
    AffectNet Dataset
    """
    def __init__(self, data_path, transform=None, mode='train'):
        super().__init__(data_path, transform, mode)
        
        # AffectNet has 8 expressions: 0=neutral, 1=happiness, 2=sadness, 3=surprise, 4=fear, 5=disgust, 6=anger, 7=contempt
        self.expression_map = {
            'neutral': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,
            'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load AffectNet dataset"""
        # Load annotation file
        annotation_file = os.path.join(self.data_path, 'annotations.csv')
        
        if os.path.exists(annotation_file):
            df = pd.read_csv(annotation_file)
            
            for _, row in df.iterrows():
                image_name = row['image_name']  # type: ignore
                label = row['expression']  # type: ignore
                
                image_path = os.path.join(self.data_path, 'images', image_name)  # type: ignore
                if os.path.exists(image_path):
                    self.samples.append(image_path)
                    self.labels.append(label)  # type: ignore

class OuluCASIADataset(FERDataset):
    """
    Oulu-CASIA Dataset
    """
    def __init__(self, data_path, transform=None, mode='train'):
        super().__init__(data_path, transform, mode)
        
        # Oulu-CASIA has 6 expressions: 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise
        self.expression_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load Oulu-CASIA dataset"""
        # This is a simplified loading mechanism
        # In practice, you would parse the Oulu-CASIA directory structure
        subjects = os.listdir(self.data_path)
        
        for subject in subjects:
            subject_path = os.path.join(self.data_path, subject)
            if os.path.isdir(subject_path):
                expressions = os.listdir(subject_path)
                for expression in expressions:
                    expression_path = os.path.join(subject_path, expression)
                    if os.path.isdir(expression_path):
                        # Get expression label
                        if expression in self.expression_map:
                            label = self.expression_map[expression]
                            
                            # Get images
                            images = [f for f in os.listdir(expression_path) if f.endswith('.jpg')]
                            for image in images:
                                image_path = os.path.join(expression_path, image)
                                self.samples.append(image_path)
                                self.labels.append(label)

def get_transforms(image_size=224, mode='train'):
    """
    Get data transformations for training and validation
    """
    if mode == 'train':
        transform = A.Compose([  # type: ignore
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([  # type: ignore
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    return transform

def create_data_loaders(dataset_name, data_path, batch_size=32, image_size=224, 
                       train_split=0.8, num_workers=4) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for the specified dataset
    """
    # Get dataset class
    dataset_classes = {
        'ckplus': CKPlusDataset,
        'rafdb': RAFDBDataset,
        'affectnet': AffectNetDataset,
        'oulucasia': OuluCASIADataset
    }
    
    if dataset_name.lower() not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    DatasetClass = dataset_classes[dataset_name.lower()]
    
    # Check if dataset already has train/test splits
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Use provided train/test splits
        train_dataset = DatasetClass(
            data_path, 
            transform=get_transforms(image_size, 'train'),
            mode='train'
        )
        
        val_dataset = DatasetClass(
            data_path, 
            transform=get_transforms(image_size, 'val'),
            mode='val'
        )
    else:
        # Create full dataset and split manually
        full_dataset = DatasetClass(data_path, transform=None)
        
        # Split into train and validation
        train_indices, val_indices = train_test_split(
            range(len(full_dataset)), 
            train_size=train_split, 
            random_state=42,
            stratify=full_dataset.labels
        )
        
        # Create train dataset
        train_dataset = DatasetClass(
            data_path, 
            transform=get_transforms(image_size, 'train'),
            mode='train'
        )
        train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        train_dataset.labels = [full_dataset.labels[i] for i in train_indices]
        
        # Create validation dataset
        val_dataset = DatasetClass(
            data_path, 
            transform=get_transforms(image_size, 'val'),
            mode='val'
        )
        val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
        val_dataset.labels = [full_dataset.labels[i] for i in val_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_dataset_info(dataset_name):
    """
    Get dataset information
    """
    dataset_info = {
        'ckplus': {
            'num_classes': 8,
            'expressions': ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
            'expected_samples': 1000  # Approximate
        },
        'rafdb': {
            'num_classes': 7,
            'expressions': ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral'],
            'expected_samples': 15000  # Approximate
        },
        'affectnet': {
            'num_classes': 8,
            'expressions': ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt'],
            'expected_samples': 1000000  # Approximate
        },
        'oulucasia': {
            'num_classes': 6,
            'expressions': ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'],
            'expected_samples': 5000  # Approximate
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {})

# Example usage
if __name__ == "__main__":
    # Example of how to use the datasets
    data_path = "path/to/your/dataset"
    dataset_name = "ckplus"
    
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=32,
            image_size=224
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")  # type: ignore
        print(f"Validation samples: {len(val_loader.dataset)}")  # type: ignore
        
        # Test a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
            break
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the dataset path is correct and the dataset is properly formatted.") 