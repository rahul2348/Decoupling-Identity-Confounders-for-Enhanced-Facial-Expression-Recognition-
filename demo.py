#!/usr/bin/env python3
"""
Demo script for DICE-FER model - Real-time facial expression recognition
"""

import cv2
import torch
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

from dice_fer import DICEFER, load_model
from datasets import get_transforms

class FacialExpressionRecognizer:
    """Real-time facial expression recognition using DICE-FER"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = None
        self.face_cascade = None
        self.transform = None
        self.expression_names = [
            'neutral', 'anger', 'contempt', 'disgust', 
            'fear', 'happy', 'sadness', 'surprise'
        ]
        
        self._load_model(model_path)
        self._load_face_detector()
        self._setup_transforms()
    
    def _load_model(self, model_path):
        """Load the trained DICE-FER model"""
        try:
            self.model = DICEFER(num_classes=8, feature_dim=64)
            self.model = load_model(model_path, self.model, self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_face_detector(self):
        """Load OpenCV face detector"""
        try:
            # Suppress Pyright false positive for cv2.data
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                self.face_cascade = None
            else:
                print("Face detector loaded successfully")
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.face_cascade = None
    
    def _setup_transforms(self):
        """Setup image transformations"""
        self.transform = get_transforms(image_size=224, mode='val')
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        return faces
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        assert self.transform is not None, "Transform is not set"
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=face_rgb)
            face_tensor = transformed['image']
        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)
        return face_tensor
    
    def predict_expression(self, face_tensor):
        """Predict facial expression"""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
            
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            outputs = self.model(face_tensor)
            
            # Get prediction and confidence
            probabilities = torch.softmax(outputs['expression_logits'], dim=1)
            prediction = torch.argmax(outputs['expression_logits'], dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            return {
                'expression': prediction.item(),
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy()[0],
                'expression_features': outputs['expression_features'].cpu().numpy()[0],
                'identity_features': outputs['identity_features'].cpu().numpy()[0]
            }
    
    def draw_results(self, image, faces, predictions):
        """Draw detection results on image"""
        for (x, y, w, h), pred in zip(faces, predictions):
            # Draw face rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # type: ignore
            
            # Get expression name and confidence
            expression_name = self.expression_names[pred['expression']]
            confidence = pred['confidence']
            
            # Draw text
            text = f"{expression_name}: {confidence:.2f}"
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)  # type: ignore
            
            # Draw confidence bar
            bar_width = int(w * confidence)
            cv2.rectangle(image, (x, y+h+5), (x+bar_width, y+h+15), (0, 255, 0), -1)  # type: ignore
            cv2.rectangle(image, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 2)  # type: ignore
        
        return image
    
    def run_realtime_demo(self, camera_id=0):
        """Run real-time facial expression recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Real-time facial expression recognition started!")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            predictions = []
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Resize face to minimum size if needed
                if w < 64 or h < 64:
                    face_img = cv2.resize(face_img, (64, 64))
                
                try:
                    # Preprocess face
                    face_tensor = self.preprocess_face(face_img)
                    
                    # Predict expression
                    pred = self.predict_expression(face_tensor)
                    predictions.append(pred)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Draw results
            frame = self.draw_results(frame, faces, predictions)
            
            # Calculate and display FPS
            frame_count += 1
            fps_counter += 1
            if time.time() - start_time >= 1.0:
                fps = fps_counter / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # type: ignore
                fps_counter = 0
                start_time = time.time()
            
            # Display frame
            cv2.imshow('DICE-FER: Facial Expression Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"dice_fer_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path, output_path=None):
        """Process a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Detect faces
        faces = self.detect_faces(image)
        if len(faces) == 0:
            print("No faces detected in the image")
            return
        
        predictions = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Resize face to minimum size if needed
            if w < 64 or h < 64:
                face_img = cv2.resize(face_img, (64, 64))
            
            try:
                # Preprocess face
                face_tensor = self.preprocess_face(face_img)
                
                # Predict expression
                pred = self.predict_expression(face_tensor)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Draw results
        result_image = self.draw_results(image.copy(), faces, predictions)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to {output_path}")
        else:
            cv2.imshow('DICE-FER: Image Processing', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print predictions
        for i, pred in enumerate(predictions):
            expression_name = self.expression_names[pred['expression']]
            confidence = pred['confidence']
            print(f"Face {i+1}: {expression_name} (confidence: {confidence:.3f})")
    
    def analyze_features(self, image_path):
        """Analyze expression and identity features"""
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        faces = self.detect_faces(image)
        if len(faces) == 0:
            print("No faces detected in the image")
            return
        
        # Process first face
        (x, y, w, h) = faces[0]
        face_img = image[y:y+h, x:x+w]
        
        if w < 64 or h < 64:
            face_img = cv2.resize(face_img, (64, 64))
        
        try:
            face_tensor = self.preprocess_face(face_img)
            pred = self.predict_expression(face_tensor)
            
            # Plot feature analysis
            self._plot_feature_analysis(pred)
            
        except Exception as e:
            print(f"Error analyzing features: {e}")
    
    def _plot_feature_analysis(self, prediction):
        """Plot expression and identity features"""
        exp_features = prediction['expression_features']
        id_features = prediction['identity_features']
        probabilities = prediction['probabilities']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Expression features
        ax1.bar(range(len(exp_features)), exp_features)
        ax1.set_title('Expression Features')
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Feature Value')
        
        # Identity features
        ax2.bar(range(len(id_features)), id_features)
        ax2.set_title('Identity Features')
        ax2.set_xlabel('Feature Dimension')
        ax2.set_ylabel('Feature Value')
        
        # Expression probabilities
        ax3.bar(self.expression_names, probabilities)
        ax3.set_title('Expression Probabilities')
        ax3.set_xlabel('Expression')
        ax3.set_ylabel('Probability')
        ax3.tick_params(axis='x', rotation=45)
        
        # Feature correlation
        correlation = np.corrcoef(exp_features, id_features)[0, 1]
        ax4.text(0.5, 0.5, f'Feature Correlation: {correlation:.3f}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Disentanglement Analysis')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DICE-FER Demo')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'image', 'analyze'],
                       help='Demo mode')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to input image (for image mode)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save output image (for image mode)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID for realtime mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create recognizer
    try:
        recognizer = FacialExpressionRecognizer(args.model_path, args.device)
    except Exception as e:
        print(f"Error creating recognizer: {e}")
        return
    
    # Run demo based on mode
    if args.mode == 'realtime':
        recognizer.run_realtime_demo(args.camera_id)
    
    elif args.mode == 'image':
        if args.image_path is None:
            print("Error: image_path is required for image mode")
            return
        recognizer.process_image(args.image_path, args.output_path)
    
    elif args.mode == 'analyze':
        if args.image_path is None:
            print("Error: image_path is required for analyze mode")
            return
        recognizer.analyze_features(args.image_path)

if __name__ == "__main__":
    main() 