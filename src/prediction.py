"""
Prediction Module for VendorClose AI
Handles model loading and predictions
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from src.preprocessing import ImagePreprocessor


class FruitPredictor:
    """Handles fruit quality predictions"""
    
    def __init__(self, model_path=None, img_size=(160, 160), data_dir=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file
            img_size: Image size for preprocessing
            data_dir: Path to data directory to infer class names (optional)
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor(img_size=img_size)
        self.data_dir = data_dir
        self.class_names = None  # Will be set dynamically
        self.action_map = {
            'fresh': '✅ Keep overnight (Fresh - still good tomorrow)',
            'medium': '⚠️ Sell now with discount (Medium - borderline quality)',
            'rotten': '❌ Remove/discard (Rotten - will contaminate others)'
        }
        # Extended action map for specific fruit types
        self._extended_action_map = {}
    
    def _detect_class_names(self):
        """Detect class names from JSON file, data directory, or model output"""
        if self.class_names is not None:
            return
        
        # First, try to load from JSON file (for deployment)
        class_names_json = Path("models/class_names.json")
        if class_names_json.exists():
            try:
                with open(class_names_json, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Loaded {len(self.class_names)} classes from class_names.json: {self.class_names[:5]}...")
                return
            except Exception as e:
                print(f"Warning: Could not load class names from JSON: {e}")
        
        # Try to get from data directory (for local development)
        if self.data_dir:
            train_dir = Path(self.data_dir) / 'train'
            if train_dir.exists():
                class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
                if class_dirs:
                    self.class_names = sorted([d.name for d in class_dirs])
                    print(f"Detected {len(self.class_names)} classes from data directory: {self.class_names[:5]}...")
                    return
        
        # If model is loaded, infer from model output shape
        if self.model is not None:
            try:
                output_shape = self.model.output_shape
                if output_shape and len(output_shape) > 1:
                    num_classes = output_shape[-1]
                    # Try to get from model metadata if available
                    if hasattr(self.model, 'class_names'):
                        self.class_names = self.model.class_names
                    else:
                        # Generate generic class names based on count
                        if num_classes == 3:
                            self.class_names = ['fresh', 'medium', 'rotten']
                        else:
                            # For multi-class, we'll need to infer from data or use indices
                            self.class_names = [f'class_{i}' for i in range(num_classes)]
                    print(f"Detected {len(self.class_names)} classes from model output shape")
                    return
            except Exception as e:
                print(f"Warning: Could not detect classes from model: {e}")
        
        # Fallback to default 3 classes
        if self.class_names is None:
            self.class_names = ['fresh', 'medium', 'rotten']
            print("Warning: Using default 3 classes. Provide data_dir or class_names.json for accurate class names.")
    
    def _get_action(self, class_name):
        """Get action recommendation for a class"""
        # Check if it's a simple quality class
        if class_name in self.action_map:
            return self.action_map[class_name]
        
        # For specific fruit types, determine action based on prefix
        if class_name.startswith('fresh'):
            return '✅ Keep overnight (Fresh - still good tomorrow)'
        elif class_name.startswith('rotten'):
            return '❌ Remove/discard (Rotten - will contaminate others)'
        elif class_name == 'medium':
            return '⚠️ Sell now with discount (Medium - borderline quality)'
        else:
            return f'Classified as: {class_name}'
    
    def load_model(self, model_path=None, data_dir=None):
        """
        Load model from file
        
        Args:
            model_path: Path to model file (optional if set in __init__)
            data_dir: Path to data directory to infer class names (optional)
        """
        if model_path:
            self.model_path = model_path
        if data_dir:
            self.data_dir = data_dir
        
        if not self.model_path:
            raise ValueError("Model path must be provided")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
        # Detect class names after loading model
        self._detect_class_names()
    
    def predict_single(self, image_path):
        """
        Predict quality for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")
        
        # Preprocess image
        img_array = self.preprocessor.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Ensure class names are detected
        self._detect_class_names()
        
        # Get class name
        if predicted_class_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = f'class_{predicted_class_idx}'
        
        # Get action recommendation
        action = self._get_action(predicted_class)
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'action': action,
            'probabilities': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, image_paths):
        """
        Predict quality for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict_single(img_path)
                result['image_path'] = str(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_from_bytes(self, image_bytes):
        """
        Predict from image bytes (for API)
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")
        
        # Preprocess image
        img_array = self.preprocessor.preprocess_image_from_bytes(image_bytes)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Ensure class names are detected
        self._detect_class_names()
        
        # Get class name
        if predicted_class_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = f'class_{predicted_class_idx}'
        
        # Get action recommendation
        action = self._get_action(predicted_class)
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'action': action,
            'probabilities': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }

