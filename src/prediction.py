"""
Prediction Module for VendorClose AI
Handles model loading and predictions
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from src.preprocessing import ImagePreprocessor


class FruitPredictor:
    """Handles fruit quality predictions"""
    
    def __init__(self, model_path=None, img_size=(224, 224)):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file
            img_size: Image size for preprocessing
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor(img_size=img_size)
        self.class_names = ['fresh', 'medium', 'rotten']
        self.action_map = {
            'fresh': '✅ Keep overnight (Fresh - still good tomorrow)',
            'medium': '⚠️ Sell now with discount (Medium - borderline quality)',
            'rotten': '❌ Remove/discard (Rotten - will contaminate others)'
        }
    
    def load_model(self, model_path=None):
        """
        Load model from file
        
        Args:
            model_path: Path to model file (optional if set in __init__)
        """
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("Model path must be provided")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
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
        
        # Get class name
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get action recommendation
        action = self.action_map.get(predicted_class, 'Unknown')
        
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
        
        # Get class name
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get action recommendation
        action = self.action_map.get(predicted_class, 'Unknown')
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'action': action,
            'probabilities': {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
        }

