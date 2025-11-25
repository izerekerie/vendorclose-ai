"""
Data Preprocessing Module for VendorClose AI
Handles image preprocessing, augmentation, and data generation
"""

import os
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2


class ImagePreprocessor:
    """Handles image preprocessing and data augmentation"""
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize preprocessor
        
        Args:
            img_size: Tuple of (height, width) for image resizing
            batch_size: Batch size for data generators
        """
        self.img_size = img_size
        self.batch_size = batch_size
        
    def create_data_generators(self, train_dir, val_dir=None, test_dir=None, 
                              validation_split=0.2):
        """
        Create data generators with augmentation
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory (optional)
            test_dir: Path to test data directory (optional)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        # Data augmentation for training (regularization technique)
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=20,  # Random rotation up to 20 degrees
            width_shift_range=0.2,  # Random horizontal shift
            height_shift_range=0.2,  # Random vertical shift
            shear_range=0.2,  # Random shear transformation
            zoom_range=0.2,  # Random zoom
            horizontal_flip=True,  # Random horizontal flip
            fill_mode='nearest',  # Fill mode for transformations
            validation_split=validation_split  # Split for validation
        )
        
        # No augmentation for validation/test (only rescaling)
        val_test_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        # Training generator
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator
        if val_dir and val_dir != train_dir:
            val_gen = val_test_datagen.flow_from_directory(
                val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
        else:
            val_gen = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
        
        # Test generator (if test directory exists)
        test_gen = None
        if test_dir and os.path.exists(test_dir):
            test_gen = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
        
        return train_gen, val_gen, test_gen
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.img_size
        )
        
        # Convert to array and normalize
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        return img_array
    
    def preprocess_image_from_bytes(self, image_bytes):
        """
        Preprocess image from bytes (for API)
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Preprocessed image array
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalize and expand dimensions
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0
        
        return img_array


def create_medium_class_from_dataset(data_dir):
    """
    Create a 'medium' class by splitting borderline cases from fresh/rotten
    This is a preprocessing step to create the 3-class classification
    
    Args:
        data_dir: Root data directory
    """
    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    
    if not train_dir.exists():
        return
    
    # Check if medium class already exists
    medium_dir = train_dir / 'medium'
    if medium_dir.exists() and len(list(medium_dir.glob('*.jpg'))) + len(list(medium_dir.glob('*.png'))) > 0:
        print("Medium class already exists, skipping creation")
        return
    
    # Create medium directory
    medium_dir.mkdir(exist_ok=True)
    
    # Strategy: Take a subset from fresh and rotten to create medium
    # In a real scenario, this would be done by experts or using confidence scores
    fresh_dir = train_dir / 'fresh'
    rotten_dir = train_dir / 'rotten'
    
    medium_count = 0
    
    # Take 10% of fresh images as medium (borderline fresh)
    if fresh_dir.exists():
        fresh_images = list(fresh_dir.glob('*.jpg')) + list(fresh_dir.glob('*.png'))
        num_medium_from_fresh = max(1, len(fresh_images) // 10)
        
        for img_path in fresh_images[:num_medium_from_fresh]:
            shutil.copy(img_path, medium_dir / f"medium_fresh_{img_path.name}")
            medium_count += 1
    
    # Take 10% of rotten images as medium (borderline rotten)
    if rotten_dir.exists():
        rotten_images = list(rotten_dir.glob('*.jpg')) + list(rotten_dir.glob('*.png'))
        num_medium_from_rotten = max(1, len(rotten_images) // 10)
        
        for img_path in rotten_images[:num_medium_from_rotten]:
            shutil.copy(img_path, medium_dir / f"medium_rotten_{img_path.name}")
            medium_count += 1
    
    if medium_count > 0:
        print(f"Created {medium_count} medium class images")
    else:
        print("No medium class images created - ensure fresh/rotten classes exist")

