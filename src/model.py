"""
Model Creation Module for VendorClose AI
Builds CNN model with transfer learning, regularization, and optimization
"""

import os
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC
import numpy as np


class FruitQualityClassifier:
    """CNN Classifier for fruit quality (Fresh/Medium/Rotten)"""
    
    def __init__(self, num_classes=3, img_size=(224, 224), learning_rate=0.001):
        """
        Initialize classifier
        
        Args:
            num_classes: Number of output classes (3: fresh, medium, rotten)
            img_size: Input image size (height, width)
            learning_rate: Initial learning rate
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, use_pretrained=True):
        """
        Build CNN model with transfer learning
        
        Args:
            use_pretrained: Whether to use pretrained MobileNetV2 weights
            
        Returns:
            Compiled Keras model
        """
        # Input shape
        input_shape = (*self.img_size, 3)
        
        # Base model: MobileNetV2 (pretrained on ImageNet)
        if use_pretrained:
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'  # Transfer learning
            )
        else:
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=None
            )
        
        # Freeze base model layers initially (fine-tuning strategy)
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=input_shape)
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        
        # Regularization: Dropout layers
        x = Dropout(0.3)(x)  # Dropout for regularization
        
        # Batch Normalization (regularization technique)
        x = BatchNormalization()(x)
        
        # Dense layer with L2 regularization
        x = Dense(
            128, 
            activation='relu',
            kernel_regularizer=l2(0.01)  # L2 regularization
        )(x)
        
        # Another dropout layer
        x = Dropout(0.4)(x)
        
        # Batch Normalization
        x = BatchNormalization()(x)
        
        # Output layer
        outputs = Dense(
            self.num_classes, 
            activation='softmax',
            kernel_regularizer=l2(0.01)
        )(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs)
        
        return self.model
    
    def compile_model(self, optimizer_name='adam'):
        """
        Compile model with optimizer and metrics
        
        Args:
            optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Select optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=0.9
            )
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile with multiple metrics
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )
    
    def train(self, train_generator, val_generator, epochs=50, batch_size=32):
        """
        Train the model with callbacks for optimization
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Maximum number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Callbacks for optimization
        callbacks = [
            # Early stopping (prevents overfitting)
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Stop if no improvement for 10 epochs
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau (optimization technique)
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reduce LR by half
                patience=5,  # Wait 5 epochs
                min_lr=1e-7,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath='../models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Unfreeze some layers for fine-tuning (after initial training)
        # This is done after a few epochs in practice, but for simplicity
        # we'll keep base frozen for now
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def fine_tune(self, train_generator, val_generator, epochs=10):
        """
        Fine-tune the model by unfreezing base layers
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Unfreeze base model layers
        base_model = self.model.layers[1]  # MobileNetV2 is second layer
        base_model.trainable = True
        
        # Fine-tune only top layers (optional: can fine-tune all)
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(name='precision'), 
                    Recall(name='recall'), AUC(name='auc')]
        )
        
        # Train with fine-tuning
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1
        )
        
        return self.history

