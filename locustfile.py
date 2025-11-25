"""
Locust Load Testing Script for VendorClose AI API
Simulates flood of requests to test model performance
"""

from locust import HttpUser, task, between
import random
import io
from PIL import Image
import numpy as np


class FruitPredictorUser(HttpUser):
    """Simulates user making prediction requests"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Health check
        self.client.get("/health")
    
    @task(3)
    def predict_single(self):
        """Single image prediction (most common task)"""
        # Generate a dummy image
        img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), 
                                                   random.randint(0, 255), 
                                                   random.randint(0, 255)))
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Make prediction request
        files = {'file': ('test_image.png', img_bytes, 'image/png')}
        self.client.post("/predict", files=files, name="predict_single")
    
    @task(1)
    def predict_batch(self):
        """Batch prediction (less common)"""
        # Generate 3 dummy images
        files = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), 
                                                       random.randint(0, 255), 
                                                       random.randint(0, 255)))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files.append(('files', (f'test_{i}.png', img_bytes.read(), 'image/png')))
        
        self.client.post("/predict/batch", files=files, name="predict_batch")
    
    @task(1)
    def get_stats(self):
        """Get statistics"""
        self.client.get("/stats", name="get_stats")
    
    @task(1)
    def get_health(self):
        """Health check"""
        self.client.get("/health", name="health_check")


class RetrainingUser(HttpUser):
    """Simulates admin user managing retraining"""
    
    wait_time = between(5, 10)  # Less frequent
    
    @task(1)
    def get_retrain_status(self):
        """Check retraining status"""
        self.client.get("/retrain/status", name="retrain_status")
    
    @task(1)
    def get_sessions(self):
        """Get training sessions"""
        self.client.get("/sessions", name="get_sessions")
    
    @task(1)
    def upload_training_data(self):
        """Upload training data"""
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = [('files', ('train_image.png', img_bytes.read(), 'image/png'))]
        self.client.post("/upload?class_label=fresh", files=files, name="upload_data")

