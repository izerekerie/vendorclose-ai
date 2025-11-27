"""
Locust Load Testing Script for VendorClose AI API
Simulates flood of requests to test model performance
Records latency and response times for different container configurations
"""

from locust import HttpUser, task, between, events
import random
import io
from PIL import Image
import numpy as np
import time


class FruitPredictorUser(HttpUser):
    """Simulates user making prediction requests"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Health check
        response = self.client.get("/health", name="health_check_startup")
        if response.status_code != 200:
            print(f"⚠️ Health check failed: {response.status_code}")
    
    @task(5)
    def predict_single(self):
        """Single image prediction (most common task - 5x weight)"""
        # Generate a dummy image (160x160 to match model input size)
        img = Image.new('RGB', (160, 160), color=(random.randint(0, 255), 
                                                   random.randint(0, 255), 
                                                   random.randint(0, 255)))
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_data = img_bytes.read()  # Read bytes for Locust
        
        # Make prediction request with timing
        start_time = time.time()
        files = {'file': ('test_image.png', img_data, 'image/png')}
        
        with self.client.post("/predict", files=files, name="predict_single", catch_response=True) as response:
            # Record response time
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Verify we got actual class names, not generic class_0, class_1, etc.
                    class_name = result.get('class_name', result.get('class', ''))
                    if class_name.startswith('class_'):
                        response.failure(f"Got generic class name: {class_name}")
                    else:
                        response.success()
                        # Log successful prediction with class name
                        print(f"✅ Prediction: {class_name} (confidence: {result.get('confidence', 0):.2%}, latency: {response_time:.2f}ms)")
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def predict_batch(self):
        """Batch prediction (less common)"""
        # Generate 3 dummy images
        files = []
        for i in range(3):
            img = Image.new('RGB', (160, 160), color=(random.randint(0, 255), 
                                                       random.randint(0, 255), 
                                                       random.randint(0, 255)))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            img_data = img_bytes.read()  # Read bytes for Locust
            files.append(('files', (f'test_{i}.png', img_data, 'image/png')))
        
        with self.client.post("/predict/batch", files=files, name="predict_batch", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
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
        img = Image.new('RGB', (160, 160), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_data = img_bytes.read()  # Read bytes for Locust
        
        files = [('files', ('train_image.png', img_data, 'image/png'))]
        with self.client.post("/upload?class_label=fresh", files=files, name="upload_data", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

