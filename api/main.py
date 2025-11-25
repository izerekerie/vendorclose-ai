"""
FastAPI Backend for VendorClose AI
RESTful API for predictions, data upload, and model retraining
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.prediction import FruitPredictor
from src.preprocessing import ImagePreprocessor, create_medium_class_from_dataset
from src.model import FruitQualityClassifier
from src.database import TrainingDataDB

# Initialize FastAPI app
app = FastAPI(
    title="VendorClose AI API",
    description="Smart End-of-Day Fruit Scanner API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
MODEL_PATH = "models/fruit_classifier.h5"
UPLOAD_DIR = Path("uploads")
TRAIN_DIR = Path("data/train")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
TRAIN_DIR.mkdir(exist_ok=True)

# Initialize predictor
predictor = FruitPredictor(model_path=MODEL_PATH)
try:
    if os.path.exists(MODEL_PATH):
        predictor.load_model()
        model_loaded = True
    else:
        model_loaded = False
        print(f"Warning: Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Initialize database
db = TrainingDataDB()

# Global variables for retraining
retraining_status = {
    "status": "idle",
    "session_id": None,
    "progress": 0,
    "message": "Ready"
}


# Response models
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    action: str
    probabilities: dict


class UploadResponse(BaseModel):
    message: str
    files_uploaded: int
    class_label: str


class RetrainResponse(BaseModel):
    message: str
    session_id: str
    status: str


class StatusResponse(BaseModel):
    status: str
    session_id: Optional[str]
    progress: int
    message: str


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VendorClose AI API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict fruit quality from uploaded image
    
    Args:
        file: Image file to predict
        
    Returns:
        Prediction results
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Predict
        result = predictor.predict_from_bytes(image_bytes)
        
        return PredictionResponse(
            class_name=result['class'],
            confidence=result['confidence'],
            action=result['action'],
            probabilities=result['probabilities']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict fruit quality for multiple images
    
    Args:
        files: List of image files
        
    Returns:
        List of prediction results
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_bytes = await file.read()
            result = predictor.predict_from_bytes(image_bytes)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {"predictions": results, "total": len(results)}


# Upload training data endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_label: str = "fresh"
):
    """
    Upload training images for retraining
    
    Args:
        files: List of image files
        class_label: Class label (fresh/medium/rotten)
        
    Returns:
        Upload confirmation
    """
    if class_label not in ['fresh', 'medium', 'rotten']:
        raise HTTPException(
            status_code=400, 
            detail="class_label must be 'fresh', 'medium', or 'rotten'"
        )
    
    uploaded_count = 0
    
    # Create class directory
    class_dir = TRAIN_DIR / class_label
    class_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            # Save file
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = class_dir / filename
            
            with open(filepath, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Add to database
            db.add_image(
                filename=file.filename,
                filepath=str(filepath),
                class_label=class_label
            )
            
            uploaded_count += 1
        except Exception as e:
            print(f"Error uploading {file.filename}: {e}")
    
    return UploadResponse(
        message=f"Successfully uploaded {uploaded_count} images",
        files_uploaded=uploaded_count,
        class_label=class_label
    )


# Retraining function (runs in background)
def retrain_model(session_id: str):
    """Retrain model with new data"""
    global retraining_status
    
    try:
        retraining_status["status"] = "in_progress"
        retraining_status["session_id"] = session_id
        retraining_status["progress"] = 10
        retraining_status["message"] = "Initializing retraining..."
        
        # Create database session
        db.create_training_session(session_id)
        
        # Get unused images
        unused_images = db.get_unused_images()
        
        if len(unused_images) == 0:
            retraining_status["status"] = "failed"
            retraining_status["message"] = "No new images to train on"
            db.update_training_session(session_id, "failed")
            return
        
        retraining_status["progress"] = 20
        retraining_status["message"] = "Preparing data..."
        
        # Ensure medium class exists
        create_medium_class_from_dataset("data")
        
        retraining_status["progress"] = 30
        retraining_status["message"] = "Creating data generators..."
        
        # Create data generators
        preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)
        train_gen, val_gen, _ = preprocessor.create_data_generators(
            train_dir=str(TRAIN_DIR),
            val_dir=str(TRAIN_DIR)
        )
        
        retraining_status["progress"] = 40
        retraining_status["message"] = "Building model..."
        
        # Build and compile model
        classifier = FruitQualityClassifier(
            num_classes=3,
            img_size=(224, 224),
            learning_rate=0.001
        )
        classifier.build_model(use_pretrained=True)
        classifier.compile_model(optimizer_name='adam')
        
        retraining_status["progress"] = 50
        retraining_status["message"] = "Training model..."
        
        # Train model
        history = classifier.train(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=30,
            batch_size=32
        )
        
        retraining_status["progress"] = 80
        retraining_status["message"] = "Saving model..."
        
        # Save model
        model_path = f"models/fruit_classifier_retrained_{session_id}.h5"
        classifier.save_model(model_path)
        
        # Update global predictor
        global predictor
        predictor.load_model(model_path)
        
        retraining_status["progress"] = 90
        retraining_status["message"] = "Evaluating model..."
        
        # Evaluate model
        eval_results = classifier.model.evaluate(val_gen, verbose=0)
        
        metrics = {
            "loss": float(eval_results[0]),
            "accuracy": float(eval_results[1]),
            "precision": float(eval_results[2]),
            "recall": float(eval_results[3]),
            "auc": float(eval_results[4])
        }
        
        # Mark images as used
        image_ids = [img['id'] for img in unused_images]
        db.mark_images_as_used(image_ids, session_id)
        
        # Update session
        db.update_training_session(
            session_id, 
            "completed", 
            metrics=metrics,
            model_path=model_path
        )
        
        retraining_status["progress"] = 100
        retraining_status["status"] = "completed"
        retraining_status["message"] = f"Retraining completed! Accuracy: {metrics['accuracy']:.2%}"
        
    except Exception as e:
        retraining_status["status"] = "failed"
        retraining_status["message"] = f"Error: {str(e)}"
        db.update_training_session(session_id, "failed")


# Trigger retraining endpoint
@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with uploaded data
    
    Returns:
        Retraining status
    """
    if retraining_status["status"] == "in_progress":
        raise HTTPException(
            status_code=400, 
            detail="Retraining already in progress"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Start retraining in background
    background_tasks.add_task(retrain_model, session_id)
    
    return RetrainResponse(
        message="Retraining started",
        session_id=session_id,
        status="in_progress"
    )


# Retraining status endpoint
@app.get("/retrain/status", response_model=StatusResponse)
async def get_retraining_status():
    """Get current retraining status"""
    return StatusResponse(**retraining_status)


# Database statistics endpoint
@app.get("/stats")
async def get_statistics():
    """Get database and model statistics"""
    stats = db.get_statistics()
    return {
        **stats,
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH if model_loaded else None
    }


# Training sessions endpoint
@app.get("/sessions")
async def get_training_sessions(limit: int = 10):
    """Get recent training sessions"""
    sessions = db.get_training_sessions(limit=limit)
    return {"sessions": sessions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

