"""
FastAPI Backend for VendorClose AI
RESTful API for predictions, data upload, and model retraining
"""

import os
import sys
import uuid
import shutil
import json
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

# Create directories (with parents=True to create parent directories)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Initialize predictor with data directory for class detection
predictor = FruitPredictor(model_path=MODEL_PATH, data_dir=str(Path("data")))

# Enhanced model loading with better diagnostics
def load_model_with_diagnostics():
    """Load model with detailed error reporting"""
    global model_loaded

    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir.absolute()}")
        return False

    # Check if model file exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model file not found at: {model_path.absolute()}")
        print(f"   Current working directory: {os.getcwd()}")
        dir_contents = (list(models_dir.iterdir())
                        if models_dir.exists() else 'N/A')
        print(f"   Models directory contents: {dir_contents}")
        return False

    # Try to load model
    try:
        print(f"📦 Loading model from: {model_path.absolute()}")
        file_size = model_path.stat().st_size / 1024 / 1024
        print(f"   File size: {file_size:.2f} MB")
        predictor.load_model(data_dir=str(Path("data")))
        model_loaded = True
        print("✅ Model loaded successfully!")
        
        # Verify class names are loaded correctly
        if predictor.class_names:
            print(f"   Classes: {len(predictor.class_names)}")
            class_preview = predictor.class_names[:5]
            print(f"   Class names: {class_preview}...")
            
            # Check if we're getting generic class names (class_0, class_1, etc.)
            if any(name.startswith('class_') for name in predictor.class_names):
                print("   ⚠️ WARNING: Using generic class names! Check if class_names.json exists.")
                print(f"   Current working directory: {os.getcwd()}")
                json_path = Path("models/class_names.json")
                print(f"   JSON file exists: {json_path.exists()}")
                if json_path.exists():
                    print(f"   JSON file absolute path: {json_path.absolute()}")
        else:
            print("   ⚠️ WARNING: No class names detected!")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False
        return False


# Load model
model_loaded = load_model_with_diagnostics()

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
    """Root endpoint with detailed model status"""
    model_path = Path(MODEL_PATH)
    model_exists = model_path.exists()
    model_size = model_path.stat().st_size / 1024 / 1024 if model_exists else 0
    
    return {
        "message": "VendorClose AI API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "model_path": str(model_path),
        "model_exists": model_exists,
        "model_size_mb": round(model_size, 2) if model_exists else None,
        "status": "operational" if model_loaded else "model_missing",
        "working_directory": os.getcwd(),
        "models_directory_contents": [f.name for f in Path("models").iterdir()] if Path("models").exists() else []
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with model diagnostics"""
    model_path = Path(MODEL_PATH)
    model_exists = model_path.exists()
    
    health_status = "healthy" if model_loaded else "degraded"
    
    return {
        "status": health_status,
        "model_loaded": model_loaded,
        "model_exists": model_exists,
        "model_path": str(model_path),
        "model_absolute_path": str(model_path.absolute()) if model_exists else None,
        "timestamp": datetime.now().isoformat(),
        "diagnostics": {
            "models_dir_exists": Path("models").exists(),
            "model_file_exists": model_exists,
            "working_directory": os.getcwd()
        }
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
    class_label: str = "freshapples"
):
    """
    Upload training images for retraining
    
    Args:
        files: List of image files
        class_label: Class label - can be:
            - Simple: 'fresh', 'medium', 'rotten' (creates generic folders)
            - Specific: 'freshapples', 'rottenbanana', etc. (19 classes supported)
            Supported classes: freshapples, freshbanana, freshbittergroud, freshcapsicum,
            freshcucumber, freshokra, freshoranges, freshpotato, freshtomato,
            rottenapples, rottenbanana, rottenbittergroud, rottencapsicum,
            rottencucumber, rottenokra, rottenoranges, rottenpotato, rottentomato, medium
        
    Returns:
        Upload confirmation
    """
    # Valid class labels (19 specific classes + 3 generic)
    valid_classes = [
        'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum',
        'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
        'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum',
        'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato',
        'medium', 'fresh', 'rotten'  # Generic classes for backward compatibility
    ]
    
    if class_label not in valid_classes:
        raise HTTPException(
            status_code=400, 
            detail=f"class_label must be one of: {', '.join(valid_classes[:5])}... (19 classes supported)"
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
        
        # Create data generators with optimized settings (same as initial training)
        # Using batch_size=128 for faster training, img_size=160x160
        preprocessor = ImagePreprocessor(img_size=(160, 160), batch_size=128)
        train_gen, val_gen, _ = preprocessor.create_data_generators(
            train_dir=str(TRAIN_DIR),
            val_dir=str(TRAIN_DIR)
        )
        
        retraining_status["progress"] = 40
        retraining_status["message"] = "Building model..."
        
        # Get number of classes from data generator
        num_classes = len(train_gen.class_indices)
        print(f"Detected {num_classes} classes for retraining")
        
        # Build and compile model (same settings as initial training)
        classifier = FruitQualityClassifier(
            num_classes=num_classes,  # Use detected number of classes
            img_size=(160, 160),  # Same as initial training
            learning_rate=0.001
        )
        classifier.build_model(use_pretrained=True)
        classifier.compile_model(optimizer_name='adam')
        
        retraining_status["progress"] = 50
        retraining_status["message"] = "Training model (10 epochs, early stopping enabled)..."
        
        # Train model with optimized settings:
        # - 10 epochs (fast retraining)
        # - batch_size=128 (faster training)
        # - Early stopping with patience=5 (stops if no improvement for 5 epochs)
        # - Same techniques as initial training
        history = classifier.train(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=10,  # Fast retraining with 10 epochs
            batch_size=128,  # Larger batch for faster training
            early_stopping_patience=5  # Stop early if no improvement for 5 epochs
        )
        
        retraining_status["progress"] = 80
        retraining_status["message"] = "Saving model..."
        
        # Save model
        model_path = f"models/fruit_classifier_retrained_{session_id}.h5"
        classifier.save_model(model_path)
        
        # Save class names to JSON file (for deployment)
        class_names = sorted(train_gen.class_indices.keys())
        class_names_path = Path("models/class_names.json")
        class_names_path.parent.mkdir(parents=True, exist_ok=True)
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f, indent=2)
        print(f"✅ Saved {len(class_names)} class names to {class_names_path}")
        
        # Update global predictor
        global predictor
        predictor.load_model(model_path, data_dir=str(TRAIN_DIR.parent))
        
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

