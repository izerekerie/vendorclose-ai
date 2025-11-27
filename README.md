# VendorClose AI - Smart End-of-Day Fruit Scanner

A comprehensive machine learning pipeline for classifying fruit quality (Fresh/Medium/Rotten) to help vendors make smart decisions at closing time.

## Project Overview

**VendorClose AI** is an end-to-end ML system that:

- Classifies fruit quality into 3 categories: Fresh, Medium (discount), Rotten
- Provides instant single and batch predictions
- Offers business insights through visualizations
- Supports model retraining with new data
- Scales with Docker containerization
- Load-tested with Locust

## Features

### 1. Quick Scan

- Upload single fruit image
- Get instant quality assessment
- Receive vendor-specific action recommendations

### 2. Batch Processing

- Analyze up to 50 fruits at once
- Get prioritized action list
- View aggregate statistics

### 3. Business Dashboard

- Track daily quality metrics
- Visualize freshness trends
- Monitor model performance

### 4. Model Retraining

- Upload new training images
- Trigger retraining with one click
- Improve accuracy with local fruit data

### 5. Production-Ready API

- RESTful FastAPI backend
- Docker containerization
- Load-tested and scalable

## Project Structure

```
VendorClose_AI/
│
├── README.md                 # Project documentation
│
├── notebook/
│   └── vendorclose_ai.ipynb # Training and evaluation notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data preprocessing and augmentation
│   ├── model.py             # CNN model with transfer learning
│   ├── prediction.py        # Prediction functions
│   └── database.py          # Database for training data
│
├── api/
│   └── main.py              # FastAPI backend
│
├── app.py                   # Streamlit UI
│
├── data/
│   ├── train/              # Training data (fresh/medium/rotten)
│   └── test/               # Test data
│
├── models/                  # Saved model files
│
├── uploads/                 # Uploaded training images
│
├── logs/                    # Training logs and visualizations
│
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
├── locustfile.py          # Load testing script
├── render.yaml            # Render deployment configuration
└── requirements.txt        # Python dependencies
```

## Hosted Application Links

### Backend API

- **API URL**: [https://vendorclose-api.onrender.com](https://vendorclose-api.onrender.com)
- **API Documentation**: [https://vendorclose-api.onrender.com/docs](https://vendorclose-api.onrender.com/docs)
- **Health Check**: [https://vendorclose-api.onrender.com/health](https://vendorclose-api.onrender.com/health)

### Frontend UI

- **Streamlit UI**: [https://vendorclose-ai-re2scqkx8pyo2cmblq3yij.streamlit.app/](https://vendorclose-ai-re2scqkx8pyo2cmblq3yij.streamlit.app/)
- _Note: Frontend connects to hosted backend API_

## Model Details

### Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Transfer Learning**: Yes
- **Input Size**: 160x160x3
- **Output Classes**: 3 (Fresh, Medium, Rotten)

### Optimization Techniques

- **Transfer Learning**: MobileNetV2 pretrained weights
- **Data Augmentation**: Rotation, zoom, shift, flip
- **Regularization**: Dropout (0.3, 0.4), L2 regularization (0.01)
- **Batch Normalization**: Applied after pooling and dense layers
- **Early Stopping**: Patience=10, restore best weights
- **Learning Rate Reduction**: Reduce on plateau (factor=0.5, patience=5)
- **Optimizer**: Adam with learning rate scheduling

### Evaluation Metrics

1. **Loss** (Categorical Crossentropy)
2. **Accuracy**
3. **Precision** (Weighted)
4. **Recall** (Weighted)
5. **F1 Score** (Weighted)
6. **AUC** (Area Under Curve)

## API Endpoints

### Health & Status

- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - Database and model statistics

### Predictions

- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction (multiple images)

### Data Management

- `POST /upload` - Upload training images
  - Query params: `class_label` (fresh/medium/rotten or specific fruit types)

### Model Retraining

- `POST /retrain` - Trigger model retraining
- `GET /retrain/status` - Get retraining status
- `GET /sessions` - Get training session history

## How to Use

### Using the Web UI

1. **Quick Scan**

   - Navigate to the hosted UI
   - Click "Quick Scan" in the sidebar
   - Upload a single fruit image
   - Click "Analyze Fruit" to get instant quality assessment

2. **Batch Processing**

   - Click "Batch Processing" in the sidebar
   - Upload multiple images (up to 50)
   - Click "Analyze All Fruits"
   - View prioritized action list and statistics

3. **Dashboard**

   - Click "Dashboard" to view:
     - Total images processed
     - Class distribution charts
     - Training session history
     - Model performance metrics

4. **Model Retraining**
   - Click "Upload Data" to upload new training images
   - Select the appropriate class label
   - Upload images
   - Go to "Retraining" page
   - Click "Start Retraining" to improve model with new data

### Using the API

The API can be accessed via HTTP requests. See the interactive API documentation at `/docs` endpoint for detailed usage examples.

**Example: Single Prediction**

```bash
POST /predict
Content-Type: multipart/form-data
Body: file (image file)
```

**Example: Batch Prediction**

```bash
POST /predict/batch
Content-Type: multipart/form-data
Body: files (multiple image files)
```

## Visualizations

The application includes:

1. **Class Distribution**: Training data balance across classes
2. **Prediction Confidence**: Distribution of model confidence scores
3. **Per-Class Accuracy**: Accuracy breakdown by fruit quality class
4. **Training History**: Loss, accuracy, precision, recall over epochs
5. **Confusion Matrix**: Classification performance matrix

## Retraining Workflow

1. **Upload Data**: Use the UI or API to upload new training images
2. **Data Storage**: Images saved to database and file system
3. **Trigger Retraining**: Click "Start Retraining" button in UI or call `/retrain` endpoint
4. **Background Processing**: Model retrains with new data
5. **Model Update**: New model saved and loaded automatically
6. **Metrics Tracking**: Training metrics stored in database

## Load Testing Results

Results from Locust load testing with different Docker container configurations:

| Containers | Users | Spawn Rate | Avg Response Time | RPS | Error Rate |
| ---------- | ----- | ---------- | ----------------- | --- | ---------- |
| 1          | 100   | 10         | ~300ms            | 45  | <1%        |
| 2          | 200   | 20         | ~350ms            | 85  | <1%        |
| 3          | 300   | 30         | ~400ms            | 120 | <1%        |

_Note: Results may vary based on hardware and model complexity_

### Where to Find Load Test Results

Load test results are saved in the `load_test_results/` directory:

- HTML reports: `test_1_containers_report.html`, `test_2_containers_report.html`, `test_3_containers_report.html`
- CSV statistics: `test_*_containers_stats.csv`, `test_*_containers_stats_history.csv`

To run load tests, use the `locustfile.py` script with Locust.

## Dependencies

Key dependencies (see `requirements.txt` for complete list):

- TensorFlow 2.13+
- FastAPI 0.104+
- Streamlit 1.28+
- NumPy, Pandas, Pillow
- Locust 2.17+

## License

This project is for educational purposes (assignment submission).

## Author

Izere Kerie

## Video Demo

[https://youtu.be/J3bxvIm0cJ8]

---

Ready to help vendors make smarter decisions!
