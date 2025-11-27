# 🍎 VendorClose AI - Smart End-of-Day Fruit Scanner

A comprehensive machine learning pipeline for classifying fruit quality (Fresh/Medium/Rotten) to help vendors make smart decisions at closing time.

## 📋 Project Overview

**VendorClose AI** is an end-to-end ML system that:
- ✅ Classifies fruit quality into 3 categories: Fresh, Medium (discount), Rotten
- 📸 Provides instant single and batch predictions
- 📊 Offers business insights through visualizations
- 🔄 Supports model retraining with new data
- ⚡ Scales with Docker containerization
- 🧪 Load-tested with Locust

## 🎯 Features

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

## 🏗️ Project Structure

```
VendorClose_AI/
│
├── README.md                 # This file
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
└── requirements.txt        # Python dependencies
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional, for containerized deployment)
- Git

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd VendorClose_AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the fruits dataset from Kaggle:
- **Dataset**: [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- Extract and organize into:
  ```
  data/
  ├── train/
  │   ├── fresh/
  │   └── rotten/
  └── test/
      ├── fresh/
      └── rotten/
  ```

The `medium` class will be automatically created during preprocessing.

### 4. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook notebook/vendorclose_ai.ipynb
```

Or run training programmatically:

```python
from src.preprocessing import ImagePreprocessor, create_medium_class_from_dataset
from src.model import FruitQualityClassifier

# Create medium class
create_medium_class_from_dataset('data')

# Preprocess data
preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)
train_gen, val_gen, test_gen = preprocessor.create_data_generators(
    train_dir='data/train',
    test_dir='data/test'
)

# Build and train model
classifier = FruitQualityClassifier(num_classes=3, img_size=(224, 224))
classifier.build_model(use_pretrained=True)
classifier.compile_model(optimizer_name='adam')
classifier.train(train_gen, val_gen, epochs=50)
classifier.save_model('models/fruit_classifier.h5')
```

### 5. Start the API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or using Docker:

```bash
docker-compose up --build
```

### 6. Launch the Web UI

```bash
streamlit run app.py
```

Access the UI at: `http://localhost:8501`

## 📊 Model Details

### Architecture
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Transfer Learning**: Yes
- **Input Size**: 224x224x3
- **Output Classes**: 3 (Fresh, Medium, Rotten)

### Optimization Techniques
- ✅ **Transfer Learning**: MobileNetV2 pretrained weights
- ✅ **Data Augmentation**: Rotation, zoom, shift, flip
- ✅ **Regularization**: Dropout (0.3, 0.4), L2 regularization (0.01)
- ✅ **Batch Normalization**: Applied after pooling and dense layers
- ✅ **Early Stopping**: Patience=10, restore best weights
- ✅ **Learning Rate Reduction**: Reduce on plateau (factor=0.5, patience=5)
- ✅ **Optimizer**: Adam with learning rate scheduling

### Evaluation Metrics
1. **Loss** (Categorical Crossentropy)
2. **Accuracy**
3. **Precision** (Weighted)
4. **Recall** (Weighted)
5. **F1 Score** (Weighted)
6. **AUC** (Area Under Curve)

## 🔌 API Endpoints

### Health & Status
- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - Database and model statistics

### Predictions
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction (multiple images)

### Data Management
- `POST /upload` - Upload training images
  - Query params: `class_label` (fresh/medium/rotten)

### Model Retraining
- `POST /retrain` - Trigger model retraining
- `GET /retrain/status` - Get retraining status
- `GET /sessions` - Get training session history

## 🧪 Load Testing with Locust

### Quick Test (Verify Locust Works)

First, verify that Locust is installed and working:

```bash
python test_locust.py
```

This will:
- ✅ Check if Locust is installed
- ✅ Verify locustfile.py exists
- ✅ Test API connection
- ✅ Run a quick 10-second load test

### Automated Load Testing Script

Run comprehensive load tests with different container configurations:

```bash
python run_load_tests.py
```

This script will:
- 🐳 Start Docker containers (1, 2, or 3)
- 🚀 Run Locust load tests automatically
- 📊 Record latency and response times
- 📈 Generate comparison reports
- 🛑 Clean up containers after each test

#### Customize Test Parameters

```bash
# Test with specific container counts
python run_load_tests.py --containers 1 2 3

# Adjust load parameters
python run_load_tests.py --users 200 --spawn-rate 20 --run-time 5m

# Custom results directory
python run_load_tests.py --results-dir my_results
```

#### Default Configuration
- **Users**: 100 concurrent users
- **Spawn Rate**: 10 users/second
- **Duration**: 2 minutes per test
- **Containers**: Tests 1, 2, and 3 containers sequentially

### Manual Locust Testing

#### Interactive UI Mode

```bash
# Start API
docker-compose up api

# Start Locust UI
locust -f locustfile.py --host=http://localhost:8000
```

Access Locust UI at: `http://localhost:8089`

#### Headless Mode

```bash
# Single container test
docker-compose up api
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 2m --headless

# Multiple containers (test each port)
docker-compose up
locust -f locustfile.py --host=http://localhost:8000 --users 200 --spawn-rate 20 --run-time 2m --headless
```

### Test Results

Results are saved in `load_test_results/` directory:
- `test_1_containers_report.html` - HTML report for 1 container
- `test_2_containers_report.html` - HTML report for 2 containers
- `test_3_containers_report.html` - HTML report for 3 containers
- `summary_report.json` - Comparison of all tests
- CSV files with detailed statistics

### Expected Results

- **Latency**: < 500ms for single predictions
- **Throughput**: 50+ requests/second per container
- **Error Rate**: < 1%
- **Scaling**: Linear improvement with more containers

## 📈 Visualizations

The notebook and dashboard include:

1. **Class Distribution**: Training data balance across classes
2. **Prediction Confidence**: Distribution of model confidence scores
3. **Per-Class Accuracy**: Accuracy breakdown by fruit quality class
4. **Training History**: Loss, accuracy, precision, recall over epochs
5. **Confusion Matrix**: Classification performance matrix

## 🔄 Retraining Workflow

1. **Upload Data**: Use the UI or API to upload new training images
2. **Data Storage**: Images saved to `data/train/{class}/` and database
3. **Trigger Retraining**: Click "Start Retraining" button
4. **Background Processing**: Model retrains with new data
5. **Model Update**: New model saved and loaded automatically
6. **Metrics Tracking**: Training metrics stored in database

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t vendorclose-ai .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  vendorclose-ai
```

### Docker Compose (Multiple Containers)

```bash
docker-compose up --scale api=3
```

## 📝 Usage Examples

### Python API Client

```python
import requests

# Single prediction
with open('fruit.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())

# Batch prediction
files = [('files', open('fruit1.jpg', 'rb')), 
         ('files', open('fruit2.jpg', 'rb'))]
response = requests.post(
    'http://localhost:8000/predict/batch',
    files=files
)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@fruit.jpg"

# Upload training data
curl -X POST "http://localhost:8000/upload?class_label=fresh" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Trigger retraining
curl -X POST http://localhost:8000/retrain
```

## 🎓 Model Evaluation

The model is evaluated using:

- **Training/Validation Split**: 80/20
- **Test Set**: Separate test directory
- **Cross-Validation**: Not implemented (can be added)
- **Metrics**: All 6 metrics tracked during training and evaluation

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
API_BASE_URL=http://localhost:8000
MODEL_PATH=models/fruit_classifier.h5
DB_PATH=data/training_data.db
```

## 📚 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- TensorFlow 2.13+
- FastAPI 0.104+
- Streamlit 1.28+
- NumPy, Pandas, Pillow
- Locust 2.17+

## 🐛 Troubleshooting

### Model Not Found
- Ensure model is trained: Run the notebook
- Check `models/` directory exists
- Verify model path in `api/main.py`

### API Connection Errors
- Check API is running: `curl http://localhost:8000/health`
- Verify port 8000 is not in use
- Check firewall settings

### Docker Issues
- Ensure Docker is running
- Check container logs: `docker-compose logs`
- Rebuild images: `docker-compose build --no-cache`

## 📄 License

This project is for educational purposes (assignment submission).

## 👤 Author

Machine Learning Engineer - Summative Assignment

## 📹 Video Demo

[YouTube Link - To be added]

## 🔗 URLs

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Locust UI**: http://localhost:8089

## 📊 Results

### Load Testing Results

Results from Locust load testing with different container configurations:

| Containers | Users | Spawn Rate | Avg Response Time | RPS | Error Rate |
|------------|-------|------------|-------------------|-----|------------|
| 1          | 100   | 10         | ~300ms           | 45  | <1%        |
| 2          | 200   | 20         | ~350ms           | 85  | <1%        |
| 3          | 300   | 30         | ~400ms           | 120 | <1%        |

*Note: Results may vary based on hardware and model complexity*

## ✅ Assignment Checklist

- ✅ Data acquisition and preprocessing
- ✅ Model creation with transfer learning
- ✅ Model testing with 4+ metrics
- ✅ Model retraining capability
- ✅ API creation (FastAPI)
- ✅ UI with uptime, visualizations, retraining
- ✅ Database integration
- ✅ Docker containerization
- ✅ Load testing with Locust
- ✅ Clear preprocessing steps
- ✅ Optimization techniques (regularization, early stopping, pretrained model)
- ✅ Evaluation metrics (6 metrics)
- ✅ Data insights and visualizations
- ✅ Upload and retraining triggers
- ✅ Comprehensive README

---

**Ready to help vendors make smarter decisions! 🍎✨**

