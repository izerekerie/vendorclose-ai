# 🚀 Quick Start Guide

## Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

## Step 2: Download Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
2. Download the dataset
3. Extract and organize:
   ```
   data/
   ├── train/
   │   ├── fresh/
   │   └── rotten/
   └── test/
       ├── fresh/
       └── rotten/
   ```

## Step 3: Train Model

Open Jupyter notebook:
```bash
jupyter notebook notebook/vendorclose_ai.ipynb
```

Run all cells to:
- Preprocess data
- Create medium class
- Train model
- Evaluate with 6 metrics
- Generate visualizations

Model will be saved to `models/fruit_classifier.h5`

## Step 4: Start API

**Windows:**
```bash
run_api.bat
```

**Linux/Mac:**
```bash
chmod +x run_api.sh
./run_api.sh
```

**Or manually:**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

## Step 5: Start UI

**Windows:**
```bash
run_ui.bat
```

**Or manually:**
```bash
streamlit run app.py
```

UI will be available at: http://localhost:8501

## Step 6: Test API

```bash
python test_api.py
```

## Step 7: Load Testing (Optional)

```bash
# Install Locust if not already installed
pip install locust

# Run Locust
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
```

## Docker Deployment (Alternative)

```bash
# Build and run
docker-compose up --build

# Scale to multiple containers
docker-compose up --scale api=3
```

## Troubleshooting

### Model Not Found
- Train model first using the notebook
- Check `models/` directory

### Port Already in Use
- Change port in `api/main.py` or use `--port` flag
- Kill process using port: `netstat -ano | findstr :8000` (Windows)

### Import Errors
- Ensure you're in the project root directory
- Check `sys.path` includes project root
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### API Not Responding
- Check API logs for errors
- Verify model file exists
- Test with: `curl http://localhost:8000/health`

## Next Steps

1. Upload new training data via UI
2. Trigger retraining
3. Monitor dashboard for metrics
4. Use batch processing for multiple fruits

---

**Need help?** Check the main README.md for detailed documentation.

