#!/bin/bash
# Script to run the API server

echo "🚀 Starting VendorClose AI API..."
echo "=================================="

# Check if model exists
if [ ! -f "models/fruit_classifier.h5" ]; then
    echo "⚠️  Warning: Model not found at models/fruit_classifier.h5"
    echo "   Please train the model first using the notebook"
    echo ""
fi

# Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

