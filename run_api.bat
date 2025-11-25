@echo off
REM Script to run the API server on Windows

echo Starting VendorClose AI API...
echo ==================================

REM Check if model exists
if not exist "models\fruit_classifier.h5" (
    echo Warning: Model not found at models\fruit_classifier.h5
    echo Please train the model first using the notebook
    echo.
)

REM Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

