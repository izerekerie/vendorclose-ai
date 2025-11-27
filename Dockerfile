# Dockerfile for VendorClose AI API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Use a different mirror and add retries
RUN echo "Acquire::Retries \"3\";" > /etc/apt/apt.conf.d/80-retries && \
    apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST (for better caching - if requirements don't change, this layer is cached!)
COPY requirements.txt .

# Install Python dependencies (TensorFlow installs here - cached if requirements.txt unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/train data/test uploads logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

