# Use Python 3.11 official slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, PaddlePaddle, and InsightFace compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache models to avoid downloads during first API request at runtime
# This caches YOLO11, PaddleOCR english models, and InsightFace buffalo_l models
RUN python -c " \
from ultralytics import YOLO; \
YOLO('yolo11n.pt'); \
from paddleocr import PaddleOCR; \
PaddleOCR(use_angle_cls=True, lang='en', show_log=False); \
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l'); \
app.prepare(ctx_id=-1, det_size=(640,640)) \
"

# Copy project files
COPY . /app/

# Expose port
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
