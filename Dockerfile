FROM python:3.11-slim

# Install system dependencies for OpenCV/DeepFace and audio processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Pre-download DeepFace models at build time so first request isn't slow
RUN python -c "\
from deepface import DeepFace; \
import numpy as np; \
dummy = np.zeros((100,100,3), dtype=np.uint8); \
DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, silent=True); \
print('DeepFace models downloaded.')" || echo "Model pre-download skipped."

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
