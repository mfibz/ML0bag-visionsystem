FROM ultralytics/ultralytics:latest-python

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

# Create data and model directories
RUN mkdir -p data/raw data/processed models/phase1 models/phase2 models/phase3 runs

# Set Python path for module imports
ENV PYTHONPATH=/app

# Default command: show help
CMD ["python", "-m", "src.models.trainer", "--help"]
