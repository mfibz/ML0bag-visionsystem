# ML0bag Vision System

A multi-phase computer vision pipeline for luxury bag detection, condition assessment, and authentication using YOLO11 (Ultralytics).

## Overview

The system runs three sequential phases on input images or video:

1. **Phase 1 - Detection**: Detects bags in frame and identifies brand logos (24 luxury brands)
2. **Phase 2 - Condition Assessment**: Identifies defects such as scratches, tears, stains, and wear
3. **Phase 3 - Authentication**: Classifies detected bags as authentic, suspicious, or counterfeit

## Project Structure

```
configs/          Training, inference, and dataset configurations
data/             Raw and processed datasets (gitignored)
models/           Trained model weights (gitignored)
runs/             Training logs and metrics (gitignored)
src/
  config/         Central configuration management
  data/           Dataset download, conversion, and splitting
  evaluation/     Metrics and reporting
  inference/      Cascade pipeline and camera inference
  models/         Training manager
  utils/          Logging and visualization
scripts/          Shell scripts for common workflows
tests/            Unit and integration tests
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU
- 16 GB RAM minimum

### Installation

```bash
# Clone the repository
cd /workspaces/ML0bag-visionsystem

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your HuggingFace token and Roboflow API key
```

### Docker

```bash
docker build -t bag-vision .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models bag-vision
```

## Data Preparation

### Download Datasets

```bash
# Set API keys
export HF_TOKEN=your_huggingface_token
export ROBOFLOW_API_KEY=your_roboflow_key

# Download all datasets
bash scripts/download_datasets.sh

# Or download individually
python -m src.data.download --config datasets --dataset lffd
python -m src.data.download --config datasets --dataset roboflow_logos
```

### Preprocess for Training

```bash
# Preprocess all datasets for all phases
python -m src.data.preprocess all

# Or preprocess a specific phase
python -m src.data.preprocess phase1
python -m src.data.preprocess phase2
python -m src.data.preprocess phase3

# Force reprocess (overwrite existing)
python -m src.data.preprocess all --force
```

### Manual Format Conversion (if needed)

```bash
# Convert COCO annotations to YOLO format
python -m src.data.convert coco2yolo \
  --coco-json data/raw/dataset/annotations.json \
  --images-dir data/raw/dataset/images \
  --output-dir data/processed/phase1

# Split into train/val/test
python -m src.data.convert split \
  --images-dir data/processed/phase1/images \
  --labels-dir data/processed/phase1/labels \
  --output-dir data/processed/phase1

# Create YOLO data.yaml
python -m src.data.convert create-yaml \
  --dataset-dir data/processed/phase1 \
  --classes bag GUCCI LOUIS-VUITTON PRADA \
  --output data/processed/phase1/data.yaml
```

## Training

### Train Individual Phases

```bash
# Phase 1: Detection
python -m src.models.trainer train phase1

# Phase 2: Condition assessment
python -m src.models.trainer train phase2

# Phase 3: Authentication
python -m src.models.trainer train phase3
```

### Train All Phases

```bash
# Using the training script
bash scripts/train_all.sh

# Or using the trainer directly
python -m src.models.trainer train-all

# Or using Make
make train-all
```

### Resume Training

```bash
python -m src.models.trainer train phase1 --resume
```

### Validate a Trained Model

```bash
python -m src.models.trainer validate phase1
```

### Export for Deployment

```bash
python -m src.models.trainer export phase1 --format onnx
```

## Inference

### Single Image

```bash
python -m src.inference.camera --source path/to/image.jpg --save output.jpg
```

### Video File

```bash
python -m src.inference.camera --source path/to/video.mp4 --save output.mp4
```

### Webcam (Real-time)

```bash
python -m src.inference.camera --source 0
# Press 'q' to quit, 's' to save screenshot
```

### Custom Model Paths

```bash
python -m src.inference.camera \
  --source 0 \
  --device cuda
```

## Configuration

All configurations are in `configs/`:

| File | Purpose |
|------|---------|
| `training.yaml` | Training hyperparameters for all phases |
| `inference.yaml` | Model paths, thresholds, camera settings |
| `datasets.yaml` | Dataset download sources |
| `phase1_detection.yaml` | Phase 1 detailed YOLO config |
| `phase2_condition.yaml` | Phase 2 detailed YOLO config |
| `phase3_authentication.yaml` | Phase 3 detailed YOLO config |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Datasets

| Dataset | Source | Images | Use |
|---------|--------|--------|-----|
| CVandDL Bag Logo | Roboflow | 1,920 | Phase 1 (brand detection) |
| Innovatiana LFFD | HuggingFace | 12,379 | Phase 3 (authentication) |
| Kaputt | kaputt-dataset.com | 238,421 | Phase 2 (defect detection) |
| Leather Defects | Kaggle | ~125 | Phase 2 (supplement) |

See `RESEARCH_FINDINGS.md` for detailed dataset analysis.

## Model Performance

| Phase | Model | mAP50 | mAP50-95 | Notes |
|-------|-------|-------|----------|-------|
| Detection | YOLO11m | - | - | Pending training |
| Condition | YOLO11m | - | - | Pending training |
| Authentication | YOLO11s-cls | - | - | Pending training |

## License

This project is for research and educational purposes. Dataset licenses vary -- see `RESEARCH_FINDINGS.md` for details.
