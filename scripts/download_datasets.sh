#!/bin/bash
# Download all datasets for the bag vision system
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Bag Vision System - Dataset Download ==="
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Ensure Python dependencies are installed
pip install -q datasets huggingface-hub roboflow

# Run the download script
python -m src.data.download --config datasets

echo "=== Download complete ==="
echo "Raw datasets saved to: $PROJECT_ROOT/data/raw/"
