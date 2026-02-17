#!/bin/bash
# Run inference on camera, video, or image
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

SOURCE="${1:-0}"  # Default to webcam

echo "=== Bag Vision System - Inference ==="
echo "Source: $SOURCE"

python -m src.inference.camera --source "$SOURCE" "$@"
