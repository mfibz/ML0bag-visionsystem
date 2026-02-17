#!/bin/bash
# Train all three phases of the bag vision pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Bag Vision System - Multi-Phase Training ==="
cd "$PROJECT_ROOT"

# Phase 1: Detection
echo ""
echo "=========================================="
echo "Phase 1: Bag Detection + Brand Logo"
echo "=========================================="
python -m src.models.trainer train phase1

# Phase 2: Condition Assessment
echo ""
echo "=========================================="
echo "Phase 2: Condition Assessment"
echo "=========================================="
python -m src.models.trainer train phase2

# Phase 3: Authentication
echo ""
echo "=========================================="
echo "Phase 3: Authentication"
echo "=========================================="
python -m src.models.trainer train phase3

echo ""
echo "=== All phases trained ==="
echo "Models saved to: $PROJECT_ROOT/models/"
