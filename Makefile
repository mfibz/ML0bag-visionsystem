.PHONY: help install data preprocess train-all train-phase1 train-phase2 train-phase3 \
       validate infer-camera infer-image docker-build docker-train docker-data \
       clean test

PYTHON ?= python
SHELL := /bin/bash

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----

install: ## Install Python dependencies
	pip install -r requirements.txt

setup: install ## Full setup: install deps and create directories
	$(PYTHON) -c "from src.config.settings import ensure_dirs; ensure_dirs()"

# ---- Data Pipeline ----

data: ## Download all datasets
	$(PYTHON) -m src.data.download --config datasets

preprocess: ## Preprocess all datasets for training
	$(PYTHON) -m src.data.preprocess all

preprocess-force: ## Preprocess all datasets (overwrite existing)
	$(PYTHON) -m src.data.preprocess all --force

data-all: data preprocess ## Download and preprocess all data

# ---- Training ----

train-phase1: ## Train Phase 1: Bag Detection + Logo Recognition
	$(PYTHON) -m src.models.trainer train phase1

train-phase2: ## Train Phase 2: Condition Assessment
	$(PYTHON) -m src.models.trainer train phase2

train-phase3: ## Train Phase 3: Authentication
	$(PYTHON) -m src.models.trainer train phase3

train-all: ## Train all phases sequentially
	$(PYTHON) -m src.models.trainer train-all

# ---- Validation ----

validate-phase1: ## Validate Phase 1 model
	$(PYTHON) -m src.models.trainer validate phase1

validate-phase2: ## Validate Phase 2 model
	$(PYTHON) -m src.models.trainer validate phase2

validate-phase3: ## Validate Phase 3 model
	$(PYTHON) -m src.models.trainer validate phase3

# ---- Export ----

export-all: ## Export all models to ONNX
	$(PYTHON) -m src.models.trainer export phase1
	$(PYTHON) -m src.models.trainer export phase2
	$(PYTHON) -m src.models.trainer export phase3

# ---- Inference ----

infer-camera: ## Run real-time camera inference
	$(PYTHON) -m src.inference.camera --source 0

infer-video: ## Run inference on video (usage: make infer-video VIDEO=path.mp4)
	$(PYTHON) -m src.inference.camera --source $(VIDEO)

infer-image: ## Run inference on image (usage: make infer-image IMAGE=path.jpg)
	$(PYTHON) -m src.inference.camera --source $(IMAGE)

# ---- Docker ----

docker-build: ## Build Docker image
	docker compose build

docker-train: ## Run training in Docker with GPU
	docker compose run --rm train

docker-data: ## Download and preprocess data in Docker
	docker compose run --rm data-prep

docker-infer: ## Run inference in Docker
	docker compose run --rm inference

# ---- Testing ----

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

# ---- Cleanup ----

clean: ## Remove cached files and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-runs: ## Remove training runs (WARNING: deletes training logs)
	rm -rf runs/

clean-data: ## Remove processed data (keeps raw downloads)
	rm -rf data/processed/
