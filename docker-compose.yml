version: "3.8"

services:
  # Training service with GPU support
  train:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./runs:/app/runs
      - ./configs:/app/configs
    command: python -m src.models.trainer train-all

  # Inference service
  inference:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    command: python -m src.inference.camera --source 0 --no-show

  # Data download and preprocessing
  data-prep:
    build: .
    environment:
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
    command: >
      bash -c "python -m src.data.download --config datasets &&
               python -m src.data.preprocess all"
