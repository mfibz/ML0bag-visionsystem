"""Central configuration management using YAML files."""

import os
from pathlib import Path
from typing import Any

import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"


def load_config(config_name: str) -> dict[str, Any]:
    """Load a YAML configuration file from the configs directory.

    Args:
        config_name: Name of the config file (with or without .yaml extension).

    Returns:
        Dictionary of configuration values.
    """
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def get_device() -> str:
    """Determine the best available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs():
    """Create required project directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, RUNS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for sub in ["raw", "processed"]:
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    for phase in ["phase1", "phase2", "phase3"]:
        (DATA_DIR / "processed" / phase).mkdir(parents=True, exist_ok=True)
        (MODELS_DIR / phase).mkdir(parents=True, exist_ok=True)
