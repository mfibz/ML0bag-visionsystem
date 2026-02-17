"""Evaluation metrics and reporting for the bag vision pipeline."""

import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config.settings import MODELS_DIR, RUNS_DIR, get_device, load_config
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def evaluate_phase(
    phase: str,
    model_path: str | None = None,
    data_yaml: str | None = None,
    config_name: str = "training",
) -> dict:
    """Run evaluation on a trained model and collect metrics.

    Args:
        phase: Pipeline phase ('detection', 'condition', 'authentication').
        model_path: Optional model weights path (defaults to models/<phase>/best.pt).
        data_yaml: Optional data YAML path (defaults to config value).
        config_name: Training config file name.

    Returns:
        Dictionary of evaluation metrics.
    """
    if model_path is None:
        model_path = str(MODELS_DIR / phase / "best.pt")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(model_path)

    if data_yaml is None:
        config = load_config(config_name)
        phase_config = config.get(phase, {})
        data_yaml = phase_config.get("data_yaml")

    device = get_device()

    logger.info(f"Evaluating {phase} model: {model_path}")
    results = model.val(
        data=data_yaml,
        device=device,
        project=str(RUNS_DIR / phase),
        name="eval",
    )

    metrics = {}

    if hasattr(results, "box"):
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50-95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)

        # Per-class metrics
        if hasattr(results.box, "ap_class_index"):
            per_class = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                class_name = results.names.get(int(cls_idx), str(cls_idx))
                per_class[class_name] = {
                    "ap50": float(results.box.ap50[i]),
                    "ap": float(results.box.ap[i]),
                }
            metrics["per_class"] = per_class

    if hasattr(results, "top1"):
        metrics["top1_accuracy"] = float(results.top1)
        metrics["top5_accuracy"] = float(results.top5)

    logger.info(f"Evaluation results for {phase}: {json.dumps(metrics, indent=2)}")
    return metrics


def generate_report(
    phases: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """Generate a combined evaluation report for all phases.

    Args:
        phases: List of phases to evaluate (defaults to all).
        output_path: Optional path to save JSON report.

    Returns:
        Combined metrics dictionary.
    """
    if phases is None:
        phases = ["detection", "condition", "authentication"]

    report = {}
    for phase in phases:
        model_path = MODELS_DIR / phase / "best.pt"
        if model_path.exists():
            try:
                report[phase] = evaluate_phase(phase)
            except Exception as e:
                logger.warning(f"Failed to evaluate {phase}: {e}")
                report[phase] = {"error": str(e)}
        else:
            report[phase] = {"status": "no model trained"}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")

    return report
