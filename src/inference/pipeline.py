"""Multi-phase cascade inference pipeline: Detect -> Condition -> Authenticate."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.settings import PROJECT_ROOT, get_device, load_config
from src.utils.logging import setup_logger
from src.utils.visualization import draw_cascade_result

logger = setup_logger(__name__)


@dataclass
class BagResult:
    """Result from the full cascade pipeline for a single detected bag."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    bag_type: str = ""
    detection_conf: float = 0.0
    condition: str | None = None
    condition_conf: float | None = None
    auth_result: str | None = None
    auth_conf: float | None = None
    crop: np.ndarray | None = field(default=None, repr=False)


class CascadePipeline:
    """Multi-phase cascade: detect bag -> assess condition -> authenticate.

    Loads models from paths specified in configs/inference.yaml and runs
    them sequentially on detected bag regions.
    """

    def __init__(self, config_path: str | None = None, device: str | None = None):
        """Initialize the cascade pipeline.

        Args:
            config_path: Path to inference config. Defaults to configs/inference.yaml.
            device: Compute device override.
        """
        self.device = device or get_device()

        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config("inference")

        pipeline_cfg = self.config.get("pipeline", {})
        self.cascade = pipeline_cfg.get("cascade", True)
        self.crop_detections = pipeline_cfg.get("crop_detections", True)
        self.crop_padding = pipeline_cfg.get("crop_padding", 0.1)

        # Load models
        self.models = {}
        for phase in ["phase1", "phase2", "phase3"]:
            phase_cfg = self.config.get("models", {}).get(phase, {})
            weights = phase_cfg.get("weights", "")
            if weights:
                weights_path = Path(weights)
                if not weights_path.is_absolute():
                    weights_path = PROJECT_ROOT / weights_path
                if weights_path.exists():
                    logger.info(f"Loading {phase} model from {weights_path}")
                    self.models[phase] = {
                        "model": YOLO(str(weights_path)),
                        "conf": phase_cfg.get("conf_threshold", 0.5),
                        "iou": phase_cfg.get("iou_threshold", 0.45),
                        "imgsz": phase_cfg.get("imgsz", 640),
                        "task": phase_cfg.get("task", "detect"),
                    }
                else:
                    logger.warning(f"Model not found for {phase}: {weights_path}")

    def _crop_with_padding(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop a region from image with padding.

        Args:
            image: BGR image.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            Cropped image region.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        return image[cy1:cy2, cx1:cx2].copy()

    def detect(self, image: np.ndarray) -> list[BagResult]:
        """Phase 1: Detect bags in image.

        Args:
            image: BGR image array.

        Returns:
            List of BagResult with detection info.
        """
        if "phase1" not in self.models:
            logger.error("Phase 1 detection model not loaded")
            return []

        cfg = self.models["phase1"]
        results = cfg["model"].predict(
            image,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            device=self.device,
            verbose=False,
        )

        bag_results = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names.get(cls, str(cls))

                crop = self._crop_with_padding(image, (x1, y1, x2, y2))

                bag_results.append(
                    BagResult(
                        bbox=(x1, y1, x2, y2),
                        bag_type=class_name,
                        detection_conf=conf,
                        crop=crop,
                    )
                )

        return bag_results

    def assess_condition(self, bag_result: BagResult) -> BagResult:
        """Phase 2: Assess condition of a detected bag.

        Args:
            bag_result: BagResult with crop from detection.

        Returns:
            Updated BagResult with condition info.
        """
        if "phase2" not in self.models or bag_result.crop is None:
            return bag_result

        cfg = self.models["phase2"]
        task = cfg["task"]

        results = cfg["model"].predict(
            bag_result.crop,
            conf=cfg["conf"],
            imgsz=cfg["imgsz"],
            device=self.device,
            verbose=False,
        )

        if not results:
            return bag_result

        result = results[0]

        if task == "classify" and result.probs is not None:
            top_idx = int(result.probs.top1)
            top_conf = float(result.probs.top1conf)
            bag_result.condition = result.names.get(top_idx, str(top_idx))
            bag_result.condition_conf = top_conf
        elif task == "detect" and result.boxes is not None and len(result.boxes) > 0:
            # For detection-based condition: report the most confident defect
            best_idx = result.boxes.conf.argmax()
            cls = int(result.boxes.cls[best_idx])
            bag_result.condition = result.names.get(cls, str(cls))
            bag_result.condition_conf = float(result.boxes.conf[best_idx])

        return bag_result

    def authenticate(self, bag_result: BagResult) -> BagResult:
        """Phase 3: Authenticate a detected bag.

        Args:
            bag_result: BagResult with crop from detection.

        Returns:
            Updated BagResult with authentication info.
        """
        if "phase3" not in self.models or bag_result.crop is None:
            return bag_result

        cfg = self.models["phase3"]
        results = cfg["model"].predict(
            bag_result.crop,
            imgsz=cfg["imgsz"],
            device=self.device,
            verbose=False,
        )

        if results and results[0].probs is not None:
            probs = results[0].probs
            top_idx = int(probs.top1)
            top_conf = float(probs.top1conf)
            class_name = results[0].names.get(top_idx, str(top_idx))

            if top_conf >= cfg["conf"]:
                bag_result.auth_result = class_name
                bag_result.auth_conf = top_conf

        return bag_result

    def process(self, image: np.ndarray) -> list[BagResult]:
        """Run the full cascade pipeline on an image.

        Args:
            image: BGR image array.

        Returns:
            List of BagResult with full pipeline results.
        """
        t0 = time.time()

        # Phase 1: Detection
        bag_results = self.detect(image)
        logger.debug(f"Detected {len(bag_results)} bags")

        if self.cascade:
            # Phase 2: Condition
            for r in bag_results:
                self.assess_condition(r)

            # Phase 3: Authentication
            for r in bag_results:
                self.authenticate(r)

        elapsed = time.time() - t0
        logger.debug(f"Pipeline: {len(bag_results)} bags in {elapsed:.3f}s")

        return bag_results

    def annotate(self, image: np.ndarray, results: list[BagResult]) -> np.ndarray:
        """Draw all results on an image.

        Args:
            image: BGR image.
            results: List of BagResult.

        Returns:
            Annotated image.
        """
        annotated = image.copy()
        for r in results:
            annotated = draw_cascade_result(
                annotated,
                r.bbox,
                r.bag_type,
                r.condition,
                r.condition_conf,
                r.auth_result,
                r.auth_conf,
            )
        return annotated
