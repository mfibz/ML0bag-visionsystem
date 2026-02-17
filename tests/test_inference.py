"""Integration tests for the inference pipeline."""

import numpy as np
import pytest

from src.inference.pipeline import BagResult, CascadePipeline
from src.utils.visualization import draw_cascade_result, draw_detection


class TestBagResult:
    def test_default_values(self):
        result = BagResult(bbox=(0, 0, 100, 100))
        assert result.bag_type == ""
        assert result.detection_conf == 0.0
        assert result.condition is None
        assert result.auth_result is None
        assert result.crop is None

    def test_full_result(self):
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        result = BagResult(
            bbox=(10, 20, 110, 120),
            bag_type="GUCCI",
            detection_conf=0.95,
            condition="scratch",
            condition_conf=0.8,
            auth_result="authentic",
            auth_conf=0.92,
            crop=crop,
        )
        assert result.bag_type == "GUCCI"
        assert result.detection_conf == 0.95
        assert result.condition == "scratch"
        assert result.auth_result == "authentic"


class TestVisualization:
    def test_draw_detection(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = draw_detection(image, (10, 10, 100, 100), "bag", 0.9)
        assert result.shape == (200, 200, 3)
        # Check that something was drawn (not all zeros)
        assert result.sum() > 0

    def test_draw_cascade_result(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = draw_cascade_result(
            image,
            bbox=(10, 50, 150, 180),
            bag_type="PRADA",
            condition="good",
            condition_conf=0.85,
            auth_result="authentic",
            auth_conf=0.91,
        )
        assert result.shape == (200, 200, 3)
        assert result.sum() > 0

    def test_draw_cascade_partial(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = draw_cascade_result(
            image,
            bbox=(10, 50, 150, 180),
            bag_type="unknown",
        )
        assert result.shape == (200, 200, 3)


class TestCascadePipeline:
    def test_init_without_models(self):
        """Pipeline should initialize gracefully even without model files."""
        # This will log warnings about missing models but should not raise
        try:
            pipeline = CascadePipeline(config_path="inference")
            assert pipeline.device in ("cuda", "mps", "cpu")
            assert pipeline.cascade is True
        except FileNotFoundError:
            pytest.skip("Config file not found in test environment")

    def test_process_without_models(self):
        """Process should return empty list when no models are loaded."""
        try:
            pipeline = CascadePipeline(config_path="inference")
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = pipeline.process(image)
            assert isinstance(results, list)
            assert len(results) == 0  # No models loaded = no detections
        except FileNotFoundError:
            pytest.skip("Config file not found in test environment")
