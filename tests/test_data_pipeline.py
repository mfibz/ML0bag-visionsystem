"""Unit tests for data preprocessing and conversion functions."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.convert import (
    coco_to_yolo,
    create_yolo_dataset_yaml,
    split_dataset,
)


@pytest.fixture
def tmp_dir():
    """Create and clean up a temporary directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_images(tmp_dir):
    """Create sample test images."""
    img_dir = tmp_dir / "images"
    img_dir.mkdir()
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_dir / f"img_{i:03d}.jpg")
    return img_dir


@pytest.fixture
def sample_labels(tmp_dir, sample_images):
    """Create sample YOLO label files."""
    lbl_dir = tmp_dir / "labels"
    lbl_dir.mkdir()
    for img_file in sample_images.glob("*.jpg"):
        label_file = lbl_dir / (img_file.stem + ".txt")
        # Random YOLO annotations: class x_center y_center width height
        lines = [f"0 0.5 0.5 0.3 0.3", f"1 0.2 0.8 0.1 0.15"]
        label_file.write_text("\n".join(lines))
    return lbl_dir


@pytest.fixture
def sample_coco(tmp_dir, sample_images):
    """Create a sample COCO format annotation file."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "bag"},
            {"id": 2, "name": "logo"},
        ],
    }
    ann_id = 0
    for i, img_file in enumerate(sorted(sample_images.glob("*.jpg"))):
        coco["images"].append({
            "id": i,
            "file_name": img_file.name,
            "width": 100,
            "height": 100,
        })
        coco["annotations"].append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,
            "bbox": [10, 10, 30, 30],
        })
        ann_id += 1

    json_path = tmp_dir / "annotations.json"
    json_path.write_text(json.dumps(coco))
    return json_path


class TestCocoToYolo:
    def test_converts_annotations(self, sample_coco, sample_images, tmp_dir):
        output_dir = tmp_dir / "yolo_output"
        coco_to_yolo(sample_coco, sample_images, output_dir)

        labels_dir = output_dir / "labels"
        assert labels_dir.exists()
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0

    def test_creates_images_dir(self, sample_coco, sample_images, tmp_dir):
        output_dir = tmp_dir / "yolo_output"
        coco_to_yolo(sample_coco, sample_images, output_dir)

        images_dir = output_dir / "images"
        assert images_dir.exists()
        assert len(list(images_dir.glob("*.jpg"))) > 0

    def test_yolo_format_values(self, sample_coco, sample_images, tmp_dir):
        output_dir = tmp_dir / "yolo_output"
        coco_to_yolo(sample_coco, sample_images, output_dir)

        label_file = next((output_dir / "labels").glob("*.txt"))
        content = label_file.read_text().strip()
        parts = content.split()
        assert len(parts) == 5  # class x_center y_center width height
        class_idx = int(parts[0])
        assert class_idx >= 0
        # All values should be normalized (0-1)
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0


class TestSplitDataset:
    def test_splits_correctly(self, sample_images, sample_labels, tmp_dir):
        output_dir = tmp_dir / "split_output"
        split_dataset(
            sample_images,
            sample_labels,
            output_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        for split in ["train", "val", "test"]:
            assert (output_dir / "images" / split).exists()
            assert (output_dir / "labels" / split).exists()

    def test_all_images_accounted_for(self, sample_images, sample_labels, tmp_dir):
        output_dir = tmp_dir / "split_output"
        split_dataset(sample_images, sample_labels, output_dir)

        total = 0
        for split in ["train", "val", "test"]:
            total += len(list((output_dir / "images" / split).glob("*.jpg")))

        original = len(list(sample_images.glob("*.jpg")))
        assert total == original

    def test_labels_match_images(self, sample_images, sample_labels, tmp_dir):
        output_dir = tmp_dir / "split_output"
        split_dataset(sample_images, sample_labels, output_dir)

        for split in ["train", "val", "test"]:
            img_stems = {f.stem for f in (output_dir / "images" / split).glob("*.jpg")}
            lbl_stems = {f.stem for f in (output_dir / "labels" / split).glob("*.txt")}
            assert img_stems == lbl_stems


class TestCreateYoloDatasetYaml:
    def test_creates_valid_yaml(self, tmp_dir):
        output_path = tmp_dir / "data.yaml"
        create_yolo_dataset_yaml(
            dataset_dir=tmp_dir,
            class_names=["bag", "logo"],
            output_path=output_path,
        )

        assert output_path.exists()

        import yaml
        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert config["nc"] == 2
        assert config["names"][0] == "bag"
        assert config["names"][1] == "logo"
        assert "train" in config
        assert "val" in config
