"""Convert various annotation formats to YOLO format."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from src.config.settings import DATA_DIR, load_config
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def coco_to_yolo(
    coco_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    class_mapping: dict[str, int] | None = None,
) -> None:
    """Convert COCO format annotations to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON annotation file.
        images_dir: Directory containing images.
        output_dir: Output directory for YOLO format dataset.
        class_mapping: Optional mapping of category names to YOLO class indices.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category mapping
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    if class_mapping is None:
        class_mapping = {name: idx for idx, name in enumerate(sorted(set(categories.values())))}

    # Build image lookup
    images = {img["id"]: img for img in coco["images"]}

    # Create output directories
    labels_dir = output_dir / "labels"
    imgs_out_dir = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs_out_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    converted = 0
    for img_id, img_info in images.items():
        filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Copy image
        src_img = images_dir / filename
        if src_img.exists():
            shutil.copy2(src_img, imgs_out_dir / filename)

        # Convert annotations
        label_file = labels_dir / (Path(filename).stem + ".txt")
        lines = []
        for ann in img_annotations.get(img_id, []):
            cat_name = categories[ann["category_id"]]
            if cat_name not in class_mapping:
                continue

            class_idx = class_mapping[cat_name]
            x, y, w, h = ann["bbox"]  # COCO format: x_min, y_min, width, height

            # Convert to YOLO format: x_center, y_center, width, height (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    logger.info(f"Converted {converted} images from COCO to YOLO format")
    logger.info(f"Class mapping: {class_mapping}")


def voc_to_yolo(
    voc_dir: Path,
    output_dir: Path,
    class_mapping: dict[str, int] | None = None,
) -> None:
    """Convert Pascal VOC format annotations to YOLO format.

    Args:
        voc_dir: Directory containing VOC XML annotations and images.
        output_dir: Output directory for YOLO format dataset.
        class_mapping: Optional mapping of class names to indices.
    """
    xml_dir = voc_dir / "Annotations"
    img_dir = voc_dir / "JPEGImages"

    if not xml_dir.exists():
        # Try flat directory structure
        xml_dir = voc_dir
        img_dir = voc_dir

    labels_dir = output_dir / "labels"
    imgs_out_dir = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs_out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all classes if no mapping provided
    if class_mapping is None:
        all_classes = set()
        for xml_file in xml_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            for obj in tree.findall(".//object"):
                all_classes.add(obj.find("name").text)
        class_mapping = {name: idx for idx, name in enumerate(sorted(all_classes))}

    converted = 0
    for xml_file in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        filename = root.find("filename").text

        # Copy image
        src_img = img_dir / filename
        if src_img.exists():
            shutil.copy2(src_img, imgs_out_dir / filename)

        lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue

            class_idx = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            w_norm = (xmax - xmin) / img_w
            h_norm = (ymax - ymin) / img_h

            lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_file = labels_dir / (Path(filename).stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    logger.info(f"Converted {converted} images from VOC to YOLO format")


def create_yolo_dataset_yaml(
    dataset_dir: Path,
    class_names: list[str],
    output_path: Path,
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str | None = None,
) -> Path:
    """Create a YOLO dataset YAML configuration file.

    Args:
        dataset_dir: Root directory of the dataset.
        class_names: List of class names.
        output_path: Path to write the YAML file.
        train_path: Relative path to training images.
        val_path: Relative path to validation images.
        test_path: Optional relative path to test images.

    Returns:
        Path to created YAML file.
    """
    config = {
        "path": str(dataset_dir),
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    if test_path:
        config["test"] = test_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created YOLO dataset YAML: {output_path}")
    return output_path


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    """Split a dataset into train/val/test sets.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing label files.
        output_dir: Output directory for split dataset.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
    """
    import random

    random.seed(seed)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
    ]
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train : n_train + n_val],
        "test": image_files[n_train + n_val :],
    }

    for split_name, files in splits.items():
        split_img_dir = output_dir / "images" / split_name
        split_lbl_dir = output_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            shutil.copy2(img_file, split_img_dir / img_file.name)
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, split_lbl_dir / label_file.name)

        logger.info(f"{split_name}: {len(files)} images")


def main():
    parser = argparse.ArgumentParser(description="Convert and prepare datasets for YOLO training")
    subparsers = parser.add_subparsers(dest="command", help="Conversion command")

    # COCO to YOLO
    coco_parser = subparsers.add_parser("coco2yolo", help="Convert COCO to YOLO format")
    coco_parser.add_argument("--coco-json", type=str, required=True, help="Path to COCO JSON")
    coco_parser.add_argument("--images-dir", type=str, required=True, help="Images directory")
    coco_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # VOC to YOLO
    voc_parser = subparsers.add_parser("voc2yolo", help="Convert VOC to YOLO format")
    voc_parser.add_argument("--voc-dir", type=str, required=True, help="VOC dataset directory")
    voc_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Split dataset
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--images-dir", type=str, required=True)
    split_parser.add_argument("--labels-dir", type=str, required=True)
    split_parser.add_argument("--output-dir", type=str, required=True)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.05)

    # Create YAML
    yaml_parser = subparsers.add_parser("create-yaml", help="Create YOLO dataset YAML")
    yaml_parser.add_argument("--dataset-dir", type=str, required=True)
    yaml_parser.add_argument("--classes", nargs="+", required=True, help="Class names")
    yaml_parser.add_argument("--output", type=str, required=True, help="Output YAML path")

    args = parser.parse_args()

    if args.command == "coco2yolo":
        coco_to_yolo(
            Path(args.coco_json), Path(args.images_dir), Path(args.output_dir)
        )
    elif args.command == "voc2yolo":
        voc_to_yolo(Path(args.voc_dir), Path(args.output_dir))
    elif args.command == "split":
        split_dataset(
            Path(args.images_dir),
            Path(args.labels_dir),
            Path(args.output_dir),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
    elif args.command == "create-yaml":
        create_yolo_dataset_yaml(
            Path(args.dataset_dir), args.classes, Path(args.output)
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
