"""Data preprocessing pipeline for each dataset and training phase.

Converts raw downloaded datasets into YOLO-format datasets organized
by training phase (phase1=detection, phase2=condition, phase3=authentication).
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

from src.config.settings import DATA_DIR, PROJECT_ROOT, ensure_dirs
from src.data.convert import coco_to_yolo, create_yolo_dataset_yaml, split_dataset
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# Phase 1 classes: bag detection + luxury brand logos
PHASE1_CLASSES = [
    "bag",
    "BALENCIAGA",
    "BURBERRY",
    "CELINE",
    "CHANEL",
    "DIOR",
    "FENDI",
    "GUCCI",
    "HERMES",
    "LOUIS-VUITTON",
    "LOEWE",
    "GOYARD",
    "MCM",
    "MICHAEL-KORS",
    "PRADA",
    "SAINT-LAURENT",
    "VALENTINO",
    "VERSACE",
    "BOTTEGA-VENETA",
    "CHLOE",
    "COACH",
    "GIVENCHY",
    "MIU-MIU",
    "MULBERRY",
    "SALVATORE-FERRAGAMO",
]

# Phase 2 classes: condition/defect types
PHASE2_CLASSES = [
    "scratch",
    "tear",
    "stain",
    "wear",
    "deformation",
    "hole",
    "discoloration",
]

# Phase 3 classes: authentication (classification)
PHASE3_CLASSES = [
    "authentic",
    "counterfeit",
]


def prepare_phase1(force: bool = False) -> Path:
    """Prepare Phase 1 (Detection) dataset from Roboflow bag logo data.

    The CVandDL bag logo dataset from Roboflow is already in YOLO format,
    so we primarily need to organize it into the expected directory structure.

    Args:
        force: Overwrite existing processed data.

    Returns:
        Path to the phase1 data.yaml file.
    """
    output_dir = DATA_DIR / "processed" / "phase1"
    data_yaml = output_dir / "data.yaml"

    if data_yaml.exists() and not force:
        logger.info("Phase 1 data already prepared, skipping (use --force to redo)")
        return data_yaml

    raw_dir = DATA_DIR / "raw" / "roboflow_logos"

    if not raw_dir.exists():
        logger.warning(
            f"Raw Roboflow logo data not found at {raw_dir}. "
            "Run 'python -m src.data.download --config datasets' first."
        )
        # Create a placeholder data.yaml so training can reference it
        return _create_placeholder_yaml(output_dir, data_yaml, PHASE1_CLASSES)

    # Roboflow exports with train/valid/test splits already
    # Copy the structure to processed directory
    for split in ["train", "valid", "test"]:
        src_imgs = raw_dir / split / "images"
        src_lbls = raw_dir / split / "labels"
        dst_imgs = output_dir / "images" / ("val" if split == "valid" else split)
        dst_lbls = output_dir / "labels" / ("val" if split == "valid" else split)

        if src_imgs.exists():
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_lbls.mkdir(parents=True, exist_ok=True)
            for f in src_imgs.iterdir():
                shutil.copy2(f, dst_imgs / f.name)
            if src_lbls.exists():
                for f in src_lbls.iterdir():
                    shutil.copy2(f, dst_lbls / f.name)

            count = len(list(dst_imgs.glob("*")))
            logger.info(f"Phase 1 {split}: {count} images")

    # Read class names from Roboflow data.yaml if available
    rf_yaml = raw_dir / "data.yaml"
    class_names = PHASE1_CLASSES
    if rf_yaml.exists():
        with open(rf_yaml) as f:
            rf_config = yaml.safe_load(f)
        if "names" in rf_config:
            if isinstance(rf_config["names"], list):
                class_names = rf_config["names"]
            elif isinstance(rf_config["names"], dict):
                class_names = [rf_config["names"][k] for k in sorted(rf_config["names"].keys())]

    create_yolo_dataset_yaml(
        dataset_dir=output_dir,
        class_names=class_names,
        output_path=data_yaml,
    )

    logger.info(f"Phase 1 dataset prepared: {data_yaml}")
    return data_yaml


def prepare_phase2(force: bool = False) -> Path:
    """Prepare Phase 2 (Condition Assessment) dataset.

    Combines Kaputt defect data (filtered) and Kaggle leather defects
    into a YOLO detection dataset for defect/condition assessment.

    Args:
        force: Overwrite existing processed data.

    Returns:
        Path to the phase2 data.yaml file.
    """
    output_dir = DATA_DIR / "processed" / "phase2"
    data_yaml = output_dir / "data.yaml"

    if data_yaml.exists() and not force:
        logger.info("Phase 2 data already prepared, skipping")
        return data_yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = []
    all_labels = []

    # Process Kaputt dataset (filter relevant defect categories)
    kaputt_dir = DATA_DIR / "raw" / "kaputt"
    if kaputt_dir.exists():
        logger.info("Processing Kaputt dataset for Phase 2...")
        relevant_categories = {"penetration", "superficial", "deformation"}
        _process_kaputt_for_condition(kaputt_dir, output_dir, relevant_categories)
    else:
        logger.warning("Kaputt dataset not found. Skipping.")

    # Process leather defects dataset
    leather_dir = DATA_DIR / "raw" / "leather_defects"
    if leather_dir.exists():
        logger.info("Processing leather defects for Phase 2...")
        _process_leather_defects(leather_dir, output_dir)
    else:
        logger.warning("Leather defects dataset not found. Skipping.")

    # Create data.yaml
    create_yolo_dataset_yaml(
        dataset_dir=output_dir,
        class_names=PHASE2_CLASSES,
        output_path=data_yaml,
    )

    logger.info(f"Phase 2 dataset prepared: {data_yaml}")
    return data_yaml


def _process_kaputt_for_condition(
    kaputt_dir: Path, output_dir: Path, categories: set[str]
) -> None:
    """Process Kaputt dataset, filtering for relevant defect categories.

    Maps Kaputt defect types to our condition classes:
    - penetration -> tear, hole
    - superficial -> scratch, stain, discoloration
    - deformation -> deformation, wear
    """
    category_map = {
        "penetration": ["tear", "hole"],
        "superficial": ["scratch", "stain"],
        "deformation": ["deformation", "wear"],
    }

    # Kaputt may have various structures; look for common patterns
    for category_dir in kaputt_dir.rglob("*"):
        if not category_dir.is_dir():
            continue
        cat_name = category_dir.name.lower()
        if cat_name not in categories:
            continue

        images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
        if not images:
            continue

        # For classification-style data, create image-level "labels"
        # by copying images into the appropriate split/class directories
        random.seed(42)
        random.shuffle(images)
        n_train = int(len(images) * 0.8)
        n_val = int(len(images) * 0.15)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        mapped_classes = category_map.get(cat_name, [cat_name])
        primary_class = mapped_classes[0]

        for split_name, split_files in splits.items():
            img_dir = output_dir / "images" / split_name
            lbl_dir = output_dir / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in split_files:
                dst_name = f"kaputt_{cat_name}_{img_file.name}"
                shutil.copy2(img_file, img_dir / dst_name)

                # Create a full-image label (entire image is the defect)
                if primary_class in PHASE2_CLASSES:
                    cls_idx = PHASE2_CLASSES.index(primary_class)
                    lbl_path = lbl_dir / (Path(dst_name).stem + ".txt")
                    with open(lbl_path, "w") as f:
                        f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

        logger.info(f"  Kaputt {cat_name}: {len(images)} images -> {primary_class}")


def _process_leather_defects(leather_dir: Path, output_dir: Path) -> None:
    """Process Kaggle leather defect classification dataset.

    Maps leather defect categories to Phase 2 classes.
    """
    leather_class_map = {
        "scratch": "scratch",
        "scratches": "scratch",
        "holes": "hole",
        "hole": "hole",
        "dirt": "stain",
        "rotten": "wear",
        "fold": "deformation",
        "folds": "deformation",
    }

    for class_dir in leather_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower().strip()
        mapped = leather_class_map.get(class_name)
        if mapped is None:
            logger.debug(f"  Skipping leather class: {class_name}")
            continue

        if mapped not in PHASE2_CLASSES:
            continue

        cls_idx = PHASE2_CLASSES.index(mapped)
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))

        random.seed(42)
        random.shuffle(images)
        n_train = int(len(images) * 0.8)
        n_val = int(len(images) * 0.15)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            img_dir = output_dir / "images" / split_name
            lbl_dir = output_dir / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in split_files:
                dst_name = f"leather_{class_name}_{img_file.name}"
                shutil.copy2(img_file, img_dir / dst_name)

                lbl_path = lbl_dir / (Path(dst_name).stem + ".txt")
                with open(lbl_path, "w") as f:
                    f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

        logger.info(f"  Leather {class_name} -> {mapped}: {len(images)} images")


def prepare_phase3(force: bool = False) -> Path:
    """Prepare Phase 3 (Authentication) dataset from LFFD data.

    The LFFD dataset has product category labels. For authentication,
    we organize images into a classification directory structure
    (authentic/counterfeit folders) suitable for YOLO classification training.

    Note: The LFFD dataset does not have explicit authentic/counterfeit labels.
    This creates a placeholder structure that needs manual annotation.

    Args:
        force: Overwrite existing processed data.

    Returns:
        Path to the phase3 dataset directory (classification format).
    """
    output_dir = DATA_DIR / "processed" / "phase3"

    if (output_dir / "train").exists() and not force:
        logger.info("Phase 3 data already prepared, skipping")
        return output_dir

    lffd_dir = DATA_DIR / "raw" / "lffd"

    if not lffd_dir.exists():
        logger.warning(
            f"LFFD dataset not found at {lffd_dir}. "
            "Run 'python -m src.data.download --config datasets' first."
        )
        # Create placeholder classification structure
        for split in ["train", "val"]:
            for cls in PHASE3_CLASSES:
                (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        logger.info("Created placeholder Phase 3 directory structure")
        return output_dir

    # LFFD is organized by product categories (classification)
    # For authentication, we need authentic vs counterfeit labels
    # Initially, treat all LFFD data as "authentic" (needs manual curation)
    logger.info("Processing LFFD dataset for Phase 3...")
    logger.info("NOTE: All LFFD images initially labeled as 'authentic'.")
    logger.info("Manual review needed to identify/add counterfeit samples.")

    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(lffd_dir.rglob(ext))

    random.seed(42)
    random.shuffle(all_images)
    n_train = int(len(all_images) * 0.8)
    n_val = int(len(all_images) * 0.15)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train : n_train + n_val],
        "test": all_images[n_train + n_val :],
    }

    for split_name, files in splits.items():
        # Create classification directory structure
        for cls in PHASE3_CLASSES:
            (output_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

        # Place all LFFD images as "authentic" initially
        for img_file in files:
            dst = output_dir / split_name / "authentic" / img_file.name
            if not dst.exists():
                shutil.copy2(img_file, dst)

        logger.info(f"Phase 3 {split_name}: {len(files)} images (as 'authentic')")

    logger.info(
        f"Phase 3 prepared at {output_dir}. "
        "Add counterfeit samples to train/counterfeit/ and val/counterfeit/ directories."
    )
    return output_dir


def _create_placeholder_yaml(
    output_dir: Path, data_yaml: Path, class_names: list[str]
) -> Path:
    """Create a placeholder data.yaml with empty directories."""
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    return create_yolo_dataset_yaml(
        dataset_dir=output_dir,
        class_names=class_names,
        output_path=data_yaml,
    )


def prepare_all(force: bool = False) -> None:
    """Prepare all phase datasets."""
    ensure_dirs()
    logger.info("Preparing all datasets...")

    logger.info("\n--- Phase 1: Detection ---")
    prepare_phase1(force)

    logger.info("\n--- Phase 2: Condition Assessment ---")
    prepare_phase2(force)

    logger.info("\n--- Phase 3: Authentication ---")
    prepare_phase3(force)

    logger.info("\nAll datasets prepared.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for training")
    parser.add_argument(
        "phase",
        nargs="?",
        choices=["phase1", "phase2", "phase3", "all"],
        default="all",
        help="Phase to prepare (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()

    if args.phase == "phase1":
        prepare_phase1(args.force)
    elif args.phase == "phase2":
        prepare_phase2(args.force)
    elif args.phase == "phase3":
        prepare_phase3(args.force)
    else:
        prepare_all(args.force)


if __name__ == "__main__":
    main()
