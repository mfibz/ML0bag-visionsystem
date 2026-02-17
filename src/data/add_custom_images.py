"""Utility to add your own bag photos to the training pipeline.

Usage:
    # Add photos for Phase 1 (detection/logos):
    python -m src.data.add_custom_images --phase 1 --source /path/to/my/photos --brand GUCCI

    # Add photos for Phase 2 (defects):
    python -m src.data.add_custom_images --phase 2 --source /path/to/defect/photos --defect scratch

    # Add photos for Phase 3 (authentication):
    python -m src.data.add_custom_images --phase 3 --source /path/to/photos --label authentic
    python -m src.data.add_custom_images --phase 3 --source /path/to/photos --label counterfeit
"""

import argparse
import shutil
from pathlib import Path

from src.config.settings import DATA_DIR, ensure_dirs
from src.data.preprocess import PHASE1_CLASSES, PHASE2_CLASSES, PHASE3_CLASSES
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _collect_images(source_dir: Path) -> list[Path]:
    """Find all image files in a directory (recursively)."""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(source_dir.rglob(f"*{ext}"))
        images.extend(source_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def add_phase1_images(source: Path, brand: str) -> int:
    """Add brand logo images for Phase 1 detection training.

    Images are copied to data/custom/phase1_detection/images/ and placeholder
    full-image labels are created. For best results, annotate with Roboflow/CVAT
    to create accurate bounding boxes.

    Args:
        source: Directory containing bag photos with the brand logo visible.
        brand: Brand name (must match one of the Phase 1 classes).

    Returns:
        Number of images added.
    """
    brand_upper = brand.upper().replace(" ", "-")

    if brand_upper not in PHASE1_CLASSES:
        logger.warning(
            f"Brand '{brand_upper}' not in known classes: {PHASE1_CLASSES}. "
            "Adding anyway - update PHASE1_CLASSES in preprocess.py if needed."
        )

    cls_idx = PHASE1_CLASSES.index(brand_upper) if brand_upper in PHASE1_CLASSES else 0

    output_imgs = DATA_DIR / "custom" / "phase1_detection" / "images"
    output_lbls = DATA_DIR / "custom" / "phase1_detection" / "labels"
    output_imgs.mkdir(parents=True, exist_ok=True)
    output_lbls.mkdir(parents=True, exist_ok=True)

    images = _collect_images(source)
    count = 0

    for img_path in images:
        dst_name = f"custom_{brand_upper}_{count:04d}{img_path.suffix.lower()}"
        dst_img = output_imgs / dst_name
        dst_lbl = output_lbls / f"custom_{brand_upper}_{count:04d}.txt"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

            # Create a full-image placeholder label
            # For accurate results, re-annotate these with Roboflow/CVAT
            with open(dst_lbl, "w") as f:
                f.write(f"{cls_idx} 0.5 0.5 0.9 0.9\n")

            count += 1

    logger.info(f"Added {count} images for {brand_upper} to Phase 1 custom data")
    logger.info(f"Location: {output_imgs}")
    logger.info(
        "TIP: For best accuracy, annotate these images with precise bounding boxes "
        "using Roboflow (roboflow.com) or CVAT (cvat.ai), then replace the label files."
    )
    return count


def add_phase2_images(source: Path, defect: str) -> int:
    """Add defect images for Phase 2 condition training.

    Args:
        source: Directory containing close-up photos of defects.
        defect: Defect type (scratch, tear, stain, wear, deformation, hole, discoloration).

    Returns:
        Number of images added.
    """
    defect_lower = defect.lower().strip()

    if defect_lower not in PHASE2_CLASSES:
        logger.error(f"Unknown defect type: '{defect_lower}'. Must be one of: {PHASE2_CLASSES}")
        return 0

    cls_idx = PHASE2_CLASSES.index(defect_lower)

    output_imgs = DATA_DIR / "custom" / "phase2_defects" / "images"
    output_lbls = DATA_DIR / "custom" / "phase2_defects" / "labels"
    output_imgs.mkdir(parents=True, exist_ok=True)
    output_lbls.mkdir(parents=True, exist_ok=True)

    images = _collect_images(source)
    count = 0

    for img_path in images:
        dst_name = f"custom_{defect_lower}_{count:04d}{img_path.suffix.lower()}"
        dst_img = output_imgs / dst_name
        dst_lbl = output_lbls / f"custom_{defect_lower}_{count:04d}.txt"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

            # Full-image label (defect covers most of the close-up photo)
            with open(dst_lbl, "w") as f:
                f.write(f"{cls_idx} 0.5 0.5 0.8 0.8\n")

            count += 1

    logger.info(f"Added {count} images for defect '{defect_lower}' to Phase 2 custom data")
    return count


def add_phase3_images(source: Path, label: str) -> int:
    """Add authentication images for Phase 3 training.

    No annotation needed - just sorts images into authentic/counterfeit folders.

    Args:
        source: Directory containing bag photos.
        label: Either 'authentic' or 'counterfeit'.

    Returns:
        Number of images added.
    """
    label_lower = label.lower().strip()

    if label_lower not in PHASE3_CLASSES:
        logger.error(f"Label must be 'authentic' or 'counterfeit', got: '{label_lower}'")
        return 0

    output_dir = DATA_DIR / "custom" / "phase3_authentication" / label_lower
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(source)
    count = 0

    for img_path in images:
        dst_name = f"custom_{label_lower}_{count:04d}{img_path.suffix.lower()}"
        dst = output_dir / dst_name

        if not dst.exists():
            shutil.copy2(img_path, dst)
            count += 1

    logger.info(f"Added {count} images as '{label_lower}' to Phase 3 custom data")
    logger.info(f"Location: {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Add your own bag photos to the training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add Gucci bag photos for logo detection:
  python -m src.data.add_custom_images --phase 1 --source ~/photos/gucci_bags --brand GUCCI

  # Add scratch defect photos:
  python -m src.data.add_custom_images --phase 2 --source ~/photos/scratches --defect scratch

  # Add verified authentic bags:
  python -m src.data.add_custom_images --phase 3 --source ~/photos/real_bags --label authentic

  # Add known fakes:
  python -m src.data.add_custom_images --phase 3 --source ~/photos/fake_bags --label counterfeit

After adding images, retrain:
  python -m src.data.preprocess all --force
  python -m src.models.trainer train phase1  # (or phase2, phase3)
        """,
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Training phase (1=detection, 2=defects, 3=authentication)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to folder containing your bag photos",
    )
    parser.add_argument(
        "--brand",
        type=str,
        help="Brand name for Phase 1 (e.g., GUCCI, LOUIS-VUITTON, CHANEL)",
    )
    parser.add_argument(
        "--defect",
        type=str,
        help="Defect type for Phase 2 (scratch, tear, stain, wear, deformation, hole, discoloration)",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label for Phase 3 (authentic or counterfeit)",
    )

    args = parser.parse_args()
    ensure_dirs()

    source = Path(args.source)
    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        return

    if args.phase == 1:
        if not args.brand:
            parser.error("--brand is required for Phase 1")
        add_phase1_images(source, args.brand)

    elif args.phase == 2:
        if not args.defect:
            parser.error("--defect is required for Phase 2")
        add_phase2_images(source, args.defect)

    elif args.phase == 3:
        if not args.label:
            parser.error("--label is required for Phase 3")
        add_phase3_images(source, args.label)

    print("\nNext steps:")
    print("  1. python -m src.data.preprocess all --force")
    print(f"  2. python -m src.models.trainer train phase{args.phase}")


if __name__ == "__main__":
    main()
