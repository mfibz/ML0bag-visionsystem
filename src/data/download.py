"""Dataset download and organization utilities."""

import argparse
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.config.settings import DATA_DIR, ensure_dirs, load_config
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def download_from_roboflow(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    output_dir: Path,
    format: str = "yolov8",
) -> Path:
    """Download a dataset from Roboflow.

    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        api_key: Roboflow API key.
        output_dir: Directory to save dataset.
        format: Export format.

    Returns:
        Path to downloaded dataset.
    """
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(format, location=str(output_dir))
        logger.info(f"Downloaded {project} v{version} to {output_dir}")
        return Path(dataset.location)
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        raise


def download_from_huggingface(
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g. 'user/dataset-name').
        output_dir: Directory to save dataset.
        token: Optional HuggingFace API token for gated datasets.

    Returns:
        Path to downloaded dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package not installed. Run: pip install datasets")
        raise

    import os

    if token is None:
        token = os.environ.get("HF_TOKEN")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading HuggingFace dataset: {repo_id}")
    dataset = load_dataset(repo_id, token=token, trust_remote_code=True)

    # Save images to disk organized by split
    for split_name, split_data in dataset.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(split_data):
            # Handle image column (common in vision datasets)
            if "image" in sample:
                img = sample["image"]
                label = sample.get("label", "unknown")
                label_dir = split_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)
                img_path = label_dir / f"{idx:06d}.jpg"
                if not img_path.exists():
                    img.save(img_path)

        logger.info(f"Saved {len(split_data)} samples to {split_dir}")

    return output_dir


def download_from_url(url: str, output_dir: Path, filename: str | None = None) -> Path:
    """Download a file from a URL using wget or curl.

    Args:
        url: Download URL.
        output_dir: Directory to save file.
        filename: Optional output filename.

    Returns:
        Path to downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    output_path = output_dir / filename
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return output_path

    logger.info(f"Downloading {url} -> {output_path}")
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), url],
            check=True,
        )
    except FileNotFoundError:
        subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            check=True,
        )

    return output_path


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract a zip or tar archive.

    Args:
        archive_path: Path to archive file.
        output_dir: Extraction directory.

    Returns:
        Path to extraction directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif archive_path.suffix in (".gz", ".tgz", ".tar"):
        import tarfile

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    logger.info(f"Extracted {archive_path.name} to {output_dir}")
    return output_dir


def download_datasets(config_name: str = "datasets") -> None:
    """Download all datasets specified in the configuration.

    Args:
        config_name: Name of the dataset config file.
    """
    ensure_dirs()
    config = load_config(config_name)

    for dataset_name, dataset_cfg in config.get("datasets", {}).items():
        logger.info(f"Processing dataset: {dataset_name}")
        source = dataset_cfg.get("source", "url")
        output_dir = DATA_DIR / "raw" / dataset_name

        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"Dataset {dataset_name} already exists, skipping")
            continue

        if source == "roboflow":
            api_key = dataset_cfg.get("api_key", "")
            if not api_key:
                import os

                api_key = os.environ.get("ROBOFLOW_API_KEY", "")
            if not api_key:
                logger.warning(
                    f"No Roboflow API key for {dataset_name}. "
                    "Set ROBOFLOW_API_KEY environment variable."
                )
                continue
            download_from_roboflow(
                workspace=dataset_cfg["workspace"],
                project=dataset_cfg["project"],
                version=dataset_cfg["version"],
                api_key=api_key,
                output_dir=output_dir,
            )
        elif source == "huggingface":
            download_from_huggingface(
                repo_id=dataset_cfg["repo_id"],
                output_dir=output_dir,
            )
        elif source == "url" or source == "web":
            url = dataset_cfg.get("url", "")
            if not url:
                logger.warning(f"No URL provided for {dataset_name}")
                continue
            archive_path = download_from_url(
                url=url,
                output_dir=DATA_DIR / "raw",
                filename=dataset_cfg.get("filename"),
            )
            if archive_path.suffix in (".zip", ".gz", ".tgz", ".tar"):
                extract_archive(archive_path, output_dir)
        else:
            logger.warning(f"Unknown source type: {source}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for bag vision system")
    parser.add_argument(
        "--config",
        type=str,
        default="datasets",
        help="Dataset config file name (default: datasets)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Download only a specific dataset by name",
    )
    args = parser.parse_args()

    if args.dataset:
        config = load_config(args.config)
        if args.dataset not in config.get("datasets", {}):
            logger.error(f"Dataset '{args.dataset}' not found in config")
            return
        filtered = {"datasets": {args.dataset: config["datasets"][args.dataset]}}
        # Write temporary config
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(filtered, f)
            temp_path = f.name
        # Monkey-patch config loading for this run
        config.update(filtered)

    download_datasets(args.config)


if __name__ == "__main__":
    main()
