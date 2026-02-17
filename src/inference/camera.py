"""Real-time camera/video inference using the cascade pipeline."""

import argparse
import time
from pathlib import Path

import cv2

from src.config.settings import PROJECT_ROOT, load_config
from src.inference.pipeline import CascadePipeline
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_camera(
    source: int | str = 0,
    config_path: str | None = None,
    save_output: str | None = None,
    device: str | None = None,
    show: bool = True,
) -> None:
    """Run real-time inference on camera or video source.

    Args:
        source: Camera index (0 for webcam) or path to video file.
        config_path: Inference configuration file name.
        save_output: Optional path to save output video.
        device: Compute device override.
        show: Whether to display the video window.
    """
    pipeline = CascadePipeline(config_path=config_path, device=device)

    config = load_config(config_path or "inference")
    cam_cfg = config.get("camera", {})

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        logger.info(f"Saving output to: {save_output}")

    logger.info(f"Starting inference on source: {source} ({width}x{height} @ {fps:.0f}fps)")
    logger.info("Press 'q' to quit, 's' to save screenshot")

    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    logger.info("End of video file")
                break

            # Run cascade pipeline
            results = pipeline.process(frame)

            # Annotate frame
            annotated = pipeline.annotate(frame, results)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            cv2.putText(
                annotated,
                f"FPS: {fps_display:.1f} | Bags: {len(results)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Bag Vision System", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("s"):
                    screenshot_path = str(PROJECT_ROOT / f"screenshot_{int(time.time())}.jpg")
                    cv2.imwrite(screenshot_path, annotated)
                    logger.info(f"Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    logger.info("Camera inference stopped")


def run_on_image(
    image_path: str,
    config_path: str | None = None,
    save_output: str | None = None,
    device: str | None = None,
    show: bool = True,
) -> None:
    """Run inference on a single image.

    Args:
        image_path: Path to input image.
        config_path: Inference config name.
        save_output: Optional output image path.
        device: Compute device override.
        show: Whether to display the result.
    """
    pipeline = CascadePipeline(config_path=config_path, device=device)

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    results = pipeline.process(image)
    annotated = pipeline.annotate(image, results)

    for i, result in enumerate(results):
        logger.info(
            f"Bag {i+1}: type={result.bag_type} conf={result.detection_conf:.2f} "
            f"condition={result.condition} auth={result.auth_result}"
        )

    if save_output:
        cv2.imwrite(save_output, annotated)
        logger.info(f"Output saved to: {save_output}")
    elif show:
        cv2.imshow("Bag Vision System", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Bag Vision System - Real-time Inference")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video/image path (default: 0 for webcam)",
    )
    parser.add_argument("--config", type=str, default=None, help="Inference config name")
    parser.add_argument("--save", type=str, help="Save output to file")
    parser.add_argument("--device", type=str, help="Compute device (cuda, cpu, mps)")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    args = parser.parse_args()

    # Determine if source is camera index or file path
    source = int(args.source) if args.source.isdigit() else args.source

    # Check if source is an image file
    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        run_on_image(
            source,
            config_path=args.config,
            save_output=args.save,
            device=args.device,
            show=not args.no_show,
        )
    else:
        run_camera(
            source=source,
            config_path=args.config,
            save_output=args.save,
            device=args.device,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
