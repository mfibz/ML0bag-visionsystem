"""Visualization utilities for bounding boxes and pipeline results."""

import cv2
import numpy as np


# Color palette for different pipeline phases
PHASE_COLORS = {
    "detection": (0, 255, 0),      # Green
    "condition": (255, 165, 0),     # Orange
    "authentication": (255, 0, 0),  # Red for fake, Blue for authentic
}

CONDITION_COLORS = {
    "excellent": (0, 200, 0),
    "good": (0, 255, 128),
    "fair": (255, 200, 0),
    "poor": (0, 0, 255),
}

AUTH_COLORS = {
    "authentic": (0, 200, 0),
    "counterfeit": (0, 0, 255),
    "uncertain": (0, 200, 255),
}


def draw_detection(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box with label on an image.

    Args:
        image: BGR image array.
        bbox: (x1, y1, x2, y2) bounding box coordinates.
        label: Class label text.
        confidence: Detection confidence score.
        color: BGR color tuple.
        thickness: Line thickness.

    Returns:
        Annotated image.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    text = f"{label} {confidence:.2f}"
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 1)

    return image


def draw_cascade_result(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    bag_type: str,
    condition: str | None = None,
    condition_conf: float | None = None,
    auth_result: str | None = None,
    auth_conf: float | None = None,
) -> np.ndarray:
    """Draw full cascade pipeline result on an image.

    Args:
        image: BGR image array.
        bbox: (x1, y1, x2, y2) bounding box.
        bag_type: Detected bag type.
        condition: Condition assessment result.
        condition_conf: Condition confidence.
        auth_result: Authentication result.
        auth_conf: Authentication confidence.

    Returns:
        Annotated image.
    """
    x1, y1, x2, y2 = bbox

    # Determine overall color based on authentication result
    if auth_result:
        color = AUTH_COLORS.get(auth_result, (128, 128, 128))
    elif condition:
        color = CONDITION_COLORS.get(condition, (128, 128, 128))
    else:
        color = PHASE_COLORS["detection"]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Build multi-line label
    lines = [f"Type: {bag_type}"]
    if condition and condition_conf is not None:
        lines.append(f"Condition: {condition} ({condition_conf:.2f})")
    if auth_result and auth_conf is not None:
        lines.append(f"Auth: {auth_result} ({auth_conf:.2f})")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    y_offset = y1 - 5
    for line in reversed(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, 1)
        cv2.rectangle(image, (x1, y_offset - th - 4), (x1 + tw + 4, y_offset + 2), color, -1)
        cv2.putText(image, line, (x1 + 2, y_offset), font, font_scale, (255, 255, 255), 1)
        y_offset -= th + 8

    return image
