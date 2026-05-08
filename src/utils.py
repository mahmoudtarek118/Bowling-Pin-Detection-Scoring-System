"""
Utility Functions
=================
Shared helper functions used across the bowling analysis pipeline.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ── Logging Setup ─────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    return logger


# ── Video Helpers ─────────────────────────────────────────────
def open_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file and return the capture object.

    Raises FileNotFoundError if the video cannot be opened.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Extract key properties from an opened video capture."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or config.OUTPUT_VIDEO_FPS,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def create_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str = None,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for the output annotated video."""
    codec = codec or config.OUTPUT_VIDEO_CODEC
    fourcc = cv2.VideoWriter_fourcc(*codec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer


# ── Geometry Helpers ──────────────────────────────────────────
def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    """Calculate the area of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    """Calculate height/width aspect ratio of a bounding box.

    Returns a large value for tall-narrow boxes (standing pins)
    and a small value for wide-short boxes (fallen pins).
    """
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return height / width


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ── Drawing Helpers ───────────────────────────────────────────
def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 4,
) -> np.ndarray:
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def overlay_transparent(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    alpha: float = 0.7,
) -> np.ndarray:
    """Overlay a semi-transparent rectangle on the frame."""
    h, w = overlay.shape[:2]
    # Ensure we don't exceed frame bounds
    h = min(h, background.shape[0] - y)
    w = min(w, background.shape[1] - x)
    if h <= 0 or w <= 0:
        return background

    roi = background[y : y + h, x : x + w]
    blended = cv2.addWeighted(overlay[:h, :w], alpha, roi, 1 - alpha, 0)
    background[y : y + h, x : x + w] = blended
    return background


# ── File Helpers ──────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_files(directory: str, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    """List all video files in a directory."""
    dirpath = Path(directory)
    if not dirpath.exists():
        return []
    return sorted(
        p for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
