"""
YOLO Pin Detector
=================
Wrapper around Ultralytics YOLO for bowling pin detection.
Handles model loading, inference, and result parsing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single object detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence score [0, 1].
        class_id: Class index (0=bowling-ball, 1=bowling-pins, 2=sweep board).
        class_name: Human-readable class name.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Height / Width ratio. High = standing, Low = fallen."""
        return self.height / max(1, self.width)


class PinDetector:
    """YOLO-based bowling pin detector.

    Loads a trained YOLO model and runs inference on video frames,
    returning structured Detection objects for each detected pin.
    """

    def __init__(
        self,
        model_path: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        image_size: int = None,
        device: str = "auto",
    ):
        """Initialize the detector.

        Args:
            model_path: Path to trained YOLO .pt weights. If None, uses config default.
            confidence: Minimum detection confidence. If None, uses config default.
            iou_threshold: NMS IoU threshold. If None, uses config default.
            image_size: Input image size for inference. If None, uses config default.
            device: Compute device ('auto', 'cpu', '0' for GPU 0, etc.).
        """
        self.model_path = model_path or str(config.TRAINED_MODEL)
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        self.image_size = image_size or config.IMAGE_SIZE
        self.device = device
        self.model = None
        self._class_names = config.CLASS_NAMES

    def load_model(self) -> None:
        """Load the YOLO model from disk.

        Raises FileNotFoundError if model weights don't exist.
        """
        from ultralytics import YOLO

        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(
                f"Trained model not found at {model_path}. "
                f"Attempting to load as a base model name (e.g. 'yolo11m.pt')..."
            )
            # Allow loading pretrained base models by name
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(str(model_path))

        logger.info(f"Model loaded: {self.model_path}")

        # Update class names from model if available
        if hasattr(self.model, "names") and self.model.names:
            self._class_names = list(self.model.names.values())
            logger.info(f"Model classes: {self._class_names}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (from OpenCV).

        Returns:
            List of Detection objects for each detected pin.
        """
        if self.model is None:
            self.load_model()

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=config.MAX_DETECTIONS,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Map class name
                cls_name = (
                    self._class_names[cls_id]
                    if cls_id < len(self._class_names)
                    else f"class_{cls_id}"
                )

                detections.append(
                    Detection(
                        bbox=tuple(xyxy),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection lists (one per frame).
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=config.MAX_DETECTIONS,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for result in results:
            frame_detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = (
                        self._class_names[cls_id]
                        if cls_id < len(self._class_names)
                        else f"class_{cls_id}"
                    )
                    frame_detections.append(
                        Detection(
                            bbox=tuple(xyxy),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )
            all_detections.append(frame_detections)

        return all_detections

    @property
    def class_names(self) -> List[str]:
        """Return the class names this model detects."""
        return self._class_names
