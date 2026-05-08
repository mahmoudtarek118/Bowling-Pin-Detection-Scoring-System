"""
Tests for Pin Detector
=======================
Tests for the YOLO detection wrapper.
Note: Full detection tests require model weights.
These tests cover the Detection dataclass and wrapper logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.detector import Detection, PinDetector


class TestDetection:
    """Test the Detection dataclass."""

    def test_properties(self):
        det = Detection(bbox=(10, 20, 50, 100), confidence=0.95,
                        class_id=0, class_name="pin")
        assert det.width == 40
        assert det.height == 80
        assert det.center == (30.0, 60.0)
        assert det.area == 3200
        assert det.aspect_ratio == 2.0  # 80/40

    def test_standing_aspect_ratio(self):
        """Standing pin should have high aspect ratio (tall & narrow)."""
        det = Detection(bbox=(0, 0, 20, 80), confidence=0.9,
                        class_id=0, class_name="pin")
        assert det.aspect_ratio == 4.0  # 80/20

    def test_fallen_aspect_ratio(self):
        """Fallen pin should have low aspect ratio (wide & short)."""
        det = Detection(bbox=(0, 0, 80, 20), confidence=0.9,
                        class_id=0, class_name="pin")
        assert det.aspect_ratio == 0.25  # 20/80

    def test_zero_width_protection(self):
        """Aspect ratio shouldn't crash on zero-width bbox."""
        det = Detection(bbox=(10, 10, 10, 50), confidence=0.5,
                        class_id=0, class_name="pin")
        # Width is 0, but max(1, width) protects against division by zero
        assert det.aspect_ratio == 40.0  # 40/1


class TestPinDetector:
    """Test PinDetector wrapper (without model loading)."""

    def test_initialization(self):
        """Test detector can be initialized with default config."""
        detector = PinDetector()
        assert detector.confidence > 0
        assert detector.iou_threshold > 0
        assert detector.model is None

    def test_custom_params(self):
        """Test detector with custom parameters."""
        detector = PinDetector(
            confidence=0.5,
            iou_threshold=0.6,
            image_size=1280,
            device="cpu",
        )
        assert detector.confidence == 0.5
        assert detector.iou_threshold == 0.6
        assert detector.image_size == 1280
        assert detector.device == "cpu"

    def test_class_names(self):
        """Test default class names from config."""
        detector = PinDetector()
        assert "pin" in detector.class_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
