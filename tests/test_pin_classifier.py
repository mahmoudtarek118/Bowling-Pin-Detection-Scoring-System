"""
Tests for Pin Classifier
=========================
Tests for aspect ratio heuristics and classification logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.detector import Detection
from src.pin_classifier import PinClassifier, STANDING, FALLEN, UNKNOWN
import config


class TestPinClassifier:
    """Test pin state classification logic."""

    def setup_method(self):
        self.classifier = PinClassifier(
            use_model_class=True,
            use_aspect_ratio=True,
            use_temporal_smoothing=False,  # Disable for unit tests
        )

    def _make_detection(self, bbox, class_name="pin", confidence=0.9):
        """Helper to create a Detection with given bbox and class."""
        class_id = 0 if "standing" in class_name else (1 if "fallen" in class_name else 0)
        return Detection(
            bbox=bbox, confidence=confidence,
            class_id=class_id, class_name=class_name,
        )

    def test_standing_from_model_class(self):
        """Model says standing → classified as standing."""
        det = self._make_detection((10, 10, 30, 80), "standing_pin", 0.95)
        states = self.classifier.classify([det])
        assert states[0].state == STANDING

    def test_fallen_from_model_class(self):
        """Model says fallen → classified as fallen."""
        det = self._make_detection((10, 10, 80, 30), "fallen_pin", 0.90)
        states = self.classifier.classify([det])
        assert states[0].state == FALLEN

    def test_standing_from_aspect_ratio(self):
        """Tall narrow bbox → standing (when model gives generic class)."""
        # height=80, width=20 → aspect ratio = 4.0
        det = self._make_detection((100, 100, 120, 180), "pin", 0.85)
        classifier = PinClassifier(use_model_class=True, use_aspect_ratio=True)
        states = classifier.classify([det])
        assert states[0].state == STANDING

    def test_fallen_from_aspect_ratio(self):
        """Wide short bbox → fallen (when model gives generic class)."""
        # height=15, width=60 → aspect ratio = 0.25
        det = self._make_detection((100, 100, 160, 115), "pin", 0.85)
        classifier = PinClassifier(use_model_class=True, use_aspect_ratio=True)
        states = classifier.classify([det])
        assert states[0].state == FALLEN

    def test_multiple_detections(self):
        """Test classification of multiple pins at once."""
        detections = [
            self._make_detection((10, 10, 30, 80), "pin"),   # AR=3.5 → standing
            self._make_detection((50, 50, 120, 70), "pin"),   # AR=0.29 → fallen
            self._make_detection((150, 10, 170, 90), "pin"),  # AR=4.0 → standing
        ]
        states = self.classifier.classify(detections)
        assert len(states) == 3
        assert states[0].state == STANDING
        assert states[1].state == FALLEN
        assert states[2].state == STANDING

    def test_pin_counts(self):
        """Test pin counting from classified states."""
        detections = [
            self._make_detection((10, 10, 30, 80), "pin"),   # AR=3.5 → standing
            self._make_detection((50, 50, 120, 70), "pin"),   # AR=0.29 → fallen
            self._make_detection((150, 10, 170, 90), "pin"),  # AR=4.0 → standing
        ]
        states = self.classifier.classify(detections)
        counts = self.classifier.get_pin_counts(states)
        assert counts["standing"] == 2
        assert counts["fallen"] == 1
        assert counts["total"] == 3

    def test_empty_detections(self):
        """No detections → empty result."""
        states = self.classifier.classify([])
        assert len(states) == 0

    def test_reset(self):
        """Test classifier reset clears temporal state."""
        self.classifier._confirmed_states[1] = STANDING
        self.classifier.reset()
        assert len(self.classifier._confirmed_states) == 0
        assert len(self.classifier._state_history) == 0


class TestTemporalSmoothing:
    """Test temporal smoothing with tracking."""

    def setup_method(self):
        self.classifier = PinClassifier(
            use_model_class=True,
            use_aspect_ratio=True,
            use_temporal_smoothing=True,
            stability_frames=3,
        )

    def _make_standing_detection(self):
        """Create a pin detection with standing aspect ratio (tall & narrow)."""
        return Detection(
            bbox=(10, 10, 30, 80), confidence=0.9,
            class_id=0, class_name="pin",
        )

    def _make_fallen_detection(self):
        """Create a pin detection with fallen aspect ratio (wide & short)."""
        return Detection(
            bbox=(10, 10, 80, 25), confidence=0.9,
            class_id=0, class_name="pin",
        )

    def test_state_persists_during_flicker(self):
        """State doesn't change from a single-frame flicker."""
        det_standing = self._make_standing_detection()
        det_fallen = self._make_fallen_detection()

        # Establish standing state
        self.classifier.classify_with_tracking([det_standing], [1])
        self.classifier.classify_with_tracking([det_standing], [1])

        # Single flicker to fallen
        states = self.classifier.classify_with_tracking([det_fallen], [1])
        # Should still be standing (not enough consecutive fallen frames)
        assert states[0].state == STANDING

    def test_state_changes_after_stability(self):
        """State changes after sufficient consecutive frames."""
        det_standing = self._make_standing_detection()
        det_fallen = self._make_fallen_detection()

        # Establish standing
        self.classifier.classify_with_tracking([det_standing], [1])

        # Consistent fallen for stability_frames
        for _ in range(3):
            states = self.classifier.classify_with_tracking([det_fallen], [1])

        assert states[0].state == FALLEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
