"""
Tests for Pin Classifier
=========================
Tests for the 3-class model classification logic.
With the new model, every "bowling-pins" detection is STANDING.
Non-pin objects (bowling-ball, sweep board) are filtered out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.detector import Detection
from src.pin_classifier import PinClassifier, STANDING, FALLEN
import config


class TestPinClassifier:
    """Test pin state classification logic."""

    def setup_method(self):
        self.classifier = PinClassifier(use_temporal_smoothing=False)

    def _make_pin(self, bbox=(10, 10, 30, 80), confidence=0.9):
        """Create a bowling-pins detection."""
        return Detection(
            bbox=bbox, confidence=confidence,
            class_id=config.PIN_CLASS, class_name="bowling-pins",
        )

    def _make_ball(self, bbox=(100, 100, 140, 140), confidence=0.85):
        """Create a bowling-ball detection."""
        return Detection(
            bbox=bbox, confidence=confidence,
            class_id=config.BALL_CLASS, class_name="bowling-ball",
        )

    def _make_sweep(self, bbox=(0, 200, 640, 280), confidence=0.75):
        """Create a sweep board detection."""
        return Detection(
            bbox=bbox, confidence=confidence,
            class_id=config.SWEEP_CLASS, class_name="sweep board",
        )

    def test_pin_classified_as_standing(self):
        """Every bowling-pins detection should be classified as STANDING."""
        det = self._make_pin()
        states = self.classifier.classify([det])
        assert len(states) == 1
        assert states[0].state == STANDING

    def test_ball_filtered_out(self):
        """Bowling ball detections should NOT appear in pin states."""
        detections = [self._make_pin(), self._make_ball()]
        states = self.classifier.classify(detections)
        assert len(states) == 1  # Only the pin, not the ball
        assert states[0].state == STANDING

    def test_sweep_filtered_out(self):
        """Sweep board detections should NOT appear in pin states."""
        detections = [self._make_pin(), self._make_sweep()]
        states = self.classifier.classify(detections)
        assert len(states) == 1

    def test_multiple_pins(self):
        """Test classification of multiple standing pins."""
        detections = [
            self._make_pin((10, 10, 30, 80)),
            self._make_pin((50, 10, 70, 80)),
            self._make_pin((90, 10, 110, 80)),
        ]
        states = self.classifier.classify(detections)
        assert len(states) == 3
        assert all(ps.state == STANDING for ps in states)

    def test_mixed_detections(self):
        """Test filtering with a mix of pins, balls, and sweep boards."""
        detections = [
            self._make_pin((10, 10, 30, 80)),
            self._make_ball(),
            self._make_pin((50, 10, 70, 80)),
            self._make_sweep(),
            self._make_pin((90, 10, 110, 80)),
        ]
        states = self.classifier.classify(detections)
        assert len(states) == 3  # 3 pins only
        assert all(ps.state == STANDING for ps in states)

    def test_pin_counts_full_rack(self):
        """10 pins detected = 10 standing, 0 fallen."""
        detections = [self._make_pin() for _ in range(10)]
        states = self.classifier.classify(detections)
        counts = self.classifier.get_pin_counts(states)
        assert counts["standing"] == 10
        assert counts["fallen"] == 0
        assert counts["total"] == 10

    def test_pin_counts_partial(self):
        """3 pins detected = 3 standing, 7 fallen."""
        detections = [self._make_pin() for _ in range(3)]
        states = self.classifier.classify(detections)
        counts = self.classifier.get_pin_counts(states)
        assert counts["standing"] == 3
        assert counts["fallen"] == 7
        assert counts["total"] == 3

    def test_pin_counts_none_detected(self):
        """0 pins detected = 0 standing, 10 fallen (strike!)."""
        states = self.classifier.classify([])
        counts = self.classifier.get_pin_counts(states)
        assert counts["standing"] == 0
        assert counts["fallen"] == 10

    def test_filter_detections(self):
        """Test the filter_detections helper method."""
        detections = [
            self._make_pin(),
            self._make_ball(),
            self._make_sweep(),
            self._make_pin(),
        ]
        filtered = self.classifier.filter_detections(detections)
        assert len(filtered["pins"]) == 2
        assert len(filtered["balls"]) == 1
        assert len(filtered["sweep"]) == 1

    def test_has_sweep_board(self):
        """Test sweep board detection helper."""
        detections_with_sweep = [self._make_pin(), self._make_sweep()]
        detections_without_sweep = [self._make_pin(), self._make_ball()]
        assert self.classifier.has_sweep_board(detections_with_sweep)
        assert not self.classifier.has_sweep_board(detections_without_sweep)

    def test_empty_detections(self):
        """No detections → empty result."""
        states = self.classifier.classify([])
        assert len(states) == 0

    def test_reset(self):
        """Test classifier reset clears temporal state."""
        self.classifier._count_history.append(5)
        self.classifier.reset()
        assert len(self.classifier._count_history) == 0
        assert len(self.classifier._state_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
