"""
Pin State Classifier
====================
Determines pin states from YOLO detections.

With the 3-class model (bowling-ball, bowling-pins, sweep board):
- Every "bowling-pins" detection is a STANDING pin.
- Fallen pins are NOT detected by the model — they simply disappear.
- Score = (pins before throw) - (pins after throw).
- "bowling-ball" and "sweep board" detections are filtered out.

Temporal smoothing is still used to stabilize the standing-pin count
across frames and prevent flickering.
"""

import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.detector import Detection
from src.utils import get_logger

logger = get_logger(__name__)

STANDING = "standing"
FALLEN = "fallen"
UNKNOWN = "unknown"

# Class names from the model that we recognize
PIN_CLASS_NAME = "bowling-pins"
BALL_CLASS_NAME = "bowling-ball"
SWEEP_CLASS_NAME = "sweep board"


@dataclass
class PinState:
    """State of a single detected pin."""
    detection: Detection
    state: str
    state_confidence: float
    track_id: Optional[int] = None


class PinClassifier:
    """Classify detected pins as standing using direct model output.

    With the 3-class model, classification is trivial:
    - "bowling-pins" → STANDING (the model only detects standing pins)
    - "bowling-ball" → filtered out (not a pin)
    - "sweep board"  → filtered out (not a pin)

    Fallen pins are inferred by subtraction (10 - standing_count).
    """

    def __init__(self, use_temporal_smoothing=True, stability_frames=None,
                 **kwargs):
        """Initialize classifier.

        Args:
            use_temporal_smoothing: Smooth pin counts across frames.
            stability_frames: Number of frames for temporal smoothing.
            **kwargs: Ignored (for backward compat with old args).
        """
        self.use_temporal_smoothing = use_temporal_smoothing
        self.stability_frames = stability_frames or config.STATE_STABILITY_FRAMES
        # Rolling window of recent standing-pin counts for smoothing
        self._count_history: deque = deque(maxlen=self.stability_frames * 3)
        self._confirmed_count: Optional[int] = None
        # Per-track state history (kept for backward compat)
        self._state_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.stability_frames * 2)
        )
        self._confirmed_states: Dict[int, str] = {}

    def filter_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Separate detections by class into pins, balls, and sweep boards.

        Args:
            detections: All detections from the model.

        Returns:
            Dict with keys "pins", "balls", "sweep" each containing
            a list of Detection objects.
        """
        result = {"pins": [], "balls": [], "sweep": []}
        for det in detections:
            name = det.class_name.lower()
            if "pin" in name:
                result["pins"].append(det)
            elif "ball" in name:
                result["balls"].append(det)
            elif "sweep" in name:
                result["sweep"].append(det)
        return result

    def classify(self, detections: List[Detection]) -> List[PinState]:
        """Classify pin detections (no tracking).

        Every bowling-pins detection is classified as STANDING.
        Non-pin detections are automatically filtered out.
        """
        filtered = self.filter_detections(detections)
        pin_states = []
        for det in filtered["pins"]:
            pin_states.append(PinState(
                detection=det, state=STANDING,
                state_confidence=det.confidence,
            ))
        return pin_states

    def classify_with_tracking(self, detections: List[Detection],
                                track_ids: List[int]) -> List[PinState]:
        """Classify with tracking IDs for temporal smoothing.

        All bowling-pins detections are STANDING. Track IDs are
        preserved for display purposes.
        """
        filtered = self.filter_detections(detections)
        pins = filtered["pins"]

        # Build a mapping from detection to track_id
        # track_ids correspond to original detections list order
        det_to_track = {}
        for det, tid in zip(detections, track_ids):
            det_to_track[id(det)] = tid

        pin_states = []
        for det in pins:
            track_id = det_to_track.get(id(det))
            pin_states.append(PinState(
                detection=det, state=STANDING,
                state_confidence=det.confidence,
                track_id=track_id,
            ))
        return pin_states

    def get_pin_counts(self, pin_states: List[PinState]) -> Dict[str, int]:
        """Count pins by state.

        Standing = number of bowling-pins detections.
        Fallen = TOTAL_PINS - standing (inferred by subtraction).
        """
        standing = len(pin_states)

        # Apply temporal smoothing to the count
        if self.use_temporal_smoothing:
            standing = self._smooth_count(standing)

        # Cap at TOTAL_PINS
        standing = min(standing, config.TOTAL_PINS)
        fallen = config.TOTAL_PINS - standing

        return {
            STANDING: standing,
            FALLEN: fallen,
            "total": standing,
        }

    def _smooth_count(self, raw_count: int) -> int:
        """Smooth the standing pin count using a rolling median.

        This prevents single-frame detection flickers from
        causing wild score swings.
        """
        self._count_history.append(raw_count)
        if len(self._count_history) < 3:
            return raw_count
        # Use median of recent counts to filter outliers
        return int(np.median(list(self._count_history)))

    def reset(self):
        """Reset all temporal state (call between games/videos)."""
        self._count_history.clear()
        self._confirmed_count = None
        self._state_history.clear()
        self._confirmed_states.clear()

    def has_sweep_board(self, detections: List[Detection]) -> bool:
        """Check if a sweep board is detected in this frame.

        The sweep board appears between frames to clear fallen pins
        and reset the lane. This provides a reliable reset signal.
        """
        filtered = self.filter_detections(detections)
        return len(filtered["sweep"]) > 0
