"""
Pin State Classifier
====================
Determines whether detected pins are standing or fallen using:
1. YOLO class output (primary)
2. Bounding box aspect ratio heuristics (secondary/fallback)
3. Temporal consistency checks (smoothing across frames)
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


@dataclass
class PinState:
    """State of a single detected pin."""
    detection: Detection
    state: str
    state_confidence: float
    track_id: Optional[int] = None


class PinClassifier:
    """Classify detected pins as standing or fallen via multi-signal approach."""

    def __init__(self, use_model_class=True, use_aspect_ratio=True,
                 use_temporal_smoothing=True, stability_frames=None):
        self.use_model_class = use_model_class
        self.use_aspect_ratio = use_aspect_ratio
        self.use_temporal_smoothing = use_temporal_smoothing
        self.stability_frames = stability_frames or config.STATE_STABILITY_FRAMES
        self._state_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.stability_frames * 2)
        )
        self._confirmed_states: Dict[int, str] = {}

    def classify(self, detections: List[Detection]) -> List[PinState]:
        """Classify detections for a single frame (no tracking)."""
        pin_states = []
        for det in detections:
            state, confidence = self._determine_state(det)
            pin_states.append(PinState(detection=det, state=state, state_confidence=confidence))
        return pin_states

    def classify_with_tracking(self, detections: List[Detection],
                                track_ids: List[int]) -> List[PinState]:
        """Classify with tracking IDs for temporal smoothing."""
        pin_states = []
        for det, track_id in zip(detections, track_ids):
            raw_state, raw_confidence = self._determine_state(det)
            if self.use_temporal_smoothing and track_id is not None:
                smoothed_state = self._smooth_state(track_id, raw_state)
            else:
                smoothed_state = raw_state
            pin_states.append(PinState(
                detection=det, state=smoothed_state,
                state_confidence=raw_confidence, track_id=track_id,
            ))
        return pin_states

    def _determine_state(self, detection: Detection) -> Tuple[str, float]:
        """Determine pin state from available signals with weighted voting.

        With a 1-class model (all pins detected as "pin"), the model class
        signal returns UNKNOWN and aspect-ratio heuristics become the
        sole decision maker. With a 2-class model (standing_pin/fallen_pin),
        model class takes priority (70%) with aspect ratio as backup (30%).
        """
        signals = []

        if self.use_model_class:
            model_state, model_conf = self._from_model_class(detection)
            if model_state != UNKNOWN:
                signals.append((model_state, model_conf, 0.7))

        if self.use_aspect_ratio:
            ar_state, ar_conf = self._from_aspect_ratio(detection)
            if ar_state != UNKNOWN:
                # When model class is unavailable (1-class model), give
                # aspect ratio full weight so it becomes the sole signal.
                ar_weight = 1.0 if not signals else 0.3
                signals.append((ar_state, ar_conf, ar_weight))

        if not signals:
            return UNKNOWN, 0.0

        standing_score = sum(c * w for s, c, w in signals if s == STANDING)
        fallen_score = sum(c * w for s, c, w in signals if s == FALLEN)
        total = standing_score + fallen_score
        if total == 0:
            return UNKNOWN, 0.0
        if standing_score >= fallen_score:
            return STANDING, standing_score / total
        return FALLEN, fallen_score / total

    def _from_model_class(self, detection: Detection) -> Tuple[str, float]:
        """Determine state from YOLO class label.

        Returns UNKNOWN for generic class names like "pin" — this triggers
        the aspect-ratio fallback in _determine_state().
        """
        name = detection.class_name.lower()
        if "standing" in name:
            return STANDING, detection.confidence
        elif "fallen" in name or "down" in name:
            return FALLEN, detection.confidence
        return UNKNOWN, 0.0

    def _from_aspect_ratio(self, detection: Detection) -> Tuple[str, float]:
        """Determine state from bounding box aspect ratio."""
        ar = detection.aspect_ratio
        if ar >= config.STANDING_MIN_ASPECT_RATIO:
            conf = min(1.0, ar / (config.STANDING_MIN_ASPECT_RATIO * 1.5))
            return STANDING, conf
        elif ar <= config.FALLEN_MAX_ASPECT_RATIO:
            conf = min(1.0, config.FALLEN_MAX_ASPECT_RATIO / max(0.1, ar))
            return FALLEN, conf
        return UNKNOWN, 0.0

    def _smooth_state(self, track_id: int, raw_state: str) -> str:
        """Apply temporal smoothing to prevent flickering."""
        history = self._state_history[track_id]
        history.append(raw_state)
        if track_id not in self._confirmed_states:
            self._confirmed_states[track_id] = raw_state
            return raw_state
        current_confirmed = self._confirmed_states[track_id]
        if len(history) >= self.stability_frames:
            recent = list(history)[-self.stability_frames:]
            if all(s == raw_state for s in recent) and raw_state != current_confirmed:
                self._confirmed_states[track_id] = raw_state
                return raw_state
        return current_confirmed

    def reset(self):
        """Reset all temporal state (call between games/videos)."""
        self._state_history.clear()
        self._confirmed_states.clear()

    def get_pin_counts(self, pin_states: List[PinState]) -> Dict[str, int]:
        """Count pins by state."""
        counts = {STANDING: 0, FALLEN: 0, UNKNOWN: 0}
        for ps in pin_states:
            counts[ps.state] = counts.get(ps.state, 0) + 1
        counts["total"] = len(pin_states)
        return counts
