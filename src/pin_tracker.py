"""
Pin Tracker
===========
Multi-object tracking for bowling pins using ByteTrack.
Assigns persistent IDs to detected pins across video frames.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.detector import Detection
from src.utils import get_logger

logger = get_logger(__name__)


class PinTracker:
    """Track bowling pins across frames using ByteTrack via supervision.

    Assigns persistent IDs to each detected pin so we can monitor
    state changes (standing → fallen) over time.
    """

    def __init__(
        self,
        track_thresh: float = None,
        track_buffer: int = None,
        match_thresh: float = None,
    ):
        self.track_thresh = track_thresh or config.TRACK_THRESH
        self.track_buffer = track_buffer or config.TRACK_BUFFER
        self.match_thresh = match_thresh or config.MATCH_THRESH
        self._tracker = None
        self._initialized = False

    def _init_tracker(self):
        """Lazy-initialize the ByteTrack tracker."""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30,
            )
            self._initialized = True
            logger.info("ByteTrack tracker initialized")
        except ImportError:
            logger.warning(
                "supervision library not installed. "
                "Tracking disabled — using sequential IDs instead."
            )
            self._initialized = False

    def update(self, detections: List[Detection]) -> List[int]:
        """Update tracker with new detections and return persistent track IDs.

        Args:
            detections: List of Detection objects for the current frame.

        Returns:
            List of track IDs corresponding to each detection.
            If tracking is unavailable, returns sequential indices.
        """
        if not self._initialized:
            self._init_tracker()

        if not detections:
            return []

        if self._tracker is None:
            # Fallback: no tracking, just assign sequential IDs
            return list(range(len(detections)))

        try:
            import supervision as sv

            # Convert detections to supervision format
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

            # Update tracker
            tracked = self._tracker.update_with_detections(sv_detections)

            # Map tracked detections back to original order
            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                track_ids = tracked.tracker_id.tolist()
            else:
                track_ids = list(range(len(detections)))

            return track_ids

        except Exception as e:
            logger.warning(f"Tracking failed: {e}. Using sequential IDs.")
            return list(range(len(detections)))

    def reset(self):
        """Reset the tracker state (call between videos/games)."""
        self._tracker = None
        self._initialized = False
        logger.info("Tracker reset")
