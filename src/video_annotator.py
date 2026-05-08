"""
Video Annotator
===============
Draws detection bounding boxes, pin state labels, and a HUD scoreboard
onto video frames for the annotated output video.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.pin_classifier import PinState, STANDING, FALLEN, UNKNOWN
from src.score_calculator import BowlingScoreCalculator
from src.utils import get_logger, draw_text_with_background

logger = get_logger(__name__)


class VideoAnnotator:
    """Annotate video frames with detection boxes, labels, and HUD overlay."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        frame: np.ndarray,
        pin_states: List[PinState],
        frame_number: int = 0,
        pin_counts: Optional[Dict] = None,
        score_calculator: Optional[BowlingScoreCalculator] = None,
        throw_info: Optional[Dict] = None,
        fps: float = 30.0,
    ) -> np.ndarray:
        """Draw all annotations on a single frame.

        Args:
            frame: BGR image (modified in-place and returned).
            pin_states: List of PinState objects for this frame.
            frame_number: Current video frame number.
            pin_counts: Dict with standing/fallen/total counts.
            score_calculator: Score calculator for displaying scorecard.
            throw_info: Dict from ThrowDetector with throw event info.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # 1. Draw bounding boxes and labels
        annotated = self._draw_detections(annotated, pin_states)

        # 2. Draw HUD panel
        annotated = self._draw_hud(
            annotated, frame_number, pin_counts, score_calculator, throw_info, fps
        )

        # 3. Draw mini pin layout
        if pin_counts:
            annotated = self._draw_pin_layout(annotated, pin_states)

        return annotated

    def _draw_detections(self, frame: np.ndarray,
                         pin_states: List[PinState]) -> np.ndarray:
        """Draw bounding boxes and labels for each detected pin."""
        for ps in pin_states:
            det = ps.detection
            x1, y1, x2, y2 = det.bbox

            # Choose color based on state
            if ps.state == STANDING:
                color = config.COLOR_STANDING
            elif ps.state == FALLEN:
                color = config.COLOR_FALLEN
            else:
                color = config.COLOR_UNKNOWN

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          config.BBOX_THICKNESS)

            # Label text
            label = f"{ps.state} {det.confidence:.0%}"
            if ps.track_id is not None:
                label = f"#{ps.track_id} {label}"

            # Draw label with background
            frame = draw_text_with_background(
                frame, label, (x1, y1 - 5),
                font_scale=config.FONT_SCALE,
                color=config.COLOR_HUD_TEXT,
                bg_color=color,
                thickness=config.FONT_THICKNESS,
                padding=3,
            )

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_number: int,
        pin_counts: Optional[Dict],
        score_calculator: Optional[BowlingScoreCalculator],
        throw_info: Optional[Dict],
        fps: float = 30.0,
    ) -> np.ndarray:
        """Draw the heads-up display panel with stats and score."""
        h, w = frame.shape[:2]
        hud_w = int(w * config.HUD_WIDTH_RATIO)
        hud_h = 220  # Fixed height for HUD
        hud_x = w - hud_w - 10
        hud_y = 10

        # Semi-transparent background
        overlay = np.full((hud_h, hud_w, 3), config.COLOR_HUD_BG, dtype=np.uint8)
        roi = frame[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w]
        if roi.shape[:2] == overlay.shape[:2]:
            blended = cv2.addWeighted(overlay, config.HUD_OPACITY, roi,
                                      1 - config.HUD_OPACITY, 0)
            frame[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w] = blended

        # Draw border
        cv2.rectangle(frame, (hud_x, hud_y),
                       (hud_x + hud_w, hud_y + hud_h),
                       config.COLOR_HUD_ACCENT, 1)

        # Text content
        text_x = hud_x + 10
        text_y = hud_y + 25
        line_height = 24

        # Title
        cv2.putText(frame, "BOWLING ANALYSIS",
                     (text_x, text_y), self.font, 0.55,
                     config.COLOR_HUD_ACCENT, 1, cv2.LINE_AA)
        text_y += line_height + 5

        # Frame number
        cv2.putText(frame, f"Video Frame: {frame_number}",
                     (text_x, text_y), self.font, 0.4,
                     config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)
        text_y += line_height

        # Throw event indicator
        if throw_info and throw_info.get("throw_detected"):
            cv2.putText(frame, f"THROW! -{throw_info['pins_knocked']} pins",
                         (text_x, text_y), self.font, 0.5,
                         (0, 165, 255), 2, cv2.LINE_AA)
            text_y += line_height

        # Score
        if score_calculator:
            score = score_calculator.get_total_score()
            cv2.putText(frame, f"Score: {score}",
                         (text_x, text_y), self.font, 0.55,
                         config.COLOR_HUD_ACCENT, 1, cv2.LINE_AA)
            text_y += line_height

        # Time
        time_sec = int(frame_number / fps) if fps > 0 else 0
        mins, secs = divmod(time_sec, 60)
        time_str = f"Time: {mins:02d}:{secs:02d}"
        cv2.putText(frame, time_str,
                     (text_x, text_y), self.font, 0.45,
                     config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        return frame

    def _draw_pin_layout(self, frame: np.ndarray,
                          pin_states: List[PinState]) -> np.ndarray:
        """Draw a mini 10-pin triangle diagram showing pin states.

        Standard 10-pin layout (viewed from above / front):
            7  8  9  10
              4  5  6
                2  3
                  1
        """
        h, w = frame.shape[:2]
        # Position in bottom-left corner
        layout_x = 15
        layout_y = h - 130
        pin_radius = 8
        spacing = 22

        # Standard pin positions in the triangle
        pin_positions = {
            1: (0, 3),
            2: (-1, 2), 3: (1, 2),
            4: (-2, 1), 5: (0, 1), 6: (2, 1),
            7: (-3, 0), 8: (-1, 0), 9: (1, 0), 10: (3, 0),
        }

        # Background
        bg_w, bg_h = 180, 120
        overlay = np.full((bg_h, bg_w, 3), (30, 30, 30), dtype=np.uint8)
        y1 = max(0, layout_y)
        y2 = min(h, layout_y + bg_h)
        x1 = max(0, layout_x)
        x2 = min(w, layout_x + bg_w)
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            oh = y2 - y1
            ow = x2 - x1
            blended = cv2.addWeighted(overlay[:oh, :ow], 0.7, roi, 0.3, 0)
            frame[y1:y2, x1:x2] = blended

        # Draw "PIN LAYOUT" label
        cv2.putText(frame, "PIN LAYOUT", (layout_x + 5, layout_y + 15),
                     self.font, 0.35, config.COLOR_HUD_ACCENT, 1, cv2.LINE_AA)

        # Determine which pins are standing vs fallen
        # For now, use number of standing/fallen as approximation
        standing_count = sum(1 for ps in pin_states if ps.state == STANDING)
        fallen_count = sum(1 for ps in pin_states if ps.state == FALLEN)

        center_x = layout_x + bg_w // 2
        center_y = layout_y + bg_h // 2 + 10

        for pin_num, (px, py) in pin_positions.items():
            cx = center_x + px * spacing // 2
            cy = center_y + py * spacing // 2

            if pin_num <= standing_count:
                color = config.COLOR_STANDING
                fill = -1  # filled circle
            else:
                color = (80, 80, 80)  # dim gray for fallen
                fill = 1  # outline only

            cv2.circle(frame, (cx, cy), pin_radius, color, fill)
            # Pin number
            cv2.putText(frame, str(pin_num), (cx - 4, cy + 3),
                         self.font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
