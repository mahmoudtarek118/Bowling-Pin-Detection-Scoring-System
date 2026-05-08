"""
Bowling Score Calculator
========================
Implements standard 10-pin bowling scoring rules including
strikes, spares, open frames, and the 10th frame bonus.
Also includes throw-event detection from pin count sequences.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ThrowResult:
    """Result of a single bowling throw."""
    pins_knocked: int        # Pins knocked down in this throw
    standing_pins: int       # Pins still standing after throw
    frame_number: int        # Which bowling frame (1-10)
    throw_in_frame: int      # Which throw within the frame (1, 2, or 3)
    is_strike: bool = False
    is_spare: bool = False


@dataclass
class FrameScore:
    """Score for a single bowling frame."""
    frame_number: int
    throws: List[int] = field(default_factory=list)  # pins knocked per throw
    score: Optional[int] = None          # frame score (None if pending bonus)
    cumulative_score: Optional[int] = None
    is_strike: bool = False
    is_spare: bool = False
    is_complete: bool = False


class BowlingScoreCalculator:
    """Calculate bowling scores from a sequence of throw results.

    Implements standard 10-pin bowling rules:
    - 10 frames per game
    - Strike: all 10 pins on first throw → 10 + next 2 throws
    - Spare: all 10 pins on two throws → 10 + next 1 throw
    - Open: sum of pins knocked down
    - 10th frame: bonus throws for strikes/spares (up to 3 total)
    """

    def __init__(self):
        self.frames: List[FrameScore] = []
        self.rolls: List[int] = []    # flat list of all rolls
        self._current_frame = 1
        self._current_throw_in_frame = 1

    def add_throw(self, pins_knocked: int) -> ThrowResult:
        """Record a throw and update scoring.

        Args:
            pins_knocked: Number of pins knocked down in this throw (0-10).

        Returns:
            ThrowResult with details about this throw.

        Raises:
            ValueError: If pins_knocked is invalid.
        """
        if pins_knocked < 0 or pins_knocked > config.TOTAL_PINS:
            raise ValueError(
                f"Invalid pin count: {pins_knocked}. Must be 0-{config.TOTAL_PINS}."
            )

        self.rolls.append(pins_knocked)

        # Determine frame context
        frame_num = self._current_frame
        throw_num = self._current_throw_in_frame

        is_strike = False
        is_spare = False

        # Ensure frame exists
        if not self.frames or self.frames[-1].frame_number != frame_num:
            self.frames.append(FrameScore(frame_number=frame_num))

        current_frame = self.frames[-1]
        current_frame.throws.append(pins_knocked)

        if frame_num < 10:
            # Normal frames (1-9)
            if throw_num == 1 and pins_knocked == config.TOTAL_PINS:
                # Strike
                is_strike = True
                current_frame.is_strike = True
                current_frame.is_complete = True
                self._current_frame += 1
                self._current_throw_in_frame = 1
            elif throw_num == 2:
                if sum(current_frame.throws) == config.TOTAL_PINS:
                    is_spare = True
                    current_frame.is_spare = True
                current_frame.is_complete = True
                self._current_frame += 1
                self._current_throw_in_frame = 1
            else:
                self._current_throw_in_frame = 2
        else:
            # 10th frame special rules
            if throw_num == 1:
                if pins_knocked == config.TOTAL_PINS:
                    is_strike = True
                    current_frame.is_strike = True
                self._current_throw_in_frame = 2
            elif throw_num == 2:
                if current_frame.is_strike:
                    # First was strike, check second
                    if pins_knocked == config.TOTAL_PINS:
                        is_strike = True
                    self._current_throw_in_frame = 3
                elif sum(current_frame.throws) == config.TOTAL_PINS:
                    is_spare = True
                    current_frame.is_spare = True
                    self._current_throw_in_frame = 3
                else:
                    current_frame.is_complete = True
            elif throw_num == 3:
                if pins_knocked == config.TOTAL_PINS:
                    is_strike = True
                current_frame.is_complete = True

        standing = config.TOTAL_PINS - pins_knocked
        result = ThrowResult(
            pins_knocked=pins_knocked,
            standing_pins=standing,
            frame_number=frame_num,
            throw_in_frame=throw_num,
            is_strike=is_strike,
            is_spare=is_spare,
        )

        # Recalculate all scores
        self._recalculate_scores()

        return result

    def _recalculate_scores(self):
        """Recalculate all frame scores from the rolls list."""
        rolls = self.rolls
        roll_idx = 0
        cumulative = 0

        for frame in self.frames:
            if roll_idx >= len(rolls):
                break

            if frame.frame_number < 10:
                if frame.is_strike:
                    score = 10
                    # Add bonus (next 2 rolls)
                    if roll_idx + 1 < len(rolls):
                        score += rolls[roll_idx + 1]
                    else:
                        frame.score = None
                        frame.cumulative_score = None
                        roll_idx += 1
                        continue
                    if roll_idx + 2 < len(rolls):
                        score += rolls[roll_idx + 2]
                    else:
                        frame.score = None
                        frame.cumulative_score = None
                        roll_idx += 1
                        continue
                    frame.score = score
                    roll_idx += 1
                elif frame.is_spare:
                    score = 10
                    if roll_idx + 2 < len(rolls):
                        score += rolls[roll_idx + 2]
                    else:
                        frame.score = None
                        frame.cumulative_score = None
                        roll_idx += 2
                        continue
                    frame.score = score
                    roll_idx += 2
                else:
                    frame.score = sum(frame.throws)
                    roll_idx += len(frame.throws)
            else:
                # 10th frame: just sum all throws (no bonus lookups)
                frame.score = sum(frame.throws)
                roll_idx += len(frame.throws)

            if frame.score is not None:
                cumulative += frame.score
                frame.cumulative_score = cumulative

    def get_total_score(self) -> Optional[int]:
        """Get the current total score. None if game is incomplete."""
        if not self.frames:
            return 0
        last_scored = None
        for f in self.frames:
            if f.cumulative_score is not None:
                last_scored = f.cumulative_score
        return last_scored or 0

    def get_frame_scores(self) -> List[Dict]:
        """Get detailed score breakdown for all frames."""
        results = []
        for f in self.frames:
            results.append({
                "frame": f.frame_number,
                "throws": f.throws,
                "score": f.score,
                "cumulative": f.cumulative_score,
                "strike": f.is_strike,
                "spare": f.is_spare,
                "complete": f.is_complete,
            })
        return results

    def is_game_complete(self) -> bool:
        """Check if the game is fully complete (10 frames scored)."""
        return (
            len(self.frames) == 10
            and self.frames[-1].is_complete
            and all(f.score is not None for f in self.frames)
        )

    def reset(self):
        """Reset the calculator for a new game."""
        self.frames.clear()
        self.rolls.clear()
        self._current_frame = 1
        self._current_throw_in_frame = 1

    def __str__(self) -> str:
        lines = ["═" * 60, "  BOWLING SCORECARD", "═" * 60]
        for f in self.frames:
            throw_str = " ".join(
                "X" if (i == 0 and v == 10) else
                "/" if (i == 1 and sum(f.throws[:2]) == 10) else
                "-" if v == 0 else str(v)
                for i, v in enumerate(f.throws)
            )
            score_str = str(f.cumulative_score) if f.cumulative_score is not None else "..."
            lines.append(f"  Frame {f.frame_number:2d}: [{throw_str:>6s}]  Score: {score_str}")
        lines.append("═" * 60)
        lines.append(f"  TOTAL: {self.get_total_score()}")
        lines.append("═" * 60)
        return "\n".join(lines)


class ThrowDetector:
    """Detect bowling throw events from pin count time series.

    Monitors the standing pin count across frames and detects
    sudden drops (throws) and resets (new bowling frames).
    """

    def __init__(self):
        self.prev_standing = config.TOTAL_PINS
        self.stable_count = 0
        self.last_stable_standing = config.TOTAL_PINS
        self.throw_detected = False
        self.reset_detected = False
        self._settle_frames = config.THROW_SETTLE_FRAMES
        self._reset_frames = config.RESET_CONFIRM_FRAMES
        self._reset_count = 0

    def update(self, standing_pins: int) -> Dict[str, any]:
        """Update with a new frame's standing pin count.

        Args:
            standing_pins: Number of standing pins detected in current frame.

        Returns:
            Dict with keys:
              - "throw_detected": bool — True if a throw event was confirmed
              - "pins_knocked": int — Pins knocked in this throw (0 if no throw)
              - "reset_detected": bool — True if pins were reset to 10
              - "standing": int — Current standing count
        """
        result = {
            "throw_detected": False,
            "pins_knocked": 0,
            "reset_detected": False,
            "standing": standing_pins,
        }

        # Check for pin reset (all pins back up)
        if standing_pins == config.TOTAL_PINS:
            self._reset_count += 1
            if self._reset_count >= self._reset_frames:
                result["reset_detected"] = True
                self.last_stable_standing = config.TOTAL_PINS
                self._reset_count = 0
                self.throw_detected = False
                logger.debug("Pin reset detected")
        else:
            self._reset_count = 0

        # Check for NEW throw (sudden drop compared to previous frame)
        # Only trigger on the transition, not on every frame below baseline
        if standing_pins < self.prev_standing:
            drop = self.prev_standing - standing_pins
            if drop >= config.MIN_PIN_CHANGE_FOR_THROW:
                self.throw_detected = True
                self.stable_count = 0

        # Check for stabilization after throw
        if standing_pins == self.prev_standing:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.throw_detected and self.stable_count >= self._settle_frames:
            pins_knocked = self.last_stable_standing - standing_pins
            if pins_knocked > 0:
                result["throw_detected"] = True
                result["pins_knocked"] = pins_knocked
                logger.info(
                    f"Throw detected: {pins_knocked} pins knocked "
                    f"({self.last_stable_standing} → {standing_pins})"
                )
                self.last_stable_standing = standing_pins
            self.throw_detected = False
            self.stable_count = 0

        self.prev_standing = standing_pins
        return result

    def reset(self):
        """Reset the throw detector state."""
        self.prev_standing = config.TOTAL_PINS
        self.stable_count = 0
        self.last_stable_standing = config.TOTAL_PINS
        self.throw_detected = False
        self.reset_detected = False
        self._reset_count = 0
