"""
Tests for Bowling Score Calculator
===================================
Comprehensive tests for scoring logic including:
- Perfect game (300)
- Gutter game (0)
- All spares
- Mixed games
- 10th frame special cases
- Throw detection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.score_calculator import BowlingScoreCalculator, ThrowDetector
import config


class TestBowlingScoreCalculator:
    """Test standard bowling scoring rules."""

    def setup_method(self):
        self.calc = BowlingScoreCalculator()

    def test_perfect_game(self):
        """Perfect game: 12 strikes = 300."""
        for _ in range(12):
            self.calc.add_throw(10)
        assert self.calc.get_total_score() == 300
        assert self.calc.is_game_complete()

    def test_gutter_game(self):
        """All gutters: 20 throws of 0 = 0."""
        for _ in range(20):
            self.calc.add_throw(0)
        assert self.calc.get_total_score() == 0
        assert self.calc.is_game_complete()

    def test_all_ones(self):
        """All ones: 20 throws of 1 = 20."""
        for _ in range(20):
            self.calc.add_throw(1)
        assert self.calc.get_total_score() == 20

    def test_single_spare(self):
        """Spare in first frame, then all gutters."""
        self.calc.add_throw(5)
        self.calc.add_throw(5)  # Spare
        self.calc.add_throw(3)  # Bonus for spare
        for _ in range(17):
            self.calc.add_throw(0)
        assert self.calc.get_total_score() == 16  # (10+3) + 3 + 0*17

    def test_single_strike(self):
        """Strike in first frame, then all gutters."""
        self.calc.add_throw(10)  # Strike
        self.calc.add_throw(3)   # Bonus 1
        self.calc.add_throw(4)   # Bonus 2
        for _ in range(16):
            self.calc.add_throw(0)
        assert self.calc.get_total_score() == 24  # (10+3+4) + (3+4) + 0*16

    def test_all_spares_with_5s(self):
        """All spares with 5/5 pattern: 10 frames × (10+5) = 150."""
        for _ in range(10):
            self.calc.add_throw(5)
            self.calc.add_throw(5)
        self.calc.add_throw(5)  # Bonus throw in 10th
        assert self.calc.get_total_score() == 150

    def test_10th_frame_strike_bonus(self):
        """Strike in 10th frame gets 2 bonus throws."""
        # 9 frames of 0
        for _ in range(18):
            self.calc.add_throw(0)
        # 10th frame: strike + 2 bonus
        self.calc.add_throw(10)
        self.calc.add_throw(10)
        self.calc.add_throw(10)
        assert self.calc.get_total_score() == 30

    def test_mixed_game(self):
        """Test a realistic mixed game.
        Frame 1: 1, 4 = 5
        Frame 2: 4, 5 = 9
        Frame 3: 6, 4 = spare → 10 + 5 = 15
        Frame 4: 5, 5 = spare → 10 + 10 = 20
        Frame 5: 10 = strike → 10 + 0 + 1 = 11
        Frame 6: 0, 1 = 1
        Frame 7: 7, 3 = spare → 10 + 6 = 16
        Frame 8: 6, 4 = spare → 10 + 10 = 20
        Frame 9: 10 = strike → 10 + 2 + 8 = 20
        Frame 10: 2, 8, 6 = spare + 6 = 16
        Total: 5+9+15+20+11+1+16+20+20+16 = 133
        """
        throws = [1, 4, 4, 5, 6, 4, 5, 5, 10, 0, 1, 7, 3, 6, 4, 10, 2, 8, 6]
        for t in throws:
            self.calc.add_throw(t)
        assert self.calc.get_total_score() == 133

    def test_frame_scores(self):
        """Test that frame-by-frame scores are returned correctly."""
        self.calc.add_throw(3)
        self.calc.add_throw(4)
        scores = self.calc.get_frame_scores()
        assert len(scores) == 1
        assert scores[0]["frame"] == 1
        assert scores[0]["throws"] == [3, 4]
        assert scores[0]["score"] == 7

    def test_reset(self):
        """Test calculator reset for new game."""
        self.calc.add_throw(10)
        self.calc.reset()
        assert self.calc.get_total_score() == 0
        assert len(self.calc.frames) == 0
        assert len(self.calc.rolls) == 0

    def test_invalid_pin_count(self):
        """Test that invalid pin counts raise ValueError."""
        with pytest.raises(ValueError):
            self.calc.add_throw(-1)
        with pytest.raises(ValueError):
            self.calc.add_throw(11)

    def test_str_representation(self):
        """Test scorecard string output."""
        self.calc.add_throw(10)
        self.calc.add_throw(3)
        self.calc.add_throw(4)
        output = str(self.calc)
        assert "BOWLING SCORECARD" in output
        assert "X" in output


class TestThrowDetector:
    """Test throw event detection from pin count sequences."""

    def setup_method(self):
        self.detector = ThrowDetector()

    def test_no_throw_stable(self):
        """No throw when pins remain stable."""
        for _ in range(20):
            result = self.detector.update(10)
            assert not result["throw_detected"]

    def test_throw_detection(self):
        """Detect a throw when pins drop and stabilize."""
        # All pins standing
        for _ in range(5):
            self.detector.update(10)

        # Ball hits — pins drop to 7
        self.detector.update(7)

        # Wait for stabilization
        detected = False
        for _ in range(config.THROW_SETTLE_FRAMES + 5):
            result = self.detector.update(7)
            if result["throw_detected"]:
                detected = True
                assert result["pins_knocked"] == 3
                break

        assert detected, "Throw should have been detected"

    def test_reset_detection(self):
        """Detect pin reset (back to 10)."""
        # After a throw
        self.detector.update(7)
        for _ in range(config.THROW_SETTLE_FRAMES + 1):
            self.detector.update(7)

        # Reset pins
        detected_reset = False
        for _ in range(config.RESET_CONFIRM_FRAMES + 1):
            result = self.detector.update(10)
            if result["reset_detected"]:
                detected_reset = True
                break

        assert detected_reset, "Reset should have been detected"

    def test_reset_state(self):
        """Test detector reset."""
        self.detector.update(5)
        self.detector.reset()
        assert self.detector.prev_standing == config.TOTAL_PINS
        assert self.detector.last_stable_standing == config.TOTAL_PINS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
