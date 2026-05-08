"""
Frame Extractor
===============
Extract frames from bowling videos for dataset annotation.
Saves frames as numbered PNG files that can be uploaded to Roboflow.
"""

import cv2
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_logger, open_video, get_video_properties, ensure_dir

logger = get_logger(__name__)


class FrameExtractor:
    """Extract frames from video files at configurable intervals.

    Attributes:
        interval: Extract every Nth frame.
        output_dir: Directory to save extracted frames.
    """

    def __init__(
        self,
        interval: int = None,
        output_dir: str = None,
    ):
        self.interval = interval or config.FRAME_EXTRACT_INTERVAL
        self.output_dir = Path(output_dir) if output_dir else config.FRAMES_DIR
        ensure_dir(self.output_dir)

    def extract(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        prefix: str = "",
    ) -> int:
        """Extract frames from a video file.

        Args:
            video_path: Path to the input video file.
            max_frames: Maximum number of frames to extract (None = no limit).
            start_frame: Frame index to start extraction from.
            prefix: Filename prefix for extracted frames (e.g. video name).

        Returns:
            Number of frames extracted.
        """
        cap = open_video(video_path)
        props = get_video_properties(cap)

        logger.info(
            f"Video: {video_path} | {props['width']}x{props['height']} | "
            f"{props['fps']:.1f} FPS | {props['total_frames']} total frames"
        )

        # Generate prefix from video filename if not provided
        if not prefix:
            prefix = Path(video_path).stem + "_"

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        extracted = 0
        frame_idx = start_frame

        pbar = tqdm(
            total=min(props["total_frames"] - start_frame, max_frames or float("inf")),
            desc="Extracting frames",
            unit="frame",
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.interval == 0:
                # Save frame
                filename = f"{prefix}frame_{frame_idx:06d}.png"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), frame)
                extracted += 1

                if max_frames and extracted >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        logger.info(f"Extracted {extracted} frames to {self.output_dir}")
        return extracted

    def extract_key_moments(
        self,
        video_path: str,
        prefix: str = "",
        diff_threshold: float = 30.0,
    ) -> int:
        """Extract frames where significant changes occur (pin falls).

        Uses frame differencing to detect moments of high motion,
        which typically correspond to ball impact and pin falls.

        Args:
            video_path: Path to the input video file.
            prefix: Filename prefix for extracted frames.
            diff_threshold: Mean pixel difference threshold to trigger extraction.

        Returns:
            Number of key-moment frames extracted.
        """
        cap = open_video(video_path)
        if not prefix:
            prefix = Path(video_path).stem + "_key_"

        extracted = 0
        frame_idx = 0
        prev_gray = None

        logger.info(f"Extracting key moments from: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = diff.mean()

                if mean_diff > diff_threshold:
                    filename = f"{prefix}frame_{frame_idx:06d}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    extracted += 1
                    logger.debug(f"Key moment at frame {frame_idx} (diff={mean_diff:.1f})")

            prev_gray = gray
            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {extracted} key-moment frames to {self.output_dir}")
        return extracted
