"""
Extract Frames CLI
==================
Extract frames from bowling videos for dataset annotation.

Usage:
    python scripts/extract_frames.py --video data/raw/game1.mp4
    python scripts/extract_frames.py --video data/raw/game1.mp4 --interval 10 --max 200
    python scripts/extract_frames.py --video data/raw/game1.mp4 --key-moments
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.frame_extractor import FrameExtractor
from src.utils import get_logger, get_video_files

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from bowling videos for annotation."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file, or directory of videos.",
    )
    parser.add_argument(
        "--output", type=str, default=str(config.FRAMES_DIR),
        help=f"Output directory for extracted frames (default: {config.FRAMES_DIR}).",
    )
    parser.add_argument(
        "--interval", type=int, default=config.FRAME_EXTRACT_INTERVAL,
        help=f"Extract every Nth frame (default: {config.FRAME_EXTRACT_INTERVAL}).",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of frames to extract.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Frame index to start extraction from.",
    )
    parser.add_argument(
        "--key-moments", action="store_true",
        help="Extract frames with significant motion (pin falls) only.",
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=30.0,
        help="Motion diff threshold for key-moment extraction (default: 30.0).",
    )
    args = parser.parse_args()

    extractor = FrameExtractor(
        interval=args.interval,
        output_dir=args.output,
    )

    video_path = Path(args.video)
    if video_path.is_dir():
        videos = get_video_files(str(video_path))
        logger.info(f"Found {len(videos)} videos in {video_path}")
        total = 0
        for vp in videos:
            if args.key_moments:
                count = extractor.extract_key_moments(
                    str(vp), diff_threshold=args.diff_threshold,
                )
            else:
                count = extractor.extract(
                    str(vp), max_frames=args.max, start_frame=args.start,
                )
            total += count
        logger.info(f"Total frames extracted: {total}")
    else:
        if args.key_moments:
            count = extractor.extract_key_moments(
                str(video_path), diff_threshold=args.diff_threshold,
            )
        else:
            count = extractor.extract(
                str(video_path), max_frames=args.max, start_frame=args.start,
            )
        logger.info(f"Frames extracted: {count}")


if __name__ == "__main__":
    main()
