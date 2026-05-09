"""
Run Inference — Full Bowling Analysis Pipeline
===============================================
Orchestrates the complete pipeline: video → detection → tracking →
classification → scoring → annotated output video.

Usage:
    python scripts/run_inference.py --video data/videos/game1.mp4
    python scripts/run_inference.py --video data/videos/game1.mp4 --model models/best.pt
    python scripts/run_inference.py --video data/videos/game1.mp4 --output output/result.mp4
"""

import argparse
import sys
import time
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.detector import PinDetector
from src.pin_classifier import PinClassifier
from src.pin_tracker import PinTracker
from src.score_calculator import BowlingScoreCalculator, ThrowDetector
from src.video_annotator import VideoAnnotator
from src.utils import (
    get_logger, open_video, get_video_properties,
    create_video_writer, ensure_dir,
)

logger = get_logger(__name__)


def run_analysis(
    video_path: str,
    model_path: str = None,
    output_path: str = None,
    confidence: float = None,
    image_size: int = None,
    device: str = "auto",
    show_preview: bool = False,
):
    """Run the complete bowling analysis pipeline on a video.

    Args:
        video_path: Path to input bowling video.
        model_path: Path to trained YOLO weights (.pt).
        output_path: Path for output annotated video.
        confidence: Detection confidence threshold.
        image_size: YOLO input image size.
        device: Compute device ('auto', 'cpu', '0', etc.).
        show_preview: Show live preview window during processing.
    """
    # ── Setup ──────────────────────────────────────────────────
    logger.info(f"{'=' * 60}")
    logger.info(f"  BOWLING ANALYSIS PIPELINE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Input:  {video_path}")
    logger.info(f"  Model:  {model_path or config.TRAINED_MODEL}")
    logger.info(f"  Output: {output_path or 'auto'}")

    # Open video
    cap = open_video(video_path)
    props = get_video_properties(cap)
    logger.info(
        f"  Video:  {props['width']}x{props['height']} @ "
        f"{props['fps']:.1f} FPS, {props['total_frames']} frames"
    )

    # Auto-generate output path if not provided
    if not output_path:
        stem = Path(video_path).stem
        output_path = str(config.ANNOTATED_VIDEO_DIR / f"{stem}_analyzed.mp4")
    ensure_dir(Path(output_path).parent)

    # Create video writer
    writer = create_video_writer(
        output_path, props["width"], props["height"], props["fps"],
    )

    # Initialize components
    detector = PinDetector(
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    detector.load_model()

    classifier = PinClassifier(
        use_temporal_smoothing=True,
    )

    tracker = PinTracker()
    score_calc = BowlingScoreCalculator()
    throw_detector = ThrowDetector()
    annotator = VideoAnnotator()

    # ── Processing Loop ────────────────────────────────────────
    frame_idx = 0
    start_time = time.time()

    pbar = tqdm(total=props["total_frames"], desc="Analyzing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect ALL objects (pins, balls, sweep boards)
        all_detections = detector.detect(frame)

        # 2. Filter: separate pins from non-pin objects
        filtered = classifier.filter_detections(all_detections)
        pin_detections = filtered["pins"]
        has_sweep = len(filtered["sweep"]) > 0

        # 3. Track pins only (assign persistent IDs)
        track_ids = tracker.update(pin_detections)

        # 4. Classify pin states
        pin_states = classifier.classify_with_tracking(pin_detections, track_ids)

        # 5. Count pins
        pin_counts = classifier.get_pin_counts(pin_states)

        # 6. Freeze pin counts while sweep board is active to prevent false drops
        if has_sweep:
            standing_count = throw_detector.prev_standing
            # Update HUD so it doesn't show 0 pins during sweep
            pin_counts["standing"] = standing_count
            pin_counts["fallen"] = config.TOTAL_PINS - standing_count
            pin_counts["total"] = standing_count
        else:
            standing_count = pin_counts.get("standing", 0)

        # 7. Detect throw events and update score
        throw_info = throw_detector.update(standing_count)
        if throw_info["throw_detected"]:
            result = score_calc.add_throw(throw_info["pins_knocked"])
            logger.info(
                f"Frame {frame_idx}: {result.pins_knocked} pins knocked | "
                f"{'STRIKE!' if result.is_strike else ''}"
                f"{'SPARE!' if result.is_spare else ''}"
                f" | Score: {score_calc.get_total_score()}"
            )

        if throw_info["reset_detected"]:
            logger.info(f"Frame {frame_idx}: Pin reset detected")

        # 8. Annotate frame (pass all detections for ball/sweep visualization)
        annotated = annotator.annotate_frame(
            frame=frame,
            pin_states=pin_states,
            frame_number=frame_idx,
            pin_counts=pin_counts,
            score_calculator=score_calc,
            throw_info=throw_info,
            fps=props["fps"],
            extra_detections=filtered["balls"] + filtered["sweep"],
        )

        # 7. Write annotated frame
        writer.write(annotated)

        # Optional: show live preview
        if show_preview:
            preview = cv2.resize(annotated, (960, 540))
            cv2.imshow("Bowling Analysis", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Preview closed by user")
                break

        frame_idx += 1
        pbar.update(1)

    # ── Cleanup ────────────────────────────────────────────────
    pbar.close()
    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_processed = frame_idx / elapsed if elapsed > 0 else 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ANALYSIS COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Frames processed: {frame_idx}")
    logger.info(f"  Processing speed: {fps_processed:.1f} FPS")
    logger.info(f"  Elapsed time:     {elapsed:.1f}s")
    logger.info(f"  Output saved to:  {output_path}")

    # Print final scorecard
    if score_calc.frames:
        print(f"\n{score_calc}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run full bowling analysis pipeline on a video."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input bowling video.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Path to trained YOLO model (default: {config.TRAINED_MODEL}).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for output annotated video (default: auto-generated).",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help=f"Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help=f"YOLO input image size (default: {config.IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: 'auto', 'cpu', '0' for GPU 0, etc.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview window during processing.",
    )
    args = parser.parse_args()

    run_analysis(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        confidence=args.confidence,
        image_size=args.imgsz,
        device=args.device,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
