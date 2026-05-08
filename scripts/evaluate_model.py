"""
Evaluate Model CLI
==================
Evaluate trained YOLO model on the test set and report metrics.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/best.pt --data data/dataset/data.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model on test set."
    )
    parser.add_argument(
        "--model", type=str, default=str(config.TRAINED_MODEL),
        help=f"Path to trained model weights (default: {config.TRAINED_MODEL}).",
    )
    parser.add_argument(
        "--data", type=str, default=str(config.DATASET_YAML),
        help=f"Path to data.yaml (default: {config.DATASET_YAML}).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMAGE_SIZE,
        help=f"Input image size (default: {config.IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD}).",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    logger.info("Running validation...")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    if hasattr(results, 'box'):
        box = results.box
        print(f"  mAP@0.5:      {box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {box.map:.4f}")

        if hasattr(box, 'maps') and box.maps is not None:
            class_names = model.names if hasattr(model, 'names') else {}
            print("\n  Per-class mAP@0.5:")
            for i, m in enumerate(box.maps):
                name = class_names.get(i, f"class_{i}")
                print(f"    {name}: {m:.4f}")

    print("=" * 60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
