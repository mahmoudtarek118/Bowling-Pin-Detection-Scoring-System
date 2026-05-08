"""
Train Model CLI
===============
Train YOLO model on the bowling pin dataset.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --model yolo11m.pt --epochs 150 --imgsz 1280
    python scripts/train_model.py --resume runs/train/bowling_pins/weights/last.pt
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
        description="Train YOLO model for bowling pin detection."
    )
    parser.add_argument(
        "--model", type=str, default=config.BASE_MODEL,
        help=f"Base model to start from (default: {config.BASE_MODEL}).",
    )
    parser.add_argument(
        "--data", type=str, default=str(config.DATASET_YAML),
        help=f"Path to data.yaml (default: {config.DATASET_YAML}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.TRAINING_EPOCHS,
        help=f"Training epochs (default: {config.TRAINING_EPOCHS}).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMAGE_SIZE,
        help=f"Input image size (default: {config.IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--batch", type=int, default=config.TRAINING_BATCH_SIZE,
        help=f"Batch size (default: {config.TRAINING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to train on ('0' for GPU, 'cpu' for CPU).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to last.pt to resume training from.",
    )
    parser.add_argument(
        "--name", type=str, default="bowling_pins",
        help="Training run name.",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    # Load model
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        logger.info(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    # Verify dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(
            f"Dataset config not found: {data_path}\n"
            f"Please prepare your dataset first. See README.md for instructions."
        )
        sys.exit(1)

    logger.info(f"Starting training...")
    logger.info(f"  Model:   {args.model}")
    logger.info(f"  Data:    {args.data}")
    logger.info(f"  Epochs:  {args.epochs}")
    logger.info(f"  ImgSize: {args.imgsz}")
    logger.info(f"  Batch:   {args.batch}")
    logger.info(f"  Device:  {args.device}")

    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=config.TRAINING_PATIENCE,
        optimizer=config.TRAINING_OPTIMIZER,
        lr0=config.TRAINING_LR,
        augment=True,
        device=args.device,
        project="runs/train",
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # Copy best weights to models directory
    best_weights = Path("runs/train") / args.name / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        dest = config.MODELS_DIR / "best.pt"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_weights), str(dest))
        logger.info(f"Best model saved to: {dest}")

    logger.info("Training complete!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
