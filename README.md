# 🎳 Bowling Pin Detection & Scoring System

A computer vision pipeline that detects bowling pins from prerecorded videos,
classifies them as standing or fallen, counts pins, calculates scores, and
generates annotated output videos — all powered by YOLO object detection.

---

## Features

| Feature | Description |
|:---|:---|
| **Pin Detection** | YOLO-based detection of individual bowling pins |
| **State Classification** | Standing vs. fallen classification via model + heuristics |
| **Pin Tracking** | ByteTrack multi-object tracking for persistent pin IDs |
| **Pin Counting** | Automatic standing/fallen pin count per frame |
| **Score Calculation** | Full 10-pin bowling scoring (strikes, spares, 10th frame) |
| **Annotated Video** | Output video with bounding boxes, HUD, and scorecard |

---

## Project Structure

```
cv2/
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── data/
│   ├── raw/                   # Raw bowling videos
│   ├── frames/                # Extracted frames for annotation
│   ├── dataset/               # YOLO-format annotated dataset
│   └── videos/                # Input videos for analysis
├── models/
│   └── best.pt                # Trained YOLO model weights
├── src/
│   ├── detector.py            # YOLO detection wrapper
│   ├── pin_classifier.py      # Standing/fallen classification
│   ├── pin_tracker.py         # ByteTrack multi-object tracking
│   ├── score_calculator.py    # Bowling score engine
│   ├── frame_extractor.py     # Frame extraction for annotation
│   ├── video_annotator.py     # Bounding box & HUD drawing
│   └── utils.py               # Shared helper functions
├── scripts/
│   ├── extract_frames.py      # CLI: extract frames from videos
│   ├── setup_dataset.py       # CLI: Roboflow dataset downloader & fixer
│   ├── train_model.py         # CLI: train YOLO model locally
│   ├── evaluate_model.py      # CLI: evaluate model metrics
│   └── run_inference.py       # CLI: full analysis pipeline
├── tests/                     # Unit tests
└── output/annotated_videos/   # Output annotated videos
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Roboflow)

1. Record bowling videos (side-view angle)
2. Extract frames:
   ```bash
   python scripts/extract_frames.py --video data/raw/game1.mp4 --interval 5
   ```
3. Upload frames to [Roboflow](https://roboflow.com)
4. Annotate with **1 class**: `pin`
5. Download the dataset using the setup script:
   ```bash
   python scripts/setup_dataset.py --download --url "YOUR_ROBOFLOW_URL"
   ```

### 3. Train Model

> **Note:** Training on a CPU is extremely slow. It is highly recommended to train using Google Colab with a free T4 GPU.

**Option A (Colab - Recommended):**
Zip your dataset, upload to Colab, and run the Ultralytics training command. Download the resulting `best.pt` and place it in `models/best.pt`.

**Option B (Local):**
```bash
python scripts/train_model.py --model yolo11m.pt --epochs 100 --imgsz 640
```

### 4. Run Analysis

```bash
python scripts/run_inference.py --video data/videos/game1.mp4 --model models/best.pt
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `CONFIDENCE_THRESHOLD` | 0.25 | Minimum detection confidence |
| `IOU_THRESHOLD` | 0.45 | NMS IoU threshold |
| `IMAGE_SIZE` | 640 | YOLO input size (640 or 1280) |
| `BASE_MODEL` | `yolo11m.pt` | Base pretrained model |
| `STANDING_MIN_ASPECT_RATIO` | 1.5 | Aspect ratio for standing pins |
| `FALLEN_MAX_ASPECT_RATIO` | 1.2 | Aspect ratio for fallen pins |
| `THROW_SETTLE_FRAMES` | 30 | Frames to wait for pins to settle before scoring |

---

## Dataset Annotation Guide

### Classes
- **1 Class**: `pin` (The system uses bounding box aspect-ratio heuristics to determine if a pin is standing or fallen, so you only need to annotate "pin").

### Annotation Rules
1. Draw **tight bounding boxes** around each visible pin.
2. Annotate **every** visible pin, even partially occluded ones.
3. Keep the bounding boxes as tight as possible, as the aspect ratio (height vs width) is used by `pin_classifier.py` to calculate standing vs fallen states.
4. Aim for **500+ annotated images** minimum.

### Recommended Split
- Train: 70% | Validation: 20% | Test: 10%

---

## Tech Stack

- **Detection**: YOLO11 (Ultralytics)
- **Tracking**: ByteTrack (supervision)
- **Vision**: OpenCV, NumPy
- **ML**: PyTorch
- **Dataset**: Roboflow
