"""
Bowling Pin Detection & Scoring — Central Configuration
========================================================
All tuneable parameters, paths, and constants live here.
Modify values here instead of scattering magic numbers throughout the code.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Directory Paths ───────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
DATASET_DIR = DATA_DIR / "dataset"
INPUT_VIDEO_DIR = DATA_DIR / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
ANNOTATED_VIDEO_DIR = OUTPUT_DIR / "annotated_videos"

# ── Dataset Configuration ────────────────────────────────────
DATASET_YAML = DATASET_DIR / "data.yaml"
CLASS_NAMES = ["bowling-ball", "bowling-pins", "sweep board"]
NUM_CLASSES = len(CLASS_NAMES)

# Class indices (must match model/data.yaml ordering)
BALL_CLASS = 0       # bowling-ball
PIN_CLASS = 1        # bowling-pins (standing pins only)
SWEEP_CLASS = 2      # sweep board

# Train / Val / Test split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# ── Model Configuration ──────────────────────────────────────
# Base model to start from (pretrained YOLO11-medium)
BASE_MODEL = "yolo11m.pt"

# Trained model weights path (updated after training)
TRAINED_MODEL = MODELS_DIR / "best.pt"

# Input image size for training & inference
# Use 640 for speed, 1280 for accuracy (especially small / distant pins)
IMAGE_SIZE = 640

# ── Detection Thresholds ─────────────────────────────────────
# Minimum confidence to accept a detection
CONFIDENCE_THRESHOLD = 0.25

# IoU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# Maximum detections per frame (10 pins + some margin)
MAX_DETECTIONS = 20

# Number of consecutive frames a pin must maintain state before confirmed
STATE_STABILITY_FRAMES = 3

# ── Tracking Configuration ────────────────────────────────────
# ByteTrack parameters
TRACK_THRESH = 0.25          # Detection threshold for tracking
TRACK_BUFFER = 30            # Frames to keep lost tracks alive
MATCH_THRESH = 0.8           # IOU matching threshold

# ── Bowling Game Rules ────────────────────────────────────────
TOTAL_PINS = 10
TOTAL_FRAMES = 10            # Standard 10-frame bowling game
MAX_THROWS_PER_FRAME = 2     # Normal frames
MAX_THROWS_10TH_FRAME = 3    # 10th frame (bonus throws)

# ── Throw Detection ──────────────────────────────────────────
# Minimum pin-count drop to register as a throw event
MIN_PIN_CHANGE_FOR_THROW = 1

# Frames of stable pin count needed to confirm throw is complete
THROW_SETTLE_FRAMES = 30

# Frames of all-10-pins needed to confirm a reset
RESET_CONFIRM_FRAMES = 5

# ── Video Output Configuration ────────────────────────────────
OUTPUT_VIDEO_CODEC = "mp4v"   # FourCC codec (mp4v, XVID, avc1)
OUTPUT_VIDEO_FPS = 30.0       # Fallback FPS if source FPS unavailable

# ── Annotation Colors (BGR for OpenCV) ────────────────────────
COLOR_STANDING = (0, 255, 0)      # Green — standing pins
COLOR_BALL = (0, 165, 255)        # Orange — bowling ball
COLOR_SWEEP = (255, 200, 0)       # Cyan — sweep board
COLOR_HUD_BG = (20, 20, 20)      # Dark gray
COLOR_HUD_TEXT = (255, 255, 255)  # White
COLOR_HUD_ACCENT = (0, 200, 255) # Orange-ish

# Bounding box / text styling
BBOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# HUD panel dimensions (relative to frame)
HUD_WIDTH_RATIO = 0.25       # 25% of frame width
HUD_OPACITY = 0.7

# ── Frame Extraction ──────────────────────────────────────────
# Extract every Nth frame from video
FRAME_EXTRACT_INTERVAL = 5

# ── Training Hyperparameters ──────────────────────────────────
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 16
TRAINING_PATIENCE = 20       # Early stopping patience
TRAINING_OPTIMIZER = "AdamW"
TRAINING_LR = 0.001

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL = "INFO"
