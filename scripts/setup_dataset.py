"""
Roboflow Dataset Setup
======================
Helper script for setting up, downloading, and integrating datasets
from Roboflow into the bowling pin detection project.

Supports three workflows:
  1. Download an existing public dataset from Roboflow Universe
  2. Upload extracted frames to a new Roboflow project for annotation
  3. Download your own annotated dataset after annotation is complete

Usage:
    # Step 1: Search for existing datasets
    python scripts/setup_dataset.py --search "bowling pin"

    # Step 2a: Download a public dataset
    python scripts/setup_dataset.py --download --workspace <WORKSPACE> --project <PROJECT> --version <VERSION>

    # Step 2b: Upload your frames to a new Roboflow project
    python scripts/setup_dataset.py --upload --api-key <KEY> --workspace <WORKSPACE> --project <PROJECT>

    # Step 3: Download your annotated dataset
    python scripts/setup_dataset.py --download --api-key <KEY> --workspace <WORKSPACE> --project <PROJECT> --version <VERSION>

    # Step 4: Verify the downloaded dataset structure
    python scripts/setup_dataset.py --verify
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def search_datasets(query: str):
    """Print a URL to search Roboflow Universe for datasets."""
    search_url = f"https://universe.roboflow.com/search?q={query.replace(' ', '%20')}"
    print("\n" + "=" * 65)
    print("  [SEARCH] ROBOFLOW UNIVERSE")
    print("=" * 65)
    print(f"\n  Open this URL in your browser to find datasets:\n")
    print(f"  --> {search_url}\n")
    print("  Recommended search terms:")
    print("    * bowling pin")
    print("    * bowling pin detection")
    print("    * bowling pin standing fallen")
    print("    * ten pin bowling")
    print("\n  When you find a dataset, note the:")
    print("    * Workspace name  (in the URL)")
    print("    * Project name    (in the URL)")
    print("    * Version number  (e.g., 1, 2, 3)")
    print("\n  Then run:")
    print("    python scripts/setup_dataset.py --download \\")
    print("        --workspace <WORKSPACE> --project <PROJECT> --version <VERSION>")
    print("=" * 65)


def download_dataset(api_key: str, workspace: str, project: str,
                     version: int, output_dir: str = None):
    """Download a dataset from Roboflow in YOLOv8 format.

    Args:
        api_key: Roboflow API key (None for public datasets).
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        output_dir: Where to save the dataset.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[ERROR] roboflow package not installed. Run:")
        print("   pip install roboflow")
        sys.exit(1)

    output_dir = output_dir or str(config.DATASET_DIR)

    print("\n" + "=" * 65)
    print("  [DOWNLOAD] DATASET FROM ROBOFLOW")
    print("=" * 65)
    print(f"  Workspace: {workspace}")
    print(f"  Project:   {project}")
    print(f"  Version:   {version}")
    print(f"  Format:    YOLOv8")
    print(f"  Output:    {output_dir}")
    print("=" * 65)

    # Connect to Roboflow
    if api_key:
        rf = Roboflow(api_key=api_key)
    else:
        rf = Roboflow()

    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    # Download in YOLOv8 format
    dataset = ver.download("yolov8", location=output_dir)

    print(f"\n[OK] Dataset downloaded to: {output_dir}")
    print(f"   Format: YOLOv8")

    # Verify and fix structure
    _fix_dataset_structure(output_dir)

    return dataset


def upload_frames(api_key: str, workspace: str, project: str,
                  frames_dir: str = None):
    """Upload extracted frames to a Roboflow project for annotation.

    Args:
        api_key: Roboflow API key.
        workspace: Roboflow workspace name.
        project: Roboflow project name/slug.
        frames_dir: Directory containing extracted frames.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[ERROR] roboflow package not installed. Run:")
        print("   pip install roboflow")
        sys.exit(1)

    frames_dir = frames_dir or str(config.FRAMES_DIR)
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        print(f"\n[ERROR] Frames directory not found: {frames_dir}")
        print("   Extract frames first:")
        print("   python scripts/extract_frames.py --video data/raw/your_video.mp4")
        sys.exit(1)

    # Find image files
    image_files = sorted(
        p for p in frames_path.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )

    if not image_files:
        print(f"\n[ERROR] No image files found in: {frames_dir}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  [UPLOAD] FRAMES TO ROBOFLOW")
    print("=" * 65)
    print(f"  Workspace:   {workspace}")
    print(f"  Project:     {project}")
    print(f"  Frames dir:  {frames_dir}")
    print(f"  Image count: {len(image_files)}")
    print("=" * 65)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    uploaded = 0
    failed = 0
    for i, img_path in enumerate(image_files, 1):
        try:
            proj.upload(str(img_path))
            uploaded += 1
            if i % 10 == 0:
                print(f"  Uploaded {i}/{len(image_files)}...")
        except Exception as e:
            logger.warning(f"Failed to upload {img_path.name}: {e}")
            failed += 1

    print(f"\n[OK] Upload complete: {uploaded} uploaded, {failed} failed")
    print(f"\n  Next steps:")
    print(f"  1. Go to https://app.roboflow.com/{workspace}/{project}")
    print(f"  2. Annotate images with classes: standing_pin, fallen_pin")
    print(f"  3. Generate a version with augmentations")
    print(f"  4. Download with:")
    print(f"     python scripts/setup_dataset.py --download "
          f"--api-key {api_key} --workspace {workspace} "
          f"--project {project} --version 1")


def _fix_dataset_structure(dataset_dir: str):
    """Ensure the dataset follows the expected directory structure.

    Roboflow sometimes nests the dataset in a subdirectory or uses
    different folder names. This normalizes it to match data.yaml.
    """
    dataset_path = Path(dataset_dir)

    # Check if train/valid/test dirs exist at the expected locations
    expected_structure = {
        "images/train": False,
        "images/val": False,
        "labels/train": False,
        "labels/val": False,
    }

    for subdir in expected_structure:
        if (dataset_path / subdir).exists():
            expected_structure[subdir] = True

    # Common Roboflow export structure: train/images, valid/images, test/images
    # We need: images/train, images/val, labels/train, labels/val
    roboflow_dirs = {
        "train": dataset_path / "train",
        "valid": dataset_path / "valid",
        "test": dataset_path / "test",
    }

    needs_restructure = any(d.exists() for d in roboflow_dirs.values())

    if needs_restructure and not all(expected_structure.values()):
        print("\n  [*] Restructuring dataset to match expected layout...")

        for split_name, rf_dir in roboflow_dirs.items():
            if not rf_dir.exists():
                continue

            # Map 'valid' → 'val'
            target_split = "val" if split_name == "valid" else split_name

            # Move images
            rf_images = rf_dir / "images"
            if rf_images.exists():
                target_images = dataset_path / "images" / target_split
                ensure_dir(target_images)
                for f in rf_images.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(target_images / f.name))

            # Move labels
            rf_labels = rf_dir / "labels"
            if rf_labels.exists():
                target_labels = dataset_path / "labels" / target_split
                ensure_dir(target_labels)
                for f in rf_labels.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(target_labels / f.name))

            # Clean up empty Roboflow dirs
            try:
                shutil.rmtree(str(rf_dir))
            except Exception:
                pass

        print("  [OK] Restructured successfully")

    # Update data.yaml to point to correct paths
    _update_data_yaml(dataset_path)


def _update_data_yaml(dataset_path: Path):
    """Update or create data.yaml with correct paths and classes."""
    yaml_path = dataset_path / "data.yaml"

    # Check if Roboflow created its own data.yaml
    rf_yaml = None
    for candidate in [yaml_path, dataset_path / "dataset.yaml"]:
        if candidate.exists():
            rf_yaml = candidate
            break

    # Read existing to check class names
    existing_classes = None
    if rf_yaml and rf_yaml.exists():
        try:
            import yaml
            with open(rf_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    existing_classes = data['names']
                    logger.info(f"Found classes in Roboflow yaml: {existing_classes}")
        except Exception:
            pass

    # Write our standardized data.yaml
    classes = existing_classes or config.CLASS_NAMES
    if isinstance(classes, dict):
        nc = len(classes)
        names_block = "\n".join(f"  {k}: {v}" for k, v in classes.items())
    elif isinstance(classes, list):
        nc = len(classes)
        names_block = "\n".join(f"  {i}: {v}" for i, v in enumerate(classes))
    else:
        nc = config.NUM_CLASSES
        names_block = "\n".join(f"  {i}: {v}" for i, v in enumerate(config.CLASS_NAMES))

    yaml_content = f"""# Bowling Pin Detection Dataset
# Auto-generated by setup_dataset.py

train: ./images/train
val: ./images/val
test: ./images/test

nc: {nc}

names:
{names_block}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"Updated data.yaml at: {yaml_path}")


def verify_dataset(dataset_dir: str = None):
    """Verify the dataset structure and report statistics."""
    dataset_path = Path(dataset_dir or str(config.DATASET_DIR))

    print("\n" + "=" * 65)
    print("  DATASET VERIFICATION")
    print("=" * 65)
    print(f"  Location: {dataset_path}\n")

    issues = []
    stats = {}

    # Check data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        print(f"  [OK] data.yaml found")
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            print(f"     Classes: {data.get('names', 'N/A')}")
            print(f"     NC: {data.get('nc', 'N/A')}")
        except Exception as e:
            print(f"     [WARN] Could not parse: {e}")
    else:
        print(f"  [MISSING] data.yaml NOT FOUND")
        issues.append("Missing data.yaml")

    # Check splits
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        stats[split] = {"images": img_count, "labels": lbl_count}

        if img_count == 0:
            status = "[ ] empty"
            if split != "test":
                issues.append(f"{split} has no images")
        elif img_count != lbl_count:
            status = f"[!] mismatch"
            issues.append(f"{split}: {img_count} images but {lbl_count} labels")
        else:
            status = "[OK]"

        print(f"  {status} {split:5s}: {img_count:4d} images, {lbl_count:4d} labels")

    # Label analysis
    total_labels = 0
    class_counts = {}
    for split in ["train", "val", "test"]:
        lbl_dir = dataset_path / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            try:
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                            total_labels += 1
            except Exception:
                pass

    if total_labels > 0:
        print(f"\n  Annotation Statistics:")
        print(f"     Total annotations: {total_labels}")
        for cls_id, count in sorted(class_counts.items()):
            cls_name = (config.CLASS_NAMES[cls_id]
                       if cls_id < len(config.CLASS_NAMES)
                       else f"class_{cls_id}")
            pct = count / total_labels * 100
            print(f"     {cls_name} (id={cls_id}): {count} ({pct:.1f}%)")

    # Summary
    print(f"\n{'=' * 65}")
    if issues:
        print(f"  [WARN] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"     * {issue}")
    else:
        print(f"  [OK] Dataset looks good!")
        total_images = sum(s["images"] for s in stats.values())
        print(f"     Total images: {total_images}")
        print(f"     Total annotations: {total_labels}")
    print(f"{'=' * 65}\n")

    return len(issues) == 0


def print_full_guide():
    """Print the complete step-by-step Roboflow setup guide."""
    lines = [
        "",
        "=" * 66,
        "  ROBOFLOW DATASET SETUP -- COMPLETE GUIDE",
        "=" * 66,
        "",
        "This guide walks you through creating a bowling pin detection dataset",
        "using Roboflow. There are TWO paths:",
        "",
        "  * PATH A: Use an existing public dataset (fastest)",
        "  * PATH B: Create your own dataset from scratch (most accurate)",
        "",
        "=" * 66,
        "",
        "  PATH A: USE AN EXISTING PUBLIC DATASET",
        "-" * 50,
        "",
        "  Step 1: Browse Roboflow Universe",
        "  ---------------------------------",
        "  Open: https://universe.roboflow.com/search?q=bowling+pin",
        "",
        "  Look for datasets with:",
        "    [+] Bounding box annotations",
        "    [+] Reasonable image count (200+)",
        '    [+] Classes like "pin", "bowling_pin", "standing", "fallen"',
        "",
        "  Step 2: Get project details from the dataset URL",
        "  -------------------------------------------------",
        "  URL format: universe.roboflow.com/<workspace>/<project>/dataset/<version>",
        "",
        "  Step 3: Download the dataset",
        "  ----------------------------",
        "    python scripts/setup_dataset.py --download \\",
        "        --workspace <WORKSPACE> \\",
        "        --project <PROJECT> \\",
        "        --version <VERSION>",
        "",
        "  Step 4: Verify",
        "  --------------",
        "    python scripts/setup_dataset.py --verify",
        "",
        "=" * 66,
        "",
        "  PATH B: CREATE YOUR OWN DATASET",
        "-" * 50,
        "",
        "  Step 1: Collect bowling videos",
        "  ------------------------------",
        "  * Record 5-10 bowling sessions from a side-view angle",
        "  * Include YouTube CC-licensed bowling clips for variety",
        "  * Place videos in: data/raw/",
        "",
        "  Step 2: Extract frames",
        "  ----------------------",
        "    # Extract every 5th frame",
        "    python scripts/extract_frames.py --video data/raw/ --interval 5",
        "",
        "    # Or extract key moments only (motion-based)",
        "    python scripts/extract_frames.py --video data/raw/ --key-moments",
        "",
        "  Step 3: Create a Roboflow account",
        "  ---------------------------------",
        "  * Go to https://app.roboflow.com and sign up (free tier works)",
        "  * Note your API key from Settings > API Keys",
        "",
        "  Step 4: Create a project",
        "  ------------------------",
        '  * Click "Create New Project"',
        "  * Project type: Object Detection",
        "  * Project name: bowling-pin-detection",
        "  * Add classes: standing_pin, fallen_pin",
        "",
        "  Step 5: Upload frames",
        "  ---------------------",
        "    python scripts/setup_dataset.py --upload \\",
        "        --api-key YOUR_API_KEY \\",
        "        --workspace YOUR_WORKSPACE \\",
        "        --project bowling-pin-detection",
        "",
        "  Or drag & drop images at: https://app.roboflow.com",
        "",
        "  Step 6: Annotate images",
        "  -----------------------",
        "  In the Roboflow annotation tool:",
        "",
        "    [GREEN] standing_pin:",
        "      * Pin is upright or nearly upright",
        "      * Draw a TIGHT bounding box around the pin",
        "      * Include partially occluded pins",
        "",
        "    [RED] fallen_pin:",
        "      * Pin is tilted >45 degrees, lying flat, or mid-fall",
        "      * Draw a tight box around visible portion",
        "      * Include pins mid-flight during a strike",
        "",
        "  TIPS:",
        "    * Annotate EVERY visible pin in EVERY frame",
        "    * After ~100 images, use Label Assist to speed up",
        "    * Be consistent -- same rules for every image",
        "    * Include diverse scenarios:",
        "        - Full rack (10 standing)",
        "        - Partial knockdowns (splits)",
        "        - All down (strike)",
        "        - Motion blur during pin fall",
        "",
        "  Step 7: Generate a dataset version",
        "  -----------------------------------",
        "  In Roboflow:",
        '    * Click "Generate" on your project',
        "    * Preprocessing:",
        "        - Auto-Orient: ON",
        "        - Resize: 640x640 (Stretch)",
        "    * Augmentations (recommended):",
        "        - Rotation: +/-15 degrees",
        "        - Brightness: +/-15%",
        "        - Noise: Up to 2%",
        "    * Split: 70% train / 20% valid / 10% test",
        '    * Click "Generate"',
        "",
        "  Step 8: Download the dataset",
        "  ----------------------------",
        "    python scripts/setup_dataset.py --download \\",
        "        --api-key YOUR_API_KEY \\",
        "        --workspace YOUR_WORKSPACE \\",
        "        --project bowling-pin-detection \\",
        "        --version 1",
        "",
        "  Step 9: Verify",
        "  --------------",
        "    python scripts/setup_dataset.py --verify",
        "",
        "  Step 10: Train!",
        "  ---------------",
        "    python scripts/train_model.py --epochs 100 --imgsz 640",
        "",
        "=" * 66,
        "",
        "  ANNOTATION BEST PRACTICES",
        "-" * 50,
        "",
        "  Rule                | Details",
        "  --------------------|------------------------------------",
        "  Tight boxes         | No excess padding around pins",
        "  Every pin           | Don't skip partially hidden pins",
        "  Consistent classes  | standing_pin vs fallen_pin only",
        "  500+ images min     | More data = better accuracy",
        "  Diverse scenarios   | Full rack, splits, strikes, spares",
        "  Various lighting    | Bright, dim, colored lane lights",
        "  Label Assist        | Use after 100 manual annotations",
        "",
        "=" * 66,
        "",
    ]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Roboflow dataset setup for bowling pin detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup guide
  python scripts/setup_dataset.py --guide

  # Search for existing datasets
  python scripts/setup_dataset.py --search "bowling pin"

  # Download a public dataset
  python scripts/setup_dataset.py --download --workspace my-ws --project bowling-pins --version 1

  # Upload frames for annotation
  python scripts/setup_dataset.py --upload --api-key KEY --workspace my-ws --project bowling-pins

  # Verify downloaded dataset
  python scripts/setup_dataset.py --verify
        """,
    )

    # Actions
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--guide", action="store_true",
                        help="Print the complete setup guide.")
    action.add_argument("--search", type=str, metavar="QUERY",
                        help="Search Roboflow Universe for datasets.")
    action.add_argument("--download", action="store_true",
                        help="Download a dataset from Roboflow.")
    action.add_argument("--upload", action="store_true",
                        help="Upload extracted frames to Roboflow.")
    action.add_argument("--verify", action="store_true",
                        help="Verify the local dataset structure.")

    # Roboflow connection
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Roboflow workspace name.")
    parser.add_argument("--project", type=str, default=None,
                        help="Roboflow project name/slug.")
    parser.add_argument("--version", type=int, default=1,
                        help="Dataset version number (default: 1).")

    # Paths
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for downloaded dataset.")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Directory containing frames to upload.")

    args = parser.parse_args()

    # Resolve API key from env if not provided
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")

    if args.guide:
        print_full_guide()

    elif args.search:
        search_datasets(args.search)

    elif args.download:
        if not args.workspace or not args.project:
            parser.error("--download requires --workspace and --project")
        download_dataset(
            api_key=api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=args.output,
        )
        verify_dataset(args.output)

    elif args.upload:
        if not api_key:
            parser.error("--upload requires --api-key or ROBOFLOW_API_KEY env var")
        if not args.workspace or not args.project:
            parser.error("--upload requires --workspace and --project")
        upload_frames(
            api_key=api_key,
            workspace=args.workspace,
            project=args.project,
            frames_dir=args.frames_dir,
        )

    elif args.verify:
        verify_dataset(args.output)


if __name__ == "__main__":
    main()
