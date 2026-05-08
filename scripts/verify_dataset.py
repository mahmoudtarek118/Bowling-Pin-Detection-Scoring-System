"""Quick script to verify dataset placement."""
import os
from pathlib import Path

base = Path(__file__).resolve().parent.parent / "data" / "dataset"

dirs = [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val", "labels/test",
]

print("=== Dataset Verification ===\n")
for d in dirs:
    p = base / d
    if p.is_dir():
        count = len([f for f in p.iterdir() if f.is_file()])
        print(f"  {d:20s} : {count} files")
    else:
        print(f"  {d:20s} : MISSING!")

# Check data.yaml exists
yaml_path = base / "data.yaml"
print(f"\n  data.yaml            : {'OK' if yaml_path.exists() else 'MISSING!'}")
print()
