#!/usr/bin/env python3
"""Convert Neovision2 CSV + MPG video files to YOLO format.

Supports two modes:

  Single-pair mode (legacy):
      python scripts/neovision2_to_yolo.py \
          --csv path/to/Neovision2-Training-Heli-001.csv \
          --video path/to/Neovision2-Training-Heli-001.mpg \
          --output /tmp/out

  Batch mode (processes all CSV+MPG pairs in one or two directories):
      python scripts/neovision2_to_yolo.py \
          --training-dir /path/to/neovision2-training-heli \
          --test-dir /path/to/neovision2-test-heli \
          --output /tmp/out

Output is flat in both modes:
    <output>/images/<stem>_frame_XXXXXX.jpg
    <output>/labels/<stem>_frame_XXXXXX.txt
    <output>/classes.txt

Class remapping applied in both modes
--------------------------------------
  Helicopter          → helicopter      (class 1)
  Plane               → fixed_wing      (class 2)
  Cyclist, Car, Truck,
  Tractor-Trailer,
  Boat, Bus           → ground_vehicle  (class 4)
  Person              → person          (class 5)
  Container           → dropped (annotation silently skipped)
  DCR                 → entire frame dropped

Out-of-frame bboxes are clipped to image bounds (correct policy).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Class remapping: source ObjectType (case-sensitive) → (dest_name, class_id)
# None means the annotation is dropped; DROP_FRAME means skip the whole frame.
# ---------------------------------------------------------------------------

DROP_FRAME = object()  # sentinel

CLASS_REMAP: dict[str, tuple[str, int] | None | object] = {
    "Helicopter": ("helicopter", 1),
    "Plane": ("fixed_wing", 2),
    "Cyclist": ("ground_vehicle", 4),
    "Car": ("ground_vehicle", 4),
    "Truck": ("ground_vehicle", 4),
    "Tractor-Trailer": ("ground_vehicle", 4),
    "Boat": ("ground_vehicle", 4),
    "Bus": ("ground_vehicle", 4),   # present in test split only
    "Person": ("person", 5),
    "Container": None,              # drop annotation
    "DCR": DROP_FRAME,              # drop entire frame
}

# classes.txt lines (index = class_id; empty string for unused slots)
_CLASS_NAMES_BY_ID: dict[int, str] = {
    dest_id: name
    for entry in CLASS_REMAP.values()
    if isinstance(entry, tuple)
    for name, dest_id in [entry]
}
_MAX_CLASS_ID = max(_CLASS_NAMES_BY_ID) if _CLASS_NAMES_BY_ID else 0
CLASSES_TXT_LINES: list[str] = [
    _CLASS_NAMES_BY_ID.get(i, "") for i in range(_MAX_CLASS_ID + 1)
]


def quad_to_aabb(row: pd.Series) -> tuple[float, float, float, float]:
    xs = [row["BoundingBox_X1"], row["BoundingBox_X2"], row["BoundingBox_X3"], row["BoundingBox_X4"]]
    ys = [row["BoundingBox_Y1"], row["BoundingBox_Y2"], row["BoundingBox_Y3"], row["BoundingBox_Y4"]]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def probe_video(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in result.stdout.strip().rstrip(",").split(","))
    return width, height


def extract_frames(video_path: Path, images_dir: Path, stem: str, max_frame: int) -> None:
    """Extract frames into images_dir as <stem>_frame_XXXXXX.jpg."""
    images_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-nostats",
            "-i", str(video_path),
            "-frames:v", str(max_frame + 1),
            "-start_number", "0",
            str(images_dir / f"{stem}_frame_%06d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def process_pair(
    csv_path: Path,
    video_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Process one CSV+MPG pair, writing into flat output_dir/images/ and output_dir/labels/.

    Image files:  output_dir/images/<stem>_frame_XXXXXX.jpg
    Label files:  output_dir/labels/<stem>_frame_XXXXXX.txt
    """
    df = pd.read_csv(csv_path)
    width, height = probe_video(video_path)
    print(f"  {video_path.name}  {width}x{height}")

    # Identify frames that must be dropped entirely (contain a DCR annotation)
    drop_frame_ids: set[int] = set()
    for _, row in df.iterrows():
        obj_type = row.get("ObjectType")
        if pd.isna(obj_type):
            continue
        if CLASS_REMAP.get(str(obj_type)) is DROP_FRAME:
            drop_frame_ids.add(int(row["Frame"]))

    by_frame: dict[int, list[str]] = {}
    skipped_ann = 0
    unknown_types: set[str] = set()

    _MISSING = object()

    for _, row in df.iterrows():
        obj_type = row.get("ObjectType")
        if pd.isna(obj_type):
            skipped_ann += 1
            continue

        obj_type_str = str(obj_type)
        frame = int(row["Frame"])

        if frame in drop_frame_ids:
            continue

        mapping = CLASS_REMAP.get(obj_type_str, _MISSING)
        if mapping is _MISSING:
            unknown_types.add(obj_type_str)
            skipped_ann += 1
            continue
        if mapping is None:
            # Container: drop annotation silently
            skipped_ann += 1
            continue
        if mapping is DROP_FRAME:
            # Should not reach here (frames pre-filtered above)
            continue

        _dest_name, class_id = mapping  # type: ignore[misc]

        x, y, w, h = quad_to_aabb(row)
        # Clip to image bounds (correct out-of-frame bbox policy)
        x = max(0.0, x)
        y = max(0.0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            skipped_ann += 1
            continue

        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        nw = w / width
        nh = h / height
        by_frame.setdefault(frame, []).append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if unknown_types:
        print(f"    WARNING: unknown ObjectType values (skipped): {sorted(unknown_types)}", file=sys.stderr)

    stem = video_path.stem

    if df["Frame"].isna().all():
        # Empty CSV — probe video for frame count so we can still extract frames
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0", str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        try:
            max_frame = int(result.stdout.strip()) - 1
        except ValueError:
            print(f"  WARNING: empty CSV and could not probe frame count — skipping {stem}", file=sys.stderr)
            return {"frames_total": 0, "frames_dropped": 0, "annotations_skipped": 0, "label_files_written": 0}
    else:
        max_frame = int(df["Frame"].max())

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {max_frame + 1} frames …")
    extract_frames(video_path, images_dir, stem, max_frame)

    written = 0
    for frame_idx in range(max_frame + 1):
        label_path = labels_dir / f"{stem}_frame_{frame_idx:06d}.txt"
        lines = by_frame.get(frame_idx, [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    return {
        "frames_total": max_frame + 1,
        "frames_dropped": len(drop_frame_ids),
        "annotations_skipped": skipped_ann,
        "label_files_written": written,
    }


def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Return sorted list of (csv_path, mpg_path) for every matching pair."""
    pairs: list[tuple[Path, Path]] = []
    for csv_path in sorted(directory.glob("*.csv")):
        mpg_path = csv_path.with_suffix(".mpg")
        if mpg_path.exists():
            pairs.append((csv_path, mpg_path))
        else:
            print(f"  WARNING: no matching .mpg for {csv_path.name} — skipping", file=sys.stderr)
    return pairs


def write_classes_txt(output_dir: Path) -> None:
    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(CLASSES_TXT_LINES) + "\n", encoding="utf-8")
    print(f"Classes file → {classes_path}")
    for i, name in enumerate(CLASSES_TXT_LINES):
        if name:
            print(f"  [{i}] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neovision2 CSV+MPG → YOLO (single-pair or batch, flat output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single-pair mode
    single = parser.add_argument_group("single-pair mode")
    single.add_argument("--csv", type=Path, help="Path to .csv annotation file")
    single.add_argument("--video", type=Path, help="Path to .mpg video file")

    # Batch mode
    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--training-dir", type=Path, help="Directory containing training CSV+MPG pairs")
    batch.add_argument("--test-dir", type=Path, help="Directory containing test CSV+MPG pairs")

    parser.add_argument("--output", required=True, type=Path, help="Output root directory")
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip a sequence if its first label file already exists in output/labels/",
    )
    args = parser.parse_args()

    use_batch = args.training_dir or args.test_dir
    use_single = args.csv or args.video

    if use_batch and use_single:
        parser.error("Specify either single-pair (--csv/--video) or batch (--training-dir/--test-dir), not both.")
    if not use_batch and not use_single:
        parser.error("Specify either single-pair (--csv/--video) or batch (--training-dir/--test-dir) arguments.")

    if use_single:
        if not args.csv or not args.video:
            parser.error("Both --csv and --video are required for single-pair mode.")
        counters = process_pair(args.csv, args.video, args.output)
        print(
            f"Done. frames={counters['frames_total']} "
            f"dropped_frames={counters['frames_dropped']} "
            f"skipped_annotations={counters['annotations_skipped']}"
        )
        write_classes_txt(args.output)
        return

    # --- Batch mode ---
    splits: list[tuple[str, Path]] = []
    if args.training_dir:
        splits.append(("training", args.training_dir))
    if args.test_dir:
        splits.append(("test", args.test_dir))

    total_pairs = 0
    total_frames = 0
    total_dropped_frames = 0
    total_skipped_ann = 0

    for split_name, split_dir in splits:
        if not split_dir.exists():
            print(f"WARNING: directory does not exist: {split_dir}", file=sys.stderr)
            continue

        pairs = find_pairs(split_dir)
        if not pairs:
            print(f"No CSV+MPG pairs found in {split_dir}")
            continue

        print(f"\n=== {split_name.upper()} ({len(pairs)} pairs) ===")

        for csv_path, mpg_path in pairs:
            # Skip check: look for the frame-0 label file in the flat labels dir
            sentinel = args.output / "labels" / f"{mpg_path.stem}_frame_000000.txt"
            if args.skip_if_exists and sentinel.exists():
                print(f"\n[{split_name}] {csv_path.stem} — already done, skipping")
                continue
            print(f"\n[{split_name}] {csv_path.stem}")
            counters = process_pair(csv_path, mpg_path, args.output)
            total_pairs += 1
            total_frames += counters["frames_total"]
            total_dropped_frames += counters["frames_dropped"]
            total_skipped_ann += counters["annotations_skipped"]
            print(
                f"    frames={counters['frames_total']} "
                f"dropped_frames={counters['frames_dropped']} "
                f"skipped_annotations={counters['annotations_skipped']}"
            )

    print(f"\n=== BATCH COMPLETE ===")
    print(f"  Pairs processed : {total_pairs}")
    print(f"  Total frames    : {total_frames}")
    print(f"  Dropped frames  : {total_dropped_frames} (contained DCR)")
    print(f"  Skipped ann.    : {total_skipped_ann} (Container, out-of-bounds, NaN)")
    write_classes_txt(args.output)


if __name__ == "__main__":
    main()
