from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
FRAME_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FRAME_FILE_TEMPLATE = "frame_{frame_index:06d}.jpg"
SEQUENCE_GROUND_TRUTH_SUFFIX = "_gt.txt"
TRACKED_OBJECT_CLASS_NAME = "object"
TRACKING_BBOX_ANNOTATION_FORMAT = "tracking_bbox"
MOT_CHALLENGE_ANNOTATION_FORMAT = "mot_challenge"
_SEQUENCE_ROW_SPLIT_PATTERN = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class FrameSequenceSource:
    sequence_name: str
    ground_truth_path: Path
    frame_files: tuple[Path, ...]
    ground_truth_format: str = TRACKING_BBOX_ANNOTATION_FORMAT


@dataclass(frozen=True)
class FrameSequenceLayout:
    frames_root: Path
    ground_truth_root: Path
    sources: tuple[FrameSequenceSource, ...]
    frame_directory_names: tuple[str, ...]
    ground_truth_sequence_names: tuple[str, ...]


@dataclass(frozen=True)
class MotGroundTruthRow:
    frame_number: int
    track_id: int
    bbox_xywh_abs: tuple[float, float, float, float]
    source_class_id: int | None
    visibility: float | None


@dataclass(frozen=True)
class PairedVideoJsonSource:
    source_name: str
    annotation_path: Path
    video_path: Path


def build_video_frame_image_rel(video_stem: str, frame_index: int) -> str:
    return (Path("images") / Path(video_stem) / FRAME_FILE_TEMPLATE.format(frame_index=frame_index)).as_posix()


def parse_video_frame_image_rel(image_rel: str) -> tuple[str, int]:
    image_path = Path(image_rel)
    if len(image_path.parts) < 3 or image_path.parts[0] != "images":
        raise ValueError(f"Unsupported video frame image path: {image_rel}")

    frame_name = image_path.name
    if not frame_name.startswith("frame_") or image_path.suffix.lower() != ".jpg":
        raise ValueError(f"Unsupported video frame filename: {image_rel}")

    frame_index = int(image_path.stem.removeprefix("frame_"))
    video_stem = Path(*image_path.parts[1:-1]).as_posix()
    if not video_stem:
        raise ValueError(f"Unsupported video frame image path: {image_rel}")
    return video_stem, frame_index


def standard_annotation_files(dataset_root: Path) -> list[Path]:
    annotations_dir = dataset_root / "annotations"
    if not annotations_dir.is_dir():
        return []
    return sorted(file for file in annotations_dir.rglob("*.txt") if file.is_file())


def discover_video_files(
    dataset_root: Path,
    *,
    max_roots: int | None = None,
    max_files: int | None = None,
) -> tuple[Path, ...]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for root_index, video_root in enumerate(_video_root_candidates(dataset_root)):
        if max_roots is not None and root_index >= max_roots:
            break
        if not video_root.is_dir():
            continue

        try:
            root_files = sorted(
                path
                for path in video_root.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
        except OSError:
            continue

        for video_file in root_files:
            if video_file in seen:
                continue
            seen.add(video_file)
            discovered.append(video_file)
            if max_files is not None and len(discovered) >= max_files:
                return tuple(discovered)

    return tuple(discovered)


def resolve_video_file(dataset_root: Path, video_stem: str) -> Path | None:
    relative_video_path = dataset_root / video_stem
    if relative_video_path.is_file() and relative_video_path.suffix.lower() in VIDEO_EXTENSIONS:
        return relative_video_path

    for extension in sorted(VIDEO_EXTENSIONS):
        candidate = relative_video_path.with_suffix(extension)
        if candidate.is_file():
            return candidate

    exact_matches = [file for file in discover_video_files(dataset_root) if file.stem == video_stem]
    if exact_matches:
        return exact_matches[0]
    return None


def discover_paired_video_json_sources(
    dataset_root: Path,
    *,
    max_sources: int | None = None,
) -> tuple[PairedVideoJsonSource, ...]:
    discovered: list[PairedVideoJsonSource] = []

    try:
        json_files = sorted(path for path in dataset_root.rglob("*.json") if path.is_file())
    except OSError:
        return ()

    for json_file in json_files:
        for extension in sorted(VIDEO_EXTENSIONS):
            video_path = json_file.with_suffix(extension)
            if not video_path.is_file():
                continue
            discovered.append(
                PairedVideoJsonSource(
                    source_name=json_file.relative_to(dataset_root).with_suffix("").as_posix(),
                    annotation_path=json_file,
                    video_path=video_path,
                )
            )
            break
        if max_sources is not None and len(discovered) >= max_sources:
            break

    return tuple(discovered)


def discover_frame_sequence_layout(dataset_root: Path) -> FrameSequenceLayout | None:
    best_layout: FrameSequenceLayout | None = None
    best_score: tuple[int, int, int] | None = None

    candidate_layouts: list[FrameSequenceLayout] = []
    mot_layout = _build_mot_challenge_sequence_layout(dataset_root)
    if mot_layout is not None:
        candidate_layouts.append(mot_layout)

    for frames_root, ground_truth_root in _sequence_root_candidates(dataset_root):
        layout = _build_frame_sequence_layout(frames_root, ground_truth_root)
        if layout is None:
            continue
        candidate_layouts.append(layout)

    for layout in candidate_layouts:
        score = _frame_sequence_layout_score(layout)
        if best_score is None or score > best_score:
            best_layout = layout
            best_score = score

    return best_layout


def _build_frame_sequence_layout(
    frames_root: Path,
    ground_truth_root: Path,
) -> FrameSequenceLayout | None:
    if not frames_root.is_dir() or not ground_truth_root.is_dir():
        return None

    frame_dir_by_name: dict[str, tuple[Path, ...]] = {}
    for frame_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        frame_files = _frame_files(frame_dir)
        if frame_files:
            frame_dir_by_name[frame_dir.name] = frame_files
    ground_truth_files = sorted(
        path
        for path in ground_truth_root.glob(f"*{SEQUENCE_GROUND_TRUTH_SUFFIX}")
        if path.is_file()
    )
    if not frame_dir_by_name or not ground_truth_files:
        return None

    frame_directory_names = tuple(sorted(frame_dir_by_name))
    ground_truth_sequence_names = tuple(
        sorted(gt_file.name.removesuffix(SEQUENCE_GROUND_TRUTH_SUFFIX) for gt_file in ground_truth_files)
    )

    sources: list[FrameSequenceSource] = []
    for ground_truth_file in ground_truth_files:
        sequence_name = ground_truth_file.name.removesuffix(SEQUENCE_GROUND_TRUTH_SUFFIX)
        matching_frame_files = frame_dir_by_name.get(sequence_name)
        if matching_frame_files is None:
            continue
        sources.append(
            FrameSequenceSource(
                sequence_name=sequence_name,
                ground_truth_path=ground_truth_file,
                frame_files=matching_frame_files,
            )
        )

    if not sources:
        return None

    return FrameSequenceLayout(
        frames_root=frames_root,
        ground_truth_root=ground_truth_root,
        sources=tuple(sources),
        frame_directory_names=frame_directory_names,
        ground_truth_sequence_names=ground_truth_sequence_names,
    )


def _build_mot_challenge_sequence_layout(dataset_root: Path) -> FrameSequenceLayout | None:
    sources: list[FrameSequenceSource] = []

    for sequence_dir in _mot_sequence_dir_candidates(dataset_root):
        source = _build_mot_challenge_sequence_source(sequence_dir)
        if source is None:
            continue
        sources.append(source)

    if not sources:
        return None

    sequence_names = tuple(sorted(source.sequence_name for source in sources))
    return FrameSequenceLayout(
        frames_root=dataset_root,
        ground_truth_root=dataset_root,
        sources=tuple(sorted(sources, key=lambda source: source.sequence_name)),
        frame_directory_names=sequence_names,
        ground_truth_sequence_names=sequence_names,
    )


def _build_mot_challenge_sequence_source(sequence_dir: Path) -> FrameSequenceSource | None:
    image_dir_name = _mot_sequence_image_dir_name(sequence_dir)
    image_dir = sequence_dir / image_dir_name
    ground_truth_path = sequence_dir / "gt" / "gt.txt"
    if not image_dir.is_dir() or not ground_truth_path.is_file():
        return None

    frame_files = _frame_files(image_dir)
    if not frame_files:
        return None

    return FrameSequenceSource(
        sequence_name=sequence_dir.name,
        ground_truth_path=ground_truth_path,
        frame_files=frame_files,
        ground_truth_format=MOT_CHALLENGE_ANNOTATION_FORMAT,
    )


def _mot_sequence_dir_candidates(dataset_root: Path) -> tuple[Path, ...]:
    candidates = [dataset_root]
    try:
        candidates.extend(sorted(path for path in dataset_root.iterdir() if path.is_dir()))
    except OSError:
        pass

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _mot_sequence_image_dir_name(sequence_dir: Path) -> str:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        return "img1"

    parser = configparser.ConfigParser()
    try:
        parser.read(seqinfo_path, encoding="utf-8")
    except configparser.Error:
        return "img1"
    return parser.get("Sequence", "imDir", fallback="img1").strip() or "img1"


def _frame_files(frame_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_IMAGE_EXTENSIONS
        )
    )


def _frame_sequence_layout_score(layout: FrameSequenceLayout) -> tuple[int, int, int]:
    matched_sequences = len(layout.sources)
    unmatched_sequences = (
        len(layout.frame_directory_names)
        + len(layout.ground_truth_sequence_names)
        - 2 * matched_sequences
    )
    exact_name_match = int(set(layout.frame_directory_names) == set(layout.ground_truth_sequence_names))
    return (
        matched_sequences,
        -unmatched_sequences,
        exact_name_match,
    )


def parse_tracking_bbox_row(row: str) -> tuple[float, float, float, float] | None:
    tokens = [token for token in _SEQUENCE_ROW_SPLIT_PATTERN.split(row.strip()) if token]
    if len(tokens) != 4:
        raise ValueError("Tracking bbox rows must contain exactly four numeric values")

    x, y, w, h = (float(token) for token in tokens)
    if x < 0 and y < 0 and w <= 0 and h <= 0:
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("Tracking bbox rows must use non-negative x/y and positive w/h")
    return x, y, w, h


def is_valid_tracking_bbox_row(row: str) -> bool:
    try:
        parse_tracking_bbox_row(row)
    except ValueError:
        return False
    return True


def parse_mot_ground_truth_row(row: str) -> MotGroundTruthRow | None:
    tokens = [token for token in _SEQUENCE_ROW_SPLIT_PATTERN.split(row.strip()) if token]
    if len(tokens) < 6:
        raise ValueError("MOT ground-truth rows must contain at least six numeric values")

    frame_number = _parse_mot_int(tokens[0], field_name="frame")
    track_id = _parse_mot_int(tokens[1], field_name="track_id")
    x = float(tokens[2])
    y = float(tokens[3])
    w = float(tokens[4])
    h = float(tokens[5])

    if frame_number <= 0:
        raise ValueError("MOT frame numbers are 1-based and must be positive")
    if w <= 0 or h <= 0:
        raise ValueError("MOT bbox rows must use positive width and height")

    confidence = float(tokens[6]) if len(tokens) >= 7 else 1.0
    if confidence <= 0:
        return None

    source_class_id = _parse_mot_int(tokens[7], field_name="class_id") if len(tokens) >= 8 else None
    if source_class_id is not None and source_class_id < 0:
        source_class_id = None

    visibility = float(tokens[8]) if len(tokens) >= 9 else None

    return MotGroundTruthRow(
        frame_number=frame_number,
        track_id=track_id,
        bbox_xywh_abs=(x, y, w, h),
        source_class_id=source_class_id,
        visibility=visibility,
    )


def is_valid_mot_ground_truth_row(row: str) -> bool:
    try:
        parse_mot_ground_truth_row(row)
    except ValueError:
        return False
    return True


def _parse_mot_int(token: str, *, field_name: str) -> int:
    value = float(token)
    if not value.is_integer():
        raise ValueError(f"MOT {field_name} must be an integer")
    return int(value)


def _sequence_root_candidates(dataset_root: Path) -> tuple[tuple[Path, Path], ...]:
    candidates: list[tuple[Path, Path]] = [
        (dataset_root / "data" / "videos", dataset_root / "data" / "videos_gt"),
        (dataset_root / "videos", dataset_root / "videos_gt"),
    ]

    try:
        child_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    except OSError:
        child_dirs = []

    child_dir_by_name = {path.name: path for path in child_dirs}
    for ground_truth_dir in child_dirs:
        base_name = _ground_truth_base_name(ground_truth_dir.name)
        if base_name is None:
            continue
        frames_dir = child_dir_by_name.get(base_name)
        if frames_dir is None:
            continue
        candidates.append((frames_dir, ground_truth_dir))

    for frames_dir in child_dirs:
        for ground_truth_dir in child_dirs:
            if frames_dir == ground_truth_dir:
                continue
            candidates.append((frames_dir, ground_truth_dir))

    unique_candidates: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _video_root_candidates(dataset_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = [
        dataset_root / "videos",
        dataset_root / "data" / "videos",
        dataset_root,
    ]

    try:
        child_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    except OSError:
        child_dirs = []

    videoish_child_dirs = [path for path in child_dirs if "video" in path.name.lower()]
    other_child_dirs = [path for path in child_dirs if path not in videoish_child_dirs]
    candidates.extend(videoish_child_dirs)
    candidates.extend(other_child_dirs)

    data_root = dataset_root / "data"
    if data_root.is_dir():
        try:
            data_child_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
        except OSError:
            data_child_dirs = []
        videoish_data_child_dirs = [path for path in data_child_dirs if "video" in path.name.lower()]
        other_data_child_dirs = [path for path in data_child_dirs if path not in videoish_data_child_dirs]
        candidates.extend(videoish_data_child_dirs)
        candidates.extend(other_data_child_dirs)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _ground_truth_base_name(directory_name: str) -> str | None:
    for suffix in ("_gt", "_GT", "GT", "gt"):
        if directory_name.endswith(suffix):
            base_name = directory_name.removesuffix(suffix)
            if base_name:
                return base_name
    return None
