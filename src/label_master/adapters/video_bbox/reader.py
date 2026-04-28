from __future__ import annotations

from bisect import bisect_left
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, TypeVar

from PIL import Image

from label_master.adapters.custom.common import (
    annotation_files_for_spec,
    build_image_rel_path,
    resolve_spec_video_file,
    split_row_tokens,
)
from label_master.adapters.video_bbox.common import (
    FRAME_FILE_TEMPLATE,
    MOT_CHALLENGE_ANNOTATION_FORMAT,
    TRACKED_OBJECT_CLASS_NAME,
    FrameSequenceLayout,
    build_video_frame_image_rel,
    discover_paired_video_json_sources,
    discover_frame_sequence_layout,
    discover_video_files,
    parse_mot_ground_truth_row,
    parse_tracking_bbox_row,
    parse_video_frame_image_rel,
    resolve_video_file,
    standard_annotation_files,
)
from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    Severity,
    SourceFormat,
    SourceMetadata,
    WarningEvent,
)
from label_master.core.domain.value_objects import ConversionError, ValidationError
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter
from label_master.format_specs.registry import (
    FormatSpec,
    TokenizedVideoParserSpec,
    resolve_builtin_format_spec,
)
from label_master.infra.filesystem import ensure_directory, safe_resolve

FrameMaterializationProgressCallback = Callable[[str, int, int], None]
_T = TypeVar("_T")


@dataclass(frozen=True)
class _PendingAnnotation:
    annotation_id: str
    image_id: str
    class_name: str
    bbox_xywh_abs: tuple[float, float, float, float]
    attributes: dict[str, str | int | float | bool | None] | None = None


@lru_cache(maxsize=1)
def _video_bbox_spec() -> FormatSpec | None:
    return resolve_builtin_format_spec("video_bbox")


def _tokenized_video_bbox_parser() -> TokenizedVideoParserSpec | None:
    spec = _video_bbox_spec()
    if spec is None or not isinstance(spec.parser, TokenizedVideoParserSpec):
        return None
    return spec.parser


def _standard_annotation_files(dataset_root: Path) -> list[Path]:
    spec = _video_bbox_spec()
    if spec is not None and isinstance(spec.parser, TokenizedVideoParserSpec):
        return annotation_files_for_spec(dataset_root, spec)
    return standard_annotation_files(dataset_root)


def _find_video_file(dataset_root: Path, video_stem: str) -> Path:
    resolved = resolve_video_file(dataset_root, video_stem)
    if resolved is not None:
        return resolved

    raise ValidationError(
        f"No source video found for annotation file stem: {video_stem}",
        context={
            "candidate_video_roots": ",".join(
                sorted({str(path.parent) for path in discover_video_files(dataset_root, max_roots=12)})
            )
        },
    )


def _find_video_file_or_none(dataset_root: Path, video_stem: str) -> Path | None:
    try:
        return _find_video_file(dataset_root, video_stem)
    except ValidationError:
        return None


def _sequence_source_frame_files(
    dataset_root: Path,
    video_stem: str,
) -> tuple[Path, ...] | None:
    sequence_layout = discover_frame_sequence_layout(dataset_root)
    if sequence_layout is None:
        return None

    for source in sequence_layout.sources:
        if source.sequence_name == video_stem:
            return source.frame_files
    return None


def _require_binary(name: str, *, error_factory: type[ValidationError] | type[ConversionError]) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise error_factory(f"Required binary is unavailable: {name}")
    return binary


def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    ffprobe = _require_binary("ffprobe", error_factory=ValidationError)
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValidationError(
            f"Unable to probe video metadata: {video_path.name}",
            context={"stderr": completed.stderr.strip()},
        )

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid ffprobe metadata payload for: {video_path.name}") from exc

    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ValidationError(f"Video metadata missing stream data: {video_path.name}")

    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValidationError(f"Video metadata stream must be an object: {video_path.name}")

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValidationError(f"Video dimensions must be positive: {video_path.name}")
    return width, height


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as opened:
            width = int(opened.width)
            height = int(opened.height)
    except OSError as exc:
        raise ValidationError(f"Video frame image could not be opened: {image_path}") from exc

    if width <= 0 or height <= 0:
        raise ValidationError(f"Video frame image dimensions must be positive: {image_path}")
    return width, height


def _clip_bbox_to_image_bounds(
    bbox_xywh_abs: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox_xywh_abs
    clipped_x = max(0.0, x)
    clipped_y = max(0.0, y)
    clipped_right = min(float(image_width), x + w)
    clipped_bottom = min(float(image_height), y + h)
    clipped_w = clipped_right - clipped_x
    clipped_h = clipped_bottom - clipped_y
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return clipped_x, clipped_y, clipped_w, clipped_h


def load_video_bbox_preview_image(dataset_root: Path, image_rel_path: str) -> Image.Image:
    video_stem, frame_index = parse_video_frame_image_rel(image_rel_path)
    if frame_index < 0:
        raise ValidationError(f"Video preview frame index must be non-negative: {image_rel_path}")

    frame_files = _sequence_source_frame_files(dataset_root, video_stem)
    if frame_files is not None:
        if frame_index >= len(frame_files):
            raise ValidationError(
                f"Video preview frame index is outside the available still-frame sequence: {image_rel_path}"
            )
        try:
            with Image.open(frame_files[frame_index]) as opened:
                return opened.convert("RGB").copy()
        except OSError as exc:
            raise ValidationError(f"Video preview image could not be opened: {frame_files[frame_index]}") from exc

    video_path = _find_video_file_or_none(dataset_root, video_stem)
    if video_path is None:
        raise ValidationError(f"Video preview source could not be resolved: {image_rel_path}")

    ffmpeg = _require_binary("ffmpeg", error_factory=ValidationError)
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        raise ValidationError(
            f"Video preview frame could not be extracted: {image_rel_path}",
            context={"stderr": completed.stderr.decode('utf-8', errors='replace').strip()},
        )

    try:
        with Image.open(BytesIO(completed.stdout)) as opened:
            return opened.convert("RGB").copy()
    except OSError as exc:
        raise ValidationError(f"Video preview frame payload could not be decoded: {image_rel_path}") from exc


def _parse_annotation_row(
    row: str,
    *,
    annotation_rel: Path,
    line_no: int,
    video_stem: str,
) -> tuple[ImageRecord, list[_PendingAnnotation]]:
    tokens = row.split()
    if len(tokens) < 2:
        raise ValidationError(
            f"Invalid video annotation row at {annotation_rel.as_posix()}:{line_no}",
            context={"row": row},
        )

    try:
        frame_index = int(tokens[0])
        instance_count = int(tokens[1])
    except ValueError as exc:
        raise ValidationError(
            f"Video annotation row must start with frame_index and instance_count: "
            f"{annotation_rel.as_posix()}:{line_no}"
        ) from exc

    if frame_index < 0 or instance_count < 0:
        raise ValidationError(
            f"Video annotation row cannot use negative frame or instance counts: "
            f"{annotation_rel.as_posix()}:{line_no}"
        )

    expected_tokens = 2 + instance_count * 5
    if len(tokens) != expected_tokens:
        raise ValidationError(
            f"Video annotation row token count mismatch at {annotation_rel.as_posix()}:{line_no}",
            context={"expected_tokens": str(expected_tokens), "actual_tokens": str(len(tokens))},
        )

    image_id = f"{video_stem}:{frame_index:06d}"
    image = ImageRecord(
        image_id=image_id,
        file_name=build_video_frame_image_rel(video_stem, frame_index),
        width=1,
        height=1,
    )

    pending: list[_PendingAnnotation] = []
    cursor = 2
    for instance_index in range(instance_count):
        try:
            x = float(tokens[cursor])
            y = float(tokens[cursor + 1])
            w = float(tokens[cursor + 2])
            h = float(tokens[cursor + 3])
        except ValueError as exc:
            raise ValidationError(
                f"Video annotation bbox fields must be numeric: {annotation_rel.as_posix()}:{line_no}"
            ) from exc

        class_name = tokens[cursor + 4]
        if w <= 0 or h <= 0:
            raise ValidationError(
                f"Video annotation bbox must use positive width/height: "
                f"{annotation_rel.as_posix()}:{line_no}"
            )
        if not class_name:
            raise ValidationError(
                f"Video annotation class name is required: {annotation_rel.as_posix()}:{line_no}"
            )

        pending.append(
            _PendingAnnotation(
                annotation_id=f"{annotation_rel.as_posix()}:{line_no:06d}:{instance_index + 1:03d}",
                image_id=image_id,
                class_name=class_name,
                bbox_xywh_abs=(x, y, w, h),
            )
        )
        cursor += 5

    return image, pending


def _parse_tokenized_annotation_row(
    row: str,
    *,
    parser: TokenizedVideoParserSpec,
    spec: FormatSpec,
    annotation_rel: Path,
    line_no: int,
    video_stem: str,
) -> tuple[ImageRecord, list[_PendingAnnotation]]:
    tokens = split_row_tokens(row, delimiter=parser.row_format.delimiter)
    header_width = max(parser.row_format.frame_index_field, parser.row_format.object_count_field)
    if len(tokens) < header_width:
        raise ValidationError(
            f"Invalid video annotation row at {annotation_rel.as_posix()}:{line_no}",
            context={"row": row},
        )

    try:
        frame_index = int(tokens[parser.row_format.frame_index_field - 1]) - parser.row_format.frame_index_base
        instance_count = int(tokens[parser.row_format.object_count_field - 1])
    except ValueError as exc:
        raise ValidationError(
            f"Video annotation row must start with frame_index and instance_count: "
            f"{annotation_rel.as_posix()}:{line_no}"
        ) from exc

    if frame_index < 0 or instance_count < 0:
        raise ValidationError(
            f"Video annotation row cannot use negative frame or instance counts: "
            f"{annotation_rel.as_posix()}:{line_no}"
        )

    expected_tokens = header_width + instance_count * parser.row_format.object_group_size
    if len(tokens) != expected_tokens:
        raise ValidationError(
            f"Video annotation row token count mismatch at {annotation_rel.as_posix()}:{line_no}",
            context={"expected_tokens": str(expected_tokens), "actual_tokens": str(len(tokens))},
        )

    image_id = f"{video_stem}:{frame_index:06d}"
    image = ImageRecord(
        image_id=image_id,
        file_name=build_image_rel_path(spec, video_stem=video_stem, frame_index=frame_index),
        width=1,
        height=1,
    )

    pending: list[_PendingAnnotation] = []
    cursor = header_width
    for instance_index in range(instance_count):
        try:
            xmin = float(tokens[cursor + parser.row_format.object_fields.xmin - 1])
            ymin = float(tokens[cursor + parser.row_format.object_fields.ymin - 1])
            width = float(tokens[cursor + parser.row_format.object_fields.width - 1])
            height = float(tokens[cursor + parser.row_format.object_fields.height - 1])
        except ValueError as exc:
            raise ValidationError(
                f"Video annotation bbox fields must be numeric: {annotation_rel.as_posix()}:{line_no}"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValidationError(
                f"Video annotation bbox must use positive width/height: "
                f"{annotation_rel.as_posix()}:{line_no}"
            )

        class_name = ""
        if parser.row_format.object_fields.class_name is not None:
            class_name = tokens[cursor + parser.row_format.object_fields.class_name - 1].strip()
        if parser.row_format.object_fields.class_id is not None:
            try:
                class_id = int(tokens[cursor + parser.row_format.object_fields.class_id - 1])
            except ValueError as exc:
                raise ValidationError(
                    f"Video annotation class_id must be an integer: {annotation_rel.as_posix()}:{line_no}"
                ) from exc
            if not class_name:
                class_name = f"class_{class_id}"

        if not class_name:
            raise ValidationError(
                f"Video annotation class name is required: {annotation_rel.as_posix()}:{line_no}"
            )

        pending.append(
            _PendingAnnotation(
                annotation_id=f"{annotation_rel.as_posix()}:{line_no:06d}:{instance_index + 1:03d}",
                image_id=image_id,
                class_name=class_name,
                bbox_xywh_abs=(xmin, ymin, width, height),
            )
        )
        cursor += parser.row_format.object_group_size

    return image, pending


def _sample_evenly(items: list[_T], limit: int) -> list[_T]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]

    selected_indices: list[int] = []
    seen: set[int] = set()
    last_index = len(items) - 1
    for position in range(limit):
        index = round(position * last_index / (limit - 1))
        if index in seen:
            continue
        seen.add(index)
        selected_indices.append(index)

    if len(selected_indices) < limit:
        for index in range(len(items)):
            if index in seen:
                continue
            seen.add(index)
            selected_indices.append(index)
            if len(selected_indices) == limit:
                break

    return [items[index] for index in selected_indices]


def _parse_paired_video_json_payload(
    annotation_path: Path,
) -> tuple[list[bool], list[object]]:
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"Unable to read paired video annotation file: {annotation_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid paired video annotation JSON: {annotation_path}") from exc

    if not isinstance(payload, dict):
        raise ValidationError(f"Paired video annotation payload must be an object: {annotation_path}")

    exist_raw = payload.get("exist")
    gt_rect_raw = payload.get("gt_rect")
    if not isinstance(exist_raw, list) or not isinstance(gt_rect_raw, list):
        raise ValidationError(
            f"Paired video annotation payload must include list fields 'exist' and 'gt_rect': {annotation_path}"
        )
    if len(exist_raw) != len(gt_rect_raw):
        raise ValidationError(
            f"Paired video annotation payload must align 'exist' and 'gt_rect' lengths: {annotation_path}"
        )

    exist: list[bool] = []
    for frame_index, value in enumerate(exist_raw):
        if isinstance(value, bool):
            exist.append(value)
            continue
        if isinstance(value, (int, float)):
            exist.append(bool(value))
            continue
        raise ValidationError(
            f"Paired video annotation 'exist' entries must be boolean-like: {annotation_path}",
            context={"frame_index": str(frame_index)},
        )

    return exist, list(gt_rect_raw)


def _parse_paired_video_json_bbox(
    rect: object,
    *,
    annotation_rel: Path,
    frame_index: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(rect, list):
        return None
    if len(rect) == 0:
        return None
    if len(rect) != 4:
        raise ValidationError(
            f"Paired video annotation bbox must contain four numeric values: "
            f"{annotation_rel.as_posix()}:{frame_index + 1}"
        )
    try:
        x, y, w, h = (float(value) for value in rect)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Paired video annotation bbox must be numeric: {annotation_rel.as_posix()}:{frame_index + 1}"
        ) from exc
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _paired_video_skip_warnings(
    skipped_annotations: dict[str, list[str]],
) -> list[WarningEvent]:
    if not skipped_annotations:
        return []

    warnings: list[WarningEvent] = []
    for source_file, reasons in sorted(skipped_annotations.items()):
        unique_reasons = sorted(set(reasons))
        sample_reasons = unique_reasons[:3]
        reason_summary = "; ".join(sample_reasons)
        if len(unique_reasons) > len(sample_reasons):
            reason_summary += f"; +{len(unique_reasons) - len(sample_reasons)} more reason(s)"
        warnings.append(
            WarningEvent(
                code="video_bbox_annotations_skipped",
                message=(
                    f"Skipped {len(reasons)} invalid paired-video annotation frame(s) from {source_file}. "
                    f"Examples: {reason_summary}."
                ),
                severity=Severity.WARNING,
                context={
                    "source_file": source_file,
                    "reason": reason_summary,
                    "skipped_frames": str(len(reasons)),
                },
            )
        )
    return warnings


def _build_dataset(
    *,
    dataset_root: Path,
    images_by_id: dict[str, ImageRecord],
    pending_annotations: list[_PendingAnnotation],
    details: dict[str, str] | None = None,
    warnings: list[WarningEvent] | None = None,
) -> AnnotationDataset:
    class_id_by_name = {
        name: index for index, name in enumerate(sorted({annotation.class_name for annotation in pending_annotations}))
    }
    categories = {
        class_id: CategoryRecord(class_id=class_id, name=name)
        for name, class_id in sorted(class_id_by_name.items(), key=lambda item: item[1])
    }

    annotations = [
        AnnotationRecord(
            annotation_id=annotation.annotation_id,
            image_id=annotation.image_id,
            class_id=class_id_by_name[annotation.class_name],
            bbox_xywh_abs=annotation.bbox_xywh_abs,
            attributes=annotation.attributes or {},
        )
        for annotation in pending_annotations
    ]

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.VIDEO_BBOX,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda ann: ann.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="video_bbox_reader",
            details=details or {},
        ),
        warnings=warnings or [],
    )


def _read_mot_frame_sequence_dataset(
    dataset_root: Path,
    sequence_layout: FrameSequenceLayout,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    images_by_id: dict[str, ImageRecord] = {}
    pending_annotations: list[_PendingAnnotation] = []

    for source in sequence_layout.sources:
        if not source.frame_files:
            raise ValidationError(f"No frame images found for MOT sequence: {source.sequence_name}")

        width, height = _image_dimensions(source.frame_files[0])
        for frame_index, _frame_file in enumerate(source.frame_files):
            image_rel = build_video_frame_image_rel(source.sequence_name, frame_index)
            if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                continue
            image_id = f"{source.sequence_name}:{frame_index:06d}"
            images_by_id[image_id] = ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
            )

        annotation_rel = source.ground_truth_path.relative_to(dataset_root)
        with source.ground_truth_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = line.strip()
                if not row:
                    continue

                try:
                    parsed = parse_mot_ground_truth_row(row)
                except ValueError as exc:
                    raise ValidationError(
                        f"Invalid MOT ground-truth row at {annotation_rel.as_posix()}:{line_no}",
                        context={"row": row},
                    ) from exc
                if parsed is None:
                    continue

                frame_index = parsed.frame_number - 1
                if frame_index >= len(source.frame_files):
                    raise ValidationError(
                        f"MOT ground-truth row references a missing frame: {source.sequence_name}",
                        context={
                            "annotation_file": annotation_rel.as_posix(),
                            "line": str(line_no),
                            "frame_number": str(parsed.frame_number),
                            "frame_count": str(len(source.frame_files)),
                        },
                    )

                bbox_xywh_abs = _clip_bbox_to_image_bounds(
                    parsed.bbox_xywh_abs,
                    image_width=width,
                    image_height=height,
                )
                if bbox_xywh_abs is None:
                    continue
                if f"{source.sequence_name}:{frame_index:06d}" not in images_by_id:
                    continue
                bbox_was_clipped = bbox_xywh_abs != parsed.bbox_xywh_abs

                pending_annotations.append(
                    _PendingAnnotation(
                        annotation_id=(
                            f"{annotation_rel.as_posix()}:{line_no:06d}:{parsed.track_id:03d}"
                        ),
                        image_id=f"{source.sequence_name}:{frame_index:06d}",
                        class_name=TRACKED_OBJECT_CLASS_NAME,
                        bbox_xywh_abs=bbox_xywh_abs,
                        attributes={
                            "mot_track_id": parsed.track_id,
                            "mot_frame_number": parsed.frame_number,
                            "mot_source_class_id": parsed.source_class_id,
                            "mot_visibility": parsed.visibility,
                            "mot_bbox_was_clipped_to_frame": bbox_was_clipped,
                            "mot_original_x": parsed.bbox_xywh_abs[0],
                            "mot_original_y": parsed.bbox_xywh_abs[1],
                            "mot_original_w": parsed.bbox_xywh_abs[2],
                            "mot_original_h": parsed.bbox_xywh_abs[3],
                        },
                    )
                )

    return _build_dataset(
        dataset_root=dataset_root,
        images_by_id=images_by_id,
        pending_annotations=pending_annotations,
    )


def _read_video_frame_sequence_dataset(
    dataset_root: Path,
    *,
    max_sources: int | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    sequence_layout = discover_frame_sequence_layout(dataset_root)
    if sequence_layout is None:
        raise ValidationError(f"No video annotation files found under: {dataset_root}")

    all_sources = list(sequence_layout.sources)
    sampled_sources = (
        _sample_evenly(all_sources, max_sources)
        if max_sources is not None and input_path_filter is None
        else all_sources
    )
    sampled_layout = (
        FrameSequenceLayout(
            frames_root=sequence_layout.frames_root,
            ground_truth_root=sequence_layout.ground_truth_root,
            sources=tuple(sampled_sources),
            frame_directory_names=sequence_layout.frame_directory_names,
            ground_truth_sequence_names=sequence_layout.ground_truth_sequence_names,
        )
        if len(sampled_sources) != len(all_sources)
        else sequence_layout
    )

    if all(
        source.ground_truth_format == MOT_CHALLENGE_ANNOTATION_FORMAT
        for source in sampled_layout.sources
    ):
        dataset = _read_mot_frame_sequence_dataset(
            dataset_root,
            sampled_layout,
            input_path_filter=input_path_filter,
        )
        details = dict(dataset.source_metadata.details)
        details["video_sources_total"] = str(len(all_sources))
        details["video_sources_loaded"] = str(len(sampled_layout.sources))
        return dataset.model_copy(update={"source_metadata": dataset.source_metadata.model_copy(update={"details": details})})

    frame_names = set(sampled_layout.frame_directory_names)
    ground_truth_names = set(sampled_layout.ground_truth_sequence_names)
    if frame_names != ground_truth_names:
        missing_ground_truth = sorted(frame_names - ground_truth_names)
        missing_frame_directories = sorted(ground_truth_names - frame_names)
        raise ValidationError(
            "Video frame sequence layout must provide one _gt.txt file per frame directory",
            context={
                "missing_ground_truth": ",".join(missing_ground_truth),
                "missing_frame_directories": ",".join(missing_frame_directories),
            },
        )

    images_by_id: dict[str, ImageRecord] = {}
    pending_annotations: list[_PendingAnnotation] = []

    for source in sampled_layout.sources:
        if not source.frame_files:
            raise ValidationError(f"No frame images found for video sequence: {source.sequence_name}")

        width, height = _image_dimensions(source.frame_files[0])
        annotation_rel = source.ground_truth_path.relative_to(dataset_root)
        frame_index = 0
        with source.ground_truth_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = line.strip()
                if not row:
                    continue
                if frame_index >= len(source.frame_files):
                    raise ValidationError(
                        f"Video frame sequence has more ground-truth rows than frames: {source.sequence_name}",
                        context={"annotation_file": annotation_rel.as_posix()},
                    )

                try:
                    bbox = parse_tracking_bbox_row(row)
                except ValueError as exc:
                    raise ValidationError(
                        f"Invalid tracking bbox row at {annotation_rel.as_posix()}:{line_no}",
                        context={"row": row},
                    ) from exc

                image_id = f"{source.sequence_name}:{frame_index:06d}"
                image_rel = build_video_frame_image_rel(source.sequence_name, frame_index)
                if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                    frame_index += 1
                    continue
                existing = images_by_id.get(image_id)
                image = ImageRecord(
                    image_id=image_id,
                    file_name=image_rel,
                    width=width,
                    height=height,
                )
                if existing is not None and existing.file_name != image.file_name:
                    raise ValidationError(
                        f"Duplicate video frame image_id with conflicting paths: {image.image_id}"
                    )
                images_by_id[image_id] = image
                if bbox is not None:
                    pending_annotations.append(
                        _PendingAnnotation(
                            annotation_id=f"{annotation_rel.as_posix()}:{line_no:06d}:001",
                            image_id=image_id,
                            class_name=TRACKED_OBJECT_CLASS_NAME,
                            bbox_xywh_abs=bbox,
                        )
                    )
                frame_index += 1

        if frame_index != len(source.frame_files):
            raise ValidationError(
                f"Video frame sequence row count must match frame count: {source.sequence_name}",
                context={
                    "frame_count": str(len(source.frame_files)),
                    "ground_truth_rows": str(frame_index),
                },
            )

    return _build_dataset(
        dataset_root=dataset_root,
        images_by_id=images_by_id,
        pending_annotations=pending_annotations,
        details={
            "video_sources_total": str(len(all_sources)),
            "video_sources_loaded": str(len(sampled_layout.sources)),
        },
    )


def _read_paired_video_json_dataset(
    dataset_root: Path,
    *,
    max_sources: int | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    all_sources = list(discover_paired_video_json_sources(dataset_root))
    if not all_sources:
        raise ValidationError(f"No paired video JSON annotations found under: {dataset_root}")

    sources = (
        _sample_evenly(all_sources, max_sources)
        if max_sources is not None and input_path_filter is None
        else all_sources
    )

    images_by_id: dict[str, ImageRecord] = {}
    pending_annotations: list[_PendingAnnotation] = []
    skipped_annotations: dict[str, list[str]] = defaultdict(list)

    for source in sources:
        annotation_rel = source.annotation_path.relative_to(dataset_root)
        exist, gt_rect = _parse_paired_video_json_payload(source.annotation_path)
        width, height = _probe_video_dimensions(source.video_path)

        for frame_index, frame_exists in enumerate(exist):
            image_id = f"{source.source_name}:{frame_index:06d}"
            image_rel = build_video_frame_image_rel(source.source_name, frame_index)
            if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                continue
            images_by_id[image_id] = ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
            )

            if not frame_exists:
                continue

            try:
                bbox = _parse_paired_video_json_bbox(
                    gt_rect[frame_index],
                    annotation_rel=annotation_rel,
                    frame_index=frame_index,
                )
            except ValidationError as exc:
                skipped_annotations[annotation_rel.as_posix()].append(str(exc))
                continue
            if bbox is None:
                skipped_annotations[annotation_rel.as_posix()].append(
                    f"{annotation_rel.as_posix()}:{frame_index + 1} uses an empty or non-positive bbox"
                )
                continue

            pending_annotations.append(
                _PendingAnnotation(
                    annotation_id=f"{annotation_rel.as_posix()}:{frame_index + 1:06d}:001",
                    image_id=image_id,
                    class_name=TRACKED_OBJECT_CLASS_NAME,
                    bbox_xywh_abs=bbox,
                    attributes={
                        "paired_video_annotation_file": annotation_rel.as_posix(),
                        "paired_video_stream_name": Path(source.source_name).name,
                    },
                )
            )

    return _build_dataset(
        dataset_root=dataset_root,
        images_by_id=images_by_id,
        pending_annotations=pending_annotations,
        details={
            "video_sources_total": str(len(all_sources)),
            "video_sources_loaded": str(len(sources)),
        },
        warnings=_paired_video_skip_warnings(skipped_annotations),
    )


def read_video_bbox_dataset(
    dataset_root: Path,
    *,
    max_sources: int | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    annotation_files = _standard_annotation_files(dataset_root)
    if not annotation_files:
        paired_json_sources = discover_paired_video_json_sources(
            dataset_root,
            max_sources=1,
        )
        if paired_json_sources:
            return _read_paired_video_json_dataset(
                dataset_root,
                max_sources=max_sources,
                input_path_filter=input_path_filter,
            )
        return _read_video_frame_sequence_dataset(
            dataset_root,
            max_sources=max_sources,
            input_path_filter=input_path_filter,
        )

    spec = _video_bbox_spec()
    tokenized_parser = _tokenized_video_bbox_parser()
    all_annotation_files = list(annotation_files)
    annotation_files = (
        _sample_evenly(all_annotation_files, max_sources)
        if max_sources is not None and input_path_filter is None
        else all_annotation_files
    )
    images_by_id: dict[str, ImageRecord] = {}
    pending_annotations: list[_PendingAnnotation] = []

    for annotation_file in annotation_files:
        annotation_rel = annotation_file.relative_to(dataset_root)
        video_stem = annotation_file.stem
        if spec is not None and tokenized_parser is not None:
            video_path = resolve_spec_video_file(dataset_root, video_stem, spec)
            if video_path is None:
                raise ValidationError(
                    f"No source video found for annotation file stem: {video_stem}",
                    context={"annotation_file": annotation_rel.as_posix()},
                )
        else:
            video_path = _find_video_file(dataset_root, video_stem)
        width, height = _probe_video_dimensions(video_path)

        with annotation_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = line.strip()
                if not row:
                    continue

                if spec is not None and tokenized_parser is not None:
                    image, parsed_annotations = _parse_tokenized_annotation_row(
                        row,
                        parser=tokenized_parser,
                        spec=spec,
                        annotation_rel=annotation_rel,
                        line_no=line_no,
                        video_stem=video_stem,
                    )
                else:
                    image, parsed_annotations = _parse_annotation_row(
                        row,
                        annotation_rel=annotation_rel,
                        line_no=line_no,
                        video_stem=video_stem,
                    )
                if not relative_path_matches_input_filter(
                    image.file_name,
                    input_path_filter=input_path_filter,
                ):
                    continue

                existing = images_by_id.get(image.image_id)
                resolved_image = image.model_copy(update={"width": width, "height": height})
                if existing is not None and existing.file_name != resolved_image.file_name:
                    raise ValidationError(
                        f"Duplicate video frame image_id with conflicting paths: {image.image_id}"
                    )
                images_by_id[image.image_id] = resolved_image

                pending_annotations.extend(parsed_annotations)

    return _build_dataset(
        dataset_root=dataset_root,
        images_by_id=images_by_id,
        pending_annotations=pending_annotations,
        details={
            "video_sources_total": str(len(all_annotation_files)),
            "video_sources_loaded": str(len(annotation_files)),
        },
    )


def _extract_video_frames_to_directory(
    *,
    ffmpeg: str,
    video_path: Path,
    frame_dir: Path,
    frame_indices: set[int],
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    max_frame_index = max(frame_indices)
    output_pattern = frame_dir / "frame_%06d.jpg"
    ensure_directory(frame_dir)

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostats",
        "-i",
        str(video_path),
        "-frames:v",
        str(max_frame_index + 1),
        "-start_number",
        "0",
    ]
    if progress_callback is None:
        completed = subprocess.run(
            command + [str(output_pattern)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise ConversionError(
                f"Unable to extract frames from source video: {video_path.name}",
                context={"stderr": completed.stderr.strip()},
            )
    else:
        process = subprocess.Popen(
            command + ["-progress", "pipe:1", str(output_pattern)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        last_reported_frame_count = 0
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line.startswith("frame="):
                continue
            try:
                frame_count = int(float(line.split("=", maxsplit=1)[1]))
            except ValueError:
                continue
            if frame_count <= last_reported_frame_count:
                continue
            last_reported_frame_count = frame_count
            progress_callback(min(frame_count, max_frame_index + 1))

        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise ConversionError(
                f"Unable to extract frames from source video: {video_path.name}",
                context={"stderr": stderr.strip()},
            )
        progress_callback(max_frame_index + 1)

    for extracted_file in frame_dir.glob("frame_*.jpg"):
        extracted_index = int(extracted_file.stem.removeprefix("frame_"))
        if extracted_index not in frame_indices:
            extracted_file.unlink()


def _dense_standard_output_directory(
    output_paths_by_frame: dict[int, Path],
    frame_indices: set[int],
) -> Path | None:
    if not output_paths_by_frame:
        return None

    sorted_indices = sorted(frame_indices)
    if sorted_indices != list(range(len(sorted_indices))):
        return None

    parents = {path.parent for path in output_paths_by_frame.values()}
    if len(parents) != 1:
        return None

    for frame_index, output_path in output_paths_by_frame.items():
        expected_name = FRAME_FILE_TEMPLATE.format(frame_index=frame_index)
        if output_path.name != expected_name:
            return None

    return next(iter(parents))


def _copy_sequence_frames_to_directory(
    *,
    sequence_name: str,
    source_frames: tuple[Path, ...],
    frame_dir: Path,
    frame_indices: set[int],
) -> None:
    for frame_index in sorted(frame_indices):
        if frame_index < 0 or frame_index >= len(source_frames):
            raise ConversionError(
                f"Requested frame index is outside the available still-frame sequence: {sequence_name}",
                context={"frame_index": str(frame_index)},
            )
        destination_path = frame_dir / FRAME_FILE_TEMPLATE.format(frame_index=frame_index)
        shutil.copy2(source_frames[frame_index], destination_path)


def _copy_sequence_frames_to_paths(
    *,
    sequence_name: str,
    source_frames: tuple[Path, ...],
    output_paths_by_frame: dict[int, Path],
    progress_callback: FrameMaterializationProgressCallback | None = None,
    total_frames: int = 0,
    completed_frames: int = 0,
) -> int:
    for frame_index, destination_path in sorted(output_paths_by_frame.items()):
        if frame_index < 0 or frame_index >= len(source_frames):
            raise ConversionError(
                f"Requested frame index is outside the available still-frame sequence: {sequence_name}",
                context={"frame_index": str(frame_index)},
            )
        ensure_directory(destination_path.parent)
        shutil.copy2(source_frames[frame_index], destination_path)
        completed_frames += 1
        if progress_callback is not None:
            progress_callback(sequence_name, completed_frames, total_frames)
    return completed_frames


def _validate_materialized_frames(
    *,
    frame_dir: Path,
    frame_indices: set[int],
    source_name: str,
) -> None:
    missing = [
        frame_index
        for frame_index in sorted(frame_indices)
        if not (frame_dir / FRAME_FILE_TEMPLATE.format(frame_index=frame_index)).exists()
    ]
    if missing:
        raise ConversionError(
            f"Frame materialization missed requested frame(s) for source: {source_name}",
            context={"missing_frames": ",".join(str(index) for index in missing)},
        )


def _validate_materialized_output_paths(
    *,
    output_paths_by_frame: dict[int, Path],
    source_name: str,
) -> None:
    missing = [
        f"{frame_index}:{destination_path.as_posix()}"
        for frame_index, destination_path in sorted(output_paths_by_frame.items())
        if not destination_path.exists()
    ]
    if missing:
        raise ConversionError(
            f"Frame materialization missed requested output file(s) for source: {source_name}",
            context={"missing_outputs": ",".join(missing)},
        )


def materialize_video_bbox_frames(
    *,
    dataset_root: Path,
    images: list[ImageRecord],
    output_root: Path,
    output_images: list[ImageRecord] | None = None,
    progress_callback: FrameMaterializationProgressCallback | None = None,
) -> None:
    frames_by_video: dict[str, set[int]] = defaultdict(set)
    output_image_by_id = {image.image_id: image for image in output_images or []}
    output_paths_by_video: dict[str, dict[int, Path]] = defaultdict(dict)
    output_owner_by_path: dict[Path, str] = {}
    sequence_layout = discover_frame_sequence_layout(dataset_root)
    sequence_sources_by_name = (
        {source.sequence_name: source.frame_files for source in sequence_layout.sources}
        if sequence_layout is not None
        else {}
    )

    for image in images:
        try:
            video_stem, frame_index = parse_video_frame_image_rel(image.file_name)
        except ValueError as exc:
            raise ConversionError(
                f"Unsupported video frame image path for materialization: {image.file_name}"
            ) from exc
        frames_by_video[video_stem].add(frame_index)
        output_image = output_image_by_id.get(image.image_id, image)
        output_image_path = Path(output_image.file_name)
        if not output_image_path.parts or output_image_path.parts[0] != "images":
            raise ConversionError(
                f"Unsupported video frame output path for materialization: {output_image.file_name}"
            )

        output_path = safe_resolve(output_root, output_image_path.as_posix())
        existing_output_path = output_paths_by_video[video_stem].get(frame_index)
        if existing_output_path is not None and existing_output_path != output_path:
            raise ConversionError(
                f"Frame {frame_index} for {video_stem} cannot be materialized into multiple output paths"
            )
        output_paths_by_video[video_stem][frame_index] = output_path
        existing_owner = output_owner_by_path.get(output_path)
        if existing_owner is not None and existing_owner != image.image_id:
            raise ConversionError(
                f"Multiple frames resolve to the same materialized output image path: {output_image.file_name}"
            )
        output_owner_by_path[output_path] = image.image_id

    ffmpeg: str | None = None
    total_requested_frames = sum(len(frame_indices) for frame_indices in frames_by_video.values())
    completed_requested_frames = 0
    for video_stem, frame_indices in sorted(frames_by_video.items()):
        output_paths = output_paths_by_video[video_stem]
        video_path = _find_video_file_or_none(dataset_root, video_stem)
        if video_path is not None:
            sorted_frame_indices = sorted(frame_indices)
            requested_frame_total = len(sorted_frame_indices)

            def _emit_video_extract_progress(extracted_frame_count: int) -> None:
                if progress_callback is None:
                    return
                source_completed = bisect_left(sorted_frame_indices, extracted_frame_count)
                progress_callback(
                    f"{video_path.name} [{source_completed}/{requested_frame_total}]",
                    completed_requested_frames + source_completed,
                    total_requested_frames,
                )

            if progress_callback is not None:
                progress_callback(
                    f"{video_path.name} [0/{requested_frame_total}]",
                    completed_requested_frames,
                    total_requested_frames,
                )

            if ffmpeg is None:
                ffmpeg = _require_binary("ffmpeg", error_factory=ConversionError)

            direct_output_dir = _dense_standard_output_directory(output_paths, frame_indices)
            if direct_output_dir is not None:
                _extract_video_frames_to_directory(
                    ffmpeg=ffmpeg,
                    video_path=video_path,
                    frame_dir=direct_output_dir,
                    frame_indices=frame_indices,
                    progress_callback=_emit_video_extract_progress,
                )
                _validate_materialized_output_paths(
                    output_paths_by_frame=output_paths,
                    source_name=video_path.name,
                )
                completed_requested_frames += requested_frame_total
                if progress_callback is not None:
                    progress_callback(
                        f"{video_path.name} [{requested_frame_total}/{requested_frame_total}]",
                        completed_requested_frames,
                        total_requested_frames,
                    )
                continue

            temp_prefix_source = video_stem.replace("/", "_").replace("\\", "_")
            temp_prefix = f"{temp_prefix_source}_"
            with tempfile.TemporaryDirectory(prefix=temp_prefix, dir=output_root) as temp_dir_raw:
                temp_dir = Path(temp_dir_raw)
                _extract_video_frames_to_directory(
                    ffmpeg=ffmpeg,
                    video_path=video_path,
                    frame_dir=temp_dir,
                    frame_indices=frame_indices,
                    progress_callback=_emit_video_extract_progress,
                )
                _validate_materialized_frames(
                    frame_dir=temp_dir,
                    frame_indices=frame_indices,
                    source_name=video_path.name,
                )
                for frame_index, output_path in sorted(output_paths.items()):
                    ensure_directory(output_path.parent)
                    extracted_path = temp_dir / FRAME_FILE_TEMPLATE.format(frame_index=frame_index)
                    shutil.copy2(extracted_path, output_path)
                _validate_materialized_output_paths(
                    output_paths_by_frame=output_paths,
                    source_name=video_path.name,
                )
            completed_requested_frames += requested_frame_total
            if progress_callback is not None:
                progress_callback(
                    f"{video_path.name} [{requested_frame_total}/{requested_frame_total}]",
                    completed_requested_frames,
                    total_requested_frames,
                )
            continue

        source_frames = sequence_sources_by_name.get(video_stem)
        if source_frames is None:
            raise ConversionError(
                f"No source video or frame sequence found for: {video_stem}",
                context={"dataset_root": str(dataset_root)},
            )
        completed_requested_frames = _copy_sequence_frames_to_paths(
            sequence_name=video_stem,
            source_frames=source_frames,
            output_paths_by_frame=output_paths,
            progress_callback=progress_callback,
            total_frames=total_requested_frames,
            completed_frames=completed_requested_frames,
        )
        _validate_materialized_output_paths(
            output_paths_by_frame=output_paths,
            source_name=video_stem,
        )
