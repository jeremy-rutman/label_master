from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from PIL import Image

from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.value_objects import ConversionError, ValidationError
from label_master.infra.filesystem import (
    InputPathFilter,
    relative_path_matches_input_filter,
    safe_resolve,
)

WAYMO_PARQUET_FORMAT_ID = "waymo_parquet_camera_boxes"
WAYMO_CAMERA_IMAGE_DIR = "camera_image"
WAYMO_CAMERA_BOX_DIR = "camera_box"
WAYMO_IMAGE_TIMESTAMP_COLUMN = "key.frame_timestamp_micros"
WAYMO_IMAGE_CAMERA_COLUMN = "key.camera_name"
WAYMO_IMAGE_BYTES_COLUMN = "[CameraImageComponent].image"
WAYMO_BOX_TYPE_COLUMN = "[CameraBoxComponent].type"
WAYMO_BOX_CENTER_X_COLUMN = "[CameraBoxComponent].box.center.x"
WAYMO_BOX_CENTER_Y_COLUMN = "[CameraBoxComponent].box.center.y"
WAYMO_BOX_SIZE_X_COLUMN = "[CameraBoxComponent].box.size.x"
WAYMO_BOX_SIZE_Y_COLUMN = "[CameraBoxComponent].box.size.y"
WAYMO_IMAGE_REQUIRED_COLUMNS = {
    WAYMO_IMAGE_TIMESTAMP_COLUMN,
    WAYMO_IMAGE_CAMERA_COLUMN,
    WAYMO_IMAGE_BYTES_COLUMN,
}
WAYMO_BOX_REQUIRED_COLUMNS = {
    WAYMO_IMAGE_TIMESTAMP_COLUMN,
    WAYMO_IMAGE_CAMERA_COLUMN,
    WAYMO_BOX_TYPE_COLUMN,
    WAYMO_BOX_CENTER_X_COLUMN,
    WAYMO_BOX_CENTER_Y_COLUMN,
    WAYMO_BOX_SIZE_X_COLUMN,
    WAYMO_BOX_SIZE_Y_COLUMN,
}
WAYMO_CAMERA_TYPE_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist",
}

WaymoMaterializationProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class _WaymoBoxRecord:
    source_type: int
    x_center: float
    y_center: float
    width: float
    height: float


def detect_waymo_parquet_dataset(dataset_root: Path, *, sample_limit: int = 500) -> float:
    segment_pairs = _discover_segment_pairs(dataset_root)
    if not segment_pairs:
        return 0.0

    sampled_pairs = _sample_evenly(segment_pairs, max(1, min(sample_limit, 8)))
    valid_pairs = 0
    matching_keys = 0

    for _segment_name, image_path, box_path in sampled_pairs:
        try:
            image_schema = pq.read_schema(image_path)
            box_schema = pq.read_schema(box_path)
        except Exception:
            continue

        if not WAYMO_IMAGE_REQUIRED_COLUMNS.issubset(set(image_schema.names)):
            continue
        if not WAYMO_BOX_REQUIRED_COLUMNS.issubset(set(box_schema.names)):
            continue

        valid_pairs += 1
        if _segment_has_matching_frame_keys(image_path, box_path):
            matching_keys += 1

    if valid_pairs == 0 or matching_keys == 0:
        return 0.0

    score = 0.55
    if valid_pairs == len(sampled_pairs):
        score += 0.25
    elif valid_pairs > 0:
        score += 0.1

    if matching_keys == len(sampled_pairs):
        score += 0.15
    elif matching_keys > 0:
        score += 0.05

    return min(score, 1.0)


def read_waymo_parquet_dataset(
    dataset_root: Path,
    *,
    input_path_filter: InputPathFilter | None = None,
    max_records: int | None = None,
) -> AnnotationDataset:
    segment_pairs = _discover_segment_pairs(dataset_root)
    if not segment_pairs:
        raise ValidationError(f"No Waymo parquet segments found under: {dataset_root}")

    box_records_by_segment: dict[str, dict[tuple[int, int], list[_WaymoBoxRecord]]] = {}
    seen_source_types: set[int] = set()
    for segment_name, _image_path, box_path in segment_pairs:
        box_records, segment_source_types = _load_box_records(box_path)
        box_records_by_segment[segment_name] = box_records
        seen_source_types.update(segment_source_types)

    type_to_class_id = {source_type: index for index, source_type in enumerate(sorted(seen_source_types))}
    categories = {
        class_id: CategoryRecord(
            class_id=class_id,
            name=WAYMO_CAMERA_TYPE_NAMES.get(source_type, f"class_{source_type}"),
        )
        for source_type, class_id in type_to_class_id.items()
    }

    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    segments_loaded = 0
    records_loaded = 0
    records_total = 0

    for segment_name, image_path, _box_path in segment_pairs:
        image_table = pq.read_table(
            image_path,
            columns=[WAYMO_IMAGE_TIMESTAMP_COLUMN, WAYMO_IMAGE_CAMERA_COLUMN, WAYMO_IMAGE_BYTES_COLUMN],
        )
        records_total += image_table.num_rows
        image_data = image_table.to_pydict()
        segment_boxes = box_records_by_segment[segment_name]
        segments_loaded += 1

        for row_index in range(image_table.num_rows):
            timestamp = int(image_data[WAYMO_IMAGE_TIMESTAMP_COLUMN][row_index])
            camera_name = int(image_data[WAYMO_IMAGE_CAMERA_COLUMN][row_index])
            image_rel = f"{segment_name}__{timestamp}__{camera_name}.jpg"
            if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                continue

            image_bytes = image_data[WAYMO_IMAGE_BYTES_COLUMN][row_index]
            width, height = _decode_image_size(image_bytes)
            image_id = f"{segment_name}:{timestamp}:{camera_name}"
            if image_id in images_by_id:
                raise ValidationError(
                    f"Duplicate Waymo parquet image id discovered: {image_id}",
                    context={"segment": segment_name},
                )

            images_by_id[image_id] = ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
            )
            records_loaded += 1

            for box_index, box_record in enumerate(segment_boxes.get((timestamp, camera_name), []), start=1):
                class_id = type_to_class_id.get(box_record.source_type)
                if class_id is None:
                    continue
                annotations.append(
                    AnnotationRecord(
                        annotation_id=f"{segment_name}:{timestamp}:{camera_name}:{row_index + 1:06d}:{box_index:03d}",
                        image_id=image_id,
                        class_id=class_id,
                        bbox_xywh_abs=(
                            box_record.x_center - (box_record.width / 2.0),
                            box_record.y_center - (box_record.height / 2.0),
                            box_record.width,
                            box_record.height,
                        ),
                        attributes={"waymo_source_type": box_record.source_type},
                    )
                )

            if max_records is not None and records_loaded >= max_records:
                break

        if max_records is not None and records_loaded >= max_records:
            break

    if not images_by_id:
        raise ValidationError(f"No Waymo parquet images matched under: {dataset_root}")

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.CUSTOM,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda annotation: annotation.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.expanduser().resolve()),
            loader="waymo_parquet_reader",
            details={
                "format_id": WAYMO_PARQUET_FORMAT_ID,
                "display_name": "Waymo Parquet Camera Boxes",
                "media_kind": "parquet_image_collection",
                "camera_image_dir": WAYMO_CAMERA_IMAGE_DIR,
                "camera_box_dir": WAYMO_CAMERA_BOX_DIR,
                "segments_loaded": str(segments_loaded),
                "segments_total": str(len(segment_pairs)),
                "records_loaded": str(records_loaded),
                "records_total": str(records_total),
            },
        ),
    )


def materialize_waymo_parquet_frames(
    *,
    dataset_root: Path,
    images: list[ImageRecord],
    output_root: Path,
    output_images: list[ImageRecord] | None = None,
    progress_callback: WaymoMaterializationProgressCallback | None = None,
) -> None:
    requested_by_segment: dict[str, dict[tuple[int, int], ImageRecord]] = {}
    output_image_by_id = {image.image_id: image for image in output_images or []}
    output_owner_by_path: dict[Path, str] = {}

    for image in images:
        segment_name, timestamp, camera_name = _parse_waymo_image_identity(image.image_id)
        requested_by_segment.setdefault(segment_name, {})[(timestamp, camera_name)] = image
        output_image = output_image_by_id.get(image.image_id, image)
        output_path = safe_resolve(output_root, output_image.file_name)
        existing_owner = output_owner_by_path.get(output_path)
        if existing_owner is not None and existing_owner != image.image_id:
            raise ConversionError(
                f"Multiple Waymo parquet images resolve to the same output path: {output_image.file_name}"
            )
        output_owner_by_path[output_path] = image.image_id

    total_requested_frames = len(images)
    completed_requested_frames = 0
    for segment_name, image_path, _box_path in _discover_segment_pairs(dataset_root):
        requested_frames = requested_by_segment.get(segment_name)
        if not requested_frames:
            continue

        image_table = pq.read_table(
            image_path,
            columns=[WAYMO_IMAGE_TIMESTAMP_COLUMN, WAYMO_IMAGE_CAMERA_COLUMN, WAYMO_IMAGE_BYTES_COLUMN],
        )
        image_data = image_table.to_pydict()
        for row_index in range(image_table.num_rows):
            timestamp = int(image_data[WAYMO_IMAGE_TIMESTAMP_COLUMN][row_index])
            camera_name = int(image_data[WAYMO_IMAGE_CAMERA_COLUMN][row_index])
            source_image = requested_frames.get((timestamp, camera_name))
            if source_image is None:
                continue

            output_image = output_image_by_id.get(source_image.image_id, source_image)
            destination_path = safe_resolve(output_root, output_image.file_name)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = image_data[WAYMO_IMAGE_BYTES_COLUMN][row_index]
            with Image.open(io.BytesIO(image_bytes)) as opened:
                opened.save(destination_path, quality=95)

            completed_requested_frames += 1
            if progress_callback is not None:
                progress_callback(segment_name, completed_requested_frames, total_requested_frames)

    if completed_requested_frames != total_requested_frames:
        raise ConversionError(
            "Waymo parquet frame materialization missed requested image(s)",
            context={"requested": str(total_requested_frames), "written": str(completed_requested_frames)},
        )


def _discover_segment_pairs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    image_root = dataset_root / WAYMO_CAMERA_IMAGE_DIR
    box_root = dataset_root / WAYMO_CAMERA_BOX_DIR
    if not image_root.is_dir() or not box_root.is_dir():
        return []

    segment_pairs: list[tuple[str, Path, Path]] = []
    for image_path in sorted(image_root.glob("*.parquet")):
        if not image_path.is_file():
            continue
        box_path = box_root / image_path.name
        if not box_path.is_file():
            continue
        segment_pairs.append((image_path.stem, image_path, box_path))
    return segment_pairs


def _load_box_records(box_path: Path) -> tuple[dict[tuple[int, int], list[_WaymoBoxRecord]], set[int]]:
    box_table = pq.read_table(
        box_path,
        columns=[
            WAYMO_IMAGE_TIMESTAMP_COLUMN,
            WAYMO_IMAGE_CAMERA_COLUMN,
            WAYMO_BOX_TYPE_COLUMN,
            WAYMO_BOX_CENTER_X_COLUMN,
            WAYMO_BOX_CENTER_Y_COLUMN,
            WAYMO_BOX_SIZE_X_COLUMN,
            WAYMO_BOX_SIZE_Y_COLUMN,
        ],
    )
    box_data = box_table.to_pydict()
    records: dict[tuple[int, int], list[_WaymoBoxRecord]] = {}
    source_types: set[int] = set()

    for row_index in range(box_table.num_rows):
        try:
            timestamp = int(box_data[WAYMO_IMAGE_TIMESTAMP_COLUMN][row_index])
            camera_name = int(box_data[WAYMO_IMAGE_CAMERA_COLUMN][row_index])
            source_type = int(box_data[WAYMO_BOX_TYPE_COLUMN][row_index])
            x_center = float(box_data[WAYMO_BOX_CENTER_X_COLUMN][row_index])
            y_center = float(box_data[WAYMO_BOX_CENTER_Y_COLUMN][row_index])
            width = float(box_data[WAYMO_BOX_SIZE_X_COLUMN][row_index])
            height = float(box_data[WAYMO_BOX_SIZE_Y_COLUMN][row_index])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"Invalid Waymo parquet box row in: {box_path.name}",
                context={"path": str(box_path)},
            ) from exc

        if width <= 0 or height <= 0:
            continue

        source_types.add(source_type)
        records.setdefault((timestamp, camera_name), []).append(
            _WaymoBoxRecord(
                source_type=source_type,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return records, source_types


def _parse_waymo_image_identity(image_id: str) -> tuple[str, int, int]:
    try:
        segment_name, timestamp_text, camera_text = image_id.rsplit(":", 2)
        return segment_name, int(timestamp_text), int(camera_text)
    except ValueError as exc:
        raise ConversionError(f"Unsupported Waymo parquet image id: {image_id}") from exc


def _decode_image_size(image_bytes: object) -> tuple[int, int]:
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValidationError("Waymo parquet image bytes must be binary data")
    with Image.open(io.BytesIO(bytes(image_bytes))) as opened:
        return int(opened.width), int(opened.height)


def _sample_evenly(paths: list[tuple[str, Path, Path]], limit: int) -> list[tuple[str, Path, Path]]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[0]]

    selected_indices: list[int] = []
    seen: set[int] = set()
    last_index = len(paths) - 1
    for position in range(limit):
        index = round(position * last_index / (limit - 1))
        if index in seen:
            continue
        seen.add(index)
        selected_indices.append(index)

    if len(selected_indices) < limit:
        for index in range(len(paths)):
            if index in seen:
                continue
            seen.add(index)
            selected_indices.append(index)
            if len(selected_indices) == limit:
                break

    return [paths[index] for index in selected_indices]


def _segment_has_matching_frame_keys(image_path: Path, box_path: Path) -> bool:
    try:
        image_table = pq.read_table(image_path, columns=[WAYMO_IMAGE_TIMESTAMP_COLUMN, WAYMO_IMAGE_CAMERA_COLUMN])
        box_table = pq.read_table(box_path, columns=[WAYMO_IMAGE_TIMESTAMP_COLUMN, WAYMO_IMAGE_CAMERA_COLUMN])
    except Exception:
        return False

    image_keys = {
        (int(timestamp), int(camera_name))
        for timestamp, camera_name in zip(
            image_table.column(WAYMO_IMAGE_TIMESTAMP_COLUMN).to_pylist(),
            image_table.column(WAYMO_IMAGE_CAMERA_COLUMN).to_pylist(),
            strict=False,
        )
    }
    for timestamp, camera_name in zip(
        box_table.column(WAYMO_IMAGE_TIMESTAMP_COLUMN).to_pylist(),
        box_table.column(WAYMO_IMAGE_CAMERA_COLUMN).to_pylist(),
        strict=False,
    ):
        if (int(timestamp), int(camera_name)) in image_keys:
            return True
    return False