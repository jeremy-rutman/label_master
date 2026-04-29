from __future__ import annotations

from pathlib import Path
from typing import Callable

from label_master.adapters.custom.common import probe_video_dimensions
from label_master.adapters.matlab_ground_truth.common import (
    find_matlab_ground_truth_files,
    load_matlab_ground_truth_payload,
    resolve_matlab_ground_truth_video_path,
)
from label_master.adapters.video_bbox.common import build_video_frame_image_rel
from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter


def read_matlab_ground_truth_dataset(
    dataset_root: Path,
    *,
    max_annotation_files: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    annotation_files = find_matlab_ground_truth_files(
        dataset_root,
        max_files=max_annotation_files if input_path_filter is None else None,
    )
    if not annotation_files:
        raise ValidationError(f"No MATLAB groundTruth annotation files found under: {dataset_root}")

    categories_by_name: dict[str, int] = {}
    categories: dict[int, CategoryRecord] = {}
    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    video_dimension_cache: dict[Path, tuple[int, int]] = {}
    loaded_annotation_files: set[Path] = set()
    total_annotation_files = len(annotation_files)
    progress_update_every = max(total_annotation_files // 100, 1) if total_annotation_files else 1

    for annotation_index, annotation_path in enumerate(annotation_files, start=1):
        payload = load_matlab_ground_truth_payload(annotation_path)
        video_path = resolve_matlab_ground_truth_video_path(
            dataset_root,
            source_video_path=payload.source_video_path,
            annotation_path=annotation_path,
        )
        if video_path is None:
            raise ValidationError(
                f"Source video could not be resolved for MATLAB annotation file: {annotation_path.name}",
                context={"source_video_path": payload.source_video_path},
            )

        if video_path not in video_dimension_cache:
            video_dimension_cache[video_path] = probe_video_dimensions(video_path)
        width, height = video_dimension_cache[video_path]
        video_stem = video_path.stem
        annotation_rel = annotation_path.relative_to(dataset_root).as_posix()

        for label_name in payload.label_names:
            if label_name in categories_by_name:
                continue
            class_id = len(categories_by_name)
            categories_by_name[label_name] = class_id
            categories[class_id] = CategoryRecord(class_id=class_id, name=label_name)

        for row_index, row in enumerate(payload.rows):
            image_id = f"{video_stem}:{row_index:06d}"
            image = ImageRecord(
                image_id=image_id,
                file_name=build_video_frame_image_rel(video_stem, row_index),
                width=width,
                height=height,
            )
            if not relative_path_matches_input_filter(image.file_name, input_path_filter=input_path_filter):
                continue
            existing = images_by_id.get(image_id)
            if existing is not None and existing != image:
                raise ValidationError(
                    f"MATLAB groundTruth image_id reused with conflicting metadata: {image_id}"
                )
            images_by_id[image_id] = image
            loaded_annotation_files.add(annotation_path)

            for label_name, bboxes in row.bboxes_by_label.items():
                class_id = categories_by_name[label_name]
                for bbox_index, bbox in enumerate(bboxes, start=1):
                    attributes: dict[str, str | int | float | bool | None] = {
                        "source_video_path": payload.source_video_path,
                    }
                    if row.timestamp_ms is not None:
                        attributes["timestamp_ms"] = float(row.timestamp_ms)
                    annotations.append(
                        AnnotationRecord(
                            annotation_id=f"{annotation_rel}:{row_index + 1:06d}:{label_name}:{bbox_index:03d}",
                            image_id=image_id,
                            class_id=class_id,
                            bbox_xywh_abs=bbox,
                            attributes=attributes,
                        )
                    )

        if progress_callback is not None and (
            annotation_index % progress_update_every == 0 or annotation_index == total_annotation_files
        ):
            progress_callback(annotation_index, total_annotation_files)
        if (
            max_annotation_files is not None
            and input_path_filter is not None
            and len(loaded_annotation_files) >= max_annotation_files
        ):
            break

    details: dict[str, str] = {
        "format_id": "matlab_ground_truth",
        "display_name": "MATLAB Ground Truth",
        "media_kind": "video_collection",
        "matlab_object_class": "groundTruth",
        "annotation_files_loaded": str(len(loaded_annotation_files) if input_path_filter is not None else len(annotation_files)),
        "annotation_files_considered": str(len(annotation_files)),
    }
    if max_annotation_files is not None:
        details["annotation_files_limit"] = str(max_annotation_files)

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.MATLAB_GROUND_TRUTH,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda annotation: annotation.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="matlab_ground_truth_reader",
            details=details,
        ),
    )
