from __future__ import annotations

import json
from pathlib import Path

from label_master.adapters.cityscapes.common import (
    LABELS_BY_NAME,
    CityscapesLabelMode,
    annotation_file_stem,
    annotation_files_for_split,
    available_split_names,
    bbox_from_polygon,
    category_table,
    normalize_label_name,
    resolve_image_path,
    resolve_labels_root,
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
from label_master.core.domain.value_objects import ValidationError
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter

DatasetLoadProgressCallback = callable


def _build_skip_warnings(skipped_files: list[tuple[str, str]]) -> list[WarningEvent]:
    if not skipped_files:
        return []

    skipped_files_payload = json.dumps(
        [{"source_file": source_file, "reason": reason} for source_file, reason in skipped_files]
    )
    if len(skipped_files) == 1:
        source_file, reason = skipped_files[0]
        return [
            WarningEvent(
                code="cityscapes_annotation_file_skipped",
                message=f"Skipped Cityscapes annotation file {source_file}: {reason}",
                severity=Severity.WARNING,
                context={
                    "source_file": source_file,
                    "reason": reason,
                    "skipped_files_json": skipped_files_payload,
                },
            )
        ]

    sample = [{"source_file": source_file, "reason": reason} for source_file, reason in skipped_files[:5]]
    return [
        WarningEvent(
            code="cityscapes_annotation_files_skipped",
            message=(
                f"Skipped {len(skipped_files)} Cityscapes annotation file(s) during load. "
                f"First example: {skipped_files[0][0]} ({skipped_files[0][1]})."
            ),
            severity=Severity.WARNING,
            context={
                "skipped_files": str(len(skipped_files)),
                "skipped_files_json": skipped_files_payload,
                "sample_skipped_files_json": json.dumps(sample),
            },
        )
    ]


def read_cityscapes_dataset(
    dataset_root: Path,
    *,
    split_names: list[str] | None = None,
    label_mode: CityscapesLabelMode = "instances",
    include_groups: bool = False,
    input_path_filter: InputPathFilter | None = None,
    progress_callback: callable | None = None,
) -> AnnotationDataset:
    labels_variant, labels_root = resolve_labels_root(dataset_root)
    resolved_splits = split_names or available_split_names(labels_root, dataset_root)
    if not resolved_splits:
        raise ValidationError(
            "No Cityscapes splits have both polygon annotations and matching leftImg8bit images.",
            context={"dataset_root": str(dataset_root)},
        )

    split_annotation_files = {
        split_name: annotation_files_for_split(labels_root, split_name)
        for split_name in resolved_splits
    }
    total_annotation_files = sum(len(files) for files in split_annotation_files.values())
    if total_annotation_files == 0:
        raise ValidationError(
            "No Cityscapes polygon annotation files were found for the requested splits.",
            context={"dataset_root": str(dataset_root), "splits": ",".join(resolved_splits)},
        )

    category_ids, category_labels = category_table(label_mode)
    categories = {
        class_id: CategoryRecord(
            class_id=class_id,
            name=label.name,
            supercategory=label.category,
        )
        for label, class_id in ((label, category_ids[label.name]) for label in category_labels)
    }

    images: list[ImageRecord] = []
    annotations: list[AnnotationRecord] = []
    image_ids: set[str] = set()
    skipped_files: list[tuple[str, str]] = []
    skipped_objects = 0
    annotation_index = 1
    processed_files = 0
    annotation_files_skipped = 0

    for split_name in resolved_splits:
        for annotation_file in split_annotation_files[split_name]:
            processed_files += 1
            if progress_callback is not None:
                progress_callback(processed_files, total_annotation_files)

            annotation_rel = annotation_file.relative_to(dataset_root).as_posix()
            stem = annotation_file_stem(annotation_file, labels_variant=labels_variant)
            if stem is None:
                skipped_files.append((annotation_rel, "unsupported Cityscapes annotation file name"))
                annotation_files_skipped += 1
                continue

            city_name = annotation_file.parent.name
            image_path = resolve_image_path(dataset_root, split_name, city_name, stem)
            if image_path is None:
                skipped_files.append((annotation_rel, "matching leftImg8bit image not found"))
                annotation_files_skipped += 1
                continue

            image_rel = image_path.relative_to(dataset_root).as_posix()
            if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                continue

            try:
                payload = json.loads(annotation_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                skipped_files.append((annotation_rel, f"invalid JSON payload: {exc}"))
                annotation_files_skipped += 1
                continue

            if not isinstance(payload, dict):
                skipped_files.append((annotation_rel, "annotation payload is not an object"))
                annotation_files_skipped += 1
                continue

            try:
                image_width = int(payload["imgWidth"])
                image_height = int(payload["imgHeight"])
            except (KeyError, TypeError, ValueError) as exc:
                skipped_files.append((annotation_rel, f"invalid image dimensions: {exc}"))
                annotation_files_skipped += 1
                continue

            raw_objects = payload.get("objects", [])
            if not isinstance(raw_objects, list):
                skipped_files.append((annotation_rel, "objects field must be a list"))
                annotation_files_skipped += 1
                continue

            image_id = stem
            if image_id not in image_ids:
                images.append(
                    ImageRecord(
                        image_id=image_id,
                        file_name=image_rel,
                        width=image_width,
                        height=image_height,
                    )
                )
                image_ids.add(image_id)

            for raw_object in raw_objects:
                if not isinstance(raw_object, dict):
                    skipped_objects += 1
                    continue
                if bool(raw_object.get("deleted", 0)):
                    skipped_objects += 1
                    continue

                normalized = normalize_label_name(str(raw_object.get("label", "")))
                if normalized is None:
                    skipped_objects += 1
                    continue

                label = LABELS_BY_NAME[normalized.name]
                if label.ignore_in_eval:
                    skipped_objects += 1
                    continue
                if label_mode == "instances" and not label.has_instances:
                    skipped_objects += 1
                    continue
                if normalized.is_group and not include_groups:
                    skipped_objects += 1
                    continue

                polygon = raw_object.get("polygon", [])
                if not isinstance(polygon, list):
                    skipped_objects += 1
                    continue
                bbox = bbox_from_polygon(
                    polygon,
                    image_width=image_width,
                    image_height=image_height,
                )
                if bbox is None:
                    skipped_objects += 1
                    continue

                category_id = category_ids.get(normalized.name)
                if category_id is None:
                    skipped_objects += 1
                    continue

                annotations.append(
                    AnnotationRecord(
                        annotation_id=str(annotation_index),
                        image_id=image_id,
                        class_id=category_id,
                        bbox_xywh_abs=bbox,
                        iscrowd=normalized.is_group,
                    )
                )
                annotation_index += 1

    warnings = _build_skip_warnings(skipped_files)
    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.CITYSCAPES,
        images=sorted(images, key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda annotation: annotation.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="cityscapes_reader",
            details={
                "labels_variant": labels_variant,
                "splits_loaded": ",".join(resolved_splits),
                "label_mode": label_mode,
                "include_groups": str(include_groups).lower(),
                "annotation_files_total": str(total_annotation_files),
                "annotation_files_loaded": str(total_annotation_files - annotation_files_skipped),
                "annotation_files_skipped": str(annotation_files_skipped),
                "skipped_objects": str(skipped_objects),
            },
        ),
        warnings=warnings,
    )