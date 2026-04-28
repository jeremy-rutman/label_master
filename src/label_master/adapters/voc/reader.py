from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from label_master.adapters.voc.common import (
    build_voc_image_index,
    discover_voc_xml_files,
    parse_voc_annotation_file,
    resolve_voc_image_path,
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


def _image_dimensions(
    annotation_width: int | None,
    annotation_height: int | None,
    image_path: Path,
    cache: dict[Path, tuple[int, int]],
) -> tuple[int, int]:
    if annotation_width is not None and annotation_height is not None:
        return annotation_width, annotation_height

    if image_path in cache:
        return cache[image_path]

    try:
        with Image.open(image_path) as opened:
            size = (int(opened.width), int(opened.height))
    except OSError as exc:
        raise ValidationError(f"VOC image could not be opened: {image_path}") from exc

    cache[image_path] = size
    return size


def _sample_evenly(paths: list[Path], limit: int) -> list[Path]:
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


def _discover_class_ids(xml_files: list[Path]) -> dict[str, int]:
    class_ids_by_name: dict[str, int] = {}
    for xml_file in xml_files:
        try:
            parsed = parse_voc_annotation_file(xml_file)
        except (ValidationError, ValueError):
            continue

        for class_name in sorted({bbox.name for bbox in parsed.objects}):
            class_ids_by_name.setdefault(class_name, len(class_ids_by_name))

    return class_ids_by_name


def read_voc_dataset(
    dataset_root: Path,
    *,
    max_xml_files: int | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    resolved_root = dataset_root.resolve()
    all_xml_files = discover_voc_xml_files(dataset_root)
    if not all_xml_files:
        raise ValidationError(f"No Pascal VOC XML files found under: {dataset_root}")
    xml_files = (
        _sample_evenly(all_xml_files, max_xml_files)
        if max_xml_files is not None and input_path_filter is None
        else all_xml_files
    )

    image_index = None

    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    class_ids_by_name = _discover_class_ids(all_xml_files)
    image_size_cache: dict[Path, tuple[int, int]] = {}
    skipped_files: list[tuple[str, str]] = []
    xml_files_loaded = 0

    for xml_file in xml_files:
        xml_rel = xml_file.relative_to(dataset_root)
        try:
            parsed = parse_voc_annotation_file(xml_file)
            image_path = resolve_voc_image_path(dataset_root, xml_file, parsed, image_index=image_index)
            if image_path is None and image_index is None:
                image_index = build_voc_image_index(dataset_root)
                image_path = resolve_voc_image_path(dataset_root, xml_file, parsed, image_index=image_index)
            if image_path is None:
                raise ValidationError(
                    f"VOC image could not be resolved: {xml_rel.as_posix()}",
                    context={"filename": parsed.filename},
                )

            image_rel = image_path.resolve().relative_to(resolved_root).as_posix()
            if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                continue
            image_id = Path(image_rel).with_suffix("").as_posix()
            width, height = _image_dimensions(parsed.width, parsed.height, image_path, image_size_cache)

            resolved_image = ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
            )
            existing = images_by_id.get(image_id)
            if existing is not None and existing != resolved_image:
                raise ValidationError(f"VOC image_id reused with conflicting metadata: {image_id}")

            pending_annotations: list[AnnotationRecord] = []
            for object_index, bbox in enumerate(parsed.objects, start=1):
                class_id = class_ids_by_name.setdefault(bbox.name, len(class_ids_by_name))
                pending_annotations.append(
                    AnnotationRecord(
                        annotation_id=f"{xml_rel.as_posix()}:{object_index}",
                        image_id=image_id,
                        class_id=class_id,
                        bbox_xywh_abs=(
                            bbox.xmin,
                            bbox.ymin,
                            bbox.xmax - bbox.xmin,
                            bbox.ymax - bbox.ymin,
                        ),
                    )
                )
        except (ValidationError, ValueError) as exc:
            reason = str(exc).replace(str(xml_file), xml_rel.as_posix())
            skipped_files.append((xml_rel.as_posix(), reason))
            continue

        images_by_id[image_id] = resolved_image
        xml_files_loaded += 1
        for pending_annotation in pending_annotations:
            annotations.append(pending_annotation)
        if max_xml_files is not None and input_path_filter is not None and xml_files_loaded >= max_xml_files:
            break

    categories = {
        class_id: CategoryRecord(class_id=class_id, name=class_name)
        for class_name, class_id in sorted(class_ids_by_name.items(), key=lambda item: item[1])
    }

    warnings: list[WarningEvent] = []
    if skipped_files:
        skipped_files_payload = json.dumps(
            [
                {"source_file": xml_file, "reason": reason}
                for xml_file, reason in skipped_files
            ]
        )
        if len(skipped_files) == 1:
            skipped_file, reason = skipped_files[0]
            warnings.append(
                WarningEvent(
                    code="voc_annotation_file_skipped",
                    message=f"Skipped Pascal VOC annotation file {skipped_file}: {reason}",
                    severity=Severity.WARNING,
                    context={
                        "xml_file": skipped_file,
                        "source_file": skipped_file,
                        "reason": reason,
                        "skipped_files_json": skipped_files_payload,
                    },
                )
            )
        else:
            sample = [{"xml_file": xml_file, "reason": reason} for xml_file, reason in skipped_files[:5]]
            warnings.append(
                WarningEvent(
                    code="voc_annotation_files_skipped",
                    message=(
                        f"Skipped {len(skipped_files)} Pascal VOC annotation file(s) during load. "
                        f"First example: {skipped_files[0][0]} ({skipped_files[0][1]})."
                    ),
                    severity=Severity.WARNING,
                    context={
                        "skipped_files": str(len(skipped_files)),
                        "skipped_files_json": skipped_files_payload,
                        "sample_skipped_files_json": json.dumps(sample),
                    },
                )
            )

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.VOC,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda annotation: annotation.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="voc_reader",
            details={
                "xml_files_loaded": str(xml_files_loaded),
                "xml_files_considered": str(len(xml_files)),
                "xml_files_total": str(len(all_xml_files)),
                **({"xml_files_limit": str(max_xml_files)} if max_xml_files is not None else {}),
                "xml_files_skipped": str(len(skipped_files)),
            },
        ),
        warnings=warnings,
    )
