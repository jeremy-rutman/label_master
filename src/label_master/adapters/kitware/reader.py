from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from label_master.adapters.kitware.common import (
    _kitware_parser,
    discover_kitware_csv_layouts,
    parse_kitware_bboxes,
    resolve_kitware_image_path,
)
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


def _image_dimensions(
    image_path: Path,
    cache: dict[Path, tuple[int, int]],
) -> tuple[int, int]:
    if image_path in cache:
        return cache[image_path]

    try:
        with Image.open(image_path) as opened:
            size = (int(opened.width), int(opened.height))
    except OSError as exc:
        raise ValidationError(f"Kitware image could not be opened: {image_path}") from exc

    cache[image_path] = size
    return size


def read_kitware_dataset(
    dataset_root: Path,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    parser = _kitware_parser()
    layouts = discover_kitware_csv_layouts(dataset_root)
    if not layouts:
        raise ValidationError(f"No Kitware annotation CSV files found under: {dataset_root}")

    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    class_names_in_order: list[str] = []
    seen_class_names: set[str] = set()
    image_size_cache: dict[Path, tuple[int, int]] = {}

    for layout in layouts:
        csv_rel = layout.csv_path.relative_to(dataset_root)

        for bbox_column in layout.bbox_columns:
            if bbox_column.class_name not in seen_class_names:
                seen_class_names.add(bbox_column.class_name)
                class_names_in_order.append(bbox_column.class_name)

        with layout.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_no, row in enumerate(reader, start=2):
                if row is None:
                    continue

                raw_image_ref = str(row.get(layout.image_field, "")).strip()
                if not raw_image_ref:
                    raise ValidationError(
                        f"Kitware row is missing imageFilename: {csv_rel.as_posix()}:{line_no}"
                    )

                image_path = resolve_kitware_image_path(dataset_root, layout.csv_path, raw_image_ref)
                if image_path is None:
                    raise ValidationError(
                        f"Kitware image could not be resolved: {csv_rel.as_posix()}:{line_no}",
                        context={"imageFilename": raw_image_ref},
                    )

                image_rel = image_path.relative_to(dataset_root).as_posix()
                if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
                    continue
                image_id = Path(image_rel).with_suffix("").as_posix()
                width, height = _image_dimensions(image_path, image_size_cache)

                existing = images_by_id.get(image_id)
                resolved_image = ImageRecord(
                    image_id=image_id,
                    file_name=image_rel,
                    width=width,
                    height=height,
                )
                if existing is not None and existing != resolved_image:
                    raise ValidationError(
                        f"Kitware image_id reused with conflicting metadata: {image_id}"
                    )
                images_by_id[image_id] = resolved_image

                for bbox_column in layout.bbox_columns:
                    raw_value = str(row.get(bbox_column.header_name, "")).strip()
                    try:
                        bboxes = parse_kitware_bboxes(raw_value, parser=parser)
                    except ValueError as exc:
                        raise ValidationError(
                            f"Invalid Kitware bbox at {csv_rel.as_posix()}:{line_no}",
                            context={"column": bbox_column.header_name, "value": raw_value},
                        ) from exc
                    if not bboxes:
                        continue

                    class_id = class_names_in_order.index(bbox_column.class_name)
                    for bbox_index, bbox in enumerate(bboxes, start=1):
                        annotations.append(
                            AnnotationRecord(
                                annotation_id=(
                                    f"{csv_rel.as_posix()}:{line_no}:{bbox_column.class_name}:{bbox_index}"
                                ),
                                image_id=image_id,
                                class_id=class_id,
                                bbox_xywh_abs=bbox,
                            )
                        )

    categories = {
        class_id: CategoryRecord(class_id=class_id, name=class_name)
        for class_id, class_name in enumerate(class_names_in_order)
    }

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.KITWARE,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda annotation: annotation.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="kitware_reader",
        ),
    )
