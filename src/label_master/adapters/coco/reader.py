from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import (
    JsonObjectDatasetParserSpec,
    resolve_builtin_format_spec,
)
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter


@lru_cache(maxsize=1)
def _coco_parser() -> JsonObjectDatasetParserSpec:
    spec = resolve_builtin_format_spec("coco")
    if spec is None or not isinstance(spec.parser, JsonObjectDatasetParserSpec):
        raise ValidationError("Built-in COCO format spec is unavailable")
    return spec.parser


def read_coco_dataset(
    dataset_root: Path,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    parser = _coco_parser()
    annotations_path = dataset_root / parser.annotations_file
    if not annotations_path.exists():
        raise ValidationError(f"COCO annotations file not found: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValidationError("COCO annotations payload must be an object")

    images_raw = payload.get(parser.images_key, [])
    annotations_raw = payload.get(parser.annotations_key, [])
    categories_raw = payload.get(parser.categories_key, [])
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list) or not isinstance(categories_raw, list):
        raise ValidationError("COCO images/annotations/categories must be arrays")

    images: list[ImageRecord] = []
    included_image_ids: set[str] = set()
    for raw in images_raw:
        if not isinstance(raw, dict):
            raise ValidationError("COCO image record must be an object")
        image_id = str(raw.get(parser.image_fields.id, "")).strip()
        file_name = str(raw.get(parser.image_fields.file_name, "")).strip()
        if not relative_path_matches_input_filter(file_name, input_path_filter=input_path_filter):
            continue
        width = int(raw.get(parser.image_fields.width, 0))
        height = int(raw.get(parser.image_fields.height, 0))
        included_image_ids.add(image_id)
        images.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
            )
        )

    categories: dict[int, CategoryRecord] = {}
    for raw in categories_raw:
        if not isinstance(raw, dict):
            raise ValidationError("COCO category record must be an object")
        if parser.category_fields.id not in raw:
            raise ValidationError("COCO category missing id")
        class_id = int(raw[parser.category_fields.id])
        categories[class_id] = CategoryRecord(
            class_id=class_id,
            name=str(raw.get(parser.category_fields.name, "")).strip(),
            supercategory=(
                str(raw.get(parser.category_fields.supercategory, "")).strip() or None
                if parser.category_fields.supercategory
                else None
            ),
        )

    annotations: list[AnnotationRecord] = []
    for raw in annotations_raw:
        if not isinstance(raw, dict):
            raise ValidationError("COCO annotation record must be an object")
        image_id = str(raw.get(parser.annotation_fields.image_id, "")).strip()
        if image_id not in included_image_ids:
            continue
        bbox = raw.get(parser.annotation_fields.bbox)
        bbox_positions = parser.bbox_fields
        if not isinstance(bbox, list):
            raise ValidationError("COCO annotation bbox must be a list")
        if len(bbox) < max(
            bbox_positions.xmin,
            bbox_positions.ymin,
            bbox_positions.width,
            bbox_positions.height,
        ):
            raise ValidationError("COCO annotation bbox must provide xmin/ymin/width/height values")
        annotations.append(
            AnnotationRecord(
                annotation_id=str(raw.get(parser.annotation_fields.id, "")).strip(),
                image_id=image_id,
                class_id=int(raw[parser.annotation_fields.class_id]),
                bbox_xywh_abs=(
                    float(bbox[bbox_positions.xmin - 1]),
                    float(bbox[bbox_positions.ymin - 1]),
                    float(bbox[bbox_positions.width - 1]),
                    float(bbox[bbox_positions.height - 1]),
                ),
                iscrowd=bool(raw.get(parser.annotation_fields.iscrowd, 0))
                if parser.annotation_fields.iscrowd
                else False,
            )
        )

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.COCO,
        images=sorted(images, key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda ann: ann.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(dataset_root=str(dataset_root.resolve()), loader="coco_reader"),
    )
