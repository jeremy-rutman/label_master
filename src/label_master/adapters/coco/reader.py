from __future__ import annotations

import json
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


def read_coco_dataset(dataset_root: Path) -> AnnotationDataset:
    annotations_path = dataset_root / "annotations.json"
    if not annotations_path.exists():
        raise ValidationError(f"COCO annotations file not found: {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValidationError("COCO annotations payload must be an object")

    images_raw = payload.get("images", [])
    annotations_raw = payload.get("annotations", [])
    categories_raw = payload.get("categories", [])
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list) or not isinstance(categories_raw, list):
        raise ValidationError("COCO images/annotations/categories must be arrays")

    images: list[ImageRecord] = []
    for raw in images_raw:
        if not isinstance(raw, dict):
            raise ValidationError("COCO image record must be an object")
        image_id = str(raw.get("id", "")).strip()
        file_name = str(raw.get("file_name", "")).strip()
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
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
        if "id" not in raw:
            raise ValidationError("COCO category missing id")
        class_id = int(raw["id"])
        categories[class_id] = CategoryRecord(
            class_id=class_id,
            name=str(raw.get("name", "")).strip(),
            supercategory=str(raw.get("supercategory", "")).strip() or None,
        )

    annotations: list[AnnotationRecord] = []
    for raw in annotations_raw:
        if not isinstance(raw, dict):
            raise ValidationError("COCO annotation record must be an object")
        bbox = raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValidationError("COCO annotation bbox must be [x, y, w, h]")
        annotations.append(
            AnnotationRecord(
                annotation_id=str(raw.get("id", "")).strip(),
                image_id=str(raw.get("image_id", "")).strip(),
                class_id=int(raw["category_id"]),
                bbox_xywh_abs=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                iscrowd=bool(raw.get("iscrowd", 0)),
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
