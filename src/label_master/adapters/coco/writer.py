from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from label_master.core.domain.entities import AnnotationDataset
from label_master.infra.filesystem import ensure_directory


def write_coco_dataset(dataset: AnnotationDataset, output_root: Path) -> Path:
    ensure_directory(output_root)
    annotations_path = output_root / "annotations.json"

    image_ids = {image.image_id for image in dataset.images}
    image_records: list[dict[str, Any]] = []
    for image in sorted(dataset.images, key=lambda item: item.image_id):
        image_records.append(
            {
                "id": image.image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }
        )

    annotation_records: list[dict[str, Any]] = []
    for annotation in dataset.deterministic_annotations():
        if annotation.image_id not in image_ids:
            continue
        x, y, w, h = annotation.bbox_xywh_abs
        annotation_records.append(
            {
                "id": annotation.annotation_id,
                "image_id": annotation.image_id,
                "category_id": annotation.class_id,
                "bbox": [x, y, w, h],
                "iscrowd": 1 if annotation.iscrowd else 0,
            }
        )

    category_records = [
        {
            "id": category.class_id,
            "name": category.name,
            "supercategory": category.supercategory or "",
        }
        for category in sorted(dataset.categories.values(), key=lambda item: item.class_id)
    ]

    payload = {
        "images": image_records,
        "annotations": annotation_records,
        "categories": category_records,
    }

    with annotations_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

    return annotations_path
