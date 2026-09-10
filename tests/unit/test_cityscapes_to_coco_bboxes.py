from __future__ import annotations

import json
from pathlib import Path

from label_master.adapters.cityscapes.reader import read_cityscapes_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format
from label_master.tools.cityscapes_to_coco_bboxes import export_cityscapes_split_to_coco


def _write_cityscapes_pair(dataset_root: Path) -> None:
    labels_dir = dataset_root / "gtCoarse" / "gtCoarse" / "train" / "sample_city"
    images_dir = dataset_root / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train" / "sample_city"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / "sample_city_000001_000001_leftImg8bit.png").write_bytes(b"fake-png")
    (labels_dir / "sample_city_000001_000001_gtCoarse_polygons.json").write_text(
        json.dumps(
            {
                "imgWidth": 100,
                "imgHeight": 60,
                "objects": [
                    {"label": "road", "polygon": [[0, 40], [99, 40], [99, 59], [0, 59]]},
                    {"label": "car", "polygon": [[10, 20], [30, 20], [30, 40], [10, 40]]},
                    {"label": "person", "polygon": [[50, 5], [55, 5], [55, 20], [50, 20]]},
                    {"label": "cargroup", "polygon": [[60, 10], [70, 10], [70, 25], [60, 25]]},
                    {"label": "car", "deleted": 1, "polygon": [[80, 10], [85, 10], [85, 15], [80, 15]]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_read_cityscapes_dataset_infers_and_filters_to_instance_labels_and_groups(tmp_path: Path) -> None:
    _write_cityscapes_pair(tmp_path)

    inference = infer_format(tmp_path, force=True)
    dataset = read_cityscapes_dataset(
        tmp_path,
        split_names=["train"],
        label_mode="instances",
        include_groups=True,
    )

    assert inference.predicted_format == SourceFormat.CITYSCAPES
    assert dataset.source_format == SourceFormat.CITYSCAPES
    assert dataset.source_metadata.details["labels_variant"] == "gtCoarse"
    assert dataset.source_metadata.details["annotation_files_skipped"] == "0"
    assert dataset.source_metadata.details["skipped_objects"] == "2"
    assert [image.file_name for image in dataset.images] == [
        "leftImg8bit_trainvaltest/leftImg8bit/train/sample_city/sample_city_000001_000001_leftImg8bit.png"
    ]
    assert [category.name for _, category in sorted(dataset.categories.items())] == [
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    assert [annotation.class_id for annotation in dataset.annotations] == [2, 0, 2]
    assert [annotation.iscrowd for annotation in dataset.annotations] == [False, False, True]
    assert [annotation.bbox_xywh_abs for annotation in dataset.annotations] == [
        (10.0, 20.0, 21.0, 21.0),
        (50.0, 5.0, 6.0, 16.0),
        (60.0, 10.0, 11.0, 16.0),
    ]


def test_export_cityscapes_split_to_coco_filters_to_instance_labels_and_groups(tmp_path: Path) -> None:
    _write_cityscapes_pair(tmp_path)
    output = tmp_path / "out" / "annotations.json"

    summary = export_cityscapes_split_to_coco(
        dataset_root=tmp_path,
        split_name="train",
        output_path=output,
        label_mode="instances",
        include_groups=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert summary.images == 1
    assert summary.annotations == 3
    assert summary.skipped_annotation_files == 0
    assert summary.skipped_objects == 2
    assert payload["images"] == [
        {
            "id": "sample_city_000001_000001",
            "file_name": (
                "leftImg8bit_trainvaltest/leftImg8bit/train/sample_city/"
                "sample_city_000001_000001_leftImg8bit.png"
            ),
            "width": 100,
            "height": 60,
        }
    ]
    assert payload["categories"] == [
        {"id": 0, "name": "person", "supercategory": "human"},
        {"id": 1, "name": "rider", "supercategory": "human"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "truck", "supercategory": "vehicle"},
        {"id": 4, "name": "bus", "supercategory": "vehicle"},
        {"id": 5, "name": "train", "supercategory": "vehicle"},
        {"id": 6, "name": "motorcycle", "supercategory": "vehicle"},
        {"id": 7, "name": "bicycle", "supercategory": "vehicle"},
    ]
    assert payload["annotations"] == [
        {
            "id": "2",
            "image_id": "sample_city_000001_000001",
            "category_id": 0,
            "bbox": [50.0, 5.0, 6.0, 16.0],
            "iscrowd": 0,
        },
        {
            "id": "1",
            "image_id": "sample_city_000001_000001",
            "category_id": 2,
            "bbox": [10.0, 20.0, 21.0, 21.0],
            "iscrowd": 0,
        },
        {
            "id": "3",
            "image_id": "sample_city_000001_000001",
            "category_id": 2,
            "bbox": [60.0, 10.0, 11.0, 16.0],
            "iscrowd": 1,
        },
    ]