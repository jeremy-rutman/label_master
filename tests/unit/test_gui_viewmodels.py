from __future__ import annotations

from pathlib import Path

from PIL import Image

from label_master.interfaces.gui.viewmodels import (
    MappingRowViewModel,
    parse_mapping_rows,
    preview_dataset_view,
)


def test_parse_mapping_rows_valid_entries() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="2", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="map", destination_class_id="10"),
            MappingRowViewModel(source_class_id="4", action="map", destination_class_id="11"),
        ]
    )

    assert parsed.errors == []
    assert parsed.class_map == {2: None, 3: 10, 4: 11}


def test_parse_mapping_rows_validation_errors() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="foo", action="map", destination_class_id="1"),
            MappingRowViewModel(source_class_id="1", action="map", destination_class_id=""),
            MappingRowViewModel(source_class_id="1", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="2", action="invalid", destination_class_id=""),
        ]
    )

    assert parsed.errors == [
        "Row 1: source_class_id must be an integer",
        "Row 2: destination_class_id is required when action is 'map'",
        "Row 3: duplicate source_class_id 1",
        "Row 4: action must be 'map' or 'drop'",
    ]
    assert parsed.class_map == {}


def test_parse_mapping_rows_invalid_destination_integer() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="9", action="map", destination_class_id="cat"),
        ]
    )

    assert parsed.errors == ["Row 1: destination_class_id must be an integer"]
    assert parsed.class_map == {}


def test_parse_mapping_rows_ignores_blank_trailing_rows() -> None:
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="", action="map", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="", action="map", destination_class_id=""),
        ]
    )

    assert parsed.errors == []
    assert parsed.class_map == {3: None}


def test_preview_dataset_view_coco_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us1/coco_minimal"),
        source_format="coco",
    )

    assert preview.source_format == "coco"
    assert preview.image_count == len(preview.images)
    assert preview.image_count > 0
    assert preview.images[0].bboxes


def test_preview_dataset_view_yolo_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us1/yolo_minimal"),
        source_format="yolo",
    )

    assert preview.source_format == "yolo"
    assert preview.image_count == len(preview.images)
    assert preview.image_count > 0
    assert preview.images[0].bboxes


def test_preview_dataset_view_yolo_with_incomplete_classes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_root = tmp_path / "yolo_missing_classes"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "0.txt").write_text("4 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("class_zero_only\n", encoding="utf-8")

    preview = preview_dataset_view(dataset_root, source_format="yolo")

    assert preview.source_format == "yolo"
    assert preview.images
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_id == 4
    assert preview.images[0].bboxes[0].class_name == "class_4"


def test_preview_dataset_view_yolo_uses_actual_image_size_when_missing_manifest(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    dataset_root = tmp_path / "yolo_missing_sizes"
    labels_dir = dataset_root / "labels"
    images_dir = dataset_root / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    Image.new("RGB", (100, 80), color="black").save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("class_zero\n", encoding="utf-8")

    preview = preview_dataset_view(dataset_root, source_format="yolo")
    bbox = preview.images[0].bboxes[0]

    assert bbox.bbox_xywh_abs[2] == 20.0
    assert bbox.bbox_xywh_abs[3] == 20.0
