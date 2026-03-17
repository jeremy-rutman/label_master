from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui.app import (
    attach_mapping_labels,
    build_identity_mapping_rows,
    describe_class_label_source,
    extract_class_labels_from_preview,
    materialize_mapping_rows,
)
from label_master.interfaces.gui.viewmodels import preview_dataset_view


def test_build_identity_mapping_rows_from_detected_classes() -> None:
    rows = build_identity_mapping_rows({3: "car", 1: "person"})
    assert rows == [
        {"source_class_id": "1", "action": "map", "destination_class_id": "1"},
        {"source_class_id": "3", "action": "map", "destination_class_id": "3"},
    ]


def test_materialize_mapping_rows_omits_blank_rows() -> None:
    rows = materialize_mapping_rows(
        [
            {"source_class_id": "", "action": "map", "destination_class_id": ""},
            {"source_class_id": "2", "action": "drop", "destination_class_id": ""},
            {"source_class_id": "3", "action": "map", "destination_class_id": "9"},
        ]
    )
    assert rows == [
        {"source_class_id": "2", "action": "drop", "destination_class_id": ""},
        {"source_class_id": "3", "action": "map", "destination_class_id": "9"},
    ]


def test_attach_mapping_labels_when_names_available() -> None:
    rows = attach_mapping_labels(
        [{"source_class_id": "3", "action": "map", "destination_class_id": "10"}],
        {3: "drone", 10: "uav"},
    )
    assert rows == [
        {
            "source_class_id": "3",
            "action": "map",
            "destination_class_id": "10",
            "source_label": "drone",
            "destination_label": "uav",
        }
    ]


def test_extract_class_labels_from_preview() -> None:
    preview = preview_dataset_view(Path("tests/fixtures/us1/coco_minimal"), source_format="coco")
    labels = extract_class_labels_from_preview(preview)

    assert labels
    assert all(isinstance(class_id, int) for class_id in labels)
    assert all(isinstance(name, str) and name for name in labels.values())


def test_describe_class_label_source_yolo_from_classes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "classes.txt").write_text("person\nvehicle\n", encoding="utf-8")

    description = describe_class_label_source(
        input_path=tmp_path,
        source_format="yolo",
        class_labels={0: "person", 1: "vehicle"},
    )

    assert description == "YOLO labels source: classes.txt."


def test_describe_class_label_source_yolo_missing_ids_fallback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "classes.txt").write_text("person\n", encoding="utf-8")

    description = describe_class_label_source(
        input_path=tmp_path,
        source_format="yolo",
        class_labels={0: "person", 2: "class_2"},
    )

    assert "YOLO labels source: classes.txt." in description
    assert "Missing class IDs (2)" in description
    assert "fallback names class_<id>" in description


def test_describe_class_label_source_yolo_without_classes_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    description = describe_class_label_source(
        input_path=tmp_path,
        source_format="yolo",
        class_labels={0: "class_0"},
    )

    assert description == "YOLO labels source: classes.txt not found; using fallback names class_<id>."
