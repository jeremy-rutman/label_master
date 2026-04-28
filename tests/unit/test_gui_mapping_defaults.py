from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui.app import (
    _apply_mapping_editor_state,
    _format_mapping_action_label,
    attach_mapping_labels,
    build_identity_mapping_rows,
    describe_class_label_source,
    extract_class_labels_from_preview,
    materialize_mapping_rows,
    normalize_mapping_rows,
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


def test_normalize_mapping_rows_preserves_destination_text_for_drop_rows() -> None:
    rows = normalize_mapping_rows(
        [
            {"source_class_id": "2", "action": "drop", "destination_class_id": "17"},
        ]
    )

    assert rows == [
        {"source_class_id": "2", "action": "drop", "destination_class_id": "17"},
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
        }
    ]


def test_attach_mapping_labels_only_adds_source_label() -> None:
    rows = attach_mapping_labels(
        [{"source_class_id": "3", "action": "drop", "destination_class_id": "10"}],
        {3: "drone", 10: "uav"},
    )

    assert rows == [
        {
            "source_class_id": "3",
            "action": "drop",
            "destination_class_id": "10",
            "source_label": "drone",
        }
    ]


def test_format_mapping_action_label_uses_keep_for_map() -> None:
    assert _format_mapping_action_label("map") == "keep"
    assert _format_mapping_action_label("drop") == "drop"


def test_apply_mapping_editor_state_preserves_destination_updates() -> None:
    rows = [
        {"source_class_id": "0", "action": "map", "destination_class_id": "0"},
        {"source_class_id": "1", "action": "map", "destination_class_id": "1"},
    ]
    editor_state = {
        "edited_rows": {
            1: {
                "destination_class_id": "10",
            }
        }
    }

    updated = _apply_mapping_editor_state(rows, editor_state)

    assert updated == [
        {"source_class_id": "0", "action": "map", "destination_class_id": "0"},
        {"source_class_id": "1", "action": "map", "destination_class_id": "10"},
    ]


def test_apply_mapping_editor_state_ignores_source_class_id_edits() -> None:
    rows = [
        {"source_class_id": "0", "action": "map", "destination_class_id": "0"},
        {"source_class_id": "1", "action": "map", "destination_class_id": "1"},
    ]
    editor_state = {
        "edited_rows": {
            1: {
                "source_class_id": "99",
                "destination_class_id": "10",
            }
        }
    }

    updated = _apply_mapping_editor_state(rows, editor_state)

    assert updated == [
        {"source_class_id": "0", "action": "map", "destination_class_id": "0"},
        {"source_class_id": "1", "action": "map", "destination_class_id": "10"},
    ]


def test_apply_mapping_editor_state_keeps_manual_destination_text() -> None:
    rows = [
        {"source_class_id": "1", "action": "map", "destination_class_id": "42"},
    ]
    editor_state = {
        "edited_rows": {
            0: {
                "destination_class_id": "77",
            }
        }
    }

    updated = _apply_mapping_editor_state(rows, editor_state)

    assert updated == [
        {"source_class_id": "1", "action": "map", "destination_class_id": "77"},
    ]


def test_apply_mapping_editor_state_handles_added_and_deleted_rows() -> None:
    rows = [
        {"source_class_id": "0", "action": "map", "destination_class_id": "0"},
        {"source_class_id": "1", "action": "map", "destination_class_id": "1"},
    ]
    editor_state = {
        "deleted_rows": [0],
        "added_rows": [
            {
                "source_class_id": "3",
                "action": "drop",
                "destination_class_id": "99",
            }
        ],
    }

    updated = _apply_mapping_editor_state(rows, editor_state)

    assert updated == [
        {"source_class_id": "1", "action": "map", "destination_class_id": "1"},
        {"source_class_id": "3", "action": "drop", "destination_class_id": "99"},
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

    assert description == "YOLO labels source: classes.txt/obj.names not found; using fallback names class_<id>."


def test_describe_class_label_source_yolo_from_obj_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "obj.names").write_text("drone\n", encoding="utf-8")

    description = describe_class_label_source(
        input_path=tmp_path,
        source_format="yolo",
        class_labels={0: "drone"},
    )

    assert description == "YOLO labels source: obj.names."


def test_describe_class_label_source_kitware() -> None:
    description = describe_class_label_source(
        input_path=Path("tests/fixtures/us4"),
        source_format="kitware",
        class_labels={0: "airplane"},
    )

    assert description == "Kitware labels source: per-directory CSV bbox columns."


def test_describe_class_label_source_voc() -> None:
    description = describe_class_label_source(
        input_path=Path("tests/fixtures/us6"),
        source_format="voc",
        class_labels={0: "UAV"},
    )

    assert description == "VOC labels source: Pascal VOC XML object names."


def test_describe_class_label_source_video_bbox() -> None:
    description = describe_class_label_source(
        input_path=Path("tests/fixtures/us5"),
        source_format="video_bbox",
        class_labels={0: "object"},
    )

    assert description == "Video bbox labels source: per-sequence tracking ground-truth text files."
