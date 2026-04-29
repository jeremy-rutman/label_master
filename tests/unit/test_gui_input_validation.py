from __future__ import annotations

import json

from label_master.core.domain.value_objects import ValidationError
from label_master.interfaces.gui.app import (
    DEFAULT_DESTINATION_FORMAT,
    DESTINATION_FORMATS,
    _build_inference_payload,
    _coerce_preview_keyboard_navigation_action,
    _consume_preview_skip_once,
    _format_oversize_image_action_label,
    _inference_payload_matches_input_path,
    _resolve_preview_index,
    default_gui_input_directory,
    format_details_yaml,
    format_run_exception_details,
    gui_state_path,
    load_gui_state,
    load_last_used_input_directory,
    load_persisted_gui_preferences,
    persist_gui_state,
    persist_last_used_input_directory,
    run_blocking_errors,
    validate_input_directory,
)


def test_default_destination_format_is_yolo() -> None:
    assert DEFAULT_DESTINATION_FORMAT == "yolo"
    assert DESTINATION_FORMATS[0] == "yolo"


def test_oversize_image_action_label_uses_drop_for_internal_ignore() -> None:
    assert _format_oversize_image_action_label("ignore") == "drop"
    assert _format_oversize_image_action_label("downscale") == "downscale"


def test_preview_skip_flag_is_consumed_once() -> None:
    session_state: dict[str, object] = {"gui_skip_preview_once": True}

    assert _consume_preview_skip_once(session_state) is True
    assert session_state == {}
    assert _consume_preview_skip_once(session_state) is False


def test_resolve_preview_index_handles_navigation_actions() -> None:
    assert _resolve_preview_index(3, max_index=9, keyboard_action="previous") == 2
    assert _resolve_preview_index(3, max_index=9, keyboard_action="next") == 4
    assert _resolve_preview_index(3, max_index=9, previous_clicked=True) == 2
    assert _resolve_preview_index(3, max_index=9, next_clicked=True) == 4
    assert _resolve_preview_index(0, max_index=9, keyboard_action="previous") == 0
    assert _resolve_preview_index(9, max_index=9, next_clicked=True) == 9


def test_build_inference_payload_tracks_input_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _build_inference_payload(
        type(
            "InferVM",
            (),
            {
                "predicted_format": "matlab_ground_truth",
                "confidence": 0.99,
                "candidates": [("matlab_ground_truth", 0.99)],
                "warnings": [],
            },
        )(),
        input_path=tmp_path,
    )

    assert payload["predicted_format"] == "matlab_ground_truth"
    assert payload["input_dir"] == str(tmp_path.resolve())


def test_inference_payload_requires_matching_input_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "predicted_format": "matlab_ground_truth",
        "confidence": 0.99,
        "candidates": [("matlab_ground_truth", 0.99)],
        "warnings": [],
        "input_dir": str(tmp_path.resolve()),
    }

    assert _inference_payload_matches_input_path(payload, tmp_path) is True
    assert _inference_payload_matches_input_path(payload, tmp_path / "other") is False
    assert _inference_payload_matches_input_path({"predicted_format": "matlab_ground_truth"}, tmp_path) is False


def test_validate_input_directory_requires_value() -> None:
    result = validate_input_directory("")
    assert result.errors == ["Input directory is required"]


def test_validate_input_directory_missing_path() -> None:
    result = validate_input_directory("/tmp/label_master_missing_path_for_gui_validation")
    assert result.errors == ["Input directory does not exist"]


def test_validate_input_directory_requires_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")

    result = validate_input_directory(str(file_path))
    assert result.errors == ["Input directory must be a directory"]


def test_validate_input_directory_requires_readable_directory(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("label_master.interfaces.gui.app._is_readable_directory", lambda path: False)

    result = validate_input_directory(str(tmp_path))
    assert result.errors == ["Input directory must be readable"]


def test_validate_input_directory_valid_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = validate_input_directory(str(tmp_path))
    assert result.errors == []
    assert result.resolved_path == tmp_path.resolve()


def test_run_blocking_errors_include_mapping_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    errors = run_blocking_errors(
        input_dir_raw=str(tmp_path),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=["Row 2: duplicate source_class_id 3"],
    )
    assert "Row 2: duplicate source_class_id 3" in errors


def test_format_run_exception_details_surfaces_validation_context() -> None:
    expected_issue_rows = [
        {
            "kind": "First issue",
            "issue": "bbox goes out of frame beyond tolerance",
            "annotation_id": "annotations/example.txt:000001:001",
            "image_id": "img-1",
            "image_file": "images/example.jpg",
            "bbox_xywh_abs": "(95.00, 40.00, 30.00, 10.00)",
            "frame_bounds": "width=100, height=50",
            "overflow_px": "right=25.00",
        },
        {
            "kind": "Sample issue",
            "issue": "has non-positive bbox size",
            "annotation_id": "annotations/example.txt:000002:001",
            "image_id": "img-2",
            "image_file": "images/example_2.jpg",
            "bbox_xywh_abs": "(10.00, 20.00, 0.00, 12.00)",
            "frame_bounds": "width=100, height=50",
        },
    ]
    exc = ValidationError(
        "Validation failed in strict mode: 17 invalid annotation(s)",
        context={
            "invalid_annotations": "17",
            "first_error": (
                "Annotation annotations/example.txt:000001:001 bbox goes out of frame beyond tolerance: "
                "image_id=img-1, image_file=images/example.jpg, bbox_xywh_abs=(95.00, 40.00, 30.00, 10.00), "
                "frame_bounds=width=100, height=50, overflow_px=right=25.00"
            ),
            "sample_errors": (
                "Annotation annotations/example.txt:000001:001 bbox goes out of frame beyond tolerance\n"
                "Annotation annotations/example.txt:000002:001 has non-positive bbox size"
            ),
            "issue_rows_json": json.dumps(expected_issue_rows),
            "predicted": "video_bbox",
        },
    )

    summary, details, issue_rows = format_run_exception_details(exc)

    assert summary == "Validation failed in strict mode: 17 invalid annotation(s)"
    assert details == [
        "Invalid annotations: 17",
        "Predicted: video_bbox",
    ]
    assert issue_rows == expected_issue_rows


def test_format_run_exception_details_handles_plain_exceptions() -> None:
    summary, details, issue_rows = format_run_exception_details(RuntimeError("plain failure"))

    assert summary == "plain failure"
    assert details == []
    assert issue_rows == []


def test_format_details_yaml_prefers_builtin_format_spec() -> None:
    rendered = format_details_yaml("coco", dataset_root=None, inference_payload={"predicted_format": "coco"})

    assert rendered is not None
    assert "format_id: coco" in rendered
    assert "kind: json_object_dataset" in rendered
    assert "images_key: images" in rendered
    assert "bbox_fields:" in rendered
    assert "xmin: 1" in rendered
    assert "predicted_format" not in rendered


def test_format_details_yaml_falls_back_to_inference_payload_without_spec() -> None:
    rendered = format_details_yaml(
        "",
        dataset_root=None,
        inference_payload={"predicted_format": "custom", "confidence": 0.91},
    )

    assert rendered == "predicted_format: custom\nconfidence: 0.91"


def test_format_details_yaml_renders_video_bbox_token_mapping() -> None:
    rendered = format_details_yaml("video_bbox", dataset_root=None, inference_payload=None)

    assert rendered is not None
    assert "format_id: video_bbox" in rendered
    assert "kind: tokenized_video" in rendered
    assert "frame_index_field: 1" in rendered
    assert "object_count_field: 2" in rendered
    assert "xmin: 1" in rendered
    assert "ymin: 2" in rendered
    assert "class_name: 5" in rendered


def test_run_blocking_errors_reject_invalid_size_gate_configuration(tmp_path) -> None:  # type: ignore[no-untyped-def]
    errors = run_blocking_errors(
        input_dir_raw=str(tmp_path),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=[],
        min_image_longest_edge_px=1200,
        max_image_longest_edge_px=1000,
        oversize_image_action="ignore",
    )

    assert "Minimum image size gate cannot exceed the maximum image size gate" in errors


def test_run_blocking_errors_require_copy_images_for_downscale(tmp_path) -> None:  # type: ignore[no-untyped-def]
    errors = run_blocking_errors(
        input_dir_raw=str(tmp_path),
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=[],
        copy_images=False,
        dry_run=False,
        max_image_longest_edge_px=1000,
        oversize_image_action="downscale",
    )

    assert "Downscaling oversized images requires 'Copy images to output' unless this is a dry run" in errors


def test_preview_keyboard_navigation_action_accepts_fresh_next_event() -> None:
    action, nonce = _coerce_preview_keyboard_navigation_action(
        {"action": "next", "nonce": 2},
        last_nonce=1,
    )

    assert action == "next"
    assert nonce == 2


def test_preview_keyboard_navigation_action_ignores_stale_event() -> None:
    action, nonce = _coerce_preview_keyboard_navigation_action(
        {"action": "previous", "nonce": 3},
        last_nonce=3,
    )

    assert action is None
    assert nonce == 3


def test_preview_keyboard_navigation_action_rejects_unknown_action() -> None:
    action, nonce = _coerce_preview_keyboard_navigation_action(
        {"action": "jump", "nonce": 4},
        last_nonce=2,
    )

    assert action is None
    assert nonce == 4


def test_persist_last_used_input_directory_writes_state_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"

    written_path = persist_last_used_input_directory("/tmp/example_dataset", state_path=state_path)

    assert written_path == state_path
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {"last_input_dir": "/tmp/example_dataset"}


def test_load_last_used_input_directory_reads_persisted_value(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    state_path.write_text(json.dumps({"last_input_dir": "/tmp/example_dataset"}), encoding="utf-8")

    assert load_last_used_input_directory(state_path) == "/tmp/example_dataset"


def test_load_last_used_input_directory_returns_none_for_missing_or_invalid_file(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    missing_path = tmp_path / "missing.json"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not-json", encoding="utf-8")

    assert load_last_used_input_directory(missing_path) is None
    assert load_last_used_input_directory(invalid_path) is None


def test_default_gui_input_directory_prefers_persisted_value(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    persist_last_used_input_directory("/tmp/last_used_dataset", state_path=state_path)
    monkeypatch.setattr("label_master.interfaces.gui.app.gui_state_path", lambda: state_path)

    assert default_gui_input_directory() == "/tmp/last_used_dataset"


def test_gui_state_path_uses_home_directory(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    assert gui_state_path() == tmp_path / ".label_master" / "gui_state.json"


def test_persist_gui_state_round_trips_full_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    payload = {
        "last_input_dir": "/tmp/input",
        "last_output_dir": "/tmp/output",
        "last_dst": "coco",
        "last_validation_mode": "permissive",
        "last_permissive_invalid_annotation_action": "drop",
        "last_allow_shared_output_dir": True,
        "last_prefix_output_filenames": True,
        "last_allow_overwrite": True,
        "last_input_path_include_substring": "train",
        "last_input_path_exclude_substring": "backup",
        "last_output_file_stem_prefix": "batchA_",
        "last_output_file_stem_suffix": "_fold1",
        "last_inference_payload": {"predicted_format": "kitware", "confidence": 1.0},
        "last_mapping_rows": [
            {"source_class_id": "3", "action": "map", "destination_class_id": "10"},
        ],
        "last_mapping_seed_signature": "dataset-signature",
    }

    written_path = persist_gui_state(payload, state_path=state_path)

    assert written_path == state_path
    assert load_gui_state(state_path) == payload


def test_persist_last_used_input_directory_preserves_other_gui_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    persist_gui_state({"last_output_dir": "/tmp/output"}, state_path=state_path)

    persist_last_used_input_directory("/tmp/input", state_path=state_path)

    assert load_gui_state(state_path) == {
        "last_input_dir": "/tmp/input",
        "last_output_dir": "/tmp/output",
    }


def test_load_persisted_gui_preferences_restores_mapping_and_output_state(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    persist_gui_state(
        {
            "last_input_dir": "/tmp/input",
            "last_output_dir": "/tmp/output",
            "last_dst": "coco",
            "last_validation_mode": "permissive",
            "last_permissive_invalid_annotation_action": "drop",
            "last_allow_shared_output_dir": False,
            "last_prefix_output_filenames": True,
            "last_allow_overwrite": True,
            "last_input_path_include_substring": "train",
            "last_input_path_exclude_substring": "backup",
            "last_output_file_stem_prefix": "batchA_",
            "last_output_file_stem_suffix": "_fold1",
            "last_correct_out_of_frame_bboxes": False,
            "last_out_of_frame_tolerance_px": 2.5,
            "last_min_image_longest_edge_px": 320,
            "last_max_image_longest_edge_px": 1440,
            "last_oversize_image_action": "downscale",
            "last_inference_payload": {"predicted_format": "kitware", "confidence": 0.99},
            "last_mapping_rows": [
                {"source_class_id": "3", "action": "map", "destination_class_id": "10"},
            ],
            "last_mapping_seed_signature": "dataset-signature",
        },
        state_path=state_path,
    )

    preferences = load_persisted_gui_preferences(state_path)

    assert preferences["gui_input_dir"] == "/tmp/input"
    assert preferences["gui_output_dir"] == "/tmp/output"
    assert preferences["gui_dst"] == "yolo"
    assert preferences["gui_validation_mode"] == "permissive"
    assert preferences["gui_permissive_invalid_annotation_action"] == "drop"
    assert preferences["gui_allow_shared_output_dir"] is False
    assert preferences["gui_prefix_output_filenames"] is True
    assert preferences["gui_allow_overwrite"] is True
    assert preferences["gui_input_path_include_substring"] == "train"
    assert preferences["gui_input_path_exclude_substring"] == "backup"
    assert preferences["gui_output_file_stem_prefix"] == "batchA_"
    assert preferences["gui_output_file_stem_suffix"] == "_fold1"
    assert preferences["gui_correct_out_of_frame_bboxes"] is False
    assert preferences["gui_out_of_frame_tolerance_px"] == 2.5
    assert preferences["gui_min_image_longest_edge_px"] == 320
    assert preferences["gui_max_image_longest_edge_px"] == 1440
    assert preferences["gui_oversize_image_action"] == "downscale"
    assert preferences["gui_inference_payload"] == {"predicted_format": "kitware", "confidence": 0.99}
    assert preferences["gui_mapping_rows"] == [
        {"source_class_id": "3", "action": "map", "destination_class_id": "10"},
    ]
    assert preferences["gui_mapping_seed_signature"] == "dataset-signature"


def test_load_persisted_gui_preferences_defaults_out_of_frame_tolerance_to_20(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    preferences = load_persisted_gui_preferences(tmp_path / "missing_state.json")

    assert preferences["gui_validation_mode"] == "strict"
    assert preferences["gui_permissive_invalid_annotation_action"] == "keep"
    assert preferences["gui_correct_out_of_frame_bboxes"] is True
    assert preferences["gui_out_of_frame_tolerance_px"] == 20.0


def test_load_persisted_gui_preferences_migrates_zero_out_of_frame_tolerance_when_enabled(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "gui_state.json"
    persist_gui_state(
        {
            "last_correct_out_of_frame_bboxes": True,
            "last_out_of_frame_tolerance_px": 0.0,
        },
        state_path=state_path,
    )

    preferences = load_persisted_gui_preferences(state_path)

    assert preferences["gui_out_of_frame_tolerance_px"] == 20.0
