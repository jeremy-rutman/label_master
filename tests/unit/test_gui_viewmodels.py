from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import label_master.interfaces.gui.viewmodels as gui_viewmodels
from label_master.core.domain.entities import SourceFormat
from label_master.interfaces.gui.viewmodels import (
    MappingRowViewModel,
    _preview_dataset_view_cached,
    build_gui_run_config,
    convert_view,
    infer_view,
    parse_mapping_rows,
    preview_dataset_view,
)


def _write_coco_dataset(
    dataset_root: Path,
    *,
    bbox: tuple[float, float, float, float],
) -> None:
    payload = {
        "images": [
            {
                "id": "img-1",
                "file_name": "images/example.jpg",
                "width": 100,
                "height": 50,
            }
        ],
        "annotations": [
            {
                "id": "ann-1",
                "image_id": "img-1",
                "category_id": 0,
                "bbox": list(bbox),
            }
        ],
        "categories": [{"id": 0, "name": "object"}],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_yolo_dataset(
    dataset_root: Path,
    *,
    rows: list[str],
    image_size: tuple[int, int] = (100, 50),
    image_rel: str = "train/images/example.jpg",
    label_rel: str = "train/labels/example.txt",
) -> None:
    image_path = dataset_root / image_rel
    label_path = dataset_root / label_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color="black").save(image_path)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text("object\n", encoding="utf-8")


def _write_anti_uav_preview_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "Anti_UAV_RGBT"
    for clip_name, rows in {
        "clip_a": {"exist": [1, 0], "gt_rect": [[10, 20, 30, 40], []]},
        "clip_b": {"exist": [0, 1], "gt_rect": [[], [5, 6, 7, 8]]},
    }.items():
        clip_root = dataset_root / "train" / clip_name
        clip_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("tests/fixtures/us9/V_BIRD_029.mp4"), clip_root / "visible.mp4")
        (clip_root / "visible.json").write_text(json.dumps(rows), encoding="utf-8")
    return dataset_root


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


def test_preview_dataset_view_yolo_sidecar_img_labels_with_obj_names(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 0.2 0.4"],
        image_rel="train/img/example.jpg",
        label_rel="train/img/example.txt",
    )
    (tmp_path / "classes.txt").unlink()
    (tmp_path / "train").mkdir(exist_ok=True)
    (tmp_path / "train" / "obj.names").write_text("drone\n", encoding="utf-8")

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.source_format == "yolo"
    assert preview.images[0].file_name == "train/img/example.jpg"
    assert preview.images[0].bboxes[0].class_name == "drone"


def test_preview_dataset_view_yolo_clips_slightly_out_of_range_normalized_bbox(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 1.01 0.5 0.2 0.4"],
    )

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 20px)."
    ]
    assert preview.images[0].file_name == "train/images/example.jpg"
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 15.0, 9.0, 20.0)


def test_preview_dataset_view_yolo_reports_large_out_of_range_normalized_bbox(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 1.6 0.4"],
    )

    preview = preview_dataset_view(tmp_path, source_format="yolo")

    assert preview.warnings == [
        "Preview loaded with 1 invalid annotation(s): bbox goes out of frame beyond the accepted 20px correction tolerance."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (-30.0, 15.0, 160.0, 20.0)


def test_preview_dataset_view_kitware_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us4"),
        source_format="kitware",
    )

    assert preview.source_format == "kitware"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 3673
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "airplane"


def test_preview_dataset_view_voc_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us6"),
        source_format="voc",
    )

    assert preview.source_format == "voc"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 5
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "UAV"
    assert preview.warnings == []


def test_preview_dataset_view_voc_caps_large_preview_sample(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _preview_dataset_view_cached.cache_clear()
    monkeypatch.setattr(gui_viewmodels, "VOC_PREVIEW_MAX_XML_FILES", 2)
    monkeypatch.setattr(gui_viewmodels, "VOC_PREVIEW_WARNING", "preview capped for test")

    preview = preview_dataset_view(
        Path("tests/fixtures/us6"),
        source_format="voc",
    )

    assert preview.source_format == "voc"
    assert preview.image_count == 2
    assert preview.warnings == ["preview capped for test"]

    _preview_dataset_view_cached.cache_clear()


def test_preview_dataset_view_video_bbox_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us5"),
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 183
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "object"


def test_preview_dataset_view_mot_video_bbox_contains_images_and_bboxes() -> None:
    preview = preview_dataset_view(
        Path("tests/fixtures/us7"),
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == len(preview.images)
    assert preview.image_count == 317
    assert preview.images[0].bboxes
    assert preview.images[0].bboxes[0].class_name == "object"
    assert preview.warnings == []


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_preview_dataset_view_video_bbox_caps_large_source_sets(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _preview_dataset_view_cached.cache_clear()
    monkeypatch.setattr(gui_viewmodels, "VIDEO_BBOX_PREVIEW_MAX_SOURCES", 1)
    monkeypatch.setattr(gui_viewmodels, "VIDEO_BBOX_PREVIEW_WARNING", "video preview capped for test")

    dataset_root = _write_anti_uav_preview_dataset(tmp_path)
    preview = preview_dataset_view(
        dataset_root,
        source_format="video_bbox",
    )

    assert preview.source_format == "video_bbox"
    assert preview.image_count == 2
    assert preview.warnings == ["video preview capped for test"]

    _preview_dataset_view_cached.cache_clear()


def test_preview_dataset_view_warns_when_bbox_is_auto_clipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    preview = preview_dataset_view(tmp_path, source_format="coco")

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 20px)."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)


def test_preview_dataset_view_explains_out_of_frame_invalid_annotations(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(95.0, 40.0, 30.0, 10.0))

    preview = preview_dataset_view(tmp_path, source_format="coco")

    assert preview.warnings == [
        "Preview loaded with 1 invalid annotation(s): bbox goes out of frame beyond the accepted 20px correction tolerance."
    ]


def test_preview_dataset_view_can_disable_out_of_frame_correction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.0, 10.0))

    preview = preview_dataset_view(
        tmp_path,
        source_format="coco",
        correct_out_of_frame_bboxes=False,
    )

    assert preview.warnings == [
        "Preview loaded with 1 invalid annotation(s): bbox goes out of frame. Enable 'Correct out-of-frame bboxes' in Step 3 to clip near-edge boxes."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 10.0, 10.0)


def test_preview_dataset_view_honors_custom_out_of_frame_tolerance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_coco_dataset(tmp_path, bbox=(91.0, 40.0, 10.6, 10.0))

    preview = preview_dataset_view(
        tmp_path,
        source_format="coco",
        out_of_frame_tolerance_px=2.0,
    )

    assert preview.warnings == [
        "Auto-corrected 1 annotation(s) whose bbox went slightly out of frame by clipping them to the image bounds (tolerance: <= 2px)."
    ]
    assert preview.images[0].bboxes[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)


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


def test_preview_dataset_view_reuses_cached_result(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    fake_validation = SimpleNamespace(
        dataset=SimpleNamespace(
            annotations=[],
            categories={},
            images=[
                SimpleNamespace(
                    image_id="image-1",
                    file_name="images/example.jpg",
                    width=100,
                    height=80,
                )
            ],
        ),
        summary=SimpleNamespace(invalid_annotations=0),
    )

    _preview_dataset_view_cached.cache_clear()

    def _fake_validate_dataset(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        calls.append("validate")
        return fake_validation

    monkeypatch.setattr(gui_viewmodels, "validate_dataset", _fake_validate_dataset)

    preview_one = preview_dataset_view(tmp_path, source_format="coco")
    preview_two = preview_dataset_view(tmp_path, source_format="coco")

    assert preview_one == preview_two
    assert calls == ["validate"]

    _preview_dataset_view_cached.cache_clear()


def test_infer_view_uses_fast_gui_sample_limit_by_default(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, int] = {}

    def _fake_infer_format(input_path, *, policy, force):  # type: ignore[no-untyped-def]
        del input_path, force
        observed["sample_limit"] = policy.sample_limit
        return SimpleNamespace(
            predicted_format=SourceFormat.KITWARE,
            confidence=1.0,
            candidates=[SimpleNamespace(format=SourceFormat.KITWARE, score=1.0)],
            warnings=[],
        )

    monkeypatch.setattr(gui_viewmodels, "infer_format", _fake_infer_format)

    vm = infer_view(tmp_path)

    assert observed["sample_limit"] == 100
    assert vm.predicted_format == "kitware"


def test_infer_view_detects_yolo_sidecar_img_layout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(
        tmp_path,
        rows=["0 0.5 0.5 0.2 0.4"],
        image_rel="train/img/example.jpg",
        label_rel="train/img/example.txt",
    )

    vm = infer_view(tmp_path)

    assert vm.predicted_format == "yolo"


def test_convert_view_passes_validation_mode_to_conversion_request(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, object] = {}

    def _fake_execute_conversion(request, *, progress_callback=None):  # type: ignore[no-untyped-def]
        del progress_callback
        observed["validation_mode"] = request.validation_mode.value
        observed["permissive_invalid_annotation_action"] = request.permissive_invalid_annotation_action.value
        observed["allow_overwrite"] = request.allow_overwrite
        observed["input_path_include_substring"] = request.input_path_include_substring
        observed["input_path_exclude_substring"] = request.input_path_exclude_substring
        return SimpleNamespace(
            report=SimpleNamespace(
                summary_counts=SimpleNamespace(
                    annotations_in=1,
                    annotations_out=1,
                    dropped=0,
                    unmapped=0,
                ),
                contention_events=[],
            ),
        )

    monkeypatch.setattr(gui_viewmodels, "execute_conversion", _fake_execute_conversion)

    vm, _result = convert_view(
        input_path=tmp_path,
        output_path=tmp_path / "out",
        src="coco",
        dst="yolo",
        map_path=None,
        unmapped_policy="error",
        dry_run=True,
        allow_overwrite=True,
        input_path_include_substring="train",
        input_path_exclude_substring="backup",
        validation_mode="permissive",
        permissive_invalid_annotation_action="drop",
    )

    assert observed["validation_mode"] == "permissive"
    assert observed["permissive_invalid_annotation_action"] == "drop"
    assert observed["allow_overwrite"] is True
    assert observed["input_path_include_substring"] == "train"
    assert observed["input_path_exclude_substring"] == "backup"
    assert vm.annotations_in == 1


def test_build_gui_run_config_includes_validation_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = build_gui_run_config(
        run_id="gui-test",
        input_path=tmp_path / "in",
        output_path=tmp_path / "out",
        src="coco",
        dst="yolo",
        map_path=None,
        unmapped_policy="error",
        dry_run=False,
        allow_overwrite=True,
        input_path_include_substring="train",
        input_path_exclude_substring="backup",
        validation_mode="permissive",
        permissive_invalid_annotation_action="drop",
    )

    assert config.validation_mode == "permissive"
    assert config.permissive_invalid_annotation_action == "drop"
    assert config.allow_overwrite is True
    assert config.input_path_include_substring == "train"
    assert config.input_path_exclude_substring == "backup"
