from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from label_master.adapters.custom.reader import read_custom_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format
from label_master.core.services.validate_service import validate_dataset
from label_master.format_specs.registry import load_builtin_format_specs

VIDEO_FIXTURE = Path("tests/fixtures/us3/provider_sample2_video")


def _write_custom_video_spec(dataset_root: Path, *, use_legacy_xy: bool = False) -> None:
    spec_root = dataset_root / "format_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    x_field = "x" if use_legacy_xy else "xmin"
    y_field = "y" if use_legacy_xy else "ymin"
    (spec_root / "custom_video_bbox.yaml").write_text(
        "\n".join(
            [
                "format_id: custom_video_bbox",
                "display_name: Custom Video BBox",
                "description: User-defined count-prefixed video bbox rows",
                "parser:",
                "  kind: tokenized_video",
                "  annotation_globs:",
                "    - odd_annotations/*.txt",
                "  video_roots:",
                "    - odd_videos",
                "  row_format:",
                "    kind: count_prefixed_objects",
                "    frame_index_field: 1",
                "    object_count_field: 2",
                "    object_group_size: 5",
                "    frame_index_base: 0",
                "    object_fields:",
                f"      {x_field}: 1",
                f"      {y_field}: 2",
                "      width: 3",
                "      height: 4",
                "      class_name: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_custom_video_dataset(dataset_root: Path, *, use_legacy_xy: bool = False) -> None:
    annotation_root = dataset_root / "odd_annotations"
    video_root = dataset_root / "odd_videos"
    annotation_root.mkdir(parents=True)
    video_root.mkdir(parents=True)
    shutil.copy2(
        VIDEO_FIXTURE / "annotations" / "00_02_45_to_00_03_10_cut.txt",
        annotation_root / "00_02_45_to_00_03_10_cut.txt",
    )
    shutil.copy2(
        VIDEO_FIXTURE / "videos" / "00_02_45_to_00_03_10_cut.mpg",
        video_root / "00_02_45_to_00_03_10_cut.mpg",
    )
    _write_custom_video_spec(dataset_root, use_legacy_xy=use_legacy_xy)


def test_builtin_input_formats_are_registered_via_yaml_specs() -> None:
    specs = load_builtin_format_specs()

    assert sorted(specs) == ["coco", "kitware", "matlab_ground_truth", "video_bbox", "voc", "yolo"]
    assert specs["coco"].parser.kind == "json_object_dataset"
    assert specs["matlab_ground_truth"].parser.kind == "built_in"
    assert specs["yolo"].parser.kind == "tokenized_image_labels"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_infer_and_read_custom_video_format_from_spec(tmp_path: Path) -> None:
    _write_custom_video_dataset(tmp_path)

    inference = infer_format(tmp_path, force=True)
    dataset = read_custom_dataset(tmp_path)
    validation = validate_dataset(tmp_path, source_format=SourceFormat.CUSTOM)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert inference.candidates[0].format == SourceFormat.CUSTOM
    assert inference.candidates[0].score > 0.8
    assert inference.candidates[0].evidence == ["format_spec:custom_video_bbox"]

    assert dataset.source_format == SourceFormat.CUSTOM
    assert len(dataset.images) == 400
    assert len(dataset.annotations) == 399
    assert dataset.images[0].file_name == "images/00_02_45_to_00_03_10_cut/frame_000000.jpg"
    assert dataset.categories[0].name == "drone"
    assert dataset.source_metadata.details["format_id"] == "custom_video_bbox"
    assert dataset.source_metadata.details["media_kind"] == "video_collection"

    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_custom_video_format_accepts_legacy_x_y_field_names(tmp_path: Path) -> None:
    _write_custom_video_dataset(tmp_path, use_legacy_xy=True)

    dataset = read_custom_dataset(tmp_path)

    assert dataset.source_format == SourceFormat.CUSTOM
    assert dataset.annotations[0].bbox_xywh_abs == (708.0, 757.0, 15.0, 13.0)
