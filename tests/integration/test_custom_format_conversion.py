from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from tests.fixtures.waymo_parquet import write_waymo_parquet_dataset

VIDEO_FIXTURE = Path("tests/fixtures/us3/provider_sample2_video")


def _write_custom_video_spec(dataset_root: Path) -> None:
    spec_root = dataset_root / "format_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    (spec_root / "custom_video_bbox.yaml").write_text(
        "\n".join(
            [
                "format_id: custom_video_bbox",
                "display_name: Custom Video BBox",
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
                "      xmin: 1",
                "      ymin: 2",
                "      width: 3",
                "      height: 4",
                "      class_name: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_custom_video_dataset(dataset_root: Path) -> None:
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
    _write_custom_video_spec(dataset_root)


def _write_bdd100k_spec(dataset_root: Path) -> None:
    spec_root = dataset_root / "format_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    (spec_root / "bdd100k_detection.yaml").write_text(
        "\n".join(
            [
                "format_id: bdd100k_detection",
                "display_name: BDD100K Detection",
                "parser:",
                "  kind: bdd100k_image_labels",
                "  annotations_file: labels/bdd100k_labels_images_train.json",
                "  image_root: images",
                "  image_name_field: name",
                "  labels_field: labels",
                "  label_id_field: id",
                "  category_field: category",
                "  bbox_field: box2d",
                "  x1_field: x1",
                "  y1_field: y1",
                "  x2_field: x2",
                "  y2_field: y2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bdd100k_dataset(dataset_root: Path) -> None:
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images" / "100k" / "train"
    labels_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(images_root / "sample.jpg")
    (labels_root / "bdd100k_labels_images_train.json").write_text(
        json.dumps(
            [
                {
                    "name": "100k/train/sample.jpg",
                    "labels": [
                        {
                            "id": "car-1",
                            "category": "car",
                            "box2d": {"x1": 10, "y1": 20, "x2": 30, "y2": 45},
                        },
                        {
                            "id": "person-1",
                            "category": "person",
                            "box2d": {"x1": 40, "y1": 5, "x2": 55, "y2": 25},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_bdd100k_spec(dataset_root)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for video fixtures",
)
def test_custom_spec_video_format_can_convert_to_yolo_with_copied_frames(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "custom_input"
    output_root = tmp_path / "converted"
    _write_custom_video_dataset(input_root)

    result = execute_conversion(
        ConvertRequest(
            run_id="custom-video-yolo",
            input_path=input_root,
            output_path=output_root,
            src_format=SourceFormat.CUSTOM,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert result.validation.inferred_format == SourceFormat.CUSTOM
    assert result.output_dataset.source_metadata.details["format_id"] == "custom_video_bbox"
    assert (output_root / "labels" / "00_02_45_to_00_03_10_cut" / "frame_000000.txt").read_text(
        encoding="utf-8"
    ).strip() == "0 0.372656 0.706944 0.007812 0.012037"
    assert (output_root / "images" / "00_02_45_to_00_03_10_cut" / "frame_000000.jpg").exists()


def test_custom_spec_bdd100k_format_can_convert_to_yolo_with_copied_images(tmp_path: Path) -> None:
    input_root = tmp_path / "bdd_input"
    output_root = tmp_path / "converted"
    _write_bdd100k_dataset(input_root)

    result = execute_conversion(
        ConvertRequest(
            run_id="bdd100k-yolo",
            input_path=input_root,
            output_path=output_root,
            src_format=SourceFormat.CUSTOM,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert result.validation.inferred_format == SourceFormat.CUSTOM
    assert result.output_dataset.source_metadata.details["format_id"] == "bdd100k_detection"
    assert (output_root / "labels" / "100k" / "train" / "sample.txt").read_text(encoding="utf-8") == (
        "0 0.2 0.65 0.2 0.5\n"
        "1 0.475 0.3 0.15 0.4\n"
    )
    assert (output_root / "images" / "100k" / "train" / "sample.jpg").exists()
    assert (output_root / "classes.txt").read_text(encoding="utf-8") == "car\nperson\n"


def test_custom_spec_bdd100k_format_can_convert_with_explicit_external_yaml(tmp_path: Path) -> None:
    input_root = tmp_path / "bdd_input"
    output_root = tmp_path / "converted"
    _write_bdd100k_dataset(input_root)
    external_spec = tmp_path / "external_specs" / "data_format.yaml"
    external_spec.parent.mkdir(parents=True, exist_ok=True)
    (input_root / "format_specs" / "bdd100k_detection.yaml").rename(external_spec)

    result = execute_conversion(
        ConvertRequest(
            run_id="bdd100k-yolo-external",
            input_path=input_root,
            output_path=output_root,
            src_format=SourceFormat.CUSTOM,
            dst_format=SourceFormat.YOLO,
            custom_format_id="bdd100k_detection",
            custom_format_path=external_spec,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert result.output_dataset.source_metadata.details["format_id"] == "bdd100k_detection"
    assert (output_root / "labels" / "100k" / "train" / "sample.txt").exists()


def test_waymo_parquet_custom_format_can_convert_to_yolo_with_copied_images(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "waymo_input"
    output_root = tmp_path / "converted"
    write_waymo_parquet_dataset(input_root)

    result = execute_conversion(
        ConvertRequest(
            run_id="waymo-parquet-yolo",
            input_path=input_root,
            output_path=output_root,
            src_format=SourceFormat.CUSTOM,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert result.validation.inferred_format == SourceFormat.CUSTOM
    assert result.output_dataset.source_metadata.details["format_id"] == "waymo_parquet_camera_boxes"
    assert (output_root / "classes.txt").read_text(encoding="utf-8") == "vehicle\ncyclist\n"
    assert (output_root / "labels" / "segment_001__111__1.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.3125 0.3125 0.125 0.1875",
        "1 0.625 0.4375 0.15625 0.25",
    ]
    assert (output_root / "labels" / "segment_001__222__2.txt").read_text(encoding="utf-8") == ""
    assert (output_root / "images" / "segment_001__111__1.jpg").exists()
    assert (output_root / "images" / "segment_001__222__2.jpg").exists()
