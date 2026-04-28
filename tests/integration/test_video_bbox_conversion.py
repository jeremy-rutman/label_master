from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.domain.value_objects import ValidationError
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
VIDEO_FIXTURE = Path("tests/fixtures/us3/provider_sample2_video")
SEQUENCE_FIXTURE = Path("tests/fixtures/us5")
MOT_FIXTURE = Path("tests/fixtures/us7")
ANTI_UAV_MP4_FIXTURE = Path("tests/fixtures/us9/V_BIRD_029.mp4")


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


def _write_anti_uav_rgbt_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "Anti_UAV_RGBT"
    for clip_name, payload in {
        "clip_a": {"exist": [1, 0], "gt_rect": [[10, 20, 30, 40], []]},
        "clip_b": {"exist": [0, 1], "gt_rect": [[], [5, 6, 7, 8]]},
    }.items():
        clip_root = dataset_root / "train" / clip_name
        clip_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ANTI_UAV_MP4_FIXTURE, clip_root / "visible.mp4")
        (clip_root / "visible.json").write_text(json.dumps(payload), encoding="utf-8")
    return dataset_root


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_cli_convert_auto_detects_video_bbox_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"

    result = RUNNER.invoke(
        app,
        [
            "convert",
            "--input",
            str(VIDEO_FIXTURE),
            "--output",
            str(output),
            "--src",
            "auto",
            "--dst",
            "coco",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    assert len(payload["images"]) == 400
    assert len(payload["annotations"]) == 399
    assert payload["categories"] == [{"id": 0, "name": "drone", "supercategory": ""}]


def test_cli_convert_auto_detects_still_frame_video_bbox_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted_sequence"

    result = RUNNER.invoke(
        app,
        [
            "convert",
            "--input",
            str(SEQUENCE_FIXTURE),
            "--output",
            str(output),
            "--src",
            "auto",
            "--dst",
            "coco",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    assert len(payload["images"]) == 183
    assert len(payload["annotations"]) == 183
    assert payload["categories"] == [{"id": 0, "name": "object", "supercategory": ""}]
    assert payload["images"][0]["file_name"] == "images/video02/frame_000000.jpg"


def test_cli_convert_auto_detects_mot_style_video_bbox_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted_mot"

    result = RUNNER.invoke(
        app,
        [
            "--report-path",
            str(tmp_path / "reports"),
            "convert",
            "--input",
            str(MOT_FIXTURE),
            "--output",
            str(output),
            "--src",
            "auto",
            "--dst",
            "coco",
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    assert len(payload["images"]) == 317
    assert len(payload["annotations"]) == 6017
    assert payload["categories"] == [{"id": 0, "name": "object", "supercategory": ""}]
    assert payload["images"][0]["file_name"] == "images/UAVSwarm-02/frame_000000.jpg"
    assert payload["annotations"][0]["bbox"] == [352.0, 20.0, 11.0, 11.0]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for video fixtures",
)
def test_video_bbox_to_yolo_extracts_frames_when_copy_images_enabled(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"

    result = execute_conversion(
        ConvertRequest(
            run_id="video-bbox-yolo",
            input_path=VIDEO_FIXTURE,
            output_path=output,
            src_format=SourceFormat.VIDEO_BBOX,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    labels_dir = output / "labels" / "00_02_45_to_00_03_10_cut"
    images_dir = output / "images" / "00_02_45_to_00_03_10_cut"

    label_files = sorted(labels_dir.glob("*.txt"))
    image_files = sorted(images_dir.glob("*.jpg"))

    assert len(label_files) == 400
    assert len(image_files) == 400
    assert (labels_dir / "frame_000000.txt").read_text(encoding="utf-8").strip() == (
        "0 0.372656 0.706944 0.007812 0.012037"
    )
    assert (labels_dir / "frame_000399.txt").read_text(encoding="utf-8") == ""
    assert image_files[0].name == "frame_000000.jpg"
    assert image_files[-1].name == "frame_000399.jpg"


def test_video_bbox_still_sequences_to_yolo_copy_existing_frames_when_copy_images_enabled(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted_sequence"

    result = execute_conversion(
        ConvertRequest(
            run_id="video-bbox-sequence-yolo",
            input_path=SEQUENCE_FIXTURE,
            output_path=output,
            src_format=SourceFormat.VIDEO_BBOX,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"

    video02_labels = sorted((output / "labels" / "video02").glob("*.txt"))
    video02_images = sorted((output / "images" / "video02").glob("*.jpg"))
    video03_labels = sorted((output / "labels" / "video03").glob("*.txt"))
    video03_images = sorted((output / "images" / "video03").glob("*.jpg"))

    assert len(video02_labels) == 83
    assert len(video02_images) == 83
    assert len(video03_labels) == 100
    assert len(video03_images) == 100
    assert (output / "labels" / "video02" / "frame_000000.txt").read_text(encoding="utf-8").strip() == (
        "0 0.7625 0.425694 0.035937 0.023611"
    )
    assert (output / "images" / "video02" / "frame_000000.jpg").exists()
    assert video03_images[-1].name == "frame_000099.jpg"


def test_video_bbox_still_sequences_can_flatten_output_layout(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "flattened_sequence"

    result = execute_conversion(
        ConvertRequest(
            run_id="video-bbox-sequence-yolo-flat",
            input_path=SEQUENCE_FIXTURE,
            output_path=output,
            src_format=SourceFormat.VIDEO_BBOX,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            flatten_output_layout=True,
        )
    )

    assert result.report.status == "completed"
    assert (output / "labels" / "video02_frame_000000.txt").exists()
    assert (output / "images" / "video02_frame_000000.jpg").exists()
    assert (output / "labels" / "video03_frame_000099.txt").exists()
    assert (output / "images" / "video03_frame_000099.jpg").exists()
    assert not (output / "labels" / "video02").exists()
    assert not (output / "images" / "video02").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for video fixtures",
)
def test_video_bbox_paired_json_video_sources_convert_to_yolo_with_images(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    input_root = _write_anti_uav_rgbt_dataset(tmp_path)
    output = tmp_path / "converted_anti_uav"

    result = execute_conversion(
        ConvertRequest(
            run_id="video-bbox-anti-uav-yolo",
            input_path=input_root,
            output_path=output,
            src_format=SourceFormat.VIDEO_BBOX,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert (output / "labels" / "train" / "clip_a" / "visible" / "frame_000000.txt").exists()
    assert (output / "images" / "train" / "clip_a" / "visible" / "frame_000000.jpg").exists()
    assert (output / "labels" / "train" / "clip_b" / "visible" / "frame_000001.txt").exists()
    assert (output / "images" / "train" / "clip_b" / "visible" / "frame_000001.jpg").exists()
    assert (output / "labels" / "train" / "clip_a" / "visible" / "frame_000001.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_conversion_report_includes_bbox_clipping_warning(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_root = tmp_path / "coco_clip"
    output_root = tmp_path / "converted"
    _write_coco_dataset(input_root, bbox=(91.0, 40.0, 10.0, 10.0))

    result = execute_conversion(
        ConvertRequest(
            run_id="coco-clip-warning",
            input_path=input_root,
            output_path=output_root,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            dry_run=True,
        )
    )

    assert result.report.status == "completed"
    assert len(result.report.warnings) == 1
    assert result.report.warnings[0].code == "validation_bbox_clipped_to_frame"
    assert "bbox went slightly out of frame" in result.report.warnings[0].message
    assert result.validation.dataset.annotations[0].bbox_xywh_abs == (91.0, 40.0, 9.0, 10.0)


def test_conversion_can_fail_when_out_of_frame_correction_is_disabled(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    input_root = tmp_path / "coco_clip_disabled"
    output_root = tmp_path / "converted"
    _write_coco_dataset(input_root, bbox=(91.0, 40.0, 10.0, 10.0))

    with pytest.raises(ValidationError):
        execute_conversion(
            ConvertRequest(
                run_id="coco-clip-disabled",
                input_path=input_root,
                output_path=output_root,
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.YOLO,
                unmapped_policy=UnmappedPolicy.ERROR,
                dry_run=True,
                correct_out_of_frame_bboxes=False,
            )
        )
