from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion

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
