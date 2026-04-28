from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from label_master.adapters.video_bbox.reader import read_video_bbox_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format

VIDEO_FIXTURE = Path("tests/fixtures/us3/provider_sample2_video")
SEQUENCE_FIXTURE = Path("tests/fixtures/us5")
MOT_FIXTURE = Path("tests/fixtures/us7")
ANTI_UAV_MP4_FIXTURE = Path("tests/fixtures/us9/V_BIRD_029.mp4")


def _write_anti_uav_rgbt_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "Anti_UAV_RGBT"
    clip_a = dataset_root / "train" / "clip_a"
    clip_b = dataset_root / "train" / "clip_b"
    clip_a.mkdir(parents=True)
    clip_b.mkdir(parents=True)

    shutil.copy2(ANTI_UAV_MP4_FIXTURE, clip_a / "visible.mp4")
    shutil.copy2(ANTI_UAV_MP4_FIXTURE, clip_b / "visible.mp4")
    (clip_a / "visible.json").write_text(
        json.dumps(
            {
                "exist": [1, 1, 0],
                "gt_rect": [[10, 20, 30, 40], [0, 0, 0, 0], []],
            }
        ),
        encoding="utf-8",
    )
    (clip_b / "visible.json").write_text(
        json.dumps(
            {
                "exist": [0, 1],
                "gt_rect": [[], [5, 6, 7, 8]],
            }
        ),
        encoding="utf-8",
    )
    return dataset_root


@pytest.mark.parametrize("fixture", [VIDEO_FIXTURE, SEQUENCE_FIXTURE, MOT_FIXTURE])
def test_infer_video_bbox_format_from_fixture(fixture: Path) -> None:
    result = infer_format(fixture, force=True)

    assert result.predicted_format == SourceFormat.VIDEO_BBOX
    assert result.candidates[0].format == SourceFormat.VIDEO_BBOX
    assert result.candidates[0].score > 0.9


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_read_video_bbox_dataset_loads_frames_and_annotations() -> None:
    dataset = read_video_bbox_dataset(VIDEO_FIXTURE)

    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 400
    assert len(dataset.annotations) == 399
    assert list(dataset.categories) == [0]
    assert dataset.categories[0].name == "drone"

    first_image = dataset.images[0]
    assert first_image.image_id == "00_02_45_to_00_03_10_cut:000000"
    assert first_image.file_name == "images/00_02_45_to_00_03_10_cut/frame_000000.jpg"
    assert first_image.width == 1920
    assert first_image.height == 1080

    first_annotation = dataset.annotations[0]
    assert first_annotation.image_id == first_image.image_id
    assert first_annotation.class_id == 0
    assert first_annotation.bbox_xywh_abs == (708.0, 757.0, 15.0, 13.0)

    last_image = dataset.images[-1]
    assert last_image.image_id == "00_02_45_to_00_03_10_cut:000399"
    assert last_image.file_name == "images/00_02_45_to_00_03_10_cut/frame_000399.jpg"
    assert all(annotation.image_id != last_image.image_id for annotation in dataset.annotations)


def test_read_video_bbox_dataset_loads_still_frame_sequences() -> None:
    dataset = read_video_bbox_dataset(SEQUENCE_FIXTURE)

    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 183
    assert len(dataset.annotations) == 183
    assert list(dataset.categories) == [0]
    assert dataset.categories[0].name == "object"

    first_image = dataset.images[0]
    assert first_image.image_id == "video02:000000"
    assert first_image.file_name == "images/video02/frame_000000.jpg"
    assert first_image.width == 1280
    assert first_image.height == 720

    first_annotation = dataset.annotations[0]
    assert first_annotation.image_id == first_image.image_id
    assert first_annotation.class_id == 0
    assert first_annotation.bbox_xywh_abs == (953.0, 298.0, 46.0, 17.0)

    last_image = dataset.images[-1]
    assert last_image.image_id == "video03:000099"
    assert last_image.file_name == "images/video03/frame_000099.jpg"
    last_annotation = next(annotation for annotation in dataset.annotations if annotation.image_id == last_image.image_id)
    assert last_annotation.class_id == 0
    assert last_annotation.bbox_xywh_abs == (708.0, 322.0, 25.0, 22.0)


def test_read_video_bbox_dataset_loads_mot_challenge_sequences() -> None:
    dataset = read_video_bbox_dataset(MOT_FIXTURE)

    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 317
    assert len(dataset.annotations) == 6017
    assert list(dataset.categories) == [0]
    assert dataset.categories[0].name == "object"

    first_image = dataset.images[0]
    assert first_image.image_id == "UAVSwarm-02:000000"
    assert first_image.file_name == "images/UAVSwarm-02/frame_000000.jpg"
    assert first_image.width == 812
    assert first_image.height == 428

    first_annotation = dataset.annotations[0]
    assert first_annotation.annotation_id == "UAVSwarm-02/gt/gt.txt:000001:001"
    assert first_annotation.image_id == first_image.image_id
    assert first_annotation.class_id == 0
    assert first_annotation.bbox_xywh_abs == (352.0, 20.0, 11.0, 11.0)
    assert first_annotation.attributes["mot_track_id"] == 1
    assert first_annotation.attributes["mot_source_class_id"] == 1

    clipped_annotation = next(
        annotation
        for annotation in dataset.annotations
        if annotation.annotation_id == "UAVSwarm-02/gt/gt.txt:002728:018"
    )
    assert clipped_annotation.bbox_xywh_abs == (438.0, 414.0, 16.0, 14.0)
    assert clipped_annotation.attributes["mot_bbox_was_clipped_to_frame"] is True
    assert clipped_annotation.attributes["mot_original_h"] == 16.0

    last_annotation = dataset.annotations[-1]
    assert last_annotation.annotation_id == "UAVSwarm-04/gt/gt.txt:003034:020"
    assert last_annotation.image_id == "UAVSwarm-04:000160"
    assert last_annotation.bbox_xywh_abs == (551.0, 375.0, 19.0, 19.0)


def test_read_video_bbox_dataset_supports_anti_uav_named_sequence_roots(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    frames_root = tmp_path / "Anti-UAV-Tracking-V0" / "video01"
    gt_root = tmp_path / "Anti-UAV-Tracking-V0GT"
    frames_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)

    Image.new("RGB", (64, 48), color="black").save(frames_root / "00001.jpg")
    Image.new("RGB", (64, 48), color="black").save(frames_root / "00002.jpg")
    (gt_root / "video01_gt.txt").write_text("10 11 12 13\n14 15 16 17\n", encoding="utf-8")

    inferred = infer_format(tmp_path, force=True)
    dataset = read_video_bbox_dataset(tmp_path)

    assert inferred.predicted_format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].score > 0.9
    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 2
    assert dataset.images[0].file_name == "images/video01/frame_000000.jpg"
    assert dataset.annotations[0].bbox_xywh_abs == (10.0, 11.0, 12.0, 13.0)


def test_read_video_bbox_dataset_supports_arbitrary_sibling_folder_names(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    frames_root = tmp_path / "sequence_bank" / "clip_a"
    gt_root = tmp_path / "labels_bucket"
    frames_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)

    Image.new("RGB", (80, 60), color="black").save(frames_root / "00001.jpg")
    Image.new("RGB", (80, 60), color="black").save(frames_root / "00002.jpg")
    (gt_root / "clip_a_gt.txt").write_text("1 2 3 4\n5 6 7 8\n", encoding="utf-8")

    inferred = infer_format(tmp_path, force=True)
    dataset = read_video_bbox_dataset(tmp_path)

    assert inferred.predicted_format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].score > 0.9
    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 2
    assert dataset.images[0].file_name == "images/clip_a/frame_000000.jpg"
    assert dataset.annotations[1].bbox_xywh_abs == (5.0, 6.0, 7.0, 8.0)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_read_video_bbox_dataset_supports_arbitrary_sibling_video_folder_names(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    dataset_root = tmp_path / "wosdetc_like"
    annotations_root = dataset_root / "annotations"
    videos_root = dataset_root / "wosdetc_train_videos"
    annotations_root.mkdir(parents=True)
    videos_root.mkdir(parents=True)

    source_annotation = VIDEO_FIXTURE / "annotations" / "00_02_45_to_00_03_10_cut.txt"
    source_video = VIDEO_FIXTURE / "videos" / "00_02_45_to_00_03_10_cut.mpg"
    shutil.copy2(source_annotation, annotations_root / source_annotation.name)
    shutil.copy2(source_video, videos_root / source_video.name)

    inferred = infer_format(dataset_root, force=True)
    dataset = read_video_bbox_dataset(dataset_root)

    assert inferred.predicted_format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].format == SourceFormat.VIDEO_BBOX
    assert inferred.candidates[0].score > 0.9
    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 400
    assert len(dataset.annotations) == 399
    assert dataset.images[0].file_name == "images/00_02_45_to_00_03_10_cut/frame_000000.jpg"
    assert dataset.categories[0].name == "drone"


def test_read_video_bbox_dataset_treats_all_negative_tracking_rows_as_empty_frames(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    frames_root = tmp_path / "frames_any_name" / "clip_b"
    gt_root = tmp_path / "ground_truth_any_name"
    frames_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)

    Image.new("RGB", (80, 60), color="black").save(frames_root / "00001.jpg")
    Image.new("RGB", (80, 60), color="black").save(frames_root / "00002.jpg")
    (gt_root / "clip_b_gt.txt").write_text("1 2 3 4\n-100 -100 -100 -100\n", encoding="utf-8")

    dataset = read_video_bbox_dataset(tmp_path)

    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 1
    assert dataset.annotations[0].image_id == "clip_b:000000"
    assert all(annotation.image_id != "clip_b:000001" for annotation in dataset.annotations)


def test_infer_video_bbox_format_from_anti_uav_rgbt_json_video_pairs(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    dataset_root = _write_anti_uav_rgbt_dataset(tmp_path)

    result = infer_format(dataset_root, force=True)

    assert result.predicted_format == SourceFormat.VIDEO_BBOX
    assert result.candidates[0].format == SourceFormat.VIDEO_BBOX
    assert result.candidates[0].score > 0.9


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_read_video_bbox_dataset_loads_anti_uav_rgbt_json_video_pairs(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    dataset_root = _write_anti_uav_rgbt_dataset(tmp_path)

    dataset = read_video_bbox_dataset(dataset_root)

    assert dataset.source_format == SourceFormat.VIDEO_BBOX
    assert len(dataset.images) == 5
    assert len(dataset.annotations) == 2
    assert dataset.images[0].file_name == "images/train/clip_a/visible/frame_000000.jpg"
    assert dataset.images[-1].file_name == "images/train/clip_b/visible/frame_000001.jpg"
    assert dataset.annotations[0].bbox_xywh_abs == (10.0, 20.0, 30.0, 40.0)
    assert dataset.annotations[-1].bbox_xywh_abs == (5.0, 6.0, 7.0, 8.0)
    assert dataset.warnings
    assert "Skipped 1 invalid paired-video annotation frame(s)" in dataset.warnings[0].message
    assert dataset.source_metadata.details["video_sources_total"] == "2"
    assert dataset.source_metadata.details["video_sources_loaded"] == "2"
