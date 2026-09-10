from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from label_master.adapters.custom.common import split_row_tokens
from label_master.adapters.custom.reader import read_custom_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format
from label_master.core.services.validate_service import validate_dataset
from label_master.format_specs.registry import load_builtin_format_specs
from tests.fixtures.waymo_parquet import write_waymo_parquet_dataset

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


def _write_neovision2_video_spec(dataset_root: Path) -> None:
    spec_root = dataset_root / "format_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    (spec_root / "neovision2_video.yaml").write_text(
        "\n".join(
            [
                "format_id: neovision2_video",
                "display_name: NeoVision2 Video",
                "description: NeoVision2 CSV video annotations with quadrilateral corners",
                "parser:",
                "  kind: tokenized_video",
                "  annotation_globs:",
                "    - annotations/*.csv",
                "  video_roots:",
                "    - videos",
                "  skip_rows: 1",
                "  row_format:",
                "    kind: single_object",
                "    delimiter: comma",
                "    frame_index_field: 1",
                "    frame_index_base: 0",
                "    class_name_field: 10",
                "    quadrilateral_fields:",
                "      x1: 2",
                "      y1: 3",
                "      x2: 4",
                "      y2: 5",
                "      x3: 6",
                "      y3: 7",
                "      x4: 8",
                "      y4: 9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_neovision2_video_dataset(dataset_root: Path) -> None:
    annotation_root = dataset_root / "annotations"
    video_root = dataset_root / "videos"
    annotation_root.mkdir(parents=True)
    video_root.mkdir(parents=True)
    shutil.copy2(
        VIDEO_FIXTURE / "videos" / "00_02_45_to_00_03_10_cut.mpg",
        video_root / "clip001.mpg",
    )
    (annotation_root / "clip001.csv").write_text(
        "\n".join(
            [
                "Frame,BoundingBox_X1,BoundingBox_Y1,BoundingBox_X2,BoundingBox_Y2,BoundingBox_X3,BoundingBox_Y3,BoundingBox_X4,BoundingBox_Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version",
                "0,10,20,30,20,30,40,10,40,Plane,FALSE,FALSE,1.0,,1.4",
                "0,50,60,68,60,68,80,50,80,Car,FALSE,FALSE,1.0,,1.4",
                "1,100,120,110,118,114,128,104,130,Helicopter,FALSE,FALSE,1.0,,1.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_neovision2_video_spec(dataset_root)


def _write_bdd100k_spec(dataset_root: Path, *, image_root: str = "images") -> None:
    spec_root = dataset_root / "format_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    (spec_root / "bdd100k_detection.yaml").write_text(
        "\n".join(
            [
                "format_id: bdd100k_detection",
                "display_name: BDD100K Detection",
                "description: BDD100K nested image records with labels[].box2d boxes",
                "parser:",
                "  kind: bdd100k_image_labels",
                "  annotations_file: labels/bdd100k_labels_images_train.json",
                f"  image_root: {image_root}",
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


def _write_bdd100k_dataset(
    dataset_root: Path,
    *,
    image_root: str = "images",
    image_names_as_basenames: bool = False,
    nested_train_dirs: bool = False,
) -> None:
    labels_root = dataset_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    if nested_train_dirs:
        sample_image_root = dataset_root / "images" / "100k" / "train" / "trainA"
        empty_image_root = dataset_root / "images" / "100k" / "train" / "trainB"
        sample_name = "sample.jpg" if image_names_as_basenames else "100k/train/trainA/sample.jpg"
        empty_name = "empty.jpg" if image_names_as_basenames else "100k/train/trainB/empty.jpg"
    else:
        sample_image_root = dataset_root / "images" / "100k" / "train"
        empty_image_root = sample_image_root
        sample_name = "sample.jpg" if image_names_as_basenames else "100k/train/sample.jpg"
        empty_name = "empty.jpg" if image_names_as_basenames else "100k/train/empty.jpg"
    sample_image_root.mkdir(parents=True, exist_ok=True)
    empty_image_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color="black").save(sample_image_root / "sample.jpg")
    Image.new("RGB", (100, 50), color="black").save(empty_image_root / "empty.jpg")
    (labels_root / "bdd100k_labels_images_train.json").write_text(
        json.dumps(
            [
                {
                    "name": sample_name,
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
                        {
                            "id": "meta-only",
                            "category": "traffic sign",
                        },
                    ],
                },
                {
                    "name": empty_name,
                    "labels": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    _write_bdd100k_spec(dataset_root, image_root=image_root)


def test_builtin_input_formats_are_registered_via_yaml_specs() -> None:
    specs = load_builtin_format_specs()

    assert sorted(specs) == [
        "cityscapes",
        "coco",
        "kitware",
        "matlab_ground_truth",
        "video_bbox",
        "voc",
        "yolo",
    ]
    assert specs["coco"].parser.kind == "json_object_dataset"
    assert specs["matlab_ground_truth"].parser.kind == "built_in"
    assert specs["yolo"].parser.kind == "tokenized_image_labels"


def test_split_row_tokens_preserves_empty_csv_columns() -> None:
    assert split_row_tokens("0,10,,Plane", delimiter="comma") == ["0", "10", "", "Plane"]


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


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required for video fixtures")
def test_infer_and_read_custom_single_object_video_format_from_spec(tmp_path: Path) -> None:
    _write_neovision2_video_dataset(tmp_path)

    inference = infer_format(tmp_path, force=True)
    dataset = read_custom_dataset(tmp_path)
    validation = validate_dataset(tmp_path, source_format=SourceFormat.CUSTOM)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert inference.candidates[0].format == SourceFormat.CUSTOM
    assert inference.candidates[0].score > 0.8
    assert inference.candidates[0].evidence == ["format_spec:neovision2_video"]

    assert dataset.source_format == SourceFormat.CUSTOM
    assert [image.file_name for image in dataset.images] == [
        "images/clip001/frame_000000.jpg",
        "images/clip001/frame_000001.jpg",
    ]
    assert len(dataset.annotations) == 3
    assert [category.name for _, category in sorted(dataset.categories.items())] == [
        "Plane",
        "Car",
        "Helicopter",
    ]
    assert dataset.annotations[0].bbox_xywh_abs == (10.0, 20.0, 20.0, 20.0)
    assert dataset.annotations[2].bbox_xywh_abs == (100.0, 118.0, 14.0, 12.0)

    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


def test_infer_and_read_custom_bdd100k_format_from_spec(tmp_path: Path) -> None:
    _write_bdd100k_dataset(tmp_path)

    inference = infer_format(tmp_path, force=True)
    dataset = read_custom_dataset(tmp_path)
    validation = validate_dataset(tmp_path, source_format=SourceFormat.CUSTOM)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert inference.candidates[0].format == SourceFormat.CUSTOM
    assert inference.candidates[0].evidence == ["format_spec:bdd100k_detection"]

    assert dataset.source_format == SourceFormat.CUSTOM
    assert [image.file_name for image in dataset.images] == [
        "images/100k/train/empty.jpg",
        "images/100k/train/sample.jpg",
    ]
    assert len(dataset.annotations) == 2
    assert [category.name for _, category in sorted(dataset.categories.items())] == ["car", "person"]
    assert dataset.annotations[0].bbox_xywh_abs == (10.0, 20.0, 20.0, 25.0)
    assert dataset.source_metadata.details["format_id"] == "bdd100k_detection"
    assert dataset.source_metadata.details["media_kind"] == "image_collection"

    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


def test_infer_and_read_custom_bdd100k_format_from_basename_records_in_nested_dirs(
    tmp_path: Path,
) -> None:
    _write_bdd100k_dataset(
        tmp_path,
        image_root="images/100k/train",
        image_names_as_basenames=True,
        nested_train_dirs=True,
    )

    inference = infer_format(tmp_path, force=True)
    dataset = read_custom_dataset(tmp_path)
    validation = validate_dataset(tmp_path, source_format=SourceFormat.CUSTOM)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert [image.file_name for image in dataset.images] == [
        "images/100k/train/trainA/sample.jpg",
        "images/100k/train/trainB/empty.jpg",
    ]
    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


def test_read_custom_bdd100k_format_from_root_level_custom_format_yaml(tmp_path: Path) -> None:
    _write_bdd100k_dataset(tmp_path)
    spec_path = tmp_path / "format_specs" / "bdd100k_detection.yaml"
    spec_path.rename(tmp_path / "custom_format.yaml")

    inference = infer_format(tmp_path, force=True)
    dataset = read_custom_dataset(tmp_path)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert inference.candidates[0].evidence == ["format_spec:bdd100k_detection"]
    assert dataset.source_metadata.details["format_id"] == "bdd100k_detection"


def test_read_custom_bdd100k_format_from_explicit_yaml_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_bdd100k_dataset(dataset_root)
    internal_spec = dataset_root / "format_specs" / "bdd100k_detection.yaml"
    external_spec = tmp_path / "external_specs" / "label_format.yaml"
    external_spec.parent.mkdir(parents=True, exist_ok=True)
    internal_spec.rename(external_spec)

    dataset = read_custom_dataset(dataset_root, format_path=external_spec)
    validation = validate_dataset(
        dataset_root,
        source_format=SourceFormat.CUSTOM,
        custom_format_path=external_spec,
    )

    assert dataset.source_metadata.details["format_id"] == "bdd100k_detection"
    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


def test_infer_and_read_waymo_parquet_custom_format(tmp_path: Path) -> None:
    dataset_root = write_waymo_parquet_dataset(tmp_path)

    inference = infer_format(dataset_root, force=True)
    dataset = read_custom_dataset(dataset_root)
    validation = validate_dataset(dataset_root, source_format=SourceFormat.CUSTOM)

    assert inference.predicted_format == SourceFormat.CUSTOM
    assert inference.candidates[0].evidence == ["format_spec:waymo_parquet_camera_boxes"]
    assert dataset.source_format == SourceFormat.CUSTOM
    assert [image.file_name for image in dataset.images] == [
        "segment_001__111__1.jpg",
        "segment_001__222__2.jpg",
    ]
    assert len(dataset.annotations) == 2
    assert [category.name for _, category in sorted(dataset.categories.items())] == [
        "vehicle",
        "cyclist",
    ]
    assert dataset.annotations[0].bbox_xywh_abs == (16.0, 7.0, 8.0, 6.0)
    assert dataset.annotations[0].class_id == 0
    assert dataset.annotations[1].class_id == 1
    assert dataset.source_metadata.details["media_kind"] == "parquet_image_collection"
    assert validation.inferred_format == SourceFormat.CUSTOM
    assert validation.summary.invalid_annotations == 0


def test_read_custom_bdd100k_format_honors_max_records(tmp_path: Path) -> None:
    _write_bdd100k_dataset(tmp_path)

    dataset = read_custom_dataset(tmp_path, max_records=1)

    assert len(dataset.images) == 1
    assert dataset.source_metadata.details["records_loaded"] == "1"
    assert dataset.source_metadata.details["records_total"] == "2"
