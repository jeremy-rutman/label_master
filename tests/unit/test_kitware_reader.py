from __future__ import annotations

from pathlib import Path

from PIL import Image

from label_master.adapters.kitware.common import parse_kitware_bboxes
from label_master.adapters.kitware.reader import read_kitware_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format
from label_master.format_specs.registry import CsvBracketBBoxDatasetParserSpec

FIXTURE = Path("tests/fixtures/us4")


def test_infer_kitware_format_from_nested_fixture() -> None:
    result = infer_format(FIXTURE, force=True)

    assert result.predicted_format == SourceFormat.KITWARE
    assert result.candidates[0].format == SourceFormat.KITWARE
    assert result.candidates[0].score > 0.9


def test_read_kitware_dataset_loads_nested_directories() -> None:
    dataset = read_kitware_dataset(FIXTURE)

    assert dataset.source_format == SourceFormat.KITWARE
    assert len(dataset.images) == 3673
    assert len(dataset.annotations) == 3697
    assert list(dataset.categories) == [0, 1, 2, 3]
    assert [dataset.categories[index].name for index in sorted(dataset.categories)] == [
        "airplane",
        "bird",
        "drone",
        "helicopter",
    ]

    first_image = dataset.images[0]
    assert first_image.image_id == "data/Training_data_001/V_AIRPLANE_001_1_001"
    assert first_image.file_name == "data/Training_data_001/V_AIRPLANE_001_1_001.png"
    assert first_image.width == 640
    assert first_image.height == 512

    first_annotation = next(annotation for annotation in dataset.annotations if annotation.image_id == first_image.image_id)
    assert first_annotation.class_id == 0
    assert first_annotation.bbox_xywh_abs == (313.0, 205.0, 36.0, 24.0)

    last_image = dataset.images[-1]
    assert last_image.file_name == "data/Training_data_003/V_HELICOPTER_003_4_300.png"
    last_annotation = next(annotation for annotation in dataset.annotations if annotation.image_id == last_image.image_id)
    assert last_annotation.class_id == 3


def test_read_kitware_dataset_supports_multiple_bboxes_in_one_cell_and_windows_paths(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    image_dir = tmp_path / "Training_data_V" / "Training_data_003"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "V_BIRD_003_2_001.png"
    Image.new("RGB", (320, 240), color="black").save(image_path)

    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "\n".join(
            [
                "imageFilename,place_bbox,bird_bbox,drone_bbox,helicopter_bbox",
                r"Training_data_V\Training_data_003\V_BIRD_003_2_001.png,[],[65 25 30 27;197 87 23 22],[],[]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = read_kitware_dataset(tmp_path)

    assert dataset.source_format == SourceFormat.KITWARE
    assert len(dataset.images) == 1
    assert dataset.images[0].file_name == "Training_data_V/Training_data_003/V_BIRD_003_2_001.png"
    assert len(dataset.annotations) == 2
    assert [annotation.class_id for annotation in dataset.annotations] == [1, 1]
    assert [annotation.bbox_xywh_abs for annotation in dataset.annotations] == [
        (65.0, 25.0, 30.0, 27.0),
        (197.0, 87.0, 23.0, 22.0),
    ]


def test_parse_kitware_bboxes_honors_explicit_bbox_field_mapping() -> None:
    parser = CsvBracketBBoxDatasetParserSpec.model_validate(
        {
            "kind": "csv_bracket_bbox_dataset",
            "csv_globs": ["**/*.csv"],
            "image_field_aliases": ["imageFilename"],
            "bbox_column_class_map": {"bird_bbox": "bird"},
            "bbox_fields": {
                "width": 1,
                "height": 2,
                "xmin": 3,
                "ymin": 4,
            },
        }
    )

    assert parse_kitware_bboxes("[30 27 65 25]", parser=parser) == [(65.0, 25.0, 30.0, 27.0)]
