from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
FIXTURE = Path("tests/fixtures/us4")


def test_cli_convert_auto_detects_kitware_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"

    result = RUNNER.invoke(
        app,
        [
            "convert",
            "--input",
            str(FIXTURE),
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
    assert len(payload["images"]) == 3673
    assert len(payload["annotations"]) == 3697
    assert payload["images"][0]["file_name"] == "data/Training_data_001/V_AIRPLANE_001_1_001.png"
    assert payload["categories"] == [
        {"id": 0, "name": "airplane", "supercategory": ""},
        {"id": 1, "name": "bird", "supercategory": ""},
        {"id": 2, "name": "drone", "supercategory": ""},
        {"id": 3, "name": "helicopter", "supercategory": ""},
    ]


def test_kitware_to_yolo_preserves_nested_paths_when_copy_images_enabled(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"

    result = execute_conversion(
        ConvertRequest(
            run_id="kitware-yolo",
            input_path=FIXTURE,
            output_path=output,
            src_format=SourceFormat.KITWARE,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"

    first_label = output / "labels" / "data" / "Training_data_001" / "V_AIRPLANE_001_1_001.txt"
    first_image = output / "images" / "data" / "Training_data_001" / "V_AIRPLANE_001_1_001.png"

    assert first_label.read_text(encoding="utf-8").strip() == "0 0.517188 0.423828 0.05625 0.046875"
    assert first_image.exists()


def test_kitware_to_yolo_prefixes_nested_output_filenames(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "prefixed"

    result = execute_conversion(
        ConvertRequest(
            run_id="kitware-yolo-prefixed",
            input_path=FIXTURE,
            output_path=output,
            src_format=SourceFormat.KITWARE,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            output_file_name_prefix="us4",
        )
    )

    assert result.report.status == "completed"

    first_label = output / "labels" / "data" / "Training_data_001" / "us4_V_AIRPLANE_001_1_001.txt"
    first_image = output / "images" / "data" / "Training_data_001" / "us4_V_AIRPLANE_001_1_001.png"

    assert first_label.read_text(encoding="utf-8").strip() == "0 0.517188 0.423828 0.05625 0.046875"
    assert first_image.exists()


def test_kitware_to_yolo_shared_output_layout_is_flat_and_prefixes_by_immediate_parent(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "shared_flat"

    result = execute_conversion(
        ConvertRequest(
            run_id="kitware-yolo-flat-shared",
            input_path=FIXTURE,
            output_path=output,
            src_format=SourceFormat.KITWARE,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            output_file_name_prefix="us4",
            output_file_stem_prefix="batchA_",
            output_file_stem_suffix="_fold1",
            flatten_output_layout=True,
        )
    )

    assert result.report.status == "completed"
    assert (output / "batchA_us4_classes_fold1.txt").exists()

    first_label = output / "labels" / "batchA_data_Training_data_001_us4_V_AIRPLANE_001_1_001_fold1.txt"
    first_image = output / "images" / "batchA_data_Training_data_001_us4_V_AIRPLANE_001_1_001_fold1.png"

    assert first_label.read_text(encoding="utf-8").strip() == "0 0.517188 0.423828 0.05625 0.046875"
    assert first_image.exists()
    assert not (output / "labels" / "data").exists()
    assert not (output / "images" / "data").exists()
