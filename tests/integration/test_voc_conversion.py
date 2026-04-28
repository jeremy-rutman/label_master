from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.interfaces.cli.main import app

RUNNER = CliRunner()
FIXTURE = Path("tests/fixtures/us6")


def _write_voc_image_and_xml(
    dataset_root: Path,
    *,
    split: str,
    stem: str,
    bbox: tuple[int, int, int, int],
) -> None:
    image_dir = dataset_root / split / "img"
    xml_dir = dataset_root / split / "xml"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 100), color="black").save(image_dir / f"{stem}.jpg")
    xmin, ymin, xmax, ymax = bbox
    (xml_dir / f"{stem}.xml").write_text(
        f"""<annotation>
    <filename>{stem}.jpg</filename>
    <path>./{split}/img/{stem}.jpg</path>
    <size><width>100</width><height>100</height></size>
    <object>
        <name>UAV</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
</annotation>
""",
        encoding="utf-8",
    )


def test_cli_convert_auto_detects_voc_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
    assert len(payload["images"]) == 5
    assert len(payload["annotations"]) == 5
    assert payload["images"][0]["file_name"] == "data/img/00001.jpg"
    assert payload["categories"] == [{"id": 0, "name": "UAV", "supercategory": ""}]


def test_voc_to_yolo_copies_images_and_labels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "converted"

    result = execute_conversion(
        ConvertRequest(
            run_id="voc-yolo",
            input_path=FIXTURE,
            output_path=output,
            src_format=SourceFormat.VOC,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    assert (output / "images" / "data" / "img" / "00001.jpg").exists()
    assert (output / "labels" / "data" / "img" / "00001.txt").read_text(encoding="utf-8").strip() == (
        "0 0.507812 0.792361 0.053125 0.084722"
    )


def test_voc_to_yolo_flattened_layout_keeps_split_context_for_duplicate_basenames(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    input_root = tmp_path / "voc_split"
    output = tmp_path / "flattened"

    _write_voc_image_and_xml(input_root, split="train", stem="00001", bbox=(10, 10, 20, 20))
    _write_voc_image_and_xml(input_root, split="val", stem="00001", bbox=(30, 30, 50, 50))
    _write_voc_image_and_xml(input_root, split="test", stem="00001", bbox=(60, 60, 90, 90))

    result = execute_conversion(
        ConvertRequest(
            run_id="voc-yolo-flat-split-collision",
            input_path=input_root,
            output_path=output,
            src_format=SourceFormat.VOC,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            flatten_output_layout=True,
        )
    )

    assert result.report.status == "completed"
    assert not (output / "labels" / "img_00001.txt").exists()
    assert (output / "labels" / "train_img_00001.txt").read_text(encoding="utf-8").strip().splitlines() == [
        "0 0.15 0.15 0.1 0.1"
    ]
    assert (output / "labels" / "val_img_00001.txt").read_text(encoding="utf-8").strip().splitlines() == [
        "0 0.4 0.4 0.2 0.2"
    ]
    assert (output / "labels" / "test_img_00001.txt").read_text(encoding="utf-8").strip().splitlines() == [
        "0 0.75 0.75 0.3 0.3"
    ]
    assert (output / "images" / "train_img_00001.jpg").exists()
    assert (output / "images" / "val_img_00001.jpg").exists()
    assert (output / "images" / "test_img_00001.jpg").exists()
