from __future__ import annotations

import json
from pathlib import Path

import label_master.adapters.voc.detector as voc_detector
import pytest
from label_master.adapters.voc.reader import read_voc_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.value_objects import ValidationError
from label_master.core.services.infer_service import infer_format

FIXTURE = Path("tests/fixtures/us6")


def _write_voc_pair(dataset_root: Path, stem: str, class_name: str) -> None:
    image_path = dataset_root / "images" / f"{stem}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"placeholder")

    xml_path = dataset_root / "annotations" / f"{stem}.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(
        f"""<annotation>
    <filename>{stem}.jpg</filename>
    <path>images/{stem}.jpg</path>
    <size><width>100</width><height>100</height></size>
    <object>
        <name>{class_name}</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>20</ymin>
            <xmax>30</xmax>
            <ymax>40</ymax>
        </bndbox>
    </object>
</annotation>
""",
        encoding="utf-8",
    )


def test_infer_voc_format_from_fixture() -> None:
    result = infer_format(FIXTURE, force=True)

    assert result.predicted_format == SourceFormat.VOC
    assert result.candidates[0].format == SourceFormat.VOC
    assert result.candidates[0].score > 0.9


def test_read_voc_dataset_loads_fixture() -> None:
    dataset = read_voc_dataset(FIXTURE)

    assert dataset.source_format == SourceFormat.VOC
    assert len(dataset.images) == 5
    assert len(dataset.annotations) == 5
    assert list(dataset.categories) == [0]
    assert dataset.categories[0].name == "UAV"

    first_image = dataset.images[0]
    assert first_image.image_id == "data/img/00001"
    assert first_image.file_name == "data/img/00001.jpg"
    assert first_image.width == 1280
    assert first_image.height == 720

    first_annotation = dataset.annotations[0]
    assert first_annotation.annotation_id == "data/xml/00001.xml:1"
    assert first_annotation.image_id == first_image.image_id
    assert first_annotation.class_id == 0
    assert first_annotation.bbox_xywh_abs == (616.0, 540.0, 68.0, 61.0)

    last_annotation = dataset.annotations[-1]
    assert last_annotation.annotation_id == "data/xml/00005.xml:1"
    assert last_annotation.bbox_xywh_abs == (666.0, 353.0, 23.0, 19.0)


def test_detect_voc_fixture_does_not_require_full_image_index(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        voc_detector,
        "build_voc_image_index",
        lambda path: (_ for _ in ()).throw(AssertionError("image index should not be built for this fixture")),
    )

    score = voc_detector.detect_voc(FIXTURE)

    assert score > 0.9


def test_read_voc_dataset_can_cap_sample_size() -> None:
    dataset = read_voc_dataset(FIXTURE, max_xml_files=3)

    assert dataset.source_format == SourceFormat.VOC
    assert len(dataset.images) == 3
    assert len(dataset.annotations) == 3
    assert dataset.source_metadata.details["xml_files_loaded"] == "3"
    assert dataset.source_metadata.details["xml_files_total"] == "5"
    assert dataset.source_metadata.details["xml_files_limit"] == "3"


def test_read_voc_dataset_sample_uses_full_class_id_table(tmp_path: Path) -> None:
    _write_voc_pair(tmp_path, "001", "zebra")
    _write_voc_pair(tmp_path, "002", "antelope")
    _write_voc_pair(tmp_path, "003", "buffalo")

    dataset = read_voc_dataset(tmp_path, max_xml_files=2)

    assert len(dataset.annotations) == 2
    assert {category.class_id: category.name for category in dataset.categories.values()} == {
        0: "zebra",
        1: "antelope",
        2: "buffalo",
    }
    assert [annotation.annotation_id for annotation in dataset.annotations] == [
        "annotations/001.xml:1",
        "annotations/003.xml:1",
    ]
    assert [annotation.class_id for annotation in dataset.annotations] == [0, 2]


def test_read_voc_dataset_reports_parse_reason(tmp_path: Path) -> None:
    xml_dir = tmp_path / "val" / "xml"
    xml_dir.mkdir(parents=True)
    (xml_dir / "00991.xml").write_text(
        """<annotation>
    <size><width>1920</width><height>1080</height></size>
</annotation>
""",
        encoding="utf-8",
    )

    dataset = read_voc_dataset(tmp_path)

    assert dataset.images == []
    assert dataset.annotations == []
    assert [warning.model_dump(mode="json") for warning in dataset.warnings] == [
        {
            "code": "voc_annotation_file_skipped",
            "message": (
                "Skipped Pascal VOC annotation file val/xml/00991.xml: "
                "VOC annotation is missing filename: val/xml/00991.xml"
            ),
            "severity": "warning",
            "context": {
                "xml_file": "val/xml/00991.xml",
                "source_file": "val/xml/00991.xml",
                "reason": "VOC annotation is missing filename: val/xml/00991.xml",
                "skipped_files_json": json.dumps(
                    [
                        {
                            "source_file": "val/xml/00991.xml",
                            "reason": "VOC annotation is missing filename: val/xml/00991.xml",
                        }
                    ]
                ),
            },
        }
    ]


def test_read_voc_dataset_skips_degenerate_bbox_file(tmp_path: Path) -> None:
    xml_dir = tmp_path / "val" / "xml"
    xml_dir.mkdir(parents=True)
    image_path = tmp_path / "val" / "00991.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"placeholder")
    (xml_dir / "00991.xml").write_text(
        """<annotation>
    <filename>00991.jpg</filename>
    <path>./val/00991.jpg</path>
    <size><width>1920</width><height>1080</height></size>
    <object>
        <name>UAV</name>
        <bndbox>
            <xmin>1056</xmin>
            <ymin>443</ymin>
            <xmax>1059</xmax>
            <ymax>443</ymax>
        </bndbox>
    </object>
</annotation>
""",
        encoding="utf-8",
    )

    dataset = read_voc_dataset(tmp_path)

    assert dataset.images == []
    assert dataset.annotations == []
    assert dataset.source_metadata.details["xml_files_skipped"] == "1"
    assert dataset.warnings[0].code == "voc_annotation_file_skipped"
    assert "VOC bbox must have xmax > xmin and ymax > ymin" in dataset.warnings[0].message
