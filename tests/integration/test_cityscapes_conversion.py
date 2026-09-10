from __future__ import annotations

import json
from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion


def _write_cityscapes_pair(dataset_root: Path) -> None:
    labels_dir = dataset_root / "gtCoarse" / "gtCoarse" / "train" / "sample_city"
    images_dir = dataset_root / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train" / "sample_city"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / "sample_city_000001_000001_leftImg8bit.png").write_bytes(b"fake-png")
    (labels_dir / "sample_city_000001_000001_gtCoarse_polygons.json").write_text(
        json.dumps(
            {
                "imgWidth": 100,
                "imgHeight": 60,
                "objects": [
                    {"label": "car", "polygon": [[10, 20], [30, 20], [30, 40], [10, 40]]},
                    {"label": "person", "polygon": [[50, 5], [55, 5], [55, 20], [50, 20]]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cityscapes_to_yolo_conversion_copies_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "cityscapes"
    output_root = tmp_path / "converted"
    _write_cityscapes_pair(dataset_root)

    result = execute_conversion(
        ConvertRequest(
            run_id="cityscapes-yolo",
            input_path=dataset_root,
            output_path=output_root,
            src_format=SourceFormat.CITYSCAPES,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    label_path = (
        output_root
        / "labels"
        / "leftImg8bit_trainvaltest"
        / "leftImg8bit"
        / "train"
        / "sample_city"
        / "sample_city_000001_000001_leftImg8bit.txt"
    )
    image_path = (
        output_root
        / "images"
        / "leftImg8bit_trainvaltest"
        / "leftImg8bit"
        / "train"
        / "sample_city"
        / "sample_city_000001_000001_leftImg8bit.png"
    )

    assert result.report.status == "completed"
    assert result.validation.dataset.source_format == SourceFormat.CITYSCAPES
    assert label_path.exists()
    assert image_path.exists()

    label_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(label_lines) == 2
    assert {line.split()[0] for line in label_lines} == {"0", "2"}

    classes_lines = [line.strip() for line in (output_root / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert classes_lines[:3] == ["person", "rider", "car"]