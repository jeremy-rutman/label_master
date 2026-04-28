from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import label_master.adapters.matlab_ground_truth.reader as matlab_ground_truth_reader
from label_master.adapters.matlab_ground_truth.reader import read_matlab_ground_truth_dataset
from label_master.core.domain.entities import SourceFormat
from label_master.core.services.infer_service import infer_format
from label_master.core.services.validate_service import validate_dataset
from label_master.interfaces.gui.viewmodels import (
    MATLAB_GROUND_TRUTH_PREVIEW_WARNING,
    preview_dataset_view,
)

FIXTURE = Path("tests/fixtures/us8/V_DRONE_001_LABELS.mat")


def _write_matlab_ground_truth_dataset(dataset_root: Path) -> None:
    video_root = dataset_root / "Video_V"
    video_root.mkdir(parents=True)
    shutil.copy2(FIXTURE, video_root / FIXTURE.name)
    (video_root / "V_DRONE_001.mp4").write_bytes(b"placeholder")


def test_infer_matlab_ground_truth_format_from_fixture(tmp_path: Path) -> None:
    _write_matlab_ground_truth_dataset(tmp_path)

    result = infer_format(tmp_path, force=True)

    assert result.predicted_format == SourceFormat.MATLAB_GROUND_TRUTH
    assert result.candidates[0].format == SourceFormat.MATLAB_GROUND_TRUTH
    assert result.candidates[0].score > 0.9


def test_read_matlab_ground_truth_dataset_loads_bbox_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_matlab_ground_truth_dataset(tmp_path)
    monkeypatch.setattr(matlab_ground_truth_reader, "probe_video_dimensions", lambda path: (1920, 1080))

    dataset = read_matlab_ground_truth_dataset(tmp_path)

    assert dataset.source_format == SourceFormat.MATLAB_GROUND_TRUTH
    assert len(dataset.images) == 301
    assert len(dataset.annotations) == 301
    assert [dataset.categories[index].name for index in sorted(dataset.categories)] == [
        "AIRPLANE",
        "BIRD",
        "DRONE",
        "HELICOPTER",
    ]

    first_image = dataset.images[0]
    assert first_image.image_id == "V_DRONE_001:000000"
    assert first_image.file_name == "images/V_DRONE_001/frame_000000.jpg"
    assert first_image.width == 1920
    assert first_image.height == 1080

    first_annotation = dataset.annotations[0]
    assert first_annotation.class_id == 2
    assert first_annotation.image_id == first_image.image_id
    assert first_annotation.bbox_xywh_abs == pytest.approx(
        (124.92949676513672, 22.944549560546875, 75.84915924072266, 39.2028923034668)
    )
    assert first_annotation.attributes["timestamp_ms"] == pytest.approx(0.0)
    assert str(first_annotation.attributes["source_video_path"]).endswith("V_DRONE_001.mp4")

    last_annotation = dataset.annotations[-1]
    assert last_annotation.image_id == "V_DRONE_001:000300"
    assert last_annotation.bbox_xywh_abs == pytest.approx((321.0, 122.0, 52.0, 29.0))


def test_read_matlab_ground_truth_dataset_reports_file_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_matlab_ground_truth_dataset(tmp_path)
    monkeypatch.setattr(matlab_ground_truth_reader, "probe_video_dimensions", lambda path: (1920, 1080))
    observed: list[tuple[int, int]] = []

    read_matlab_ground_truth_dataset(
        tmp_path,
        progress_callback=lambda completed, total: observed.append((completed, total)),
    )

    assert observed == [(1, 1)]


def test_validate_and_preview_matlab_ground_truth_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_matlab_ground_truth_dataset(tmp_path)
    monkeypatch.setattr(matlab_ground_truth_reader, "probe_video_dimensions", lambda path: (1920, 1080))

    validation = validate_dataset(tmp_path, source_format=SourceFormat.MATLAB_GROUND_TRUTH)
    preview = preview_dataset_view(tmp_path, source_format="matlab_ground_truth")

    assert validation.inferred_format == SourceFormat.MATLAB_GROUND_TRUTH
    assert validation.summary.invalid_annotations == 0
    assert preview.source_format == "matlab_ground_truth"
    assert preview.image_count == 301
    assert preview.images[0].bboxes[0].class_name == "DRONE"
    assert preview.warnings == [MATLAB_GROUND_TRUTH_PREVIEW_WARNING]
