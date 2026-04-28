from __future__ import annotations

from pathlib import Path

import label_master.core.services.infer_service as infer_service
from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import InferencePolicy


def test_infer_format_passes_sample_limit_to_detectors(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    observed: list[tuple[str, int]] = []

    def _record(name: str, score: float):  # type: ignore[no-untyped-def]
        def _detector(path: Path, *, sample_limit: int) -> float:
            assert path == tmp_path
            observed.append((name, sample_limit))
            return score

        return _detector

    monkeypatch.setattr(infer_service, "detect_coco", _record("coco", 0.9))
    monkeypatch.setattr(infer_service, "detect_kitware", _record("kitware", 0.1))
    monkeypatch.setattr(infer_service, "detect_matlab_ground_truth", _record("matlab_ground_truth", 0.0))
    monkeypatch.setattr(infer_service, "detect_voc", _record("voc", 0.0))
    monkeypatch.setattr(infer_service, "detect_video_bbox", _record("video_bbox", 0.0))
    monkeypatch.setattr(infer_service, "detect_yolo", _record("yolo", 0.0))
    monkeypatch.setattr(infer_service, "detect_custom_format", lambda path, *, sample_limit: (0.0, None))

    result = infer_service.infer_format(
        tmp_path,
        policy=InferencePolicy(sample_limit=17, min_confidence=0.5),
        force=True,
    )

    assert observed == [
        ("coco", 17),
        ("kitware", 17),
        ("matlab_ground_truth", 17),
        ("video_bbox", 17),
        ("voc", 17),
        ("yolo", 17),
    ]
    assert result.predicted_format == SourceFormat.COCO
