from __future__ import annotations

from pathlib import Path
from typing import Callable

from label_master.adapters.coco.detector import detect_coco
from label_master.adapters.custom.detector import detect_custom_format
from label_master.adapters.kitware.detector import detect_kitware
from label_master.adapters.matlab_ground_truth.detector import detect_matlab_ground_truth
from label_master.adapters.video_bbox.detector import detect_video_bbox
from label_master.adapters.voc.detector import detect_voc
from label_master.adapters.yolo.detector import detect_yolo
from label_master.core.domain.entities import (
    InferenceCandidate,
    InferenceResult,
    Severity,
    SourceFormat,
    WarningEvent,
)
from label_master.core.domain.policies import InferencePolicy
from label_master.core.domain.value_objects import InferenceError
from label_master.format_specs.registry import load_builtin_format_specs


def _builtin_detectors() -> dict[str, Callable[..., float]]:
    return {
        "coco": detect_coco,
        "kitware": detect_kitware,
        "matlab_ground_truth": detect_matlab_ground_truth,
        "voc": detect_voc,
        "video_bbox": detect_video_bbox,
        "yolo": detect_yolo,
    }


def infer_format(
    input_path: Path,
    *,
    policy: InferencePolicy | None = None,
    force: bool = False,
) -> InferenceResult:
    policy = policy or InferencePolicy()
    detectors = _builtin_detectors()

    candidates: list[InferenceCandidate] = []
    for format_id, spec in load_builtin_format_specs().items():
        detector = detectors.get(format_id)
        if detector is None:
            continue
        score = detector(input_path, sample_limit=policy.sample_limit)
        candidates.append(
            InferenceCandidate(
                format=SourceFormat(format_id),
                score=score,
                evidence=[f"format_spec:{spec.format_id}"],
            )
        )

    custom_score, custom_spec_id = detect_custom_format(input_path, sample_limit=policy.sample_limit)
    candidates.append(
        InferenceCandidate(
            format=SourceFormat.CUSTOM,
            score=custom_score,
            evidence=[f"format_spec:{custom_spec_id}"] if custom_spec_id else ["format_spec:custom"],
        )
    )
    candidates.sort(key=lambda item: item.score, reverse=True)

    top = candidates[0]
    second = candidates[1]
    warnings: list[WarningEvent] = []

    if top.score == 0:
        predicted = SourceFormat.UNKNOWN
        confidence = 0.0
    elif policy.is_ambiguous(top.score, second.score):
        predicted = SourceFormat.AMBIGUOUS
        confidence = top.score
        warnings.append(
            WarningEvent(
                code="inference_ambiguous",
                message="Format inference is ambiguous between top candidates",
                severity=Severity.WARNING,
                context={
                    "top": top.format.value,
                    "top_score": str(top.score),
                    "second": second.format.value,
                    "second_score": str(second.score),
                },
            )
        )
    elif top.score < policy.min_confidence:
        predicted = SourceFormat.UNKNOWN
        confidence = top.score
        warnings.append(
            WarningEvent(
                code="inference_low_confidence",
                message="Format inference confidence below configured threshold",
                severity=Severity.WARNING,
                context={"score": str(top.score), "threshold": str(policy.min_confidence)},
            )
        )
    else:
        predicted = top.format
        confidence = top.score

    result = InferenceResult(
        predicted_format=predicted,
        confidence=confidence,
        candidates=candidates,
        warnings=warnings,
    )

    if result.predicted_format == SourceFormat.AMBIGUOUS and not force:
        raise InferenceError(
            "Ambiguous source format; pass force=True to override",
            context={"input_path": str(input_path)},
        )

    return result
