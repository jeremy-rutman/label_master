from __future__ import annotations

from pathlib import Path

from label_master.adapters.coco.detector import detect_coco
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


def infer_format(
    input_path: Path,
    *,
    policy: InferencePolicy | None = None,
    force: bool = False,
) -> InferenceResult:
    policy = policy or InferencePolicy()

    coco_score = detect_coco(input_path)
    yolo_score = detect_yolo(input_path)

    candidates = [
        InferenceCandidate(format=SourceFormat.COCO, score=coco_score, evidence=["coco_detector"]),
        InferenceCandidate(format=SourceFormat.YOLO, score=yolo_score, evidence=["yolo_detector"]),
    ]
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
