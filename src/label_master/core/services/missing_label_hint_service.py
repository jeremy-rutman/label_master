from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import TokenizedImageLabelsParserSpec, resolve_builtin_format_spec
from label_master.infra.filesystem import (
    InputPathFilter,
    atomic_write_json,
    build_input_path_filter,
    ensure_directory,
    iter_files,
    relative_path_matches_input_filter,
)


@dataclass(frozen=True)
class HintDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xywh_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class MissingLabelHint:
    image_rel_path: str
    suggested_label_rel_path: str
    detections: list[HintDetection]


@dataclass(frozen=True)
class MissingLabelHintResult:
    model_path: Path
    hints_output_dir: Path
    report_path: Path
    scanned_images: int
    images_with_existing_labels: int
    missing_label_images: int
    hinted_images: int
    hint_files_written: int
    total_detections: int
    hints: list[MissingLabelHint]


HintPredictor = Callable[[Path], list[HintDetection]]


def _yolo_parser() -> TokenizedImageLabelsParserSpec:
    spec = resolve_builtin_format_spec("yolo")
    if spec is None or not isinstance(spec.parser, TokenizedImageLabelsParserSpec):
        raise ValidationError("Built-in YOLO format spec is unavailable")
    return spec.parser


def _discover_images(
    dataset_root: Path,
    parser: TokenizedImageLabelsParserSpec,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> list[Path]:
    files = iter_files(dataset_root, suffixes=parser.image_extensions)
    image_paths: list[Path] = []
    for file_path in files:
        image_rel = file_path.relative_to(dataset_root).as_posix()
        if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
            continue
        image_paths.append(file_path)
    return image_paths


def _derive_label_rel_path(image_rel_path: Path, parser: TokenizedImageLabelsParserSpec) -> Path:
    label_rel_text = image_rel_path.with_suffix(".txt").as_posix()
    candidates = [label_rel_text]
    for rewrite in parser.path_rewrites:
        replaced = label_rel_text.replace(rewrite.to_text, rewrite.from_text)
        if replaced != label_rel_text:
            candidates.append(replaced)

    ordered_unique: list[str] = []
    for candidate in candidates:
        if candidate not in ordered_unique:
            ordered_unique.append(candidate)

    ordered_unique.sort(
        key=lambda candidate: (
            0 if "labels/" in candidate or candidate.startswith("labels") else 1,
            candidate,
        )
    )
    return Path(ordered_unique[0])


def _write_hint_label_file(label_path: Path, detections: list[HintDetection]) -> None:
    lines = [
        (
            f"{detection.class_id} "
            f"{detection.bbox_xywh_normalized[0]:.6f} "
            f"{detection.bbox_xywh_normalized[1]:.6f} "
            f"{detection.bbox_xywh_normalized[2]:.6f} "
            f"{detection.bbox_xywh_normalized[3]:.6f}"
        )
        for detection in detections
    ]
    ensure_directory(label_path.parent)
    content = "\n".join(lines).strip()
    label_path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def _build_ultralytics_predictor(
    *,
    model_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
) -> HintPredictor:
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Detector hints require ultralytics. Install it in this environment "
            "and retry, for example: pip install ultralytics"
        ) from exc

    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Detector model file not found: {model_path}")

    model = YOLO(str(model_path))
    raw_names = getattr(model, "names", {})
    class_name_map = {
        int(class_id): str(name)
        for class_id, name in (raw_names.items() if isinstance(raw_names, dict) else [])
    }

    def _predict(image_path: Path) -> list[HintDetection]:
        predictions = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_detections_per_image,
            verbose=False,
        )

        detections: list[HintDetection] = []
        for prediction in predictions:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue

            classes = [int(value) for value in boxes.cls.tolist()]
            confidences = [float(value) for value in boxes.conf.tolist()]
            xywhn_rows = boxes.xywhn.tolist()

            for class_id, confidence, xywhn in zip(classes, confidences, xywhn_rows, strict=True):
                if len(xywhn) != 4:
                    continue
                detections.append(
                    HintDetection(
                        class_id=class_id,
                        class_name=class_name_map.get(class_id, f"class_{class_id}"),
                        confidence=confidence,
                        bbox_xywh_normalized=(
                            float(xywhn[0]),
                            float(xywhn[1]),
                            float(xywhn[2]),
                            float(xywhn[3]),
                        ),
                    )
                )

        return detections

    return _predict


def generate_missing_yolo_label_hints(
    *,
    dataset_root: Path,
    detector_model_path: Path,
    hints_output_dir: Path,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections_per_image: int = 200,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    predictor: HintPredictor | None = None,
) -> MissingLabelHintResult:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    parser = _yolo_parser()
    input_path_filter = build_input_path_filter(
        include_substring=input_path_include_substring,
        exclude_substring=input_path_exclude_substring,
    )
    image_paths = _discover_images(
        resolved_root,
        parser,
        input_path_filter=input_path_filter,
    )
    resolved_hints_output_dir = hints_output_dir.expanduser().resolve()
    ensure_directory(resolved_hints_output_dir)

    active_predictor = predictor or _build_ultralytics_predictor(
        model_path=detector_model_path.expanduser().resolve(),
        confidence_threshold=max(0.0, min(1.0, float(confidence_threshold))),
        iou_threshold=max(0.0, min(1.0, float(iou_threshold))),
        max_detections_per_image=max(1, int(max_detections_per_image)),
    )

    hints: list[MissingLabelHint] = []
    scanned_images = 0
    images_with_existing_labels = 0
    missing_label_images = 0
    hinted_images = 0
    hint_files_written = 0
    total_detections = 0

    for image_path in image_paths:
        scanned_images += 1
        image_rel_path = image_path.relative_to(resolved_root)
        label_rel_path = _derive_label_rel_path(image_rel_path, parser)
        existing_label_path = resolved_root / label_rel_path

        if existing_label_path.exists():
            images_with_existing_labels += 1
            continue

        missing_label_images += 1
        detections = active_predictor(image_path)
        if not detections:
            continue

        hinted_images += 1
        total_detections += len(detections)
        hint_label_path = resolved_hints_output_dir / label_rel_path
        _write_hint_label_file(hint_label_path, detections)
        hint_files_written += 1
        hints.append(
            MissingLabelHint(
                image_rel_path=image_rel_path.as_posix(),
                suggested_label_rel_path=label_rel_path.as_posix(),
                detections=detections,
            )
        )

    report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(resolved_root),
        "model_path": str(detector_model_path.expanduser().resolve()),
        "hints_output_dir": str(resolved_hints_output_dir),
        "summary": {
            "scanned_images": scanned_images,
            "images_with_existing_labels": images_with_existing_labels,
            "missing_label_images": missing_label_images,
            "hinted_images": hinted_images,
            "hint_files_written": hint_files_written,
            "total_detections": total_detections,
        },
        "hints": [
            {
                "image_rel_path": hint.image_rel_path,
                "suggested_label_rel_path": hint.suggested_label_rel_path,
                "detections": [
                    {
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox_xywh_normalized": list(detection.bbox_xywh_normalized),
                    }
                    for detection in hint.detections
                ],
            }
            for hint in hints
        ],
    }
    report_path = resolved_hints_output_dir / "missing_label_hints.report.json"
    atomic_write_json(report_path, report_payload)

    return MissingLabelHintResult(
        model_path=detector_model_path.expanduser().resolve(),
        hints_output_dir=resolved_hints_output_dir,
        report_path=report_path,
        scanned_images=scanned_images,
        images_with_existing_labels=images_with_existing_labels,
        missing_label_images=missing_label_images,
        hinted_images=hinted_images,
        hint_files_written=hint_files_written,
        total_detections=total_detections,
        hints=hints,
    )
