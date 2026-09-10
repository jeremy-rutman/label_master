from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Sequence

from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import (
    TokenizedImageLabelsParserSpec,
    resolve_builtin_format_spec,
)
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


AuditProposalAction = Literal["add", "remove", "adjust"]


@dataclass(frozen=True)
class ExistingYoloLabel:
    label_index: int
    class_id: int
    class_name: str
    bbox_xywh_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloBBoxAuditProposal:
    proposal_id: str
    action: AuditProposalAction
    class_id: int
    class_name: str
    confidence: float | None
    match_iou: float | None
    existing_label_index: int | None
    existing_bbox_xywh_normalized: tuple[float, float, float, float] | None
    proposed_bbox_xywh_normalized: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class YoloBBoxAuditItem:
    image_rel_path: str
    label_rel_path: str
    existing_label_count: int
    proposals: list[YoloBBoxAuditProposal]


@dataclass(frozen=True)
class YoloBBoxAuditResult:
    model_path: Path
    report_output_dir: Path
    report_path: Path
    scanned_labeled_images: int
    images_with_proposals: int
    add_proposals: int
    remove_proposals: int
    adjust_proposals: int
    items: list[YoloBBoxAuditItem]


@dataclass(frozen=True)
class SingleImageYoloDetectorReview:
    image_rel_path: str
    label_rel_path: str
    has_existing_label_file: bool
    existing_label_count: int
    proposals: list[YoloBBoxAuditProposal]


@dataclass(frozen=True)
class CombinedYoloDetectorReviewResult:
    missing_label_hints: MissingLabelHintResult
    bbox_audit: YoloBBoxAuditResult


@dataclass(frozen=True)
class YoloBBoxAuditApprovalResult:
    approved_label_files: int
    applied_proposals: int
    label_paths: list[str]


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


def _discover_label_files(
    dataset_root: Path,
    parser: TokenizedImageLabelsParserSpec,
    *,
    input_path_filter: InputPathFilter | None = None,
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in parser.label_globs:
        for path in sorted(dataset_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            image_rel_path = _resolve_image_rel_path(dataset_root, path, parser)
            if not relative_path_matches_input_filter(image_rel_path, input_path_filter=input_path_filter):
                continue
            seen.add(path)
            files.append(path)
    return files


def _resolve_image_rel_path(
    dataset_root: Path,
    label_path: Path,
    parser: TokenizedImageLabelsParserSpec,
) -> str:
    label_rel = label_path.relative_to(dataset_root)
    image_rel_text = label_rel.with_suffix("").as_posix()
    for rewrite in parser.path_rewrites:
        image_rel_text = image_rel_text.replace(rewrite.from_text, rewrite.to_text)

    for extension in parser.image_extensions:
        candidate = dataset_root / f"{image_rel_text}{extension}"
        if candidate.exists() and candidate.is_file():
            return f"{image_rel_text}{extension}"
    return f"{image_rel_text}.jpg"


def _load_yolo_class_names(dataset_root: Path) -> dict[int, str]:
    parser = _yolo_parser()
    candidate_files = [dataset_root / parser.classes_file_name, dataset_root / "obj.names"]
    candidate_files.extend(sorted(dataset_root.glob("**/obj.names")))
    classes_file = next((path for path in candidate_files if path.exists() and path.is_file()), None)
    if classes_file is None:
        return {}

    class_names: dict[int, str] = {}
    with classes_file.open("r", encoding="utf-8") as handle:
        for class_id, line in enumerate(handle):
            name = line.strip()
            if name:
                class_names[class_id] = name
    return class_names


def _read_existing_yolo_labels(
    label_path: Path,
    *,
    class_names: dict[int, str],
) -> list[ExistingYoloLabel]:
    labels: list[ExistingYoloLabel] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for label_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != 5:
                raise ValidationError(f"YOLO label row must have 5 columns: {label_path}")
            try:
                class_id = int(tokens[0])
                cx, cy, w, h = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))
            except ValueError as exc:
                raise ValidationError(f"YOLO label row must contain numeric bbox fields: {label_path}") from exc
            labels.append(
                ExistingYoloLabel(
                    label_index=label_index,
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"class_{class_id}"),
                    bbox_xywh_normalized=(cx, cy, w, h),
                )
            )
    return labels


def load_existing_yolo_labels_for_rel_path(
    *,
    dataset_root: Path,
    label_rel_path: str,
) -> list[ExistingYoloLabel]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    resolved_label_path = resolved_root / Path(label_rel_path)
    if not resolved_label_path.exists() or not resolved_label_path.is_file():
        return []

    class_names = _load_yolo_class_names(resolved_root)
    return _read_existing_yolo_labels(resolved_label_path, class_names=class_names)


def load_yolo_class_names_for_dataset_root(dataset_root: Path) -> dict[int, str]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")
    return _load_yolo_class_names(resolved_root)


def _normalized_box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1 = left[0] - (left[2] / 2.0)
    left_y1 = left[1] - (left[3] / 2.0)
    left_x2 = left[0] + (left[2] / 2.0)
    left_y2 = left[1] + (left[3] / 2.0)
    right_x1 = right[0] - (right[2] / 2.0)
    right_y1 = right[1] - (right[3] / 2.0)
    right_x2 = right[0] + (right[2] / 2.0)
    right_y2 = right[1] + (right[3] / 2.0)

    intersection_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    if intersection <= 0.0:
        return 0.0

    left_area = max(left[2], 0.0) * max(left[3], 0.0)
    right_area = max(right[2], 0.0) * max(right[3], 0.0)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def write_yolo_hint_label_file(label_path: Path, detections: Sequence[HintDetection]) -> None:
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


def _write_hint_label_file(label_path: Path, detections: list[HintDetection]) -> None:
    write_yolo_hint_label_file(label_path, detections)


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


@lru_cache(maxsize=4)
def _cached_ultralytics_predictor(
    model_path_text: str,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
) -> HintPredictor:
    return _build_ultralytics_predictor(
        model_path=Path(model_path_text),
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
    )


def _get_cached_ultralytics_predictor(
    *,
    model_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
) -> HintPredictor:
    normalized_model_path = str(model_path.expanduser().resolve())
    normalized_confidence = round(max(0.0, min(1.0, float(confidence_threshold))), 6)
    normalized_iou = round(max(0.0, min(1.0, float(iou_threshold))), 6)
    normalized_max_detections = max(1, int(max_detections_per_image))
    return _cached_ultralytics_predictor(
        normalized_model_path,
        normalized_confidence,
        normalized_iou,
        normalized_max_detections,
    )


def _existing_label_to_detection(label: ExistingYoloLabel) -> HintDetection:
    return HintDetection(
        class_id=label.class_id,
        class_name=label.class_name,
        confidence=1.0,
        bbox_xywh_normalized=label.bbox_xywh_normalized,
    )


def _build_yolo_bbox_audit_proposals(
    *,
    existing_labels: list[ExistingYoloLabel],
    detections: list[HintDetection],
    match_iou_threshold: float,
    correction_iou_threshold: float,
) -> list[YoloBBoxAuditProposal]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(existing_labels):
        for detection_index, detection in enumerate(detections):
            if label.class_id != detection.class_id:
                continue
            iou = _normalized_box_iou(label.bbox_xywh_normalized, detection.bbox_xywh_normalized)
            if iou >= match_iou_threshold:
                candidate_pairs.append((iou, label_index, detection_index))

    matched_labels: set[int] = set()
    matched_detections: set[int] = set()
    proposals: list[YoloBBoxAuditProposal] = []

    for iou, label_index, detection_index in sorted(candidate_pairs, reverse=True):
        if label_index in matched_labels or detection_index in matched_detections:
            continue
        matched_labels.add(label_index)
        matched_detections.add(detection_index)
        if iou >= correction_iou_threshold:
            continue

        label = existing_labels[label_index]
        detection = detections[detection_index]
        proposals.append(
            YoloBBoxAuditProposal(
                proposal_id=f"adjust:{label.label_index}:{detection_index}",
                action="adjust",
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                match_iou=iou,
                existing_label_index=label.label_index,
                existing_bbox_xywh_normalized=label.bbox_xywh_normalized,
                proposed_bbox_xywh_normalized=detection.bbox_xywh_normalized,
            )
        )

    for label in existing_labels:
        if label.label_index in {existing_labels[index].label_index for index in matched_labels}:
            continue
        proposals.append(
            YoloBBoxAuditProposal(
                proposal_id=f"remove:{label.label_index}",
                action="remove",
                class_id=label.class_id,
                class_name=label.class_name,
                confidence=None,
                match_iou=None,
                existing_label_index=label.label_index,
                existing_bbox_xywh_normalized=label.bbox_xywh_normalized,
                proposed_bbox_xywh_normalized=None,
            )
        )

    for detection_index, detection in enumerate(detections):
        if detection_index in matched_detections:
            continue
        proposals.append(
            YoloBBoxAuditProposal(
                proposal_id=f"add:{detection_index}",
                action="add",
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                match_iou=None,
                existing_label_index=None,
                existing_bbox_xywh_normalized=None,
                proposed_bbox_xywh_normalized=detection.bbox_xywh_normalized,
            )
        )

    action_order = {"remove": 0, "adjust": 1, "add": 2}
    return sorted(
        proposals,
        key=lambda proposal: (
            action_order.get(proposal.action, 99),
            proposal.existing_label_index if proposal.existing_label_index is not None else 1_000_000,
            proposal.class_id,
            proposal.proposal_id,
        ),
    )


def _build_add_only_proposals(detections: list[HintDetection]) -> list[YoloBBoxAuditProposal]:
    return [
        YoloBBoxAuditProposal(
            proposal_id=f"add:{detection_index}",
            action="add",
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            match_iou=None,
            existing_label_index=None,
            existing_bbox_xywh_normalized=None,
            proposed_bbox_xywh_normalized=detection.bbox_xywh_normalized,
        )
        for detection_index, detection in enumerate(detections)
    ]


def discover_yolo_review_image_paths(
    *,
    dataset_root: Path,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> list[str]:
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
    return sorted(path.relative_to(resolved_root).as_posix() for path in image_paths)


def generate_yolo_detector_review_for_image(
    *,
    dataset_root: Path,
    image_rel_path: str,
    detector_model_path: Path,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections_per_image: int = 200,
    match_iou_threshold: float = 0.30,
    correction_iou_threshold: float = 0.85,
    predictor: HintPredictor | None = None,
) -> SingleImageYoloDetectorReview:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    resolved_image_rel_path = Path(image_rel_path)
    resolved_image_path = resolved_root / resolved_image_rel_path
    if not resolved_image_path.exists() or not resolved_image_path.is_file():
        raise FileNotFoundError(f"YOLO image file not found for detector review: {resolved_image_path}")

    parser = _yolo_parser()
    label_rel_path = _derive_label_rel_path(resolved_image_rel_path, parser)
    label_path = resolved_root / label_rel_path
    active_predictor = predictor or _get_cached_ultralytics_predictor(
        model_path=detector_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
    )
    detections = active_predictor(resolved_image_path)

    if label_path.exists() and label_path.is_file():
        class_names = _load_yolo_class_names(resolved_root)
        existing_labels = _read_existing_yolo_labels(label_path, class_names=class_names)
        proposals = _build_yolo_bbox_audit_proposals(
            existing_labels=existing_labels,
            detections=detections,
            match_iou_threshold=max(0.0, min(1.0, float(match_iou_threshold))),
            correction_iou_threshold=max(0.0, min(1.0, float(correction_iou_threshold))),
        )
        return SingleImageYoloDetectorReview(
            image_rel_path=resolved_image_rel_path.as_posix(),
            label_rel_path=label_rel_path.as_posix(),
            has_existing_label_file=True,
            existing_label_count=len(existing_labels),
            proposals=proposals,
        )

    return SingleImageYoloDetectorReview(
        image_rel_path=resolved_image_rel_path.as_posix(),
        label_rel_path=label_rel_path.as_posix(),
        has_existing_label_file=False,
        existing_label_count=0,
        proposals=_build_add_only_proposals(detections),
    )


def generate_yolo_bbox_audit(
    *,
    dataset_root: Path,
    detector_model_path: Path,
    report_output_dir: Path,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections_per_image: int = 200,
    max_labeled_images: int = 100,
    match_iou_threshold: float = 0.30,
    correction_iou_threshold: float = 0.85,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    predictor: HintPredictor | None = None,
) -> YoloBBoxAuditResult:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    parser = _yolo_parser()
    class_names = _load_yolo_class_names(resolved_root)
    input_path_filter = build_input_path_filter(
        include_substring=input_path_include_substring,
        exclude_substring=input_path_exclude_substring,
    )
    label_files = _discover_label_files(
        resolved_root,
        parser,
        input_path_filter=input_path_filter,
    )
    if max_labeled_images > 0:
        label_files = label_files[:max_labeled_images]

    resolved_report_output_dir = report_output_dir.expanduser().resolve()
    ensure_directory(resolved_report_output_dir)

    active_predictor = predictor or _build_ultralytics_predictor(
        model_path=detector_model_path.expanduser().resolve(),
        confidence_threshold=max(0.0, min(1.0, float(confidence_threshold))),
        iou_threshold=max(0.0, min(1.0, float(iou_threshold))),
        max_detections_per_image=max(1, int(max_detections_per_image)),
    )

    scanned_labeled_images = 0
    images_with_proposals = 0
    add_proposals = 0
    remove_proposals = 0
    adjust_proposals = 0
    items: list[YoloBBoxAuditItem] = []

    for label_path in label_files:
        scanned_labeled_images += 1
        image_rel_path = _resolve_image_rel_path(resolved_root, label_path, parser)
        image_path = resolved_root / image_rel_path
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"YOLO image file not found for label file: {label_path}")

        existing_labels = _read_existing_yolo_labels(label_path, class_names=class_names)
        detections = active_predictor(image_path)
        proposals = _build_yolo_bbox_audit_proposals(
            existing_labels=existing_labels,
            detections=detections,
            match_iou_threshold=max(0.0, min(1.0, float(match_iou_threshold))),
            correction_iou_threshold=max(0.0, min(1.0, float(correction_iou_threshold))),
        )
        if not proposals:
            continue

        images_with_proposals += 1
        add_proposals += sum(proposal.action == "add" for proposal in proposals)
        remove_proposals += sum(proposal.action == "remove" for proposal in proposals)
        adjust_proposals += sum(proposal.action == "adjust" for proposal in proposals)
        items.append(
            YoloBBoxAuditItem(
                image_rel_path=image_rel_path,
                label_rel_path=label_path.relative_to(resolved_root).as_posix(),
                existing_label_count=len(existing_labels),
                proposals=proposals,
            )
        )

    report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(resolved_root),
        "model_path": str(detector_model_path.expanduser().resolve()),
        "report_output_dir": str(resolved_report_output_dir),
        "summary": {
            "scanned_labeled_images": scanned_labeled_images,
            "images_with_proposals": images_with_proposals,
            "add_proposals": add_proposals,
            "remove_proposals": remove_proposals,
            "adjust_proposals": adjust_proposals,
        },
        "items": [
            {
                "image_rel_path": item.image_rel_path,
                "label_rel_path": item.label_rel_path,
                "existing_label_count": item.existing_label_count,
                "proposals": [
                    {
                        "proposal_id": proposal.proposal_id,
                        "action": proposal.action,
                        "class_id": proposal.class_id,
                        "class_name": proposal.class_name,
                        "confidence": proposal.confidence,
                        "match_iou": proposal.match_iou,
                        "existing_label_index": proposal.existing_label_index,
                        "existing_bbox_xywh_normalized": list(proposal.existing_bbox_xywh_normalized)
                        if proposal.existing_bbox_xywh_normalized is not None
                        else None,
                        "proposed_bbox_xywh_normalized": list(proposal.proposed_bbox_xywh_normalized)
                        if proposal.proposed_bbox_xywh_normalized is not None
                        else None,
                    }
                    for proposal in item.proposals
                ],
            }
            for item in items
        ],
    }
    report_path = resolved_report_output_dir / "yolo_bbox_audit.report.json"
    atomic_write_json(report_path, report_payload)

    return YoloBBoxAuditResult(
        model_path=detector_model_path.expanduser().resolve(),
        report_output_dir=resolved_report_output_dir,
        report_path=report_path,
        scanned_labeled_images=scanned_labeled_images,
        images_with_proposals=images_with_proposals,
        add_proposals=add_proposals,
        remove_proposals=remove_proposals,
        adjust_proposals=adjust_proposals,
        items=items,
    )


def apply_yolo_bbox_audit(
    *,
    dataset_root: Path,
    approved_items: Sequence[YoloBBoxAuditItem],
) -> YoloBBoxAuditApprovalResult:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    class_names = _load_yolo_class_names(resolved_root)
    label_paths: list[str] = []
    applied_proposals = 0

    for item in approved_items:
        if not item.proposals:
            continue

        label_path = resolved_root / item.label_rel_path
        if not label_path.exists() or not label_path.is_file():
            raise FileNotFoundError(f"Label file not found for bbox audit approval: {label_path}")

        existing_labels = _read_existing_yolo_labels(label_path, class_names=class_names)
        remove_indexes = {
            proposal.existing_label_index
            for proposal in item.proposals
            if proposal.action == "remove" and proposal.existing_label_index is not None
        }
        adjust_map = {
            proposal.existing_label_index: HintDetection(
                class_id=proposal.class_id,
                class_name=proposal.class_name,
                confidence=proposal.confidence or 1.0,
                bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
            )
            for proposal in item.proposals
            if proposal.action == "adjust"
            and proposal.existing_label_index is not None
            and proposal.proposed_bbox_xywh_normalized is not None
        }
        add_detections = [
            HintDetection(
                class_id=proposal.class_id,
                class_name=proposal.class_name,
                confidence=proposal.confidence or 1.0,
                bbox_xywh_normalized=proposal.proposed_bbox_xywh_normalized,
            )
            for proposal in item.proposals
            if proposal.action == "add" and proposal.proposed_bbox_xywh_normalized is not None
        ]

        final_detections: list[HintDetection] = []
        for label in existing_labels:
            if label.label_index in remove_indexes:
                continue
            replacement = adjust_map.get(label.label_index)
            if replacement is not None:
                final_detections.append(replacement)
            else:
                final_detections.append(_existing_label_to_detection(label))

        final_detections.extend(add_detections)
        write_yolo_hint_label_file(label_path, final_detections)
        label_paths.append(str(label_path))
        applied_proposals += len(item.proposals)

    return YoloBBoxAuditApprovalResult(
        approved_label_files=len(label_paths),
        applied_proposals=applied_proposals,
        label_paths=label_paths,
    )


def generate_yolo_detector_review(
    *,
    dataset_root: Path,
    detector_model_path: Path,
    hints_output_dir: Path,
    report_output_dir: Path,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections_per_image: int = 200,
    max_labeled_images: int = 100,
    match_iou_threshold: float = 0.30,
    correction_iou_threshold: float = 0.85,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
    predictor: HintPredictor | None = None,
) -> CombinedYoloDetectorReviewResult:
    resolved_model_path = detector_model_path.expanduser().resolve()
    normalized_confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
    normalized_iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
    normalized_max_detections = max(1, int(max_detections_per_image))

    active_predictor = predictor or _build_ultralytics_predictor(
        model_path=resolved_model_path,
        confidence_threshold=normalized_confidence_threshold,
        iou_threshold=normalized_iou_threshold,
        max_detections_per_image=normalized_max_detections,
    )

    bbox_audit = generate_yolo_bbox_audit(
        dataset_root=dataset_root,
        detector_model_path=resolved_model_path,
        report_output_dir=report_output_dir,
        confidence_threshold=normalized_confidence_threshold,
        iou_threshold=normalized_iou_threshold,
        max_detections_per_image=normalized_max_detections,
        max_labeled_images=max_labeled_images,
        match_iou_threshold=match_iou_threshold,
        correction_iou_threshold=correction_iou_threshold,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        predictor=active_predictor,
    )
    missing_label_hints = generate_missing_yolo_label_hints(
        dataset_root=dataset_root,
        detector_model_path=resolved_model_path,
        hints_output_dir=hints_output_dir,
        confidence_threshold=normalized_confidence_threshold,
        iou_threshold=normalized_iou_threshold,
        max_detections_per_image=normalized_max_detections,
        input_path_include_substring=input_path_include_substring,
        input_path_exclude_substring=input_path_exclude_substring,
        predictor=active_predictor,
    )
    return CombinedYoloDetectorReviewResult(
        missing_label_hints=missing_label_hints,
        bbox_audit=bbox_audit,
    )


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
        write_yolo_hint_label_file(hint_label_path, detections)
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
