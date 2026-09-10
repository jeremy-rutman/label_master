from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, MutableMapping
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
import yaml
from PIL import Image, ImageDraw, ImageFont

from label_master.adapters.custom.detector import detect_custom_format
from label_master.adapters.video_bbox.reader import load_video_bbox_preview_image
from label_master.core.domain.policies import (
    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
    DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    InvalidAnnotationAction,
    OutOfFrameBBoxPolicy,
    ValidationMode,
)
from label_master.core.services.classification_service import (
    ClassificationConfig,
    apply_classification_label,
    clear_classification_label,
    discover_classification_image_paths,
    find_default_classification_config_path,
    image_labels,
    load_classification_config,
    load_classification_labels,
    next_unlabeled_index,
    resolve_classification_labels_path,
    save_classification_labels,
    summarize_classification_labels,
)
from label_master.core.services.convert_service import (
    derive_output_filename_prefix,
    sanitize_output_file_stem_affix,
)
from label_master.format_specs.registry import (
    custom_format_spec_entries,
    load_custom_format_spec_from_path,
    resolve_builtin_format_spec,
    resolve_custom_format_spec,
)
from label_master.infra.filesystem import (
    atomic_write_json,
    ensure_directory,
    normalize_input_path_filter_substring,
)
from label_master.infra.reporting import build_run_warnings_payload
from label_master.interfaces.gui import system_actions
from label_master.interfaces.gui.system_actions import OutputDirectoryOpenResult
from label_master.interfaces.gui.viewmodels import (
    BBoxAuditItemViewModel,
    BBoxAuditProposalViewModel,
    DetectorReviewEditorBoxViewModel,
    DetectorReviewItemViewModel,
    MappingRowViewModel,
    MissingLabelHintItemViewModel,
    apply_detector_review_bbox_strategy,
    approve_detector_review_edited_boxes_view,
    build_detector_review_editor_state_view,
    build_gui_run_config,
    convert_view,
    detector_review_final_bbox_xywh_normalized,
    generate_detector_review_item_view,
    infer_view,
    list_detector_review_image_paths_view,
    parse_mapping_rows,
    preview_dataset_view,
)

LOCALHOST_VALUES = {"127.0.0.1", "localhost", "::1"}
SOURCE_FORMATS = ["auto", "cityscapes", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
DESTINATION_FORMATS = ["yolo", "coco"]
DEFAULT_DESTINATION_FORMAT = "yolo"
UNMAPPED_POLICIES = ["error", "drop", "identity"]
VALIDATION_MODES = [ValidationMode.STRICT.value, ValidationMode.PERMISSIVE.value]
PERMISSIVE_INVALID_ANNOTATION_ACTIONS = [
    InvalidAnnotationAction.KEEP.value,
    InvalidAnnotationAction.DROP.value,
]
MAPPING_ACTIONS = ["map", "drop", "drop_frame"]
OVERSIZE_IMAGE_ACTIONS = ["ignore", "downscale"]
DEFAULT_INPUT_DIR = "tests/fixtures/us1/coco_minimal"
DEFAULT_OUTPUT_DIR = "/tmp/label_master_gui_output"
DEFAULT_MAPPING_ROWS: list[dict[str, str]] = []
GUI_STATE_FILE_NAME = "gui_state.json"
PREVIEW_MAX_IMAGE_DIMENSION = 1600
DEFAULT_PREVIEW_SCAN_LIMIT = 100
MAX_PREVIEW_SCAN_LIMIT = 1_000_000
RUN_STATUSES = {"idle", "running", "completed", "failed"}
RUN_EVENTS = {"start", "complete", "fail", "reset"}
RUN_PROGRESS_BY_STATUS = {
    "idle": 0,
    "running": 15,
    "completed": 100,
    "failed": 100,
}
RUN_INTERRUPTED_DETAIL = "Conversion interrupted before completion. You can run conversion again."
STREAMLIT_CONTROL_FLOW_EXCEPTION_NAMES = {"StopException", "RerunException"}
PREVIEW_CLASS_EXAMPLES_PER_CLASS = 3
CLASSIFICATION_KEYBOARD_ACTIONS = {"previous", "next", "class", "clear"}
CLASSIFICATION_CONFIG_EXAMPLE = """\
# classification.yaml (place in the dataset root, or point the GUI at it)
classes:
  - key: "1"
    name: helicopter
  - key: "2"
    name: airplane
  - key: "3"
    name: bird
  - key: "0"
    name: none
# optional: allow several classes per image (keys toggle each class)
multi_label: false
# optional: where labels are stored, relative to the dataset root
labels_file: classification_labels.json
"""
_PREVIEW_KEYBOARD_NAV_COMPONENT = components.declare_component(
    "preview_keyboard_nav",
    path=Path(__file__).resolve().parent / "components" / "preview_keyboard_nav",
)
_BBOX_EDITOR_COMPONENT = components.declare_component(
    "bbox_editor",
    path=Path(__file__).resolve().parent / "components" / "bbox_editor",
)


@dataclass(frozen=True)
class DirectoryValidationResult:
    resolved_path: Path | None
    errors: list[str]


@dataclass(frozen=True)
class InputDirectoryBrowseState:
    input_dir_raw: str
    browse_available: bool
    browse_message: str | None


@dataclass(frozen=True)
class OutputDirectoryBrowseState:
    output_dir_raw: str
    browse_available: bool
    browse_message: str | None


@dataclass(frozen=True)
class CustomFormatFileBrowseState:
    custom_format_path_raw: str
    browse_available: bool
    browse_message: str | None


@dataclass(frozen=True)
class PathBrowseState:
    path_raw: str
    browse_available: bool
    browse_message: str | None


@dataclass(frozen=True)
class BBoxEditorImagePayload:
    image_data_url: str
    original_width: int
    original_height: int
    display_width: int
    display_height: int


@dataclass(frozen=True)
class CustomFormatOption:
    format_id: str
    display_name: str
    path: Path
    description: str | None


@dataclass(frozen=True)
class RunSummaryMetrics:
    images_processed: int
    annotations_converted: int
    warning_count: int
    error_count: int


@dataclass(frozen=True)
class ClassExampleImage:
    file_name: str
    annotation_count: int
    overlay_labels: tuple[tuple[float, float, float, float, str], ...]


@dataclass(frozen=True)
class ClassExampleGroup:
    class_id: int
    class_name: str
    image_count: int
    examples: tuple[ClassExampleImage, ...]


def is_localhost_binding(address: str | None) -> bool:
    if address is None:
        return True
    return address in LOCALHOST_VALUES


def export_run_config(payload: dict[str, Any], output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    atomic_write_json(output_path, payload)
    return output_path


def export_json_artifact(payload: dict[str, Any], output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    atomic_write_json(output_path, payload)
    return output_path


def gui_state_path() -> Path:
    return Path.home() / ".label_master" / GUI_STATE_FILE_NAME


def load_gui_state(state_path: Path | None = None) -> dict[str, Any]:
    path = state_path or gui_state_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def persist_gui_state(
    payload: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> Path | None:
    path = state_path or gui_state_path()
    try:
        atomic_write_json(path, payload)
    except OSError:
        return None
    return path


def load_last_used_input_directory(state_path: Path | None = None) -> str | None:
    payload = load_gui_state(state_path)
    last_input_dir = payload.get("last_input_dir")
    if not isinstance(last_input_dir, str):
        return None

    normalized = last_input_dir.strip()
    return normalized or None


def persist_last_used_input_directory(
    input_directory: Path | str,
    *,
    state_path: Path | None = None,
) -> Path | None:
    normalized = str(input_directory).strip()
    if not normalized:
        return None

    payload = load_gui_state(state_path)
    payload["last_input_dir"] = normalized
    return persist_gui_state(payload, state_path=state_path)


def default_gui_input_directory() -> str:
    return load_last_used_input_directory() or DEFAULT_INPUT_DIR


def _coerce_out_of_frame_tolerance_px(
    raw_value: Any,
    *,
    out_of_frame_bbox_policy: str,
) -> float:
    tolerance_px = max(0.0, _coerce_float(raw_value, DEFAULT_OUT_OF_FRAME_TOLERANCE_PX))
    if out_of_frame_bbox_policy in {"correct", "warn"} and tolerance_px <= 0.0:
        return DEFAULT_OUT_OF_FRAME_TOLERANCE_PX
    return tolerance_px


def load_persisted_gui_preferences(state_path: Path | None = None) -> dict[str, Any]:
    payload = load_gui_state(state_path)

    output_dir = _coerce_text(payload.get("last_output_dir")) or DEFAULT_OUTPUT_DIR
    destination_format = DEFAULT_DESTINATION_FORMAT
    validation_mode = _coerce_text(payload.get("last_validation_mode")) or ValidationMode.STRICT.value
    if validation_mode not in VALIDATION_MODES:
        validation_mode = ValidationMode.STRICT.value
    permissive_invalid_annotation_action = (
        _coerce_text(payload.get("last_permissive_invalid_annotation_action"))
        or InvalidAnnotationAction.KEEP.value
    )
    if permissive_invalid_annotation_action not in PERMISSIVE_INVALID_ANNOTATION_ACTIONS:
        permissive_invalid_annotation_action = InvalidAnnotationAction.KEEP.value
    allow_shared_output_dir = payload.get("last_allow_shared_output_dir")
    if not isinstance(allow_shared_output_dir, bool):
        allow_shared_output_dir = True
    prefix_output_filenames = payload.get("last_prefix_output_filenames")
    if not isinstance(prefix_output_filenames, bool):
        prefix_output_filenames = False
    allow_overwrite = payload.get("last_allow_overwrite")
    if not isinstance(allow_overwrite, bool):
        allow_overwrite = False
    input_path_include_substring = normalize_input_path_filter_substring(
        _coerce_text(payload.get("last_input_path_include_substring"))
    )
    input_path_exclude_substring = normalize_input_path_filter_substring(
        _coerce_text(payload.get("last_input_path_exclude_substring"))
    )
    output_file_stem_prefix = _coerce_text(payload.get("last_output_file_stem_prefix"))
    output_file_stem_suffix = _coerce_text(payload.get("last_output_file_stem_suffix"))
    out_of_frame_bbox_policy_raw = _coerce_text(payload.get("last_out_of_frame_bbox_policy"))
    if out_of_frame_bbox_policy_raw not in {p.value for p in OutOfFrameBBoxPolicy}:
        out_of_frame_bbox_policy_raw = DEFAULT_OUT_OF_FRAME_BBOX_POLICY
    out_of_frame_tolerance_px = _coerce_out_of_frame_tolerance_px(
        payload.get("last_out_of_frame_tolerance_px"),
        out_of_frame_bbox_policy=out_of_frame_bbox_policy_raw,
    )
    min_image_longest_edge_px = max(
        0,
        int(_coerce_float(payload.get("last_min_image_longest_edge_px"), DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX)),
    )
    max_image_longest_edge_px = max(
        0,
        int(_coerce_float(payload.get("last_max_image_longest_edge_px"), DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX)),
    )
    preview_scan_limit = max(
        0,
        int(_coerce_float(payload.get("last_preview_scan_limit"), DEFAULT_PREVIEW_SCAN_LIMIT)),
    )
    preview_scan_limit = min(preview_scan_limit, MAX_PREVIEW_SCAN_LIMIT)
    oversize_image_action = _coerce_text(payload.get("last_oversize_image_action")) or "ignore"
    if oversize_image_action not in OVERSIZE_IMAGE_ACTIONS:
        oversize_image_action = "ignore"
    missing_label_detector_model_path = _coerce_text(payload.get("last_missing_label_detector_model_path"))
    missing_label_hints_output_dir = _coerce_text(payload.get("last_missing_label_hints_output_dir"))
    missing_label_confidence_threshold = min(
        1.0,
        max(0.0, _coerce_float(payload.get("last_missing_label_confidence_threshold"), 0.25)),
    )
    missing_label_iou_threshold = min(
        1.0,
        max(0.0, _coerce_float(payload.get("last_missing_label_iou_threshold"), 0.45)),
    )
    missing_label_max_detections_per_image = max(
        1,
        int(_coerce_float(payload.get("last_missing_label_max_detections_per_image"), 200)),
    )
    bbox_audit_output_dir = _coerce_text(payload.get("last_bbox_audit_output_dir"))
    bbox_audit_match_iou_threshold = min(
        1.0,
        max(0.0, _coerce_float(payload.get("last_bbox_audit_match_iou_threshold"), 0.30)),
    )
    bbox_audit_correction_iou_threshold = min(
        1.0,
        max(0.0, _coerce_float(payload.get("last_bbox_audit_correction_iou_threshold"), 0.85)),
    )
    bbox_audit_max_labeled_images = max(
        1,
        int(_coerce_float(payload.get("last_bbox_audit_max_labeled_images"), 100)),
    )

    inference_payload = payload.get("last_inference_payload")
    if not isinstance(inference_payload, dict):
        inference_payload = None

    custom_format_id = _coerce_text(payload.get("last_custom_format_id")) or None
    custom_format_path = _coerce_text(payload.get("last_custom_format_path")) or None
    mapping_seed_signature = _coerce_text(payload.get("last_mapping_seed_signature")) or None
    last_input_dir = _coerce_text(payload.get("last_input_dir"))
    classification_config_path = _coerce_text(payload.get("last_classification_config_path"))
    classification_advance_on_label = payload.get("last_classification_advance_on_label")
    if not isinstance(classification_advance_on_label, bool):
        classification_advance_on_label = True
    classification_show_bboxes = payload.get("last_classification_show_bboxes")
    if not isinstance(classification_show_bboxes, bool):
        classification_show_bboxes = True

    return {
        "gui_input_dir": last_input_dir or DEFAULT_INPUT_DIR,
        "gui_output_dir": output_dir,
        "gui_dst": destination_format,
        "gui_validation_mode": validation_mode,
        "gui_permissive_invalid_annotation_action": permissive_invalid_annotation_action,
        "gui_allow_shared_output_dir": allow_shared_output_dir,
        "gui_prefix_output_filenames": prefix_output_filenames,
        "gui_allow_overwrite": allow_overwrite,
        "gui_input_path_include_substring": input_path_include_substring,
        "gui_input_path_exclude_substring": input_path_exclude_substring,
        "gui_output_file_stem_prefix": output_file_stem_prefix,
        "gui_output_file_stem_suffix": output_file_stem_suffix,
        "gui_out_of_frame_bbox_policy": out_of_frame_bbox_policy_raw,
        "gui_out_of_frame_tolerance_px": out_of_frame_tolerance_px,
        "gui_min_image_longest_edge_px": min_image_longest_edge_px,
        "gui_max_image_longest_edge_px": max_image_longest_edge_px,
        "gui_preview_scan_limit": preview_scan_limit,
        "gui_oversize_image_action": oversize_image_action,
        "gui_missing_label_detector_model_path": missing_label_detector_model_path,
        "gui_missing_label_hints_output_dir": missing_label_hints_output_dir,
        "gui_missing_label_confidence_threshold": missing_label_confidence_threshold,
        "gui_missing_label_iou_threshold": missing_label_iou_threshold,
        "gui_missing_label_max_detections_per_image": missing_label_max_detections_per_image,
        "gui_bbox_audit_output_dir": bbox_audit_output_dir,
        "gui_bbox_audit_match_iou_threshold": bbox_audit_match_iou_threshold,
        "gui_bbox_audit_correction_iou_threshold": bbox_audit_correction_iou_threshold,
        "gui_bbox_audit_max_labeled_images": bbox_audit_max_labeled_images,
        "gui_inference_payload": inference_payload,
        "gui_custom_format_id": custom_format_id,
        "gui_custom_format_path": custom_format_path,
        "gui_mapping_rows": normalize_mapping_rows(payload.get("last_mapping_rows")),
        "gui_mapping_seed_signature": mapping_seed_signature,
        "gui_classification_config_path": classification_config_path,
        "gui_classification_advance_on_label": classification_advance_on_label,
        "gui_classification_show_bboxes": classification_show_bboxes,
        "gui_last_persisted_input_dir": last_input_dir,
        "gui_last_persisted_state_payload": json.dumps(payload, sort_keys=True) if payload else "",
    }


def persist_generated_class_map(
    class_map: dict[int, int | None],
    *,
    run_id: str,
    reports_dir: Path = Path("reports"),
) -> Path:
    ensure_directory(reports_dir)
    payload = {"class_map": {str(key): value for key, value in sorted(class_map.items())}}
    path = reports_dir / f"{run_id}.gui.class_map.json"
    atomic_write_json(path, payload)
    return path


def finalize_generated_class_map(
    pending_map_path: Path,
    *,
    run_id: str,
    reports_dir: Path = Path("reports"),
) -> Path:
    ensure_directory(reports_dir)
    final_path = reports_dir / f"{run_id}.gui.class_map.json"
    pending_map_path.replace(final_path)
    return final_path


def _preview_image_cache_token(dataset_root: Path, image_rel_path: str) -> int:
    try:
        return (dataset_root / image_rel_path).stat().st_mtime_ns
    except OSError:
        return -1


OverlayBBox = (
    tuple[float, float, float, float, str]
    | tuple[float, float, float, float, str, str]
)


def _normalize_overlay_bbox(
    bbox: OverlayBBox,
) -> tuple[float, float, float, float, str, str]:
    if len(bbox) == 5:
        x, y, w, h, label = bbox
        color = "red"
    elif len(bbox) == 6:
        x, y, w, h, label, color = bbox
    else:
        raise ValueError("Overlay bbox entries must have 5 or 6 values")

    return float(x), float(y), float(w), float(h), str(label), str(color)


def _overlay_legend_item(label: str, color: str) -> str:
    safe_label = escape(label)
    safe_color = escape(color)
    return (
        '<span style="display:inline-flex;align-items:center;margin-right:1rem;">'
        f'<span style="display:inline-block;width:0.85rem;height:0.85rem;background:{safe_color};'
        'border:1px solid rgba(0, 0, 0, 0.35);margin-right:0.35rem;"></span>'
        f"{safe_label}</span>"
    )


@lru_cache(maxsize=16)
def _load_overlay_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=256)
def _render_preview_overlay_cached(
    *,
    dataset_root: str,
    image_rel_path: str,
    bboxes: tuple[tuple[float, float, float, float, str, str], ...],
    image_cache_token: int,
) -> tuple[bytes | None, tuple[str, ...]]:
    del image_cache_token

    image_path = Path(dataset_root) / image_rel_path
    try:
        if image_path.exists():
            with Image.open(image_path) as opened:
                rgb_image = opened.convert("RGB")
        else:
            rgb_image = load_video_bbox_preview_image(Path(dataset_root), image_rel_path)
    except ValueError:
        return None, (f"Preview image not found: {image_rel_path}",)
    except Exception as exc:
        return None, (f"Preview image could not be loaded: {exc}",)

    width, height = rgb_image.size
    scale = min(1.0, PREVIEW_MAX_IMAGE_DIMENSION / max(width, height))
    if scale < 1.0:
        canvas = rgb_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        canvas = rgb_image.copy()

    draw = ImageDraw.Draw(canvas)
    line_width = max(1, int(round(3 * scale)))
    font_size = max(16, int(round(22 * scale)))
    font = _load_overlay_font(font_size)
    label_padding_x = max(4, int(round(6 * scale)))
    label_padding_y = max(3, int(round(4 * scale)))
    for x, y, w, h, label, color in bboxes:
        scaled_x = x * scale
        scaled_y = y * scale
        scaled_w = w * scale
        scaled_h = h * scale
        x2 = scaled_x + scaled_w
        y2 = scaled_y + scaled_h
        draw.rectangle((scaled_x, scaled_y, x2, y2), outline=color, width=line_width)
        text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label, font=font)
        label_width = max(40, int((text_right - text_left) + (label_padding_x * 2)))
        label_height = int((text_bottom - text_top) + (label_padding_y * 2))
        label_x = max(0, int(scaled_x))
        label_y = max(0, int(scaled_y) - label_height)
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height), fill=color)
        draw.text(
            (label_x + label_padding_x, label_y + label_padding_y),
            label,
            fill="white",
            font=font,
        )

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue(), ()


def render_preview_overlay(
    *,
    dataset_root: Path,
    image_rel_path: str,
    bboxes: list[OverlayBBox],
) -> tuple[Image.Image | None, list[str]]:
    overlay_bytes, warnings = _render_preview_overlay_cached(
        dataset_root=str(dataset_root.expanduser().resolve()),
        image_rel_path=image_rel_path,
        bboxes=tuple(_normalize_overlay_bbox(bbox) for bbox in bboxes),
        image_cache_token=_preview_image_cache_token(dataset_root, image_rel_path),
    )
    if overlay_bytes is None:
        return None, list(warnings)

    with Image.open(BytesIO(overlay_bytes)) as opened:
        return opened.copy(), list(warnings)


@lru_cache(maxsize=256)
def _load_bbox_editor_image_payload_cached(
    *,
    dataset_root: str,
    image_rel_path: str,
    image_cache_token: int,
) -> tuple[BBoxEditorImagePayload | None, tuple[str, ...]]:
    del image_cache_token

    image_path = Path(dataset_root) / image_rel_path
    try:
        if image_path.exists():
            with Image.open(image_path) as opened:
                rgb_image = opened.convert("RGB")
        else:
            rgb_image = load_video_bbox_preview_image(Path(dataset_root), image_rel_path)
    except ValueError:
        return None, (f"Preview image not found: {image_rel_path}",)
    except Exception as exc:
        return None, (f"Preview image could not be loaded: {exc}",)

    original_width, original_height = rgb_image.size
    scale = min(1.0, PREVIEW_MAX_IMAGE_DIMENSION / max(original_width, original_height))
    if scale < 1.0:
        display_image = rgb_image.resize(
            (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        display_image = rgb_image.copy()

    buffer = BytesIO()
    display_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        BBoxEditorImagePayload(
            image_data_url=f"data:image/png;base64,{encoded}",
            original_width=original_width,
            original_height=original_height,
            display_width=display_image.size[0],
            display_height=display_image.size[1],
        ),
        (),
    )


def load_bbox_editor_image_payload(
    *,
    dataset_root: Path,
    image_rel_path: str,
) -> tuple[BBoxEditorImagePayload | None, list[str]]:
    payload, warnings = _load_bbox_editor_image_payload_cached(
        dataset_root=str(dataset_root.expanduser().resolve()),
        image_rel_path=image_rel_path,
        image_cache_token=_preview_image_cache_token(dataset_root, image_rel_path),
    )
    return payload, list(warnings)


def _default_missing_label_hints_output_dir(output_dir_raw: str) -> Path:
    output_dir_text = _coerce_text(output_dir_raw) or DEFAULT_OUTPUT_DIR
    return Path(output_dir_text).expanduser() / "missing_label_hints"


def _resolve_missing_label_hints_output_dir(
    output_dir_raw: str,
    hints_output_dir_raw: str,
) -> Path:
    hints_output_text = _coerce_text(hints_output_dir_raw)
    if hints_output_text:
        return Path(hints_output_text).expanduser()
    return _default_missing_label_hints_output_dir(output_dir_raw)


def _missing_label_review_key(
    kind: str,
    label_rel_path: str,
    detection_index: int | None = None,
) -> str:
    normalized = (
        label_rel_path.replace("\\", "/")
        .replace("/", "__")
        .replace(".", "_")
        .replace(":", "_")
    )
    suffix = f"_{detection_index}" if detection_index is not None else ""
    return f"gui_missing_label_{kind}_{normalized}{suffix}"


def _selected_missing_label_hint_detections(
    hint: MissingLabelHintItemViewModel,
    session_state: MutableMapping[str, Any],
) -> list[Any]:
    return [
        detection
        for index, detection in enumerate(hint.detections)
        if bool(
            session_state.get(
                _missing_label_review_key(
                    "detection",
                    hint.suggested_label_rel_path,
                    detection_index=index,
                ),
                True,
            )
        )
    ]


def _materialize_approved_missing_label_hints(
    hints: list[MissingLabelHintItemViewModel],
    session_state: MutableMapping[str, Any],
) -> list[MissingLabelHintItemViewModel]:
    approved_hints: list[MissingLabelHintItemViewModel] = []
    for hint in hints:
        if not bool(
            session_state.get(
                _missing_label_review_key("approve", hint.suggested_label_rel_path),
                False,
            )
        ):
            continue

        selected_detections = _selected_missing_label_hint_detections(hint, session_state)
        if not selected_detections:
            continue

        approved_hints.append(
            MissingLabelHintItemViewModel(
                image_rel_path=hint.image_rel_path,
                suggested_label_rel_path=hint.suggested_label_rel_path,
                detections=selected_detections,
            )
        )

    return approved_hints


def _default_bbox_audit_output_dir(output_dir_raw: str) -> Path:
    output_dir_text = _coerce_text(output_dir_raw) or DEFAULT_OUTPUT_DIR
    return Path(output_dir_text).expanduser() / "bbox_audit"


def _resolve_bbox_audit_output_dir(
    output_dir_raw: str,
    audit_output_dir_raw: str,
) -> Path:
    audit_output_text = _coerce_text(audit_output_dir_raw)
    if audit_output_text:
        return Path(audit_output_text).expanduser()
    return _default_bbox_audit_output_dir(output_dir_raw)


def _bbox_audit_review_key(
    kind: str,
    label_rel_path: str,
    proposal_index: int | None = None,
) -> str:
    normalized = (
        label_rel_path.replace("\\", "/")
        .replace("/", "__")
        .replace(".", "_")
        .replace(":", "_")
    )
    suffix = f"_{proposal_index}" if proposal_index is not None else ""
    return f"gui_bbox_audit_{kind}_{normalized}{suffix}"


def _selected_bbox_audit_proposals(
    item: BBoxAuditItemViewModel,
    session_state: MutableMapping[str, Any],
) -> list[BBoxAuditProposalViewModel]:
    return [
        proposal
        for index, proposal in enumerate(item.proposals)
        if bool(
            session_state.get(
                _bbox_audit_review_key(
                    "proposal",
                    item.label_rel_path,
                    proposal_index=index,
                ),
                True,
            )
        )
    ]


def _materialize_approved_bbox_audits(
    items: list[BBoxAuditItemViewModel],
    session_state: MutableMapping[str, Any],
) -> list[BBoxAuditItemViewModel]:
    approved_items: list[BBoxAuditItemViewModel] = []
    for item in items:
        if not bool(
            session_state.get(
                _bbox_audit_review_key("approve", item.label_rel_path),
                False,
            )
        ):
            continue

        selected_proposals = _selected_bbox_audit_proposals(item, session_state)
        if not selected_proposals:
            continue

        approved_items.append(
            BBoxAuditItemViewModel(
                image_rel_path=item.image_rel_path,
                label_rel_path=item.label_rel_path,
                existing_label_count=item.existing_label_count,
                proposals=selected_proposals,
            )
        )

    return approved_items


def _detector_review_key(
    kind: str,
    label_rel_path: str,
    proposal_index: int | None = None,
) -> str:
    normalized = (
        label_rel_path.replace("\\", "/")
        .replace("/", "__")
        .replace(".", "_")
        .replace(":", "_")
    )
    suffix = f"_{proposal_index}" if proposal_index is not None else ""
    return f"gui_detector_review_{kind}_{normalized}{suffix}"


def _selected_detector_review_proposals(
    item: DetectorReviewItemViewModel,
    session_state: MutableMapping[str, Any],
) -> list[BBoxAuditProposalViewModel]:
    return [
        proposal
        for index, proposal in enumerate(item.proposals)
        if bool(
            session_state.get(
                _detector_review_key(
                    "proposal",
                    item.label_rel_path,
                    proposal_index=index,
                ),
                True,
            )
        )
    ]


def _materialize_approved_detector_review_items(
    items: list[DetectorReviewItemViewModel],
    session_state: MutableMapping[str, Any],
) -> list[DetectorReviewItemViewModel]:
    approved_items: list[DetectorReviewItemViewModel] = []
    for item in items:
        if not bool(
            session_state.get(
                _detector_review_key("approve", item.label_rel_path),
                False,
            )
        ):
            continue

        selected_proposals = _selected_detector_review_proposals(item, session_state)
        if not selected_proposals:
            continue

        approved_items.append(
            DetectorReviewItemViewModel(
                source_kind=item.source_kind,
                image_rel_path=item.image_rel_path,
                label_rel_path=item.label_rel_path,
                existing_label_count=item.existing_label_count,
                proposals=selected_proposals,
            )
        )

    return approved_items


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _format_bbox_text(
    bbox_xywh_normalized: tuple[float, float, float, float] | None,
) -> str:
    if bbox_xywh_normalized is None:
        return ""
    return "(" + ", ".join(f"{value:.3f}" for value in bbox_xywh_normalized) + ")"


def _editor_box_to_dict(box: DetectorReviewEditorBoxViewModel) -> dict[str, Any]:
    return {
        "box_id": box.box_id,
        "class_id": box.class_id,
        "class_name": box.class_name,
        "cx": box.bbox_xywh_normalized[0],
        "cy": box.bbox_xywh_normalized[1],
        "w": box.bbox_xywh_normalized[2],
        "h": box.bbox_xywh_normalized[3],
        "source": box.source,
    }


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    text = _coerce_text(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _coerce_editor_boxes(
    raw_boxes: Any,
    *,
    class_name_map: dict[int, str],
) -> list[DetectorReviewEditorBoxViewModel]:
    if not isinstance(raw_boxes, list):
        return []

    normalized_boxes: list[DetectorReviewEditorBoxViewModel] = []
    for index, raw_box in enumerate(raw_boxes):
        if not isinstance(raw_box, dict):
            continue
        try:
            class_id = int(raw_box.get("class_id"))
        except (TypeError, ValueError):
            continue

        width = max(0.0, min(1.0, _coerce_float(raw_box.get("w"), 0.0)))
        height = max(0.0, min(1.0, _coerce_float(raw_box.get("h"), 0.0)))
        if width <= 0.0 or height <= 0.0:
            continue

        center_x = max(0.0, min(1.0, _coerce_float(raw_box.get("cx"), 0.5)))
        center_y = max(0.0, min(1.0, _coerce_float(raw_box.get("cy"), 0.5)))
        box_id = _coerce_text(raw_box.get("box_id")) or f"manual:{index + 1}"
        class_name = class_name_map.get(class_id) or _coerce_text(raw_box.get("class_name")) or f"class_{class_id}"
        source = _coerce_text(raw_box.get("source")) or "manual"
        normalized_boxes.append(
            DetectorReviewEditorBoxViewModel(
                box_id=box_id,
                class_id=class_id,
                class_name=class_name,
                bbox_xywh_normalized=(center_x, center_y, width, height),
                source=source,
            )
        )

    return normalized_boxes


def normalize_mapping_rows(value: Any) -> list[dict[str, str]]:
    rows_raw: list[dict[str, Any]] = []
    if hasattr(value, "to_dict"):
        to_dict = value.to_dict
        if callable(to_dict):
            try:
                records = to_dict(orient="records")
                if isinstance(records, list):
                    rows_raw = [record for record in records if isinstance(record, dict)]
            except TypeError:
                rows_raw = []
    elif isinstance(value, list):
        rows_raw = [record for record in value if isinstance(record, dict)]

    normalized: list[dict[str, str]] = []
    for row in rows_raw:
        source_class_id = _coerce_text(row.get("source_class_id"))
        destination_class_id = _coerce_text(row.get("destination_class_id"))
        action = _coerce_text(row.get("action")).lower() or "map"
        normalized_action = action if action in MAPPING_ACTIONS else "map"

        if not source_class_id and not destination_class_id:
            continue

        normalized.append(
            {
                "source_class_id": source_class_id,
                "action": normalized_action,
                "destination_class_id": destination_class_id,
            }
        )

    return normalized


_normalize_mapping_rows = normalize_mapping_rows


def _apply_mapping_editor_state(
    rows: list[dict[str, str]],
    editor_state: Any,
) -> list[dict[str, str]]:
    materialized_rows = [dict(row) for row in rows]
    if not isinstance(editor_state, dict):
        return materialized_rows

    deleted_rows = editor_state.get("deleted_rows", [])
    if isinstance(deleted_rows, list):
        delete_indexes = sorted(
            {
                row_index
                for row_index in deleted_rows
                if isinstance(row_index, int) and 0 <= row_index < len(materialized_rows)
            },
            reverse=True,
        )
        for row_index in delete_indexes:
            del materialized_rows[row_index]

    edited_rows = editor_state.get("edited_rows", {})
    if isinstance(edited_rows, dict):
        for row_index_raw, updates in edited_rows.items():
            if not isinstance(updates, dict):
                continue
            try:
                row_index = int(row_index_raw)
            except (TypeError, ValueError):
                continue
            if row_index < 0 or row_index >= len(materialized_rows):
                continue
            for key in ("action", "destination_class_id"):
                if key in updates:
                    materialized_rows[row_index][key] = _coerce_text(updates.get(key))

    added_rows = editor_state.get("added_rows", [])
    if isinstance(added_rows, list):
        for added_row in added_rows:
            if not isinstance(added_row, dict):
                continue
            materialized_rows.append(
                {
                    "source_class_id": _coerce_text(added_row.get("source_class_id")),
                    "action": _coerce_text(added_row.get("action")).lower() or "map",
                    "destination_class_id": _coerce_text(added_row.get("destination_class_id")),
                }
            )

    return materialized_rows


def materialize_mapping_rows(value: Any) -> list[dict[str, str]]:
    rows = normalize_mapping_rows(value)
    return [row for row in rows if row["source_class_id"] or row["destination_class_id"]]


def mapping_rows_to_viewmodels(rows: list[dict[str, str]]) -> list[MappingRowViewModel]:
    return [
        MappingRowViewModel(
            source_class_id=row["source_class_id"],
            action=row["action"],
            destination_class_id=row["destination_class_id"],
        )
        for row in rows
    ]


def _parse_int_or_none(value: str) -> int | None:
    text = _coerce_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def extract_class_labels_from_preview(preview_vm: Any) -> dict[int, str]:
    labels: dict[int, str] = {}
    images = getattr(preview_vm, "images", [])
    if not isinstance(images, list):
        return labels

    for image in images:
        bboxes = getattr(image, "bboxes", [])
        if not isinstance(bboxes, list):
            continue
        for bbox in bboxes:
            class_id = getattr(bbox, "class_id", None)
            class_name = getattr(bbox, "class_name", None)
            if isinstance(class_id, int) and isinstance(class_name, str) and class_name:
                labels.setdefault(class_id, class_name)

    return {key: labels[key] for key in sorted(labels)}


def build_class_example_groups(
    preview_vm: Any,
    *,
    examples_per_class: int = PREVIEW_CLASS_EXAMPLES_PER_CLASS,
) -> list[ClassExampleGroup]:
    if examples_per_class <= 0:
        return []

    images = getattr(preview_vm, "images", [])
    if not isinstance(images, list):
        return []

    class_names: dict[int, str] = {}
    image_counts: dict[int, int] = {}
    examples_by_class: dict[int, list[ClassExampleImage]] = {}

    for image in images:
        file_name = getattr(image, "file_name", None)
        bboxes = getattr(image, "bboxes", None)
        if not isinstance(file_name, str) or not file_name or not isinstance(bboxes, list):
            continue

        labels_by_class: dict[int, list[tuple[float, float, float, float, str]]] = {}
        for bbox in bboxes:
            class_id = getattr(bbox, "class_id", None)
            class_name = getattr(bbox, "class_name", None)
            bbox_xywh_abs = getattr(bbox, "bbox_xywh_abs", None)
            if not isinstance(class_id, int):
                continue
            if not isinstance(class_name, str) or not class_name:
                class_name = f"class_{class_id}"
            if not isinstance(bbox_xywh_abs, tuple | list) or len(bbox_xywh_abs) != 4:
                continue

            try:
                x = float(bbox_xywh_abs[0])
                y = float(bbox_xywh_abs[1])
                w = float(bbox_xywh_abs[2])
                h = float(bbox_xywh_abs[3])
            except (TypeError, ValueError):
                continue

            class_names.setdefault(class_id, class_name)
            labels_by_class.setdefault(class_id, []).append((x, y, w, h, f"{class_id}:{class_name}"))

        for class_id, overlay_labels in sorted(labels_by_class.items()):
            image_counts[class_id] = image_counts.get(class_id, 0) + 1
            class_examples = examples_by_class.setdefault(class_id, [])
            if len(class_examples) >= examples_per_class:
                continue
            class_examples.append(
                ClassExampleImage(
                    file_name=file_name,
                    annotation_count=len(overlay_labels),
                    overlay_labels=tuple(overlay_labels),
                )
            )

    return [
        ClassExampleGroup(
            class_id=class_id,
            class_name=class_names[class_id],
            image_count=image_counts.get(class_id, 0),
            examples=tuple(examples_by_class.get(class_id, [])),
        )
        for class_id in sorted(class_names)
        if examples_by_class.get(class_id)
    ]


def build_identity_mapping_rows(class_labels: dict[int, str]) -> list[dict[str, str]]:
    return [
        {
            "source_class_id": str(class_id),
            "action": "map",
            "destination_class_id": str(class_id),
        }
        for class_id in sorted(class_labels)
    ]


def attach_mapping_labels(
    rows: list[dict[str, str]],
    class_labels: dict[int, str],
) -> list[dict[str, str]]:
    with_labels: list[dict[str, str]] = []
    for row in rows:
        source_id = _parse_int_or_none(row["source_class_id"])
        with_labels.append(
            {
                **row,
                "source_label": class_labels.get(source_id, "") if source_id is not None else "",
            }
        )
    return with_labels


def _format_mapping_action_label(action: str) -> str:
    normalized = _coerce_text(action).lower()
    if normalized == "map":
        return "keep"
    if normalized == "drop":
        return "drop"
    if normalized == "drop_frame":
        return "drop frame"
    return normalized or "keep"


def _format_oversize_image_action_label(action: str) -> str:
    normalized = _coerce_text(action).lower()
    if normalized == "ignore":
        return "drop"
    return normalized or "drop"


def _mapping_widget_key(source_class_id: str, field: str) -> str:
    return f"gui_mapping_{field}_{source_class_id}"


def _sync_mapping_row_widget_state(
    rows: list[dict[str, str]],
    *,
    overwrite: bool = False,
) -> None:
    for row in rows:
        source_class_id = _coerce_text(row.get("source_class_id"))
        if not source_class_id:
            continue

        action_key = _mapping_widget_key(source_class_id, "action")
        destination_key = _mapping_widget_key(source_class_id, "destination_class_id")
        action_value = _coerce_text(row.get("action")).lower() or "map"
        destination_value = _coerce_text(row.get("destination_class_id"))

        if overwrite or action_key not in st.session_state:
            st.session_state[action_key] = action_value
        if overwrite or destination_key not in st.session_state:
            st.session_state[destination_key] = destination_value


def _mapping_display_cell(
    text: str,
    *,
    muted: bool = False,
) -> str:
    content = escape(text) if text else "&nbsp;"
    class_name = "lm-mapping-cell muted" if muted else "lm-mapping-cell"
    return f"<div class='{class_name}'>{content}</div>"


def _inject_mapping_table_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            border-right: 1px solid #d0d7de !important;
            border-bottom: 1px solid #d0d7de !important;
        }
        div[data-testid="stDataFrame"] [role="row"] [role="gridcell"]:first-child,
        div[data-testid="stDataFrame"] [role="row"] [role="columnheader"]:first-child {
            border-left: 1px solid #d0d7de !important;
        }
        div[data-testid="stDataFrame"] [role="row"]:first-child [role="columnheader"] {
            border-top: 1px solid #d0d7de !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_compact_layout_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.9rem;
            padding-bottom: 0.9rem;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 0.5rem;
        }
        div[data-testid="stImage"] img {
            width: auto !important;
            max-width: 100%;
            max-height: 68vh;
            object-fit: contain;
            margin: 0 auto;
            display: block;
        }
        .lm-mapping-header {
            color: #475467;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0 0.2rem;
        }
        .lm-mapping-cell {
            min-height: 2.4rem;
            border: 1px solid #d0d7de;
            border-radius: 0.45rem;
            padding: 0.45rem 0.55rem;
            background: #ffffff;
            display: flex;
            align-items: center;
        }
        .lm-mapping-cell.muted {
            color: #98a2b3;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_preview_table_alignment_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"] *,
        div[data-testid="stDataFrame"] [role="gridcell"] * {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _coerce_preview_keyboard_navigation_action(
    payload: Any,
    *,
    last_nonce: int,
) -> tuple[str | None, int]:
    if not isinstance(payload, dict):
        return None, last_nonce

    raw_nonce = _coerce_text(payload.get("nonce"))
    try:
        nonce = int(raw_nonce)
    except (TypeError, ValueError):
        return None, last_nonce

    if nonce <= last_nonce:
        return None, last_nonce

    action = _coerce_text(payload.get("action")).lower()
    if action not in {"previous", "next"}:
        return None, nonce

    return action, nonce


def _resolve_preview_index(
    current_index: int,
    *,
    max_index: int,
    keyboard_action: str | None = None,
    previous_clicked: bool = False,
    next_clicked: bool = False,
) -> int:
    resolved_index = min(max(current_index, 0), max_index)

    if keyboard_action == "previous" and resolved_index > 0:
        resolved_index -= 1
    elif keyboard_action == "next" and resolved_index < max_index:
        resolved_index += 1
    elif previous_clicked and resolved_index > 0:
        resolved_index -= 1
    elif next_clicked and resolved_index < max_index:
        resolved_index += 1

    return min(max(resolved_index, 0), max_index)


def _preview_keyboard_navigation_action(
    *,
    enabled: bool,
    previous_disabled: bool,
    next_disabled: bool,
) -> str | None:
    payload = _PREVIEW_KEYBOARD_NAV_COMPONENT(
        enabled=enabled,
        previous_disabled=previous_disabled,
        next_disabled=next_disabled,
        default=None,
        key="gui_preview_keyboard_nav",
        tab_index=-1,
    )
    last_nonce = int(st.session_state.get("gui_preview_keyboard_event_nonce", 0))
    action, nonce = _coerce_preview_keyboard_navigation_action(payload, last_nonce=last_nonce)
    st.session_state["gui_preview_keyboard_event_nonce"] = nonce
    return action


def _coerce_classification_keyboard_action(
    payload: Any,
    *,
    last_nonce: int,
) -> tuple[str | None, str | None, int]:
    if not isinstance(payload, dict):
        return None, None, last_nonce

    try:
        nonce = int(_coerce_text(payload.get("nonce")))
    except (TypeError, ValueError):
        return None, None, last_nonce

    if nonce <= last_nonce:
        return None, None, last_nonce

    action = _coerce_text(payload.get("action")).lower()
    if action not in CLASSIFICATION_KEYBOARD_ACTIONS:
        return None, None, nonce

    key = _coerce_text(payload.get("key")).lower() or None
    if action == "class" and key is None:
        return None, None, nonce
    return action, key, nonce


def _classification_keyboard_action(
    *,
    enabled: bool,
    previous_disabled: bool,
    next_disabled: bool,
    class_keys: list[str],
) -> tuple[str | None, str | None]:
    payload = _PREVIEW_KEYBOARD_NAV_COMPONENT(
        enabled=enabled,
        previous_disabled=previous_disabled,
        next_disabled=next_disabled,
        class_keys=list(class_keys),
        clear_enabled=True,
        default=None,
        key="gui_classification_keyboard_nav",
        tab_index=-1,
    )
    last_nonce = int(st.session_state.get("gui_classification_keyboard_event_nonce", 0))
    action, key, nonce = _coerce_classification_keyboard_action(payload, last_nonce=last_nonce)
    st.session_state["gui_classification_keyboard_event_nonce"] = nonce
    return action, key


def resolve_classification_config_candidate(
    config_path_raw: str,
    *,
    dataset_root: Path | None,
) -> Path | None:
    """Explicit config path wins; otherwise look for a default file in the dataset root."""

    normalized = config_path_raw.strip()
    if normalized:
        return Path(normalized).expanduser()
    if dataset_root is None:
        return None
    return find_default_classification_config_path(dataset_root)


def classification_index_after_label(
    current_index: int,
    *,
    max_index: int,
    advance_on_label: bool,
    multi_label: bool,
) -> int:
    if advance_on_label and not multi_label and current_index < max_index:
        return current_index + 1
    return current_index


def _classification_dataset_signature(
    dataset_root: Path,
    *,
    include_substring: str | None,
    exclude_substring: str | None,
) -> str:
    return f"{dataset_root.expanduser().resolve()}::{include_substring or ''}::{exclude_substring or ''}"


def _format_payload_yaml(payload: dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=False)
    return str(dumped).strip()


def format_details_yaml(
    source_format: str | None,
    *,
    dataset_root: Path | None,
    inference_payload: dict[str, Any] | None = None,
    custom_format_id: str | None = None,
    custom_format_path: Path | None = None,
) -> str | None:
    normalized_format = _coerce_text(source_format)
    spec = None

    if normalized_format == "custom" and dataset_root is not None:
        if custom_format_path is not None:
            spec = load_custom_format_spec_from_path(custom_format_path)
        if spec is None and custom_format_id:
            spec = resolve_custom_format_spec(custom_format_id, dataset_root)
        if spec is None:
            score, spec_id = detect_custom_format(dataset_root, sample_limit=100)
            if score > 0 and spec_id:
                spec = resolve_custom_format_spec(spec_id, dataset_root)
    elif normalized_format:
        spec = resolve_builtin_format_spec(normalized_format)

    if spec is not None:
        payload = spec.model_dump(mode="python", exclude_none=True)
        dumped = yaml.safe_dump(payload, sort_keys=False)
        return str(dumped).strip()

    if isinstance(inference_payload, dict):
        return _format_payload_yaml(inference_payload)

    return None


def _resolved_input_dir_token(input_path: Path) -> str:
    return str(input_path.expanduser().resolve())


def _custom_format_options(dataset_root: Path | None) -> list[CustomFormatOption]:
    if dataset_root is None:
        return []
    return [
        CustomFormatOption(
            format_id=entry.spec.format_id,
            display_name=entry.spec.display_name,
            path=entry.path,
            description=entry.spec.description,
        )
        for entry in custom_format_spec_entries(dataset_root)
    ]


def _default_custom_format_id(
    dataset_root: Path,
    *,
    options: list[CustomFormatOption],
) -> str | None:
    if not options:
        return None
    valid_ids = {option.format_id for option in options}
    score, detected_id = detect_custom_format(dataset_root, sample_limit=100)
    if score > 0 and detected_id in valid_ids:
        return detected_id
    return options[0].format_id


def _selected_custom_format_option(
    options: list[CustomFormatOption],
    format_id: str | None,
) -> CustomFormatOption | None:
    if not format_id:
        return None
    for option in options:
        if option.format_id == format_id:
            return option
    return None


def _format_custom_format_option(option: CustomFormatOption) -> str:
    return f"{option.path.name} ({option.display_name})"


def _explicit_custom_format_option(raw_path: str | None) -> tuple[CustomFormatOption | None, str | None]:
    path_value = (raw_path or "").strip()
    if not path_value:
        return None, None

    candidate = Path(path_value).expanduser()
    if not candidate.exists():
        return None, f"Custom format YAML does not exist: {candidate}"
    if not candidate.is_file():
        return None, f"Custom format YAML path is not a file: {candidate}"
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        return None, f"Custom format YAML must end with .yaml or .yml: {candidate.name}"

    try:
        spec = load_custom_format_spec_from_path(candidate)
    except Exception as exc:
        return None, str(exc)

    resolved_path = candidate.resolve()
    return (
        CustomFormatOption(
            format_id=spec.format_id,
            display_name=spec.display_name,
            path=resolved_path,
            description=spec.description,
        ),
        None,
    )


def _build_inference_payload(infer_vm: Any, *, input_path: Path) -> dict[str, Any]:
    return {
        "predicted_format": infer_vm.predicted_format,
        "confidence": infer_vm.confidence,
        "candidates": infer_vm.candidates,
        "warnings": infer_vm.warnings,
        "input_dir": _resolved_input_dir_token(input_path),
    }


def _inference_payload_matches_input_path(payload: Any, input_path: Path) -> bool:
    if not isinstance(payload, dict):
        return False
    return _coerce_text(payload.get("input_dir")) == _resolved_input_dir_token(input_path)


def _sync_inference_state_for_input_path(input_path: Path) -> None:
    payload = st.session_state.get("gui_inference_payload")
    if payload is not None and not _inference_payload_matches_input_path(payload, input_path):
        st.session_state["gui_inference_payload"] = None

    error = st.session_state.get("gui_inference_error")
    error_input_dir = _coerce_text(st.session_state.get("gui_inference_error_input_dir"))
    if error and error_input_dir != _resolved_input_dir_token(input_path):
        st.session_state["gui_inference_error"] = None
        st.session_state["gui_inference_error_input_dir"] = None


def _maybe_auto_infer_for_preview(input_path: Path) -> None:
    _sync_inference_state_for_input_path(input_path)

    if _coerce_text(st.session_state.get("gui_src")) != "auto":
        return
    if st.session_state.get("gui_inference_payload") is not None:
        return
    if st.session_state.get("gui_inference_error"):
        return

    try:
        infer_vm = infer_view(input_path)
    except Exception as exc:
        st.session_state["gui_inference_error"] = str(exc)
        st.session_state["gui_inference_error_input_dir"] = _resolved_input_dir_token(input_path)
        return

    st.session_state["gui_inference_payload"] = _build_inference_payload(infer_vm, input_path=input_path)
    st.session_state["gui_inference_error"] = None
    st.session_state["gui_inference_error_input_dir"] = None


def _mark_preview_skip_for_output_only_change() -> None:
    st.session_state["gui_skip_preview_once"] = True


def _consume_preview_skip_once(session_state: Any) -> bool:
    return bool(session_state.pop("gui_skip_preview_once", False))


def _class_labels_from_state() -> dict[int, str]:
    raw = st.session_state.get("gui_class_labels")
    if not isinstance(raw, dict):
        return {}
    labels: dict[int, str] = {}
    for key, value in raw.items():
        try:
            class_id = int(str(key))
        except ValueError:
            continue
        if isinstance(value, str) and value:
            labels[class_id] = value
    return {key: labels[key] for key in sorted(labels)}


def _store_class_labels(class_labels: dict[int, str]) -> None:
    st.session_state["gui_class_labels"] = {str(key): value for key, value in class_labels.items()}


def _seed_identity_rows_for_dataset(dataset_signature: str, class_labels: dict[int, str]) -> None:
    if not class_labels:
        return
    if st.session_state.get("gui_mapping_seed_signature") == dataset_signature:
        return

    st.session_state["gui_mapping_rows"] = build_identity_mapping_rows(class_labels)
    st.session_state["gui_mapping_seed_signature"] = dataset_signature
    st.session_state.pop("gui_mapping_editor", None)
    _sync_mapping_row_widget_state(st.session_state["gui_mapping_rows"], overwrite=True)


def describe_class_label_source(
    *,
    input_path: Path,
    source_format: str | None,
    class_labels: dict[int, str],
) -> str:
    if not class_labels:
        return "Class labels are not available yet. Load preview data first."

    if source_format == "yolo":
        label_source_path = None
        for candidate in [input_path / "classes.txt", input_path / "obj.names", *sorted(input_path.glob("**/obj.names"))]:
            if candidate.exists() and candidate.is_file():
                label_source_path = candidate
                break
        if label_source_path is not None:
            try:
                lines = [
                    line.strip()
                    for line in label_source_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                lines = []

            provided_ids = set(range(len(lines)))
            observed_ids = set(class_labels)
            fallback_ids = sorted(observed_ids - provided_ids)
            source_name = label_source_path.name
            if fallback_ids:
                fallback_text = ", ".join(str(class_id) for class_id in fallback_ids)
                return (
                    f"YOLO labels source: {source_name}. "
                    f"Missing class IDs ({fallback_text}) use fallback names class_<id>."
                )
            return f"YOLO labels source: {source_name}."

        return "YOLO labels source: classes.txt/obj.names not found; using fallback names class_<id>."

    if source_format == "coco":
        return "COCO labels source: categories from annotations.json."

    if source_format == "cityscapes":
        return "Cityscapes labels source: polygon instance labels normalized from Cityscapes JSON annotations."

    if source_format == "kitware":
        return "Kitware labels source: per-directory CSV bbox columns."

    if source_format == "matlab_ground_truth":
        return "MATLAB labels source: groundTruth LabelData tables inside .mat files."

    if source_format == "voc":
        return "VOC labels source: Pascal VOC XML object names."

    if source_format == "custom":
        return "Custom labels source: user-defined YAML format spec."

    if source_format == "video_bbox":
        return "Video bbox labels source: per-sequence tracking ground-truth text files."

    return "Label source follows detected preview format."


def _is_readable_directory(path: Path) -> bool:
    return os.access(path, os.R_OK | os.X_OK)


def validate_input_directory(input_dir_raw: str) -> DirectoryValidationResult:
    input_value = input_dir_raw.strip()
    if not input_value:
        return DirectoryValidationResult(resolved_path=None, errors=["Input directory is required"])

    input_path = Path(input_value).expanduser()
    if not input_path.exists():
        return DirectoryValidationResult(resolved_path=None, errors=["Input directory does not exist"])
    if not input_path.is_dir():
        return DirectoryValidationResult(
            resolved_path=None,
            errors=["Input directory must be a directory"],
        )
    if not _is_readable_directory(input_path):
        return DirectoryValidationResult(
            resolved_path=None,
            errors=["Input directory must be readable"],
        )

    return DirectoryValidationResult(resolved_path=input_path.resolve(), errors=[])


def run_blocking_errors(
    *,
    input_dir_raw: str,
    output_dir_raw: str,
    src: str,
    dst: str,
    mapping_errors: list[str],
    dry_run: bool = False,
    copy_images: bool = False,
    min_image_longest_edge_px: int = DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    max_image_longest_edge_px: int = DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    oversize_image_action: str = "ignore",
) -> list[str]:
    errors: list[str] = []

    directory_validation = validate_input_directory(input_dir_raw)
    errors.extend(directory_validation.errors)

    if not output_dir_raw.strip():
        errors.append("Output directory is required")

    if src not in SOURCE_FORMATS:
        errors.append(f"Source format must be one of: {', '.join(SOURCE_FORMATS)}")

    if dst not in DESTINATION_FORMATS:
        errors.append(f"Destination format must be one of: {', '.join(DESTINATION_FORMATS)}")

    if min_image_longest_edge_px > 0 and max_image_longest_edge_px > 0 and min_image_longest_edge_px > max_image_longest_edge_px:
        errors.append("Minimum image size gate cannot exceed the maximum image size gate")

    if max_image_longest_edge_px > 0 and oversize_image_action == "downscale" and not copy_images and not dry_run:
        errors.append("Downscaling oversized images requires 'Copy images to output' unless this is a dry run")

    errors.extend(mapping_errors)
    return errors


def _validate_run_inputs(
    *,
    input_dir_raw: str,
    output_dir_raw: str,
    src: str,
    dst: str,
    mapping_errors: list[str],
) -> list[str]:
    return run_blocking_errors(
        input_dir_raw=input_dir_raw,
        output_dir_raw=output_dir_raw,
        src=src,
        dst=dst,
        mapping_errors=mapping_errors,
    )


def _resolve_preview_source_format(src: str, inferred_format: str | None) -> str | None:
    if src in {"cityscapes", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"}:
        return src
    if inferred_format in {"cityscapes", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"}:
        return inferred_format
    return None


def _resolve_directory_browse_initial_directory(
    current_directory: str,
    *,
    allow_parent_fallback: bool = False,
) -> Path | None:
    current_value = current_directory.strip()
    if not current_value:
        return None

    candidate = Path(current_value).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()

    if allow_parent_fallback:
        parent = candidate.parent
        if parent.exists() and parent.is_dir():
            return parent.resolve()

    return None


def attempt_input_directory_browse(current_input_dir: str) -> InputDirectoryBrowseState:
    initial_directory = _resolve_directory_browse_initial_directory(current_input_dir)

    result = system_actions.browse_for_directory(
        initial_directory=initial_directory,
        dialog_title="Select input directory",
    )
    input_dir_raw = current_input_dir
    if result.selected_path is not None:
        input_dir_raw = str(result.selected_path)

    if result.message is not None:
        message = result.message
    elif result.available:
        message = "Selected directory."
    else:
        message = "Browse unavailable. Enter a path manually."

    return InputDirectoryBrowseState(
        input_dir_raw=input_dir_raw,
        browse_available=result.available,
        browse_message=message,
    )


def _on_browse_input_directory() -> None:
    browse_state = attempt_input_directory_browse(_coerce_text(st.session_state.get("gui_input_dir", "")))
    st.session_state["gui_input_dir"] = browse_state.input_dir_raw
    st.session_state["gui_input_browse_available"] = browse_state.browse_available
    st.session_state["gui_input_browse_message"] = browse_state.browse_message


def attempt_output_directory_browse(current_output_dir: str) -> OutputDirectoryBrowseState:
    initial_directory = _resolve_directory_browse_initial_directory(
        current_output_dir,
        allow_parent_fallback=True,
    )

    result = system_actions.browse_for_directory(
        initial_directory=initial_directory,
        dialog_title="Select output directory",
    )
    output_dir_raw = current_output_dir
    if result.selected_path is not None:
        output_dir_raw = str(result.selected_path)

    if result.message is not None:
        message = result.message
    elif result.available:
        message = "Selected directory."
    else:
        message = "Browse unavailable. Enter a path manually."

    return OutputDirectoryBrowseState(
        output_dir_raw=output_dir_raw,
        browse_available=result.available,
        browse_message=message,
    )


def _on_browse_output_directory() -> None:
    browse_state = attempt_output_directory_browse(_coerce_text(st.session_state.get("gui_output_dir", "")))
    st.session_state["gui_output_dir"] = browse_state.output_dir_raw
    st.session_state["gui_output_browse_available"] = browse_state.browse_available
    st.session_state["gui_output_browse_message"] = browse_state.browse_message


def _resolve_custom_format_browse_initial_path(
    current_custom_format_path: str,
    *,
    dataset_root: Path | None = None,
) -> Path | None:
    current_value = current_custom_format_path.strip()
    if current_value:
        candidate = Path(current_value).expanduser()
        if candidate.exists():
            return candidate.resolve()
        parent = candidate.parent
        if parent.exists() and parent.is_dir():
            return parent.resolve()
    if dataset_root is not None and dataset_root.exists() and dataset_root.is_dir():
        return dataset_root.resolve()
    return None


def attempt_custom_format_file_browse(
    current_custom_format_path: str,
    *,
    dataset_root: Path | None = None,
) -> CustomFormatFileBrowseState:
    initial_path = _resolve_custom_format_browse_initial_path(
        current_custom_format_path,
        dataset_root=dataset_root,
    )
    result = system_actions.browse_for_file(
        initial_path=initial_path,
        dialog_title="Select custom format YAML",
    )
    custom_format_path_raw = current_custom_format_path
    if result.selected_path is not None:
        custom_format_path_raw = str(result.selected_path)

    if result.message is not None:
        message = result.message
    elif result.available:
        message = "Selected file."
    else:
        message = "Browse unavailable. Enter a path manually."

    return CustomFormatFileBrowseState(
        custom_format_path_raw=custom_format_path_raw,
        browse_available=result.available,
        browse_message=message,
    )


def _on_browse_custom_format_file() -> None:
    input_dir_raw = _coerce_text(st.session_state.get("gui_input_dir", ""))
    dataset_root = Path(input_dir_raw).expanduser() if input_dir_raw else None
    browse_state = attempt_custom_format_file_browse(
        _coerce_text(st.session_state.get("gui_custom_format_path", "")),
        dataset_root=dataset_root,
    )
    st.session_state["gui_custom_format_path"] = browse_state.custom_format_path_raw
    st.session_state["gui_custom_format_browse_available"] = browse_state.browse_available
    st.session_state["gui_custom_format_browse_message"] = browse_state.browse_message


def _resolve_file_browse_initial_path(current_path: str) -> Path | None:
    current_value = current_path.strip()
    if not current_value:
        return None

    candidate = Path(current_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    parent = candidate.parent
    if parent.exists() and parent.is_dir():
        return parent.resolve()

    return None


def attempt_file_path_browse(
    current_path: str,
    *,
    dialog_title: str,
    yaml_only: bool = False,
) -> PathBrowseState:
    result = system_actions.browse_for_file(
        initial_path=_resolve_file_browse_initial_path(current_path),
        dialog_title=dialog_title,
        yaml_only=yaml_only,
    )
    path_raw = current_path
    if result.selected_path is not None:
        path_raw = str(result.selected_path)

    if result.message is not None:
        message = result.message
    elif result.available:
        message = "Selected file."
    else:
        message = "Browse unavailable. Enter a path manually."

    return PathBrowseState(
        path_raw=path_raw,
        browse_available=result.available,
        browse_message=message,
    )


def attempt_directory_path_browse(
    current_path: str,
    *,
    dialog_title: str,
    allow_parent_fallback: bool = True,
) -> PathBrowseState:
    result = system_actions.browse_for_directory(
        initial_directory=_resolve_directory_browse_initial_directory(
            current_path,
            allow_parent_fallback=allow_parent_fallback,
        ),
        dialog_title=dialog_title,
    )
    path_raw = current_path
    if result.selected_path is not None:
        path_raw = str(result.selected_path)

    if result.message is not None:
        message = result.message
    elif result.available:
        message = "Selected directory."
    else:
        message = "Browse unavailable. Enter a path manually."

    return PathBrowseState(
        path_raw=path_raw,
        browse_available=result.available,
        browse_message=message,
    )


def _on_browse_missing_label_detector_model_path() -> None:
    browse_state = attempt_file_path_browse(
        _coerce_text(st.session_state.get("gui_missing_label_detector_model_path", "")),
        dialog_title="Select detector model file",
        yaml_only=False,
    )
    st.session_state["gui_missing_label_detector_model_path"] = browse_state.path_raw
    st.session_state["gui_missing_label_detector_model_browse_available"] = browse_state.browse_available
    st.session_state["gui_missing_label_detector_model_browse_message"] = browse_state.browse_message


def _on_browse_missing_label_hints_output_dir() -> None:
    browse_state = attempt_directory_path_browse(
        _coerce_text(st.session_state.get("gui_missing_label_hints_output_dir", "")),
        dialog_title="Select hints staging directory",
    )
    st.session_state["gui_missing_label_hints_output_dir"] = browse_state.path_raw
    st.session_state["gui_missing_label_hints_output_browse_available"] = browse_state.browse_available
    st.session_state["gui_missing_label_hints_output_browse_message"] = browse_state.browse_message


def _on_browse_bbox_audit_output_dir() -> None:
    browse_state = attempt_directory_path_browse(
        _coerce_text(st.session_state.get("gui_bbox_audit_output_dir", "")),
        dialog_title="Select bbox audit report directory",
    )
    st.session_state["gui_bbox_audit_output_dir"] = browse_state.path_raw
    st.session_state["gui_bbox_audit_output_browse_available"] = browse_state.browse_available
    st.session_state["gui_bbox_audit_output_browse_message"] = browse_state.browse_message


def _on_browse_classification_config_path() -> None:
    browse_state = attempt_file_path_browse(
        _coerce_text(st.session_state.get("gui_classification_config_path", "")),
        dialog_title="Select classification config (YAML or JSON)",
        yaml_only=False,
    )
    st.session_state["gui_classification_config_path"] = browse_state.path_raw
    st.session_state["gui_classification_config_browse_available"] = browse_state.browse_available
    st.session_state["gui_classification_config_browse_message"] = browse_state.browse_message


def transition_run_state(current_status: str, event: str) -> tuple[str, int]:
    normalized_status = current_status if current_status in RUN_STATUSES else "idle"
    normalized_event = event if event in RUN_EVENTS else "reset"

    if normalized_event == "start":
        return "running", RUN_PROGRESS_BY_STATUS["running"]
    if normalized_event == "complete":
        return "completed", RUN_PROGRESS_BY_STATUS["completed"]
    if normalized_event == "fail":
        return "failed", RUN_PROGRESS_BY_STATUS["failed"]
    return "idle", RUN_PROGRESS_BY_STATUS[normalized_status if normalized_event != "reset" else "idle"]


def is_streamlit_control_flow_exception(exc: Exception) -> bool:
    return exc.__class__.__name__ in STREAMLIT_CONTROL_FLOW_EXCEPTION_NAMES


def reset_gui_run_state(
    session_state: MutableMapping[str, Any],
    *,
    detail: str | None = None,
    interrupted: bool = False,
) -> tuple[str, int]:
    current_status = _coerce_text(session_state.get("gui_run_status")) or "idle"
    reset_status, reset_progress = transition_run_state(current_status, "reset")
    session_state["gui_run_status"] = reset_status
    session_state["gui_run_progress"] = reset_progress
    session_state["gui_run_error"] = None
    session_state["gui_run_error_details"] = []
    session_state["gui_run_error_issue_rows"] = []
    session_state["gui_run_detail"] = detail
    session_state["gui_run_interrupted_notice"] = interrupted
    return reset_status, reset_progress


def build_run_summary_metrics(report: dict[str, Any]) -> RunSummaryMetrics:
    summary_counts = report.get("summary_counts") if isinstance(report, dict) else None
    if not isinstance(summary_counts, dict):
        summary_counts = {}

    images_processed = int(summary_counts.get("images", 0))
    annotations_converted = int(summary_counts.get("annotations_out", 0))

    warnings_payload = report.get("warnings") if isinstance(report, dict) else None
    warning_count = 0
    error_count = 0
    if isinstance(warnings_payload, list):
        for warning in warnings_payload:
            if not isinstance(warning, dict):
                continue
            severity = str(warning.get("severity", "")).lower()
            if severity == "error":
                error_count += 1
            elif severity == "warning":
                warning_count += 1

    return RunSummaryMetrics(
        images_processed=images_processed,
        annotations_converted=annotations_converted,
        warning_count=warning_count,
        error_count=error_count,
    )


def format_run_exception_details(exc: Exception) -> tuple[str, list[str], list[dict[str, str]]]:
    summary = _coerce_text(str(exc)) or exc.__class__.__name__
    raw_context = getattr(exc, "context", None)
    if not isinstance(raw_context, dict):
        return summary, [], []

    details: list[str] = []
    issue_rows: list[dict[str, str]] = []
    invalid_annotations = _coerce_text(raw_context.get("invalid_annotations"))
    first_error = _coerce_text(raw_context.get("first_error"))
    sample_errors = _coerce_text(raw_context.get("sample_errors"))
    issue_rows_json = _coerce_text(raw_context.get("issue_rows_json"))

    if invalid_annotations:
        details.append(f"Invalid annotations: {invalid_annotations}")
    if issue_rows_json:
        try:
            parsed_issue_rows = json.loads(issue_rows_json)
        except json.JSONDecodeError:
            parsed_issue_rows = None
        if isinstance(parsed_issue_rows, list):
            for item in parsed_issue_rows:
                if not isinstance(item, dict):
                    continue
                normalized_row = {
                    str(key): value_text
                    for key, value in item.items()
                    if (value_text := _coerce_text(value))
                }
                if normalized_row:
                    issue_rows.append(normalized_row)

    if not issue_rows:
        if first_error:
            issue_rows.append({"kind": "First issue", "issue": first_error})
        if sample_errors:
            seen_sample_errors: set[str] = set()
            for line in sample_errors.splitlines():
                error_line = _coerce_text(line)
                if not error_line or error_line == first_error or error_line in seen_sample_errors:
                    continue
                seen_sample_errors.add(error_line)
                issue_rows.append({"kind": "Sample issue", "issue": error_line})

    handled_keys = {"invalid_annotations", "first_error", "sample_errors", "issue_rows_json"}
    for key in sorted(raw_context):
        if key in handled_keys:
            continue
        value = _coerce_text(raw_context.get(key))
        if not value:
            continue
        label = key.replace("_", " ").capitalize()
        details.append(f"{label}: {value}")

    return summary, details, issue_rows


def extract_run_warning_messages(report: dict[str, Any]) -> list[tuple[str, str]]:
    warnings_payload = report.get("warnings") if isinstance(report, dict) else None
    if not isinstance(warnings_payload, list):
        return []

    messages: list[tuple[str, str]] = []
    for warning in warnings_payload:
        if not isinstance(warning, dict):
            continue
        message = str(warning.get("message", "")).strip()
        if not message:
            continue
        severity = str(warning.get("severity", "warning")).lower()
        messages.append((severity, message))
    return messages


def build_annotation_distribution_rows(dataset: Any) -> list[dict[str, float | str]]:
    annotations = getattr(dataset, "annotations", [])
    if not isinstance(annotations, list):
        return []

    rows: list[dict[str, float | str]] = []
    for annotation in annotations:
        class_id = getattr(annotation, "class_id", None)
        bbox_xywh_abs = getattr(annotation, "bbox_xywh_abs", None)
        if not isinstance(class_id, int):
            continue
        if not isinstance(bbox_xywh_abs, tuple | list) or len(bbox_xywh_abs) != 4:
            continue
        try:
            width = float(bbox_xywh_abs[2])
            height = float(bbox_xywh_abs[3])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "class_id": str(class_id),
                "bbox_width": width,
                "bbox_height": height,
            }
        )
    return rows


def class_occurrence_chart_spec(annotation_rows: list[dict[str, float | str]]) -> dict[str, Any]:
    return {
        "data": {"values": annotation_rows},
        "height": 260,
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {
                "field": "class_id",
                "type": "nominal",
                "sort": "ascending",
                "title": "Class ID",
            },
            "y": {
                "aggregate": "count",
                "type": "quantitative",
                "title": "Occurrences",
            },
        },
    }


def bbox_size_histogram_spec(annotation_rows: list[dict[str, float | str]]) -> dict[str, Any]:
    return {
        "data": {"values": annotation_rows},
        "height": 320,
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "bbox_width",
                "type": "quantitative",
                "bin": {"maxbins": 18},
                "title": "BBox width",
            },
            "y": {
                "field": "bbox_height",
                "type": "quantitative",
                "bin": {"maxbins": 18},
                "title": "BBox height",
            },
            "color": {
                "aggregate": "count",
                "type": "quantitative",
                "title": "Count",
            },
        },
        "config": {
            "view": {"stroke": "transparent"},
        },
    }


def attempt_output_directory_access(output_dir_raw: str) -> OutputDirectoryOpenResult:
    output_value = output_dir_raw.strip()
    if not output_value:
        return OutputDirectoryOpenResult(
            requested_path=Path.cwd().resolve(),
            opened=False,
            message="Output directory is empty. Provide a directory path.",
        )

    return system_actions.open_output_directory(Path(output_value).expanduser())


def _build_persistable_gui_state() -> dict[str, Any]:
    persisted_input_dir = _coerce_text(st.session_state.get("gui_last_persisted_input_dir"))
    validated_input_dir = _coerce_text(st.session_state.get("gui_input_validated_path"))
    output_dir = _coerce_text(st.session_state.get("gui_output_dir")) or DEFAULT_OUTPUT_DIR
    destination_format = _coerce_text(st.session_state.get("gui_dst"))
    inference_payload = st.session_state.get("gui_inference_payload")
    mapping_seed_signature = _coerce_text(st.session_state.get("gui_mapping_seed_signature")) or None
    out_of_frame_bbox_policy = _coerce_text(
        st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)
    ) or DEFAULT_OUT_OF_FRAME_BBOX_POLICY

    if destination_format not in DESTINATION_FORMATS:
        destination_format = DEFAULT_DESTINATION_FORMAT
    if not isinstance(inference_payload, dict):
        inference_payload = None

    return {
        "last_input_dir": validated_input_dir or persisted_input_dir or DEFAULT_INPUT_DIR,
        "last_output_dir": output_dir,
        "last_custom_format_id": _coerce_text(st.session_state.get("gui_custom_format_id")) or None,
        "last_dst": destination_format,
        "last_validation_mode": (
            _coerce_text(st.session_state.get("gui_validation_mode"))
            if _coerce_text(st.session_state.get("gui_validation_mode")) in VALIDATION_MODES
            else ValidationMode.STRICT.value
        ),
        "last_permissive_invalid_annotation_action": (
            _coerce_text(st.session_state.get("gui_permissive_invalid_annotation_action"))
            if _coerce_text(st.session_state.get("gui_permissive_invalid_annotation_action"))
            in PERMISSIVE_INVALID_ANNOTATION_ACTIONS
            else InvalidAnnotationAction.KEEP.value
        ),
        "last_allow_shared_output_dir": bool(st.session_state.get("gui_allow_shared_output_dir", True)),
        "last_prefix_output_filenames": bool(st.session_state.get("gui_prefix_output_filenames", False)),
        "last_allow_overwrite": bool(st.session_state.get("gui_allow_overwrite", False)),
        "last_input_path_include_substring": normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_include_substring"))
        ),
        "last_input_path_exclude_substring": normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_exclude_substring"))
        ),
        "last_output_file_stem_prefix": sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_prefix"))
        ),
        "last_output_file_stem_suffix": sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_suffix"))
        ),
        "last_out_of_frame_bbox_policy": out_of_frame_bbox_policy,
        "last_out_of_frame_tolerance_px": _coerce_out_of_frame_tolerance_px(
            st.session_state.get("gui_out_of_frame_tolerance_px"),
            out_of_frame_bbox_policy=out_of_frame_bbox_policy,
        ),
        "last_min_image_longest_edge_px": max(
            0,
            int(
                _coerce_float(
                    st.session_state.get("gui_min_image_longest_edge_px"),
                    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
                )
            ),
        ),
        "last_max_image_longest_edge_px": max(
            0,
            int(
                _coerce_float(
                    st.session_state.get("gui_max_image_longest_edge_px"),
                    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
                )
            ),
        ),
        "last_preview_scan_limit": min(
            MAX_PREVIEW_SCAN_LIMIT,
            max(
                0,
                int(
                    _coerce_float(
                        st.session_state.get("gui_preview_scan_limit"),
                        DEFAULT_PREVIEW_SCAN_LIMIT,
                    )
                ),
            ),
        ),
        "last_missing_label_detector_model_path": (
            _coerce_text(st.session_state.get("gui_missing_label_detector_model_path")) or None
        ),
        "last_missing_label_hints_output_dir": (
            _coerce_text(st.session_state.get("gui_missing_label_hints_output_dir")) or None
        ),
        "last_missing_label_confidence_threshold": min(
            1.0,
            max(
                0.0,
                _coerce_float(st.session_state.get("gui_missing_label_confidence_threshold"), 0.25),
            ),
        ),
        "last_missing_label_iou_threshold": min(
            1.0,
            max(
                0.0,
                _coerce_float(st.session_state.get("gui_missing_label_iou_threshold"), 0.45),
            ),
        ),
        "last_missing_label_max_detections_per_image": max(
            1,
            int(
                _coerce_float(
                    st.session_state.get("gui_missing_label_max_detections_per_image"),
                    200,
                )
            ),
        ),
        "last_bbox_audit_output_dir": _coerce_text(st.session_state.get("gui_bbox_audit_output_dir")) or None,
        "last_bbox_audit_max_labeled_images": max(
            1,
            int(_coerce_float(st.session_state.get("gui_bbox_audit_max_labeled_images"), 100)),
        ),
        "last_bbox_audit_match_iou_threshold": min(
            1.0,
            max(0.0, _coerce_float(st.session_state.get("gui_bbox_audit_match_iou_threshold"), 0.30)),
        ),
        "last_bbox_audit_correction_iou_threshold": min(
            1.0,
            max(
                0.0,
                _coerce_float(st.session_state.get("gui_bbox_audit_correction_iou_threshold"), 0.85),
            ),
        ),
        "last_oversize_image_action": (
            _coerce_text(st.session_state.get("gui_oversize_image_action"))
            if _coerce_text(st.session_state.get("gui_oversize_image_action")) in OVERSIZE_IMAGE_ACTIONS
            else "ignore"
        ),
        "last_custom_format_path": _coerce_text(st.session_state.get("gui_custom_format_path")) or None,
        "last_inference_payload": inference_payload,
        "last_mapping_rows": normalize_mapping_rows(st.session_state.get("gui_mapping_rows")),
        "last_mapping_seed_signature": mapping_seed_signature,
        "last_classification_config_path": (
            _coerce_text(st.session_state.get("gui_classification_config_path")) or None
        ),
        "last_classification_advance_on_label": bool(
            st.session_state.get("gui_classification_advance_on_label", True)
        ),
        "last_classification_show_bboxes": bool(
            st.session_state.get("gui_classification_show_bboxes", True)
        ),
    }


def _remember_gui_preferences() -> None:
    payload = _build_persistable_gui_state()
    serialized_payload = json.dumps(payload, sort_keys=True)
    if _coerce_text(st.session_state.get("gui_last_persisted_state_payload")) == serialized_payload:
        return

    persisted_path = persist_gui_state(payload)
    if persisted_path is not None:
        st.session_state["gui_last_persisted_state_payload"] = serialized_payload
        st.session_state["gui_last_persisted_input_dir"] = _coerce_text(payload.get("last_input_dir"))


def _remember_last_used_input_directory(input_directory: Path) -> None:
    st.session_state["gui_last_persisted_input_dir"] = str(input_directory)


def _initialize_state() -> None:
    persisted_preferences = load_persisted_gui_preferences()
    if "gui_input_dir" not in st.session_state:
        st.session_state["gui_input_dir"] = persisted_preferences["gui_input_dir"]
    if "gui_output_dir" not in st.session_state:
        st.session_state["gui_output_dir"] = persisted_preferences["gui_output_dir"]
    if "gui_src" not in st.session_state:
        st.session_state["gui_src"] = "auto"
    if "gui_dst" not in st.session_state:
        st.session_state["gui_dst"] = persisted_preferences["gui_dst"]
    if "gui_validation_mode" not in st.session_state:
        st.session_state["gui_validation_mode"] = persisted_preferences["gui_validation_mode"]
    if "gui_permissive_invalid_annotation_action" not in st.session_state:
        st.session_state["gui_permissive_invalid_annotation_action"] = persisted_preferences[
            "gui_permissive_invalid_annotation_action"
        ]
    if "gui_allow_shared_output_dir" not in st.session_state:
        st.session_state["gui_allow_shared_output_dir"] = persisted_preferences["gui_allow_shared_output_dir"]
    if "gui_prefix_output_filenames" not in st.session_state:
        st.session_state["gui_prefix_output_filenames"] = persisted_preferences["gui_prefix_output_filenames"]
    if "gui_allow_overwrite" not in st.session_state:
        st.session_state["gui_allow_overwrite"] = persisted_preferences["gui_allow_overwrite"]
    if "gui_input_path_include_substring" not in st.session_state:
        st.session_state["gui_input_path_include_substring"] = persisted_preferences[
            "gui_input_path_include_substring"
        ]
    if "gui_input_path_exclude_substring" not in st.session_state:
        st.session_state["gui_input_path_exclude_substring"] = persisted_preferences[
            "gui_input_path_exclude_substring"
        ]
    if "gui_output_file_stem_prefix" not in st.session_state:
        st.session_state["gui_output_file_stem_prefix"] = persisted_preferences["gui_output_file_stem_prefix"]
    if "gui_output_file_stem_suffix" not in st.session_state:
        st.session_state["gui_output_file_stem_suffix"] = persisted_preferences["gui_output_file_stem_suffix"]
    if "gui_out_of_frame_bbox_policy" not in st.session_state:
        st.session_state["gui_out_of_frame_bbox_policy"] = persisted_preferences["gui_out_of_frame_bbox_policy"]
    if "gui_out_of_frame_tolerance_px" not in st.session_state:
        st.session_state["gui_out_of_frame_tolerance_px"] = persisted_preferences["gui_out_of_frame_tolerance_px"]
    elif _coerce_text(st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)) in {"correct", "warn"}:
        current_out_of_frame_tolerance_px = _coerce_float(
            st.session_state.get("gui_out_of_frame_tolerance_px"),
            DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
        )
        if current_out_of_frame_tolerance_px <= 0.0:
            st.session_state["gui_out_of_frame_tolerance_px"] = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX
    if "gui_min_image_longest_edge_px" not in st.session_state:
        st.session_state["gui_min_image_longest_edge_px"] = persisted_preferences["gui_min_image_longest_edge_px"]
    if "gui_max_image_longest_edge_px" not in st.session_state:
        st.session_state["gui_max_image_longest_edge_px"] = persisted_preferences["gui_max_image_longest_edge_px"]
    if "gui_preview_scan_limit" not in st.session_state:
        st.session_state["gui_preview_scan_limit"] = persisted_preferences["gui_preview_scan_limit"]
    if "gui_oversize_image_action" not in st.session_state:
        st.session_state["gui_oversize_image_action"] = persisted_preferences["gui_oversize_image_action"]
    if "gui_unmapped_policy" not in st.session_state:
        st.session_state["gui_unmapped_policy"] = "error"
    if "gui_dry_run" not in st.session_state:
        st.session_state["gui_dry_run"] = False
    if "gui_copy_images" not in st.session_state:
        st.session_state["gui_copy_images"] = True
    if "gui_inference_payload" not in st.session_state:
        st.session_state["gui_inference_payload"] = persisted_preferences["gui_inference_payload"]
    if "gui_custom_format_id" not in st.session_state:
        st.session_state["gui_custom_format_id"] = persisted_preferences["gui_custom_format_id"]
    if "gui_custom_format_path" not in st.session_state:
        st.session_state["gui_custom_format_path"] = persisted_preferences["gui_custom_format_path"]
    if "gui_inference_error" not in st.session_state:
        st.session_state["gui_inference_error"] = None
    if "gui_inference_error_input_dir" not in st.session_state:
        st.session_state["gui_inference_error_input_dir"] = None
    if "gui_preview_index" not in st.session_state:
        st.session_state["gui_preview_index"] = 0
    if "gui_preview_key" not in st.session_state:
        st.session_state["gui_preview_key"] = ""
    if "gui_preview_keyboard_event_nonce" not in st.session_state:
        st.session_state["gui_preview_keyboard_event_nonce"] = 0
    if "gui_mapping_rows" not in st.session_state:
        st.session_state["gui_mapping_rows"] = [dict(row) for row in persisted_preferences["gui_mapping_rows"]]
    if "gui_class_labels" not in st.session_state:
        st.session_state["gui_class_labels"] = {}
    if "gui_mapping_seed_signature" not in st.session_state:
        st.session_state["gui_mapping_seed_signature"] = persisted_preferences["gui_mapping_seed_signature"]
    if "gui_last_run" not in st.session_state:
        st.session_state["gui_last_run"] = None
    if "gui_input_browse_available" not in st.session_state:
        st.session_state["gui_input_browse_available"] = True
    if "gui_input_browse_message" not in st.session_state:
        st.session_state["gui_input_browse_message"] = None
    if "gui_output_browse_available" not in st.session_state:
        st.session_state["gui_output_browse_available"] = True
    if "gui_output_browse_message" not in st.session_state:
        st.session_state["gui_output_browse_message"] = None
    if "gui_custom_format_browse_available" not in st.session_state:
        st.session_state["gui_custom_format_browse_available"] = True
    if "gui_custom_format_browse_message" not in st.session_state:
        st.session_state["gui_custom_format_browse_message"] = None
    if "gui_missing_label_detector_model_browse_available" not in st.session_state:
        st.session_state["gui_missing_label_detector_model_browse_available"] = True
    if "gui_missing_label_detector_model_browse_message" not in st.session_state:
        st.session_state["gui_missing_label_detector_model_browse_message"] = None
    if "gui_missing_label_hints_output_browse_available" not in st.session_state:
        st.session_state["gui_missing_label_hints_output_browse_available"] = True
    if "gui_missing_label_hints_output_browse_message" not in st.session_state:
        st.session_state["gui_missing_label_hints_output_browse_message"] = None
    if "gui_bbox_audit_output_browse_available" not in st.session_state:
        st.session_state["gui_bbox_audit_output_browse_available"] = True
    if "gui_bbox_audit_output_browse_message" not in st.session_state:
        st.session_state["gui_bbox_audit_output_browse_message"] = None
    if "gui_input_validation_errors" not in st.session_state:
        st.session_state["gui_input_validation_errors"] = []
    if "gui_input_validated_path" not in st.session_state:
        st.session_state["gui_input_validated_path"] = None
    if "gui_run_status" not in st.session_state:
        st.session_state["gui_run_status"] = "idle"
    if "gui_run_progress" not in st.session_state:
        st.session_state["gui_run_progress"] = 0
    if "gui_run_error" not in st.session_state:
        st.session_state["gui_run_error"] = None
    if "gui_run_error_details" not in st.session_state:
        st.session_state["gui_run_error_details"] = []
    if "gui_run_error_issue_rows" not in st.session_state:
        st.session_state["gui_run_error_issue_rows"] = []
    if "gui_run_detail" not in st.session_state:
        st.session_state["gui_run_detail"] = None
    if "gui_run_interrupted_notice" not in st.session_state:
        st.session_state["gui_run_interrupted_notice"] = False
    if "gui_output_action_message" not in st.session_state:
        st.session_state["gui_output_action_message"] = None
    if "gui_last_persisted_input_dir" not in st.session_state:
        st.session_state["gui_last_persisted_input_dir"] = persisted_preferences["gui_last_persisted_input_dir"]
    if "gui_last_persisted_state_payload" not in st.session_state:
        st.session_state["gui_last_persisted_state_payload"] = persisted_preferences["gui_last_persisted_state_payload"]
    if "gui_missing_label_detector_model_path" not in st.session_state:
        st.session_state["gui_missing_label_detector_model_path"] = persisted_preferences[
            "gui_missing_label_detector_model_path"
        ]
    if "gui_missing_label_hints_output_dir" not in st.session_state:
        st.session_state["gui_missing_label_hints_output_dir"] = persisted_preferences[
            "gui_missing_label_hints_output_dir"
        ]
    if "gui_missing_label_confidence_threshold" not in st.session_state:
        st.session_state["gui_missing_label_confidence_threshold"] = persisted_preferences[
            "gui_missing_label_confidence_threshold"
        ]
    if "gui_missing_label_iou_threshold" not in st.session_state:
        st.session_state["gui_missing_label_iou_threshold"] = persisted_preferences[
            "gui_missing_label_iou_threshold"
        ]
    if "gui_missing_label_max_detections_per_image" not in st.session_state:
        st.session_state["gui_missing_label_max_detections_per_image"] = persisted_preferences[
            "gui_missing_label_max_detections_per_image"
        ]
    if "gui_missing_label_review_index" not in st.session_state:
        st.session_state["gui_missing_label_review_index"] = 0
    if "gui_missing_label_result" not in st.session_state:
        st.session_state["gui_missing_label_result"] = None
    if "gui_missing_label_error" not in st.session_state:
        st.session_state["gui_missing_label_error"] = None
    if "gui_detector_review_result" not in st.session_state:
        st.session_state["gui_detector_review_result"] = None
    if "gui_detector_review_item_cache" not in st.session_state:
        st.session_state["gui_detector_review_item_cache"] = {}
    if "gui_detector_review_cache_signature" not in st.session_state:
        st.session_state["gui_detector_review_cache_signature"] = ""
    if "gui_detector_review_image_paths" not in st.session_state:
        st.session_state["gui_detector_review_image_paths"] = []
    if "gui_detector_review_image_paths_signature" not in st.session_state:
        st.session_state["gui_detector_review_image_paths_signature"] = ""
    if "gui_detector_review_generation_error" not in st.session_state:
        st.session_state["gui_detector_review_generation_error"] = None
    if "gui_detector_review_error" not in st.session_state:
        st.session_state["gui_detector_review_error"] = None
    if "gui_detector_review_review_index" not in st.session_state:
        st.session_state["gui_detector_review_review_index"] = 0
    if "gui_detector_review_source_signature" not in st.session_state:
        st.session_state["gui_detector_review_source_signature"] = None
    if "gui_detector_review_allow_overwrite_missing" not in st.session_state:
        st.session_state["gui_detector_review_allow_overwrite_missing"] = False
    if "gui_detector_review_apply_confirm" not in st.session_state:
        st.session_state["gui_detector_review_apply_confirm"] = False
    if "gui_detector_review_last_approval" not in st.session_state:
        st.session_state["gui_detector_review_last_approval"] = None
    if "gui_missing_label_source_signature" not in st.session_state:
        st.session_state["gui_missing_label_source_signature"] = None
    if "gui_missing_label_allow_overwrite" not in st.session_state:
        st.session_state["gui_missing_label_allow_overwrite"] = False
    if "gui_missing_label_apply_confirm" not in st.session_state:
        st.session_state["gui_missing_label_apply_confirm"] = False
    if "gui_missing_label_last_approval" not in st.session_state:
        st.session_state["gui_missing_label_last_approval"] = None
    if "gui_bbox_audit_output_dir" not in st.session_state:
        st.session_state["gui_bbox_audit_output_dir"] = persisted_preferences[
            "gui_bbox_audit_output_dir"
        ]
    if "gui_bbox_audit_match_iou_threshold" not in st.session_state:
        st.session_state["gui_bbox_audit_match_iou_threshold"] = persisted_preferences[
            "gui_bbox_audit_match_iou_threshold"
        ]
    if "gui_bbox_audit_correction_iou_threshold" not in st.session_state:
        st.session_state["gui_bbox_audit_correction_iou_threshold"] = persisted_preferences[
            "gui_bbox_audit_correction_iou_threshold"
        ]
    if "gui_bbox_audit_max_labeled_images" not in st.session_state:
        st.session_state["gui_bbox_audit_max_labeled_images"] = persisted_preferences[
            "gui_bbox_audit_max_labeled_images"
        ]
    if "gui_bbox_audit_review_index" not in st.session_state:
        st.session_state["gui_bbox_audit_review_index"] = 0
    if "gui_bbox_audit_result" not in st.session_state:
        st.session_state["gui_bbox_audit_result"] = None
    if "gui_bbox_audit_error" not in st.session_state:
        st.session_state["gui_bbox_audit_error"] = None
    if "gui_bbox_audit_source_signature" not in st.session_state:
        st.session_state["gui_bbox_audit_source_signature"] = None
    if "gui_bbox_audit_apply_confirm" not in st.session_state:
        st.session_state["gui_bbox_audit_apply_confirm"] = False
    if "gui_bbox_audit_last_approval" not in st.session_state:
        st.session_state["gui_bbox_audit_last_approval"] = None

    if "gui_classification_config_path" not in st.session_state:
        st.session_state["gui_classification_config_path"] = persisted_preferences[
            "gui_classification_config_path"
        ]
    if "gui_classification_advance_on_label" not in st.session_state:
        st.session_state["gui_classification_advance_on_label"] = persisted_preferences[
            "gui_classification_advance_on_label"
        ]
    if "gui_classification_show_bboxes" not in st.session_state:
        st.session_state["gui_classification_show_bboxes"] = persisted_preferences[
            "gui_classification_show_bboxes"
        ]
    if "gui_classification_index" not in st.session_state:
        st.session_state["gui_classification_index"] = 0
    if "gui_classification_keyboard_event_nonce" not in st.session_state:
        st.session_state["gui_classification_keyboard_event_nonce"] = 0
    if "gui_classification_image_paths" not in st.session_state:
        st.session_state["gui_classification_image_paths"] = []
    if "gui_classification_image_paths_signature" not in st.session_state:
        st.session_state["gui_classification_image_paths_signature"] = ""
    if "gui_classification_last_saved" not in st.session_state:
        st.session_state["gui_classification_last_saved"] = None
    if "gui_classification_error" not in st.session_state:
        st.session_state["gui_classification_error"] = None

    if _coerce_text(st.session_state.get("gui_run_status")) == "running":
        reset_gui_run_state(
            st.session_state,
            detail="Previous run was interrupted before completion. You can run conversion again.",
            interrupted=True,
        )


def render() -> None:
    st.set_page_config(page_title="label_master", layout="wide")
    st.title("label_master")
    st.caption("Localhost-only annotation conversion workflow")

    server_address = st.query_params.get("server.address", None)
    if isinstance(server_address, list):
        server_address = server_address[0] if server_address else None

    if not is_localhost_binding(server_address):
        st.error("GUI must bind to localhost only")
        st.stop()

    _initialize_state()
    _inject_compact_layout_css()

    tabs = st.tabs([
        "1. Dataset",
        "2. Format & Preview",
        "3. Output",
        "4. Detector Review",
        "5. Label Mapping",
        "6. Review & Run",
        "7. Classification",
    ])

    with tabs[0]:
        st.subheader("Step 1: Dataset")
        st.caption("Select an input directory. Browse is optional; manual entry is always available.")

        input_cols = st.columns([4, 1])
        with input_cols[0]:
            st.text_input("Input directory", key="gui_input_dir")
        with input_cols[1]:
            st.button("Browse...", key="gui_input_dir_browse", on_click=_on_browse_input_directory)

        browse_message = st.session_state.get("gui_input_browse_message")
        if isinstance(browse_message, str) and browse_message:
            if bool(st.session_state.get("gui_input_browse_available", True)):
                st.info(browse_message)
            else:
                st.warning(browse_message)

        directory_validation = validate_input_directory(_coerce_text(st.session_state["gui_input_dir"]))
        st.session_state["gui_input_validation_errors"] = directory_validation.errors
        st.session_state["gui_input_validated_path"] = (
            str(directory_validation.resolved_path) if directory_validation.resolved_path else None
        )

        if directory_validation.errors:
            for error in directory_validation.errors:
                st.error(error)
        elif directory_validation.resolved_path is not None:
            _remember_last_used_input_directory(directory_validation.resolved_path)
            st.success(f"Using dataset directory: {directory_validation.resolved_path}")

        filter_cols = st.columns(2)
        with filter_cols[0]:
            st.text_input(
                "Only include input paths containing",
                key="gui_input_path_include_substring",
                placeholder="e.g. train or drone",
            )
        with filter_cols[1]:
            st.text_input(
                "Exclude input paths containing",
                key="gui_input_path_exclude_substring",
                placeholder="e.g. val or backup",
            )
        st.caption(
            "Filters are case-insensitive and match against the dataset-relative image path text."
        )

    with tabs[1]:
        st.subheader("Step 2: Format & Preview")
        format_controls = st.columns([3, 1], gap="small")
        with format_controls[0]:
            st.selectbox("Source format", SOURCE_FORMATS, key="gui_src")
        with format_controls[1]:
            infer_requested = st.button(
                "Infer format",
                key="gui_infer_button",
                width="stretch",
            )
        st.number_input(
            "Preview scan limit (Step 2 only, 0 = full scan)",
            min_value=0,
            max_value=MAX_PREVIEW_SCAN_LIMIT,
            step=500,
            key="gui_preview_scan_limit",
            help=(
                "Limits how many samples Step 2 scans when building preview images and class labels. "
                "Use this for very large datasets to keep preview responsive."
            ),
        )
        _inject_preview_table_alignment_css()

        input_dir_raw = _coerce_text(st.session_state["gui_input_dir"])
        input_path = Path(input_dir_raw).expanduser() if input_dir_raw else Path(".")
        skip_preview_refresh = _consume_preview_skip_once(st.session_state)
        source_format_value = _coerce_text(st.session_state.get("gui_src"))
        custom_format_options: list[CustomFormatOption] = []
        custom_format_error: str | None = None
        custom_format_path_error: str | None = None
        selected_custom_format_id: str | None = None
        selected_custom_format: CustomFormatOption | None = None
        selected_custom_format_path: Path | None = None

        if source_format_value == "custom":
            if input_dir_raw and input_path.exists() and input_path.is_dir():
                try:
                    custom_format_options = _custom_format_options(input_path)
                except Exception as exc:
                    custom_format_error = str(exc)

            if custom_format_error:
                st.error(f"Unable to load custom format YAMLs: {custom_format_error}")
            elif not input_dir_raw or not input_path.exists() or not input_path.is_dir():
                st.caption("Set a valid input directory to load custom format YAML choices.")
            elif not custom_format_options:
                st.warning(
                    "No custom format YAMLs found in the supported default locations. "
                    "You can place one at the dataset root as custom_format.yaml, data_format.yaml, "
                    "or label_format.yaml, keep one under format_specs, or browse to a YAML file below."
                )
                explicit_cols = st.columns([4, 1], gap="small")
                with explicit_cols[0]:
                    st.text_input(
                        "Custom format YAML path",
                        key="gui_custom_format_path",
                        placeholder="/path/to/custom_format.yaml",
                    )
                with explicit_cols[1]:
                    st.button(
                        "Browse...",
                        key="gui_custom_format_path_browse",
                        on_click=_on_browse_custom_format_file,
                    )

                custom_browse_message = st.session_state.get("gui_custom_format_browse_message")
                if isinstance(custom_browse_message, str) and custom_browse_message:
                    if bool(st.session_state.get("gui_custom_format_browse_available", True)):
                        st.info(custom_browse_message)
                    else:
                        st.warning(custom_browse_message)

                selected_custom_format, custom_format_path_error = _explicit_custom_format_option(
                    _coerce_text(st.session_state.get("gui_custom_format_path"))
                )
                if custom_format_path_error:
                    st.warning(custom_format_path_error)
                elif selected_custom_format is not None:
                    selected_custom_format_id = selected_custom_format.format_id
                    selected_custom_format_path = selected_custom_format.path
                    st.caption(f"Selected YAML: {selected_custom_format.path}")
                    if selected_custom_format.description:
                        st.caption(selected_custom_format.description)
                    st.caption(
                        "The selected YAML is used for preview and conversion while Source format is `custom`."
                    )
            else:
                option_ids = [option.format_id for option in custom_format_options]
                option_map = {option.format_id: option for option in custom_format_options}
                current_custom_format_id = _coerce_text(st.session_state.get("gui_custom_format_id")) or None
                if current_custom_format_id not in option_map:
                    st.session_state["gui_custom_format_id"] = (
                        _default_custom_format_id(input_path, options=custom_format_options) or option_ids[0]
                    )
                st.selectbox(
                    "Custom format YAML",
                    options=option_ids,
                    key="gui_custom_format_id",
                    format_func=lambda format_id: _format_custom_format_option(option_map[format_id]),
                )
                selected_custom_format_id = _coerce_text(st.session_state.get("gui_custom_format_id")) or None
                selected_custom_format = _selected_custom_format_option(
                    custom_format_options,
                    selected_custom_format_id,
                )
                if selected_custom_format is not None:
                    selected_custom_format_path = selected_custom_format.path
                    st.caption(f"Selected YAML: {selected_custom_format.path}")
                    if selected_custom_format.description:
                        st.caption(selected_custom_format.description)
                    st.caption("The selected YAML is used for preview and conversion while Source format is `custom`.")

        if infer_requested:
            try:
                infer_vm = infer_view(input_path)
                st.session_state["gui_inference_payload"] = _build_inference_payload(
                    infer_vm,
                    input_path=input_path,
                )
                st.session_state["gui_inference_error"] = None
                st.session_state["gui_inference_error_input_dir"] = None
            except Exception as exc:
                st.session_state["gui_inference_payload"] = None
                st.session_state["gui_inference_error"] = str(exc)
                st.session_state["gui_inference_error_input_dir"] = _resolved_input_dir_token(input_path)

        if input_dir_raw and input_path.exists() and input_path.is_dir() and not skip_preview_refresh:
            _maybe_auto_infer_for_preview(input_path)

        inference_payload = st.session_state["gui_inference_payload"]
        inference_error = st.session_state["gui_inference_error"]

        inferred_format = None
        if isinstance(inference_payload, dict):
            predicted = inference_payload.get("predicted_format")
            inferred_format = str(predicted) if predicted is not None else None

        preview_source_format = _resolve_preview_source_format(
            _coerce_text(st.session_state["gui_src"]),
            inferred_format,
        )
        preview_custom_format_id = selected_custom_format_id if source_format_value == "custom" else None
        preview_custom_format_path = selected_custom_format_path if source_format_value == "custom" else None
        custom_preview_blocker: str | None = None
        if source_format_value == "custom":
            if custom_format_error:
                custom_preview_blocker = f"Unable to load custom format YAMLs: {custom_format_error}"
            elif input_dir_raw and input_path.exists() and input_path.is_dir() and not custom_format_options:
                custom_preview_blocker = custom_format_path_error or (
                    "No custom format YAML found. Browse to a YAML file or enter a path manually."
                )
            elif input_dir_raw and input_path.exists() and input_path.is_dir() and preview_custom_format_id is None:
                custom_preview_blocker = "Choose a custom format YAML to preview this dataset."
        input_path_include_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_include_substring"))
        )
        input_path_exclude_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_exclude_substring"))
        )
        out_of_frame_bbox_policy = _coerce_text(
            st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)
        ) or DEFAULT_OUT_OF_FRAME_BBOX_POLICY
        out_of_frame_tolerance_px = max(
            0.0,
            _coerce_float(
                st.session_state.get("gui_out_of_frame_tolerance_px"),
                DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
            ),
        )
        preview_scan_limit = min(
            MAX_PREVIEW_SCAN_LIMIT,
            max(
                0,
                int(
                    _coerce_float(
                        st.session_state.get("gui_preview_scan_limit"),
                        DEFAULT_PREVIEW_SCAN_LIMIT,
                    )
                ),
            ),
        )
        if preview_scan_limit > 0:
            st.caption(
                f"Preview scan limit is active: Step 2 scans up to `{preview_scan_limit}` samples for preview."
            )
        elif source_format_value == "custom":
            st.warning(
                "Preview scan limit is `0`, so Step 2 will scan the full custom dataset. "
                "Large BDD100K datasets can take a while; set a non-zero limit for a faster preview."
            )
        preview_vm = None
        if skip_preview_refresh:
            st.caption("Preview refresh skipped while editing output-only image size settings.")
            _preview_keyboard_navigation_action(
                enabled=False,
                previous_disabled=True,
                next_disabled=True,
            )
        elif not input_dir_raw or not input_path.exists() or not input_path.is_dir():
            _store_class_labels({})
            st.info("Set a valid input directory to load preview.")
            _preview_keyboard_navigation_action(
                enabled=False,
                previous_disabled=True,
                next_disabled=True,
            )
        elif preview_source_format is None:
            st.info("Choose a source format or run inference to load preview.")
            _preview_keyboard_navigation_action(
                enabled=False,
                previous_disabled=True,
                next_disabled=True,
            )
        elif preview_source_format == "custom" and source_format_value == "custom" and custom_preview_blocker:
            _store_class_labels({})
            st.warning(custom_preview_blocker)
            _preview_keyboard_navigation_action(
                enabled=False,
                previous_disabled=True,
                next_disabled=True,
            )
        else:
            try:
                preview_vm = preview_dataset_view(
                    input_path,
                    source_format=preview_source_format,
                    out_of_frame_bbox_policy=out_of_frame_bbox_policy,
                    out_of_frame_tolerance_px=out_of_frame_tolerance_px,
                    input_path_include_substring=input_path_include_substring,
                    input_path_exclude_substring=input_path_exclude_substring,
                    preview_scan_limit=preview_scan_limit,
                    custom_format_id=preview_custom_format_id,
                    custom_format_path=preview_custom_format_path,
                )
            except Exception as exc:
                st.error(f"Unable to load preview dataset: {exc}")
                _preview_keyboard_navigation_action(
                    enabled=False,
                    previous_disabled=True,
                    next_disabled=True,
                )
                preview_vm = None

            if preview_vm:
                preview_key = (
                    f"{input_path.resolve()}::{preview_source_format}::"
                    f"{preview_custom_format_id or ''}::"
                    f"{preview_custom_format_path or ''}::"
                    f"{input_path_include_substring or ''}::{input_path_exclude_substring or ''}"
                    f"::{preview_scan_limit}"
                )
                if st.session_state["gui_preview_key"] != preview_key:
                    st.session_state["gui_preview_key"] = preview_key
                    st.session_state["gui_preview_index"] = 0

                class_labels = extract_class_labels_from_preview(preview_vm)
                _store_class_labels(class_labels)
                class_signature = (
                    f"{preview_key}::{'|'.join(str(class_id) for class_id in sorted(class_labels))}"
                )
                _seed_identity_rows_for_dataset(class_signature, class_labels)

                for warning in preview_vm.warnings:
                    st.warning(warning)

                if not preview_vm.images:
                    st.info("No previewable images found in dataset.")
                    _preview_keyboard_navigation_action(
                        enabled=False,
                        previous_disabled=True,
                        next_disabled=True,
                    )
                else:
                    max_index = len(preview_vm.images) - 1
                    current_index = int(st.session_state["gui_preview_index"])
                    current_index = min(max(current_index, 0), max_index)
                    previous_disabled = current_index == 0
                    next_disabled = current_index == max_index
                    keyboard_action = _preview_keyboard_navigation_action(
                        enabled=True,
                        previous_disabled=previous_disabled,
                        next_disabled=next_disabled,
                    )

                    nav_cols = st.columns([1, 3, 1])
                    with nav_cols[0]:
                        previous_clicked = st.button(
                            "Previous",
                            disabled=previous_disabled,
                            key="gui_preview_prev",
                        )
                    with nav_cols[2]:
                        next_clicked = st.button(
                            "Next",
                            disabled=next_disabled,
                            key="gui_preview_next",
                        )

                    current_index = _resolve_preview_index(
                        current_index,
                        max_index=max_index,
                        keyboard_action=keyboard_action,
                        previous_clicked=previous_clicked,
                        next_clicked=next_clicked,
                    )
                    st.session_state["gui_preview_index"] = current_index

                    with nav_cols[1]:
                        st.markdown(f"**Image {current_index + 1} / {len(preview_vm.images)}**")

                    current_image = preview_vm.images[current_index]
                    overlay_labels = [
                        (
                            bbox.bbox_xywh_abs[0],
                            bbox.bbox_xywh_abs[1],
                            bbox.bbox_xywh_abs[2],
                            bbox.bbox_xywh_abs[3],
                            f"{bbox.class_id}:{bbox.class_name}",
                        )
                        for bbox in current_image.bboxes
                    ]
                    overlay, overlay_warnings = render_preview_overlay(
                        dataset_root=input_path,
                        image_rel_path=current_image.file_name,
                        bboxes=overlay_labels,
                    )
                    preview_cols = st.columns([3, 2], gap="small")
                    if overlay is None:
                        with preview_cols[0]:
                            for warning in overlay_warnings:
                                st.warning(warning)
                    else:
                        with preview_cols[0]:
                            st.caption(current_image.file_name)
                            st.image(overlay, width="stretch")

                    bbox_rows = [
                        {
                            "annotation_id": str(bbox.annotation_id),
                            "class_id": str(bbox.class_id),
                            "class_name": str(bbox.class_name),
                            "x": f"{bbox.bbox_xywh_abs[0]:.2f}",
                            "y": f"{bbox.bbox_xywh_abs[1]:.2f}",
                            "w": f"{bbox.bbox_xywh_abs[2]:.2f}",
                            "h": f"{bbox.bbox_xywh_abs[3]:.2f}",
                        }
                        for bbox in current_image.bboxes
                    ]
                    with preview_cols[1]:
                        st.caption(f"{len(bbox_rows)} annotations")
                        if bbox_rows:
                            st.dataframe(
                                bbox_rows,
                                width="stretch",
                                hide_index=True,
                                height=min(300, max(120, 38 * (len(bbox_rows) + 1))),
                            )
                        else:
                            st.caption("No bounding boxes for this image.")

                    class_example_groups = build_class_example_groups(preview_vm)
                    if class_example_groups:
                        with st.expander("Class examples", expanded=True):
                            st.caption(
                                "Showing up to "
                                f"{PREVIEW_CLASS_EXAMPLES_PER_CLASS} sample image(s) per class."
                            )
                            for class_group in class_example_groups:
                                st.markdown(f"**{class_group.class_id}: {class_group.class_name}**")
                                st.caption(
                                    f"Showing {len(class_group.examples)} of "
                                    f"{class_group.image_count} image(s) containing this class."
                                )
                                example_cols = st.columns(len(class_group.examples), gap="small")
                                for index, example in enumerate(class_group.examples):
                                    overlay, overlay_warnings = render_preview_overlay(
                                        dataset_root=input_path,
                                        image_rel_path=example.file_name,
                                        bboxes=list(example.overlay_labels),
                                    )
                                    with example_cols[index]:
                                        st.caption(example.file_name)
                                        if overlay is None:
                                            for warning in overlay_warnings:
                                                st.warning(warning)
                                        else:
                                            st.image(overlay, width="stretch")
                                        st.caption(
                                            f"{example.annotation_count} matching annotation(s)"
                                        )
        if inference_error:
            st.error(inference_error)
        format_details = format_details_yaml(
            preview_source_format,
            dataset_root=input_path if input_path.exists() and input_path.is_dir() else None,
            inference_payload=inference_payload if isinstance(inference_payload, dict) else None,
            custom_format_id=preview_custom_format_id,
            custom_format_path=preview_custom_format_path,
        )
        if format_details:
            with st.expander("Format details (YAML)", expanded=False):
                st.code(format_details, language="yaml")

    with tabs[2]:
        st.subheader("Step 3: Output")
        output_cols = st.columns([4, 1])
        with output_cols[0]:
            st.text_input("Output directory", key="gui_output_dir")
        with output_cols[1]:
            st.button("Browse...", key="gui_output_dir_browse", on_click=_on_browse_output_directory)

        output_browse_message = st.session_state.get("gui_output_browse_message")
        if isinstance(output_browse_message, str) and output_browse_message:
            if bool(st.session_state.get("gui_output_browse_available", True)):
                st.info(output_browse_message)
            else:
                st.warning(output_browse_message)

        st.selectbox("Destination format", DESTINATION_FORMATS, key="gui_dst")
        st.selectbox("Unmapped policy", UNMAPPED_POLICIES, key="gui_unmapped_policy")
        st.selectbox("Validation mode", VALIDATION_MODES, key="gui_validation_mode")
        st.selectbox(
            "If permissive: invalid annotation handling",
            PERMISSIVE_INVALID_ANNOTATION_ACTIONS,
            key="gui_permissive_invalid_annotation_action",
            disabled=_coerce_text(st.session_state.get("gui_validation_mode")) != ValidationMode.PERMISSIVE.value,
        )
        if _coerce_text(st.session_state.get("gui_validation_mode")) == ValidationMode.STRICT.value:
            st.caption("Strict mode stops the run on invalid annotations. Switch to `permissive` to continue while reporting invalid rows.")
        else:
            if (
                _coerce_text(st.session_state.get("gui_permissive_invalid_annotation_action"))
                == InvalidAnnotationAction.DROP.value
            ):
                st.caption("Permissive mode continues the run, reports invalid rows, and drops them from the output dataset.")
            else:
                st.caption("Permissive mode continues the run and reports invalid rows while keeping them in the output dataset.")
        st.checkbox(
            "Ignore input directory structure",
            key="gui_allow_shared_output_dir",
        )
        if bool(st.session_state.get("gui_allow_shared_output_dir", True)):
            st.caption(
                "Ignoring input directory structure flattens YOLO exports into `images/` and `labels/`. "
                "Flattened filenames retain source path context to avoid collisions."
            )
        st.checkbox(
            "Prefix exported filenames with input directory name",
            key="gui_prefix_output_filenames",
            disabled=not bool(st.session_state.get("gui_allow_shared_output_dir", True)),
        )
        if bool(st.session_state.get("gui_allow_shared_output_dir", True)):
            if bool(st.session_state.get("gui_prefix_output_filenames", False)):
                input_dir_raw = _coerce_text(st.session_state.get("gui_input_dir"))
                prefix_preview = derive_output_filename_prefix(
                    Path(input_dir_raw).expanduser() if input_dir_raw else Path("dataset")
                )
                st.caption(f"Exported filename prefix preview: `{prefix_preview}`")
            else:
                st.caption(
                    "Without the extra input-directory prefix, flattened filenames keep only source "
                    "path context, for example `train_img_example.jpg`."
                )
        stem_affix_cols = st.columns(2, gap="small")
        with stem_affix_cols[0]:
            st.text_input(
                "Extra file stem prefix",
                key="gui_output_file_stem_prefix",
                placeholder="batchA_",
            )
        with stem_affix_cols[1]:
            st.text_input(
                "Extra file stem suffix",
                key="gui_output_file_stem_suffix",
                placeholder="_fold1",
            )
        output_file_stem_prefix_preview = sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_prefix"))
        )
        output_file_stem_suffix_preview = sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_suffix"))
        )
        if output_file_stem_prefix_preview or output_file_stem_suffix_preview:
            st.caption(
                "Extra stem affix preview: "
                f"`{output_file_stem_prefix_preview}example{output_file_stem_suffix_preview}.jpg`"
            )
        st.selectbox(
            "Out-of-frame bbox policy",
            [p.value for p in OutOfFrameBBoxPolicy],
            key="gui_out_of_frame_bbox_policy",
            format_func=lambda v: {
                "correct": "Correct (clip to image bounds)",
                "warn": "Warn (clip + emit warning)",
                "ignore": "Ignore (keep as-is)",
                "drop": "Drop (remove annotation)",
            }.get(v, v),
        )
        st.number_input(
            "Out-of-frame correction tolerance (px)",
            key="gui_out_of_frame_tolerance_px",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            disabled=_coerce_text(
                st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)
            ) not in {"correct", "warn"},
        )
        if _coerce_text(st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)) in {"correct", "warn"}:
            tolerance_preview = _coerce_out_of_frame_tolerance_px(
                st.session_state.get("gui_out_of_frame_tolerance_px"),
                out_of_frame_bbox_policy=_coerce_text(
                    st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)
                ) or DEFAULT_OUT_OF_FRAME_BBOX_POLICY,
            )
            st.caption(
                f"Near-edge boxes are clipped to image bounds when they exceed the frame by at most `{tolerance_preview:g}` px."
            )
        elif _coerce_text(st.session_state.get("gui_out_of_frame_bbox_policy", DEFAULT_OUT_OF_FRAME_BBOX_POLICY)) == "drop":
            st.caption("Out-of-frame boxes are dropped.")
        else:
            st.caption("Out-of-frame boxes are kept as-is (no clipping or error).")

        st.number_input(
            "Drop images whose longest edge is smaller than (px)",
            key="gui_min_image_longest_edge_px",
            min_value=0,
            step=1,
            on_change=_mark_preview_skip_for_output_only_change,
        )
        st.number_input(
            "Drop/downscale images whose longest edge is larger than (px)",
            key="gui_max_image_longest_edge_px",
            min_value=0,
            step=1,
            on_change=_mark_preview_skip_for_output_only_change,
        )
        st.selectbox(
            "Too-big image action",
            OVERSIZE_IMAGE_ACTIONS,
            key="gui_oversize_image_action",
            format_func=_format_oversize_image_action_label,
            disabled=int(st.session_state.get("gui_max_image_longest_edge_px", 0)) <= 0,
            on_change=_mark_preview_skip_for_output_only_change,
        )
        st.caption(
            "The size gate uses each image's longest edge in pixels. `0` disables a threshold. "
            "Too-big images can be dropped or downscaled."
        )

    missing_label_signature = None
    if preview_source_format == "yolo" and input_dir_raw and input_path.exists() and input_path.is_dir():
        missing_label_signature = (
            f"{input_path.resolve()}::{input_path_include_substring or ''}::{input_path_exclude_substring or ''}"
        )
    if st.session_state.get("gui_missing_label_source_signature") != missing_label_signature:
        st.session_state["gui_missing_label_result"] = None
        st.session_state["gui_missing_label_error"] = None
        st.session_state["gui_detector_review_generation_error"] = None
        st.session_state["gui_missing_label_review_index"] = 0
        st.session_state["gui_missing_label_last_approval"] = None
        st.session_state["gui_missing_label_apply_confirm"] = False
        st.session_state["gui_missing_label_source_signature"] = missing_label_signature

    bbox_audit_signature = missing_label_signature
    if st.session_state.get("gui_bbox_audit_source_signature") != bbox_audit_signature:
        st.session_state["gui_bbox_audit_result"] = None
        st.session_state["gui_bbox_audit_error"] = None
        st.session_state["gui_bbox_audit_review_index"] = 0
        st.session_state["gui_bbox_audit_last_approval"] = None
        st.session_state["gui_bbox_audit_apply_confirm"] = False
        st.session_state["gui_bbox_audit_source_signature"] = bbox_audit_signature

    detector_review_signature = missing_label_signature
    if st.session_state.get("gui_detector_review_source_signature") != detector_review_signature:
        st.session_state["gui_detector_review_result"] = None
        st.session_state["gui_detector_review_item_cache"] = {}
        st.session_state["gui_detector_review_cache_signature"] = ""
        st.session_state["gui_detector_review_image_paths"] = []
        st.session_state["gui_detector_review_image_paths_signature"] = ""
        st.session_state["gui_detector_review_error"] = None
        st.session_state["gui_detector_review_generation_error"] = None
        st.session_state["gui_detector_review_review_index"] = 0
        st.session_state["gui_detector_review_last_approval"] = None
        st.session_state["gui_detector_review_apply_confirm"] = False
        st.session_state["gui_detector_review_allow_overwrite_missing"] = False
        st.session_state["gui_detector_review_source_signature"] = detector_review_signature

    with tabs[3]:
        st.subheader("Step 4: BBox Review")
        st.caption(
            "Run detector-assisted review after previewing the dataset and choosing output-related staging locations."
        )

        if missing_label_signature is None:
            st.info("Detector review is currently available for YOLO input datasets after Step 2 preview is ready.")
        else:
            detector_review_generation_error = st.session_state.get("gui_detector_review_generation_error")
            if isinstance(detector_review_generation_error, str) and detector_review_generation_error:
                st.error(detector_review_generation_error)

            st.subheader("YOLO BBox Review")
            st.caption(
                "Configure one shared detector pass, then review missing-box additions and existing-box edits in one combined list."
            )

            detector_cols = st.columns([4, 1], gap="small")
            with detector_cols[0]:
                st.text_input(
                    "Detector model path",
                    key="gui_missing_label_detector_model_path",
                    placeholder="/path/to/model.pt",
                )
            with detector_cols[1]:
                st.button(
                    "Browse...",
                    key="gui_missing_label_detector_model_path_browse",
                    on_click=_on_browse_missing_label_detector_model_path,
                )

            detector_model_browse_message = st.session_state.get(
                "gui_missing_label_detector_model_browse_message"
            )
            if isinstance(detector_model_browse_message, str) and detector_model_browse_message:
                if bool(st.session_state.get("gui_missing_label_detector_model_browse_available", True)):
                    st.info(detector_model_browse_message)
                else:
                    st.warning(detector_model_browse_message)

            default_hints_output_dir = _default_missing_label_hints_output_dir(
                _coerce_text(st.session_state.get("gui_output_dir"))
            )
            hints_output_cols = st.columns([4, 1], gap="small")
            with hints_output_cols[0]:
                st.text_input(
                    "Hints staging directory",
                    key="gui_missing_label_hints_output_dir",
                    placeholder=str(default_hints_output_dir),
                )
            with hints_output_cols[1]:
                st.button(
                    "Browse...",
                    key="gui_missing_label_hints_output_dir_browse",
                    on_click=_on_browse_missing_label_hints_output_dir,
                )

            hints_output_browse_message = st.session_state.get("gui_missing_label_hints_output_browse_message")
            if isinstance(hints_output_browse_message, str) and hints_output_browse_message:
                if bool(st.session_state.get("gui_missing_label_hints_output_browse_available", True)):
                    st.info(hints_output_browse_message)
                else:
                    st.warning(hints_output_browse_message)

            st.caption(
                "Missing-label proposals are staged outside the dataset first. Blank uses "
                f"`{default_hints_output_dir}`."
            )

            detector_settings = st.columns(3, gap="small")
            with detector_settings[0]:
                st.number_input(
                    "Confidence threshold",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f",
                    key="gui_missing_label_confidence_threshold",
                )
            with detector_settings[1]:
                st.number_input(
                    "IoU threshold",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f",
                    key="gui_missing_label_iou_threshold",
                )
            with detector_settings[2]:
                st.number_input(
                    "Max detections per image",
                    min_value=1,
                    max_value=5000,
                    step=25,
                    key="gui_missing_label_max_detections_per_image",
                )

            default_bbox_audit_output_dir = _default_bbox_audit_output_dir(
                _coerce_text(st.session_state.get("gui_output_dir"))
            )
            bbox_audit_cols = st.columns([4, 1], gap="small")
            with bbox_audit_cols[0]:
                st.text_input(
                    "Audit report directory",
                    key="gui_bbox_audit_output_dir",
                    placeholder=str(default_bbox_audit_output_dir),
                )
            with bbox_audit_cols[1]:
                st.button(
                    "Browse...",
                    key="gui_bbox_audit_output_dir_browse",
                    on_click=_on_browse_bbox_audit_output_dir,
                )

            bbox_audit_output_browse_message = st.session_state.get("gui_bbox_audit_output_browse_message")
            if isinstance(bbox_audit_output_browse_message, str) and bbox_audit_output_browse_message:
                if bool(st.session_state.get("gui_bbox_audit_output_browse_available", True)):
                    st.info(bbox_audit_output_browse_message)
                else:
                    st.warning(bbox_audit_output_browse_message)

            bbox_audit_settings = st.columns(3, gap="small")
            with bbox_audit_settings[0]:
                st.number_input(
                    "Max images to review",
                    min_value=1,
                    max_value=1_000_000,
                    step=25,
                    key="gui_bbox_audit_max_labeled_images",
                )
            with bbox_audit_settings[1]:
                st.number_input(
                    "Match IoU threshold",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f",
                    key="gui_bbox_audit_match_iou_threshold",
                )
            with bbox_audit_settings[2]:
                st.number_input(
                    "Adjust bbox when matched IoU is below",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    format="%.2f",
                    key="gui_bbox_audit_correction_iou_threshold",
                )

            detector_model_path_text = _coerce_text(
                st.session_state.get("gui_missing_label_detector_model_path")
            )
            review_queue_signature = (
                f"{input_path.resolve()}::{input_path_include_substring or ''}::"
                f"{input_path_exclude_substring or ''}::"
                f"{max(1, int(st.session_state.get('gui_bbox_audit_max_labeled_images', 100)))}"
            )
            if st.session_state.get("gui_detector_review_image_paths_signature") != review_queue_signature:
                try:
                    review_image_paths = list_detector_review_image_paths_view(
                        input_path=input_path,
                        source_format="yolo",
                        input_path_include_substring=input_path_include_substring,
                        input_path_exclude_substring=input_path_exclude_substring,
                    )
                    max_review_images = max(
                        1,
                        int(st.session_state.get("gui_bbox_audit_max_labeled_images", 100)),
                    )
                    st.session_state["gui_detector_review_image_paths"] = review_image_paths[:max_review_images]
                    st.session_state["gui_detector_review_image_paths_signature"] = review_queue_signature
                    st.session_state["gui_detector_review_review_index"] = 0
                    st.session_state["gui_detector_review_generation_error"] = None
                except Exception as exc:
                    st.session_state["gui_detector_review_image_paths"] = []
                    st.session_state["gui_detector_review_image_paths_signature"] = review_queue_signature
                    st.session_state["gui_detector_review_generation_error"] = str(exc)

            detector_review_cache_signature = json.dumps(
                {
                    "model_path": detector_model_path_text,
                    "confidence_threshold": float(
                        st.session_state.get("gui_missing_label_confidence_threshold", 0.25)
                    ),
                    "iou_threshold": float(st.session_state.get("gui_missing_label_iou_threshold", 0.45)),
                    "max_detections_per_image": max(
                        1,
                        int(st.session_state.get("gui_missing_label_max_detections_per_image", 200)),
                    ),
                    "match_iou_threshold": float(
                        st.session_state.get("gui_bbox_audit_match_iou_threshold", 0.30)
                    ),
                    "correction_iou_threshold": float(
                        st.session_state.get("gui_bbox_audit_correction_iou_threshold", 0.85)
                    ),
                },
                sort_keys=True,
            )
            if st.session_state.get("gui_detector_review_cache_signature") != detector_review_cache_signature:
                st.session_state["gui_detector_review_item_cache"] = {}
                st.session_state["gui_detector_review_cache_signature"] = detector_review_cache_signature
                st.session_state["gui_detector_review_error"] = None

            detector_review_error = st.session_state.get("gui_detector_review_error")
            if isinstance(detector_review_error, str) and detector_review_error:
                st.error(detector_review_error)

            review_image_paths = list(st.session_state.get("gui_detector_review_image_paths", []))
            if not review_image_paths:
                st.info(
                    "No detector review images are available for the current input path and filters."
                )
            else:
                max_review_index = len(review_image_paths) - 1
                current_review_index = min(
                    max(int(st.session_state.get("gui_detector_review_review_index", 0)), 0),
                    max_review_index,
                )
                st.session_state["gui_detector_review_review_index"] = current_review_index
                current_image_rel_path = review_image_paths[current_review_index]

                nav_cols = st.columns([1, 2, 2, 1], gap="small")
                with nav_cols[0]:
                    back_clicked = st.button(
                        "Back",
                        disabled=current_review_index == 0,
                        key="gui_detector_review_back",
                    )
                with nav_cols[1]:
                    rerun_current_clicked = st.button(
                        "Re-run current image inference",
                        key="gui_detector_review_rerun_current",
                        width="stretch",
                    )
                with nav_cols[2]:
                    st.markdown(
                        f"**Image {current_review_index + 1} / {len(review_image_paths)}**  \n"
                        f"{current_image_rel_path}"
                    )
                with nav_cols[3]:
                    next_clicked = st.button(
                        "Next",
                        disabled=current_review_index >= max_review_index,
                        key="gui_detector_review_next",
                    )

                if back_clicked:
                    st.session_state["gui_detector_review_review_index"] = max(0, current_review_index - 1)
                    st.rerun()
                if next_clicked:
                    st.session_state["gui_detector_review_review_index"] = min(
                        max_review_index,
                        current_review_index + 1,
                    )
                    st.rerun()

                review_item_cache = dict(st.session_state.get("gui_detector_review_item_cache", {}))
                if rerun_current_clicked:
                    review_item_cache.pop(current_image_rel_path, None)
                    st.session_state["gui_detector_review_item_cache"] = review_item_cache

                current_review_item = review_item_cache.get(current_image_rel_path)
                if current_review_item is None:
                    if not detector_model_path_text:
                        st.info("Provide a detector model path to run inference for the current image.")
                    else:
                        try:
                            with st.spinner("Running detector for the current image..."):
                                current_review_item = generate_detector_review_item_view(
                                    input_path=input_path,
                                    source_format="yolo",
                                    image_rel_path=current_image_rel_path,
                                    detector_model_path=Path(detector_model_path_text).expanduser(),
                                    confidence_threshold=float(
                                        st.session_state.get("gui_missing_label_confidence_threshold", 0.25)
                                    ),
                                    iou_threshold=float(
                                        st.session_state.get("gui_missing_label_iou_threshold", 0.45)
                                    ),
                                    max_detections_per_image=max(
                                        1,
                                        int(
                                            st.session_state.get(
                                                "gui_missing_label_max_detections_per_image",
                                                200,
                                            )
                                        ),
                                    ),
                                    match_iou_threshold=float(
                                        st.session_state.get("gui_bbox_audit_match_iou_threshold", 0.30)
                                    ),
                                    correction_iou_threshold=float(
                                        st.session_state.get("gui_bbox_audit_correction_iou_threshold", 0.85)
                                    ),
                                )
                            review_item_cache[current_image_rel_path] = current_review_item
                            st.session_state["gui_detector_review_item_cache"] = review_item_cache
                            st.session_state["gui_detector_review_error"] = None
                        except Exception as exc:
                            st.session_state["gui_detector_review_error"] = str(exc)
                            current_review_item = None

                queue_metrics = st.columns(4)
                queue_metrics[0].metric("Images in queue", len(review_image_paths))
                queue_metrics[1].metric("Current position", current_review_index + 1)
                queue_metrics[2].metric(
                    "Current proposals",
                    len(current_review_item.proposals) if current_review_item is not None else 0,
                )
                queue_metrics[3].metric(
                    "Mode",
                    (
                        "create label"
                        if current_review_item is not None and current_review_item.source_kind == "missing_label"
                        else "edit label"
                    )
                    if current_review_item is not None
                    else "pending",
                )

                if current_review_item is not None:
                    review_cols = st.columns([3, 2], gap="small")
                    with review_cols[1]:
                        if current_review_item.source_kind == "missing_label":
                            st.caption(f"Will create label file: {current_review_item.label_rel_path}")
                            st.caption("Existing labels: 0")
                        else:
                            st.caption(f"Label file: {current_review_item.label_rel_path}")
                            st.caption(f"Existing labels: {current_review_item.existing_label_count}")

                        bbox_strategy = "detector"
                        has_adjustable_proposals = any(
                            proposal.action == "adjust"
                            and proposal.existing_bbox_xywh_normalized is not None
                            and proposal.proposed_bbox_xywh_normalized is not None
                            for proposal in current_review_item.proposals
                        )
                        if has_adjustable_proposals:
                            bbox_strategy = st.radio(
                                "Adjust proposal bbox choice",
                                options=["detector", "smallest", "closest_bounds"],
                                format_func=lambda value: {
                                    "detector": "Use detector box",
                                    "smallest": "Take smallest box",
                                    "closest_bounds": "Take closest bounds",
                                }[value],
                                key=_detector_review_key(
                                    "bbox_strategy",
                                    current_review_item.label_rel_path,
                                ),
                            )
                            st.caption(
                                "These options apply to adjust proposals only. Add proposals keep the detector box, and remove proposals keep the current annotation visible."
                            )

                        selected_proposals: list[BBoxAuditProposalViewModel] = []
                        proposal_rows: list[dict[str, str]] = []
                        for proposal_index, proposal in enumerate(current_review_item.proposals):
                            apply_proposal = st.checkbox(
                                (
                                    f"Apply {proposal.action} proposal {proposal_index + 1}: "
                                    f"{proposal.class_id}:{proposal.class_name}"
                                ),
                                key=_detector_review_key(
                                    "proposal",
                                    current_review_item.label_rel_path,
                                    proposal_index=proposal_index,
                                ),
                                value=True,
                            )
                            existing_bbox = proposal.existing_bbox_xywh_normalized
                            proposed_bbox = proposal.proposed_bbox_xywh_normalized
                            final_bbox = detector_review_final_bbox_xywh_normalized(
                                proposal,
                                strategy=bbox_strategy,
                            )
                            proposal_rows.append(
                                {
                                    "apply": "yes" if apply_proposal else "no",
                                    "action": proposal.action,
                                    "class_id": str(proposal.class_id),
                                    "class_name": proposal.class_name,
                                    "confidence": (
                                        f"{proposal.confidence:.2f}" if proposal.confidence is not None else ""
                                    ),
                                    "match_iou": (
                                        f"{proposal.match_iou:.2f}" if proposal.match_iou is not None else ""
                                    ),
                                    "annotation_bbox": _format_bbox_text(existing_bbox),
                                    "detector_bbox": _format_bbox_text(proposed_bbox),
                                    "final_bbox": (
                                        _format_bbox_text(final_bbox)
                                        if final_bbox is not None
                                        else "removed"
                                    ),
                                }
                            )
                            if apply_proposal:
                                selected_proposals.append(proposal)

                        effective_selected_proposals = apply_detector_review_bbox_strategy(
                            selected_proposals,
                            strategy=bbox_strategy,
                        )
                        editor_state_view = build_detector_review_editor_state_view(
                            dataset_root=input_path,
                            review_item=current_review_item,
                            selected_proposals=selected_proposals,
                            strategy=bbox_strategy,
                        )
                        editor_class_name_map = {
                            option.class_id: option.class_name for option in editor_state_view.class_options
                        }
                        editor_signature = json.dumps(
                            {
                                "image_rel_path": current_review_item.image_rel_path,
                                "label_rel_path": current_review_item.label_rel_path,
                                "source_kind": current_review_item.source_kind,
                                "bbox_strategy": bbox_strategy,
                                "editable_boxes": [
                                    _editor_box_to_dict(box) for box in editor_state_view.editable_boxes
                                ],
                                "selected_proposals": [
                                    {
                                        "proposal_id": proposal.proposal_id,
                                        "action": proposal.action,
                                        "class_id": proposal.class_id,
                                        "bbox": proposal.proposed_bbox_xywh_normalized,
                                        "existing_index": proposal.existing_label_index,
                                    }
                                    for proposal in effective_selected_proposals
                                ],
                            },
                            sort_keys=True,
                        )
                        editor_signature_key = _detector_review_key(
                            "editor_signature",
                            current_review_item.label_rel_path,
                        )
                        editor_boxes_key = _detector_review_key(
                            "editor_boxes",
                            current_review_item.label_rel_path,
                        )
                        if st.session_state.get(editor_signature_key) != editor_signature:
                            st.session_state[editor_signature_key] = editor_signature
                            st.session_state[editor_boxes_key] = [
                                _editor_box_to_dict(box) for box in editor_state_view.editable_boxes
                            ]

                        edited_boxes = _coerce_editor_boxes(
                            st.session_state.get(editor_boxes_key, []),
                            class_name_map=editor_class_name_map,
                        )
                        edited_box_rows = [
                            {
                                "class_id": str(box.class_id),
                                "class_name": box.class_name,
                                "source": box.source,
                                "final_bbox": _format_bbox_text(box.bbox_xywh_normalized),
                            }
                            for box in edited_boxes
                        ]

                        if proposal_rows:
                            st.caption(
                                f"Selected {len(selected_proposals)} of {len(current_review_item.proposals)} proposal(s) for this image."
                            )
                            if len(effective_selected_proposals) != len(selected_proposals):
                                st.caption(
                                    f"BBox choice leaves {len(effective_selected_proposals)} effective proposal(s) after dropping no-op adjust edits."
                                )
                            st.dataframe(
                                proposal_rows,
                                width="stretch",
                                hide_index=True,
                                height=min(360, max(160, 38 * (len(proposal_rows) + 1))),
                            )
                        else:
                            st.info("No detector proposals were generated for this image.")

                        st.caption(f"Final boxes currently in editor: {len(edited_boxes)}")
                        if edited_box_rows:
                            st.dataframe(
                                edited_box_rows,
                                width="stretch",
                                hide_index=True,
                                height=min(280, max(120, 38 * (len(edited_box_rows) + 1))),
                            )
                        else:
                            st.info(
                                "The editor currently has no final boxes. Use Draw to add a box, or accept to write an empty label file."
                            )

                        st.checkbox(
                            "Allow overwriting label files for reviewed missing-label items",
                            key="gui_detector_review_allow_overwrite_missing",
                        )
                        accept_and_next_requested = st.button(
                            "Accept and go to next",
                            key="gui_detector_review_accept_next",
                            width="stretch",
                        )

                    with review_cols[0]:
                        editor_image_payload, editor_image_warnings = load_bbox_editor_image_payload(
                            dataset_root=input_path,
                            image_rel_path=current_review_item.image_rel_path,
                        )
                        if editor_image_payload is None:
                            for warning in editor_image_warnings:
                                st.warning(warning)
                        else:
                            st.caption(
                                "BBox editor: drag green boxes, use corner handles to resize, switch to Draw to create new boxes, and use the class dropdown for the selected box or the next box you draw."
                            )
                            editor_value = _BBOX_EDITOR_COMPONENT(
                                image_data_url=editor_image_payload.image_data_url,
                                original_width=editor_image_payload.original_width,
                                original_height=editor_image_payload.original_height,
                                display_width=editor_image_payload.display_width,
                                display_height=editor_image_payload.display_height,
                                annotation_boxes=[
                                    _editor_box_to_dict(box) for box in editor_state_view.annotation_boxes
                                ],
                                detector_boxes=[
                                    _editor_box_to_dict(box) for box in editor_state_view.detector_boxes
                                ],
                                editable_boxes=[
                                    _editor_box_to_dict(box) for box in edited_boxes
                                ],
                                class_options=[
                                    {
                                        "class_id": option.class_id,
                                        "class_name": option.class_name,
                                    }
                                    for option in editor_state_view.class_options
                                ],
                                show_annotations=True,
                                show_detectors=True,
                                key=_detector_review_key(
                                    "bbox_editor",
                                    current_review_item.label_rel_path,
                                ),
                            )
                            if isinstance(editor_value, dict) and "boxes" in editor_value:
                                updated_boxes = _coerce_editor_boxes(
                                    editor_value.get("boxes"),
                                    class_name_map=editor_class_name_map,
                                )
                                st.session_state[editor_boxes_key] = [
                                    _editor_box_to_dict(box) for box in updated_boxes
                                ]

                    if accept_and_next_requested:
                        try:
                            approval_vm = approve_detector_review_edited_boxes_view(
                                dataset_root=input_path,
                                review_item=current_review_item,
                                edited_boxes=edited_boxes,
                                allow_overwrite_missing_label_files=bool(
                                    st.session_state.get("gui_detector_review_allow_overwrite_missing")
                                ),
                            )
                            st.session_state["gui_detector_review_last_approval"] = approval_vm
                            st.session_state.pop(editor_signature_key, None)
                            st.session_state.pop(editor_boxes_key, None)
                            review_item_cache.pop(current_image_rel_path, None)
                            st.session_state["gui_detector_review_item_cache"] = review_item_cache
                            st.session_state["gui_detector_review_error"] = None
                            if current_review_index < max_review_index:
                                st.session_state["gui_detector_review_review_index"] = current_review_index + 1
                            st.rerun()
                        except Exception as exc:
                            st.session_state["gui_detector_review_error"] = str(exc)

                    last_approval = st.session_state.get("gui_detector_review_last_approval")
                    if last_approval is not None:
                        st.success(
                            "Approved "
                            f"{getattr(last_approval, 'approved_label_files', 0)} label file(s) "
                            f"with {getattr(last_approval, 'applied_proposals', 0)} final box(es)."
                        )
                        with st.expander("Updated label files", expanded=False):
                            st.dataframe(
                                [
                                    {"label_file": path}
                                    for path in getattr(last_approval, "label_paths", [])
                                ],
                                width="stretch",
                                hide_index=True,
                            )

    with tabs[4]:
        st.subheader("Step 5: Label Mapping")
        st.caption("Define source-to-destination mappings. Set keep/drop to 'drop' to remove a class.")
        class_labels = _class_labels_from_state()
        inference_payload = st.session_state.get("gui_inference_payload")
        inferred_format = None
        if isinstance(inference_payload, dict):
            predicted = inference_payload.get("predicted_format")
            inferred_format = str(predicted) if predicted is not None else None
        mapping_source_format = _resolve_preview_source_format(
            _coerce_text(st.session_state["gui_src"]),
            inferred_format,
        )
        validated_input_path = _coerce_text(st.session_state.get("gui_input_validated_path"))
        mapping_input_path = (
            Path(validated_input_path)
            if validated_input_path
            else Path(_coerce_text(st.session_state["gui_input_dir"])).expanduser()
        )
        st.caption(
            describe_class_label_source(
                input_path=mapping_input_path,
                source_format=mapping_source_format,
                class_labels=class_labels,
            )
        )
        if class_labels:
            st.caption("Source class IDs are read-only and come from the detected dataset classes.")
        else:
            st.caption(
                "Class labels will appear once preview data is available for the selected dataset."
            )

        current_rows = normalize_mapping_rows(st.session_state["gui_mapping_rows"])
        editor_rows = attach_mapping_labels(current_rows, class_labels)
        _sync_mapping_row_widget_state(current_rows)
        with st.container(border=True):
            header_cols = st.columns([0.9, 1.4, 1.0, 1.4], gap="small")
            header_labels = [
                "source id",
                "source label",
                "keep/drop",
                "output class",
            ]
            for column, label in zip(header_cols, header_labels, strict=True):
                column.markdown(f"<div class='lm-mapping-header'>{escape(label)}</div>", unsafe_allow_html=True)

            normalized_rows: list[dict[str, str]] = []
            for row in editor_rows:
                source_class_id = row["source_class_id"]
                action_key = _mapping_widget_key(source_class_id, "action")
                destination_key = _mapping_widget_key(source_class_id, "destination_class_id")

                row_cols = st.columns([0.9, 1.4, 1.0, 1.4], gap="small")
                row_cols[0].markdown(
                    _mapping_display_cell(source_class_id),
                    unsafe_allow_html=True,
                )
                row_cols[1].markdown(
                    _mapping_display_cell(row.get("source_label", "")),
                    unsafe_allow_html=True,
                )

                action_value = row_cols[2].selectbox(
                    "keep/drop",
                    options=MAPPING_ACTIONS,
                    key=action_key,
                    format_func=_format_mapping_action_label,
                    label_visibility="collapsed",
                )
                drop_selected = action_value == "drop"
                row_cols[3].text_input(
                    "output class",
                    key=destination_key,
                    label_visibility="collapsed",
                    disabled=drop_selected,
                )
                destination_class_id = _coerce_text(st.session_state.get(destination_key))

                normalized_rows.append(
                    {
                        "source_class_id": source_class_id,
                        "action": action_value,
                        "destination_class_id": destination_class_id,
                    }
                )

        st.session_state["gui_mapping_rows"] = normalized_rows
        materialized_rows = materialize_mapping_rows(normalized_rows)
        parsed_mappings = parse_mapping_rows(mapping_rows_to_viewmodels(materialized_rows))

        if parsed_mappings.errors:
            for error in parsed_mappings.errors:
                st.error(error)
        else:
            st.success("Mapping rows are valid.")

        if parsed_mappings.class_map:
            st.caption("Normalized class-map preview")
            st.code(
                json.dumps(
                    {
                        "class_map": {
                            str(key): value for key, value in sorted(parsed_mappings.class_map.items())
                        }
                    },
                    indent=2,
                ),
                language="json",
            )
        else:
            st.info("No mappings defined yet. Conversion follows unmapped policy behavior.")

    with tabs[5]:
        st.subheader("Step 6: Review & Run")
        st.caption("Runs are blocked until required fields and mapping rows validate.")
        st.checkbox("Dry run", key="gui_dry_run")
        st.checkbox("Copy images to output", key="gui_copy_images")
        st.checkbox("Allow overwriting existing output files", key="gui_allow_overwrite")

        input_dir_raw = _coerce_text(st.session_state["gui_input_dir"])
        output_dir_raw = _coerce_text(st.session_state["gui_output_dir"])
        src = _coerce_text(st.session_state["gui_src"])
        dst = _coerce_text(st.session_state["gui_dst"])
        selected_custom_format_id = _coerce_text(st.session_state.get("gui_custom_format_id")) or None
        selected_custom_format_input_path = _coerce_text(st.session_state.get("gui_custom_format_path")) or None
        validation_mode = _coerce_text(st.session_state["gui_validation_mode"]) or ValidationMode.STRICT.value
        permissive_invalid_annotation_action = (
            _coerce_text(st.session_state["gui_permissive_invalid_annotation_action"])
            or InvalidAnnotationAction.KEEP.value
        )
        unmapped_policy = _coerce_text(st.session_state["gui_unmapped_policy"])
        allow_shared_output_dir = bool(st.session_state["gui_allow_shared_output_dir"])
        prefix_output_filenames = bool(st.session_state["gui_prefix_output_filenames"])
        output_file_stem_prefix = sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_prefix"))
        ) or None
        output_file_stem_suffix = sanitize_output_file_stem_affix(
            _coerce_text(st.session_state.get("gui_output_file_stem_suffix"))
        ) or None
        out_of_frame_bbox_policy = _coerce_text(st.session_state["gui_out_of_frame_bbox_policy"]) or DEFAULT_OUT_OF_FRAME_BBOX_POLICY
        out_of_frame_tolerance_px = _coerce_out_of_frame_tolerance_px(
            st.session_state["gui_out_of_frame_tolerance_px"],
            out_of_frame_bbox_policy=out_of_frame_bbox_policy,
        )
        min_image_longest_edge_px = max(
            0,
            int(_coerce_float(st.session_state["gui_min_image_longest_edge_px"], DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX)),
        )
        max_image_longest_edge_px = max(
            0,
            int(_coerce_float(st.session_state["gui_max_image_longest_edge_px"], DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX)),
        )
        oversize_image_action = _coerce_text(st.session_state["gui_oversize_image_action"]) or "ignore"
        dry_run = bool(st.session_state["gui_dry_run"])
        copy_images = bool(st.session_state["gui_copy_images"])
        allow_overwrite = bool(st.session_state["gui_allow_overwrite"])
        input_path_include_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_include_substring"))
        )
        input_path_exclude_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_exclude_substring"))
        )
        output_filename_prefix = (
            derive_output_filename_prefix(Path(input_dir_raw).expanduser())
            if allow_shared_output_dir and prefix_output_filenames and input_dir_raw
            else None
        )

        review_rows = materialize_mapping_rows(st.session_state["gui_mapping_rows"])
        parsed_mappings = parse_mapping_rows(mapping_rows_to_viewmodels(review_rows))

        blocking_errors = run_blocking_errors(
            input_dir_raw=input_dir_raw,
            output_dir_raw=output_dir_raw,
            src=src,
            dst=dst,
            mapping_errors=parsed_mappings.errors,
            dry_run=dry_run,
            copy_images=copy_images,
            min_image_longest_edge_px=min_image_longest_edge_px,
            max_image_longest_edge_px=max_image_longest_edge_px,
            oversize_image_action=oversize_image_action,
        )
        resolved_custom_format_id: str | None = selected_custom_format_id
        selected_custom_format_path: Path | None = None
        if src == "custom" and input_dir_raw:
            custom_input_path = Path(input_dir_raw).expanduser()
            if custom_input_path.exists() and custom_input_path.is_dir():
                try:
                    available_custom_options = _custom_format_options(custom_input_path)
                except Exception as exc:
                    blocking_errors.append(f"Unable to load custom format YAMLs: {exc}")
                else:
                    if not available_custom_options:
                        explicit_option, explicit_error = _explicit_custom_format_option(selected_custom_format_input_path)
                        if explicit_error:
                            blocking_errors.append(explicit_error)
                        elif explicit_option is None:
                            blocking_errors.append(
                                "No custom format YAML found. Add one at the dataset root as custom_format.yaml, "
                                "data_format.yaml, or label_format.yaml, keep one under format_specs, or choose one manually."
                            )
                        else:
                            resolved_custom_format_id = explicit_option.format_id
                            selected_custom_format_path = explicit_option.path
                    else:
                        selected_option = _selected_custom_format_option(
                            available_custom_options,
                            resolved_custom_format_id,
                        )
                        if selected_option is None:
                            blocking_errors.append("Choose a custom format YAML before running conversion.")
                        else:
                            resolved_custom_format_id = selected_option.format_id
                            selected_custom_format_path = selected_option.path

        status = _coerce_text(st.session_state["gui_run_status"]) or "idle"
        progress = int(st.session_state["gui_run_progress"])
        run_detail = _coerce_text(st.session_state.get("gui_run_detail"))
        run_status_placeholder = st.empty()
        run_progress_placeholder = st.empty()
        run_detail_placeholder = st.empty()

        def render_run_progress(status_value: str, progress_value: int, detail_value: str | None) -> None:
            run_status_placeholder.markdown(f"**Run status:** `{status_value}`")
            run_progress_placeholder.progress(min(max(progress_value, 0), 100) / 100.0)
            detail_text = _coerce_text(detail_value)
            if detail_text:
                run_detail_placeholder.caption(detail_text)
            else:
                run_detail_placeholder.empty()

        render_run_progress(status, progress, run_detail)

        run_error = st.session_state.get("gui_run_error")
        run_error_details = st.session_state.get("gui_run_error_details")
        run_error_issue_rows = st.session_state.get("gui_run_error_issue_rows")
        if isinstance(run_error, str) and run_error:
            st.error(run_error)
            if (
                (isinstance(run_error_details, list) and run_error_details)
                or (isinstance(run_error_issue_rows, list) and run_error_issue_rows)
            ):
                with st.expander("Failure details", expanded=True):
                    if isinstance(run_error_issue_rows, list) and run_error_issue_rows:
                        st.dataframe(
                            run_error_issue_rows,
                            width="stretch",
                            hide_index=True,
                            height=min(260, max(120, 38 * (len(run_error_issue_rows) + 1))),
                        )
                    if isinstance(run_error_details, list) and run_error_details:
                        for detail in run_error_details:
                            st.write(detail)

        if bool(st.session_state.get("gui_run_interrupted_notice")):
            st.warning("Previous run was interrupted. Run state was reset so you can run conversion again.")
            st.session_state["gui_run_interrupted_notice"] = False

        with st.expander("Run request (JSON)", expanded=False):
            st.json(
                {
                    "input_dir": input_dir_raw,
                    "output_dir": output_dir_raw,
                    "src": src,
                    "custom_format_id": resolved_custom_format_id,
                    "custom_format_yaml": str(selected_custom_format_path) if selected_custom_format_path else None,
                    "dst": dst,
                    "validation_mode": validation_mode,
                    "unmapped_policy": unmapped_policy,
                    "allow_shared_output_dir": allow_shared_output_dir,
                    "prefix_output_filenames": prefix_output_filenames,
                    "output_filename_prefix": output_filename_prefix,
                    "output_file_stem_prefix": output_file_stem_prefix,
                    "output_file_stem_suffix": output_file_stem_suffix,
                    "out_of_frame_bbox_policy": out_of_frame_bbox_policy,
                    "out_of_frame_tolerance_px": out_of_frame_tolerance_px,
                    "min_image_longest_edge_px": min_image_longest_edge_px,
                    "max_image_longest_edge_px": max_image_longest_edge_px,
                    "oversize_image_action": oversize_image_action,
                    "dry_run": dry_run,
                    "copy_images": copy_images,
                    "allow_overwrite": allow_overwrite,
                    "input_path_include_substring": input_path_include_substring,
                    "input_path_exclude_substring": input_path_exclude_substring,
                    "mapping_rows": review_rows,
                    "parsed_class_map_size": len(parsed_mappings.class_map),
                }
            )

        if blocking_errors:
            for error in blocking_errors:
                st.error(error)

        reset_clicked = st.button(
            "Reset run state",
            disabled=status == "idle",
            help="Use this after an interrupted run (for example after pressing Stop in the app toolbar).",
        )
        if reset_clicked:
            status, progress = reset_gui_run_state(
                st.session_state,
                detail="Run state reset.",
                interrupted=False,
            )
            run_detail = _coerce_text(st.session_state.get("gui_run_detail"))
            render_run_progress(status, progress, run_detail)

        run_disabled = bool(blocking_errors) or status == "running"
        if st.button("Run conversion", disabled=run_disabled, type="primary"):
            started_status, started_progress = transition_run_state(status, "start")
            st.session_state["gui_run_status"] = started_status
            st.session_state["gui_run_progress"] = started_progress
            st.session_state["gui_run_error"] = None
            st.session_state["gui_run_error_details"] = []
            st.session_state["gui_run_error_issue_rows"] = []
            st.session_state["gui_run_detail"] = "Starting conversion..."
            st.session_state["gui_run_interrupted_notice"] = False
            st.session_state["gui_output_action_message"] = None
            render_run_progress(started_status, started_progress, "Starting conversion...")

            input_path = Path(input_dir_raw).expanduser()
            output_path = Path(output_dir_raw).expanduser()
            pending_map_path: Path | None = None
            final_map_path: Path | None = None

            def on_progress_update(message: str, percent: int) -> None:
                normalized_progress = min(max(int(percent), 0), 100)
                detail = _coerce_text(message)
                st.session_state["gui_run_status"] = "running"
                st.session_state["gui_run_progress"] = normalized_progress
                st.session_state["gui_run_detail"] = detail
                render_run_progress("running", normalized_progress, detail)

            try:
                if parsed_mappings.class_map:
                    pending_map_path = persist_generated_class_map(
                        parsed_mappings.class_map,
                        run_id=f"pending-{uuid4().hex}",
                        reports_dir=output_path,
                    )

                convert_vm, result = convert_view(
                    input_path=input_path,
                    output_path=output_path,
                    src=src,
                    dst=dst,
                    custom_format_id=resolved_custom_format_id,
                    custom_format_path=selected_custom_format_path,
                    map_path=pending_map_path,
                    unmapped_policy=unmapped_policy,
                    dry_run=dry_run,
                    validation_mode=validation_mode,
                    permissive_invalid_annotation_action=permissive_invalid_annotation_action,
                    copy_images=copy_images,
                    allow_overwrite=allow_overwrite,
                    input_path_include_substring=input_path_include_substring,
                    input_path_exclude_substring=input_path_exclude_substring,
                    output_file_name_prefix=output_filename_prefix,
                    output_file_stem_prefix=output_file_stem_prefix,
                    output_file_stem_suffix=output_file_stem_suffix,
                    flatten_output_layout=allow_shared_output_dir,
                    drop_frames_with_class_ids=parsed_mappings.drop_frames_with_class_ids,
                    out_of_frame_bbox_policy=out_of_frame_bbox_policy,
                    out_of_frame_tolerance_px=out_of_frame_tolerance_px,
                    min_image_longest_edge_px=min_image_longest_edge_px,
                    max_image_longest_edge_px=max_image_longest_edge_px,
                    oversize_image_action=oversize_image_action,
                    progress_callback=on_progress_update,
                )

                if pending_map_path:
                    final_map_path = finalize_generated_class_map(
                        pending_map_path,
                        run_id=convert_vm.run_id,
                        reports_dir=output_path,
                    )

                config = build_gui_run_config(
                    run_id=convert_vm.run_id,
                    input_path=input_path,
                    output_path=output_path,
                    src=src,
                    dst=dst,
                    custom_format_id=resolved_custom_format_id,
                    custom_format_path=selected_custom_format_path,
                    map_path=final_map_path,
                    unmapped_policy=unmapped_policy,
                    dry_run=dry_run,
                    allow_overwrite=allow_overwrite,
                    input_path_include_substring=input_path_include_substring,
                    input_path_exclude_substring=input_path_exclude_substring,
                    validation_mode=validation_mode,
                    permissive_invalid_annotation_action=permissive_invalid_annotation_action,
                    out_of_frame_bbox_policy=out_of_frame_bbox_policy,
                    out_of_frame_tolerance_px=out_of_frame_tolerance_px,
                    min_image_longest_edge_px=min_image_longest_edge_px,
                    max_image_longest_edge_px=max_image_longest_edge_px,
                    oversize_image_action=oversize_image_action,
                )
                config_path = export_run_config(
                    config.model_dump(mode="json"),
                    output_path / f"{convert_vm.run_id}.gui.config.json",
                )
                report_payload = result.report.model_dump(mode="json")
                report_path = export_json_artifact(
                    report_payload,
                    output_path / f"{convert_vm.run_id}.gui.report.json",
                )
                warnings_payload = build_run_warnings_payload(
                    report_payload,
                    dropped_annotations=result.dropped_annotations,
                )
                warnings_path = (
                    export_json_artifact(
                        warnings_payload,
                        output_path / f"{convert_vm.run_id}.gui.warnings.json",
                    )
                    if int(warnings_payload.get("warning_count", 0)) > 0
                    else None
                )
                dropped_annotations_payload = {
                    "run_id": convert_vm.run_id,
                    "dropped_annotation_count": len(result.dropped_annotations),
                    "dropped_annotations": [
                        item.model_dump(mode="json") for item in result.dropped_annotations
                    ],
                }
                dropped_annotations_path = (
                    export_json_artifact(
                        dropped_annotations_payload,
                        output_path / f"{convert_vm.run_id}.gui.dropped_annotations.json",
                    )
                    if dropped_annotations_payload["dropped_annotation_count"] > 0
                    else None
                )

                st.session_state["gui_last_run"] = {
                    "run_id": convert_vm.run_id,
                    "report": report_payload,
                    "annotation_distribution_rows": build_annotation_distribution_rows(
                        result.output_dataset
                    ),
                    "config_path": str(config_path),
                    "report_path": str(report_path),
                    "warnings_path": str(warnings_path) if warnings_path is not None else None,
                    "dropped_annotations_path": (
                        str(dropped_annotations_path) if dropped_annotations_path is not None else None
                    ),
                    "mapping_path": str(final_map_path) if final_map_path else None,
                    "output_path": str(output_path.resolve()),
                    "custom_format_id": resolved_custom_format_id,
                    "custom_format_yaml": str(selected_custom_format_path) if selected_custom_format_path else None,
                    "output_filename_prefix": output_filename_prefix,
                    "output_file_stem_prefix": output_file_stem_prefix,
                    "output_file_stem_suffix": output_file_stem_suffix,
                    "allow_overwrite": allow_overwrite,
                    "input_path_include_substring": input_path_include_substring,
                    "input_path_exclude_substring": input_path_exclude_substring,
                }
                completed_status, completed_progress = transition_run_state(started_status, "complete")
                st.session_state["gui_run_status"] = completed_status
                st.session_state["gui_run_progress"] = completed_progress
                st.session_state["gui_run_error"] = None
                st.session_state["gui_run_error_details"] = []
                st.session_state["gui_run_error_issue_rows"] = []
                st.session_state["gui_run_detail"] = "Conversion complete."
                render_run_progress(completed_status, completed_progress, "Conversion complete.")
            except Exception as exc:
                if is_streamlit_control_flow_exception(exc):
                    interrupted_status, interrupted_progress = reset_gui_run_state(
                        st.session_state,
                        detail=RUN_INTERRUPTED_DETAIL,
                        interrupted=True,
                    )
                    render_run_progress(interrupted_status, interrupted_progress, RUN_INTERRUPTED_DETAIL)
                    raise

                error_summary, error_details, error_issue_rows = format_run_exception_details(exc)
                failed_status, failed_progress = transition_run_state(started_status, "fail")
                st.session_state["gui_run_status"] = failed_status
                st.session_state["gui_run_progress"] = failed_progress
                st.session_state["gui_run_error"] = error_summary
                st.session_state["gui_run_error_details"] = error_details
                st.session_state["gui_run_error_issue_rows"] = error_issue_rows
                st.session_state["gui_run_detail"] = error_summary
                render_run_progress(failed_status, failed_progress, error_summary)
                st.error(error_summary)
                if error_issue_rows or error_details:
                    with st.expander("Failure details", expanded=True):
                        if error_issue_rows:
                            st.dataframe(
                                error_issue_rows,
                                width="stretch",
                                hide_index=True,
                                height=min(260, max(120, 38 * (len(error_issue_rows) + 1))),
                            )
                        for detail in error_details:
                            st.write(detail)
            except BaseException:
                interrupted_status, interrupted_progress = reset_gui_run_state(
                    st.session_state,
                    detail=RUN_INTERRUPTED_DETAIL,
                    interrupted=True,
                )
                render_run_progress(interrupted_status, interrupted_progress, RUN_INTERRUPTED_DETAIL)
                raise

        last_run = st.session_state["gui_last_run"]
        if isinstance(last_run, dict):
            if _coerce_text(st.session_state.get("gui_run_status")) == "completed":
                st.success(f"Run {last_run['run_id']} complete")
            metrics = build_run_summary_metrics(last_run["report"])
            metric_cols = st.columns(4)
            metric_cols[0].metric("Processed images", metrics.images_processed)
            metric_cols[1].metric("Converted labels", metrics.annotations_converted)
            metric_cols[2].metric("Warnings", metrics.warning_count)
            metric_cols[3].metric("Errors", metrics.error_count)

            warning_messages = extract_run_warning_messages(last_run["report"])
            if warning_messages:
                st.caption("Warnings")
                for severity, message in warning_messages:
                    if severity == "error":
                        st.error(message)
                    elif severity == "info":
                        st.info(message)
                    else:
                        st.warning(message)

            annotation_rows = last_run.get("annotation_distribution_rows")
            if isinstance(annotation_rows, list) and annotation_rows:
                chart_cols = st.columns(2, gap="medium")
                with chart_cols[0]:
                    st.caption("Class occurrences")
                    st.vega_lite_chart(
                        class_occurrence_chart_spec(annotation_rows),
                        width="stretch",
                    )
                with chart_cols[1]:
                    st.caption("BBox size histogram")
                    st.vega_lite_chart(
                        bbox_size_histogram_spec(annotation_rows),
                        width="stretch",
                    )

            with st.expander("Run report (JSON)", expanded=False):
                st.json(last_run["report"])
            st.code(
                json.dumps(
                    {
                        "config_export": last_run["config_path"],
                        "report_export": last_run.get("report_path"),
                        "warnings_export": last_run.get("warnings_path"),
                        "dropped_annotations_export": last_run.get("dropped_annotations_path"),
                        "mapping_file": last_run["mapping_path"],
                    },
                    indent=2,
                ),
                language="json",
            )

            open_button_key = f"gui_open_output_{last_run['run_id']}"
            if st.button("Open output directory", key=open_button_key):
                open_result = attempt_output_directory_access(_coerce_text(last_run.get("output_path")))
                st.session_state["gui_output_action_message"] = open_result.message

            output_action_message = st.session_state.get("gui_output_action_message")
            if isinstance(output_action_message, str) and output_action_message:
                if output_action_message.startswith("Opened output directory"):
                    st.success(output_action_message)
                else:
                    st.info(output_action_message)

            st.caption(f"Output directory: {last_run.get('output_path')}")

    with tabs[6]:
        st.subheader("Step 7: Classification")
        st.caption(
            "Assign a class to each image with a single keystroke. Class names and keys come from a "
            "classification config file. Labels are saved to a JSON manifest and never modify "
            "bounding-box label files, so this works alongside bbox labels or on bare image folders."
        )
        classification_config_cols = st.columns([4, 1], gap="small")
        with classification_config_cols[0]:
            st.text_input(
                "Classification config (YAML or JSON)",
                key="gui_classification_config_path",
                placeholder="Blank: use classification.yaml from the dataset root",
            )
        with classification_config_cols[1]:
            st.button(
                "Browse...",
                key="gui_classification_config_browse",
                on_click=_on_browse_classification_config_path,
            )
        classification_browse_message = st.session_state.get("gui_classification_config_browse_message")
        if isinstance(classification_browse_message, str) and classification_browse_message:
            if bool(st.session_state.get("gui_classification_config_browse_available", True)):
                st.info(classification_browse_message)
            else:
                st.warning(classification_browse_message)

        classification_dataset_ready = bool(input_dir_raw) and input_path.exists() and input_path.is_dir()
        classification_config: ClassificationConfig | None = None
        classification_config_candidate = resolve_classification_config_candidate(
            _coerce_text(st.session_state.get("gui_classification_config_path")),
            dataset_root=input_path if classification_dataset_ready else None,
        )
        if not classification_dataset_ready:
            st.info("Set a valid input directory in Step 1 to start classifying images.")
        elif classification_config_candidate is None:
            st.info(
                "No classification config found. Add `classification.yaml` to the dataset root "
                "or enter a config path above."
            )
        else:
            try:
                classification_config = load_classification_config(classification_config_candidate)
            except Exception as exc:
                st.error(f"Unable to load classification config: {exc}")

        with st.expander("Example classification config", expanded=classification_config is None):
            st.code(CLASSIFICATION_CONFIG_EXAMPLE, language="yaml")

        if classification_config is not None and classification_dataset_ready:
            st.caption(f"Config: {classification_config.source_path}")
            classification_labels_path = resolve_classification_labels_path(input_path, classification_config)

            classification_options = st.columns([1.2, 1.2, 1], gap="small")
            with classification_options[0]:
                st.checkbox(
                    "Advance to next image after labeling",
                    key="gui_classification_advance_on_label",
                    disabled=classification_config.multi_label,
                )
            with classification_options[1]:
                st.checkbox(
                    "Show bounding boxes from Step 2 preview",
                    key="gui_classification_show_bboxes",
                )
            with classification_options[2]:
                rescan_requested = st.button("Rescan images", key="gui_classification_rescan")

            classification_signature = _classification_dataset_signature(
                input_path,
                include_substring=input_path_include_substring,
                exclude_substring=input_path_exclude_substring,
            )
            if (
                rescan_requested
                or st.session_state.get("gui_classification_image_paths_signature") != classification_signature
            ):
                try:
                    st.session_state["gui_classification_image_paths"] = discover_classification_image_paths(
                        input_path,
                        input_path_include_substring=input_path_include_substring,
                        input_path_exclude_substring=input_path_exclude_substring,
                    )
                    st.session_state["gui_classification_error"] = None
                except Exception as exc:
                    st.session_state["gui_classification_image_paths"] = []
                    st.session_state["gui_classification_error"] = f"Unable to scan images: {exc}"
                st.session_state["gui_classification_image_paths_signature"] = classification_signature
                st.session_state["gui_classification_index"] = 0

            classification_image_paths: list[str] = list(st.session_state.get("gui_classification_image_paths", []))
            classification_labels: dict[str, list[str]] | None
            try:
                classification_labels = load_classification_labels(classification_labels_path)
            except Exception as exc:
                classification_labels = None
                st.error(f"Unable to read classification labels {classification_labels_path}: {exc}")

            classification_error = st.session_state.get("gui_classification_error")
            if isinstance(classification_error, str) and classification_error:
                st.error(classification_error)

            if not classification_image_paths:
                st.info("No images found in the input directory (after include/exclude filters).")
                _classification_keyboard_action(
                    enabled=False,
                    previous_disabled=True,
                    next_disabled=True,
                    class_keys=[],
                )
            elif classification_labels is not None:
                classification_max_index = len(classification_image_paths) - 1
                classification_index = int(st.session_state.get("gui_classification_index", 0))
                classification_index = min(max(classification_index, 0), classification_max_index)
                keyboard_action, keyboard_key = _classification_keyboard_action(
                    enabled=True,
                    previous_disabled=classification_index == 0,
                    next_disabled=classification_index == classification_max_index,
                    class_keys=classification_config.keys,
                )

                classification_nav = st.columns([1, 1, 1.4, 1.2, 2], gap="small")
                with classification_nav[0]:
                    classification_previous_clicked = st.button(
                        "Previous",
                        key="gui_classification_prev",
                        disabled=classification_index == 0,
                    )
                with classification_nav[1]:
                    classification_next_clicked = st.button(
                        "Next",
                        key="gui_classification_next",
                        disabled=classification_index == classification_max_index,
                    )
                with classification_nav[2]:
                    classification_next_unlabeled_clicked = st.button(
                        "Next unlabeled",
                        key="gui_classification_next_unlabeled",
                    )
                with classification_nav[3]:
                    classification_clear_clicked = st.button(
                        "Clear label",
                        key="gui_classification_clear",
                    )

                current_rel_path = classification_image_paths[classification_index]
                labels_changed = False
                if keyboard_action == "class":
                    keyed_class = classification_config.class_for_key(keyboard_key)
                    if keyed_class is not None:
                        classification_labels = apply_classification_label(
                            classification_labels,
                            current_rel_path,
                            keyed_class.name,
                            multi_label=classification_config.multi_label,
                        )
                        labels_changed = True
                        classification_index = classification_index_after_label(
                            classification_index,
                            max_index=classification_max_index,
                            advance_on_label=bool(
                                st.session_state.get("gui_classification_advance_on_label", True)
                            ),
                            multi_label=classification_config.multi_label,
                        )
                elif keyboard_action == "clear" or classification_clear_clicked:
                    if image_labels(classification_labels, current_rel_path):
                        classification_labels = clear_classification_label(
                            classification_labels, current_rel_path
                        )
                        labels_changed = True

                if labels_changed:
                    try:
                        save_classification_labels(
                            classification_labels_path,
                            classification_labels,
                            config=classification_config,
                        )
                        st.session_state["gui_classification_last_saved"] = (
                            f"Saved {current_rel_path} -> "
                            f"{', '.join(image_labels(classification_labels, current_rel_path)) or '(no label)'}"
                        )
                        st.session_state["gui_classification_error"] = None
                    except Exception as exc:
                        st.session_state["gui_classification_error"] = (
                            f"Unable to save classification labels: {exc}"
                        )
                        st.error(st.session_state["gui_classification_error"])

                classification_index = _resolve_preview_index(
                    classification_index,
                    max_index=classification_max_index,
                    keyboard_action=keyboard_action,
                    previous_clicked=classification_previous_clicked,
                    next_clicked=classification_next_clicked,
                )
                if classification_next_unlabeled_clicked:
                    unlabeled_index = next_unlabeled_index(
                        classification_image_paths,
                        classification_labels,
                        start_index=classification_index,
                    )
                    if unlabeled_index is None:
                        st.info("Every image already has a label.")
                    else:
                        classification_index = unlabeled_index
                st.session_state["gui_classification_index"] = classification_index

                with classification_nav[4]:
                    st.markdown(
                        f"**Image {classification_index + 1} / {len(classification_image_paths)}**"
                    )

                current_rel_path = classification_image_paths[classification_index]
                current_label_names = image_labels(classification_labels, current_rel_path)
                classification_summary = summarize_classification_labels(
                    classification_image_paths,
                    classification_labels,
                    config=classification_config,
                )

                preview_bboxes_by_file: dict[str, list[OverlayBBox]] = {}
                if preview_vm is not None and bool(st.session_state.get("gui_classification_show_bboxes", True)):
                    for preview_image in preview_vm.images:
                        preview_bboxes_by_file[str(preview_image.file_name).replace("\\", "/")] = [
                            (
                                bbox.bbox_xywh_abs[0],
                                bbox.bbox_xywh_abs[1],
                                bbox.bbox_xywh_abs[2],
                                bbox.bbox_xywh_abs[3],
                                f"{bbox.class_id}:{bbox.class_name}",
                            )
                            for bbox in preview_image.bboxes
                        ]

                classification_cols = st.columns([3, 2], gap="small")
                with classification_cols[0]:
                    st.caption(current_rel_path)
                    classification_overlay, classification_overlay_warnings = render_preview_overlay(
                        dataset_root=input_path,
                        image_rel_path=current_rel_path,
                        bboxes=preview_bboxes_by_file.get(current_rel_path, []),
                    )
                    if classification_overlay is None:
                        for warning in classification_overlay_warnings:
                            st.warning(warning)
                    else:
                        st.image(classification_overlay, width="stretch")

                with classification_cols[1]:
                    if current_label_names:
                        st.markdown(f"**Current label:** {escape(', '.join(current_label_names))}")
                    else:
                        st.markdown("**Current label:** _unlabeled_")
                    st.caption(
                        "Press a key to label this image. Left/Right arrows move between images; "
                        "Backspace or Delete clears the label."
                        + (" Multi-label mode: each key toggles its class." if classification_config.multi_label else "")
                    )
                    st.dataframe(
                        [
                            {
                                "key": entry.key,
                                "class": entry.name,
                                "images": classification_summary.class_counts.get(entry.name, 0),
                                "current": "yes" if entry.name in current_label_names else "",
                            }
                            for entry in classification_config.classes
                        ],
                        width="stretch",
                        hide_index=True,
                        height=min(420, max(120, 38 * (len(classification_config.classes) + 1))),
                    )
                    classification_metrics = st.columns(2, gap="small")
                    classification_metrics[0].metric(
                        "Labeled",
                        f"{classification_summary.labeled_count} / {classification_summary.image_count}",
                    )
                    classification_metrics[1].metric("Unlabeled", classification_summary.unlabeled_count)
                    st.caption(f"Labels file: {classification_labels_path}")
                    last_saved = st.session_state.get("gui_classification_last_saved")
                    if isinstance(last_saved, str) and last_saved:
                        st.success(last_saved)

    _remember_gui_preferences()


def main() -> None:
    render()


if __name__ == "__main__":
    main()
