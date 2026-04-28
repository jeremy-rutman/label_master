from __future__ import annotations

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
from PIL import Image, ImageDraw

from label_master.adapters.custom.detector import detect_custom_format
from label_master.adapters.video_bbox.reader import load_video_bbox_preview_image
from label_master.core.domain.policies import (
    DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES,
    DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX,
    DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    InvalidAnnotationAction,
    ValidationMode,
)
from label_master.core.services.convert_service import (
    derive_output_filename_prefix,
    sanitize_output_file_stem_affix,
)
from label_master.format_specs.registry import (
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
    MappingRowViewModel,
    build_gui_run_config,
    convert_view,
    infer_view,
    parse_mapping_rows,
    preview_dataset_view,
)

LOCALHOST_VALUES = {"127.0.0.1", "localhost", "::1"}
SOURCE_FORMATS = ["auto", "coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]
DESTINATION_FORMATS = ["yolo", "coco"]
DEFAULT_DESTINATION_FORMAT = "yolo"
UNMAPPED_POLICIES = ["error", "drop", "identity"]
VALIDATION_MODES = [ValidationMode.STRICT.value, ValidationMode.PERMISSIVE.value]
PERMISSIVE_INVALID_ANNOTATION_ACTIONS = [
    InvalidAnnotationAction.KEEP.value,
    InvalidAnnotationAction.DROP.value,
]
MAPPING_ACTIONS = ["map", "drop"]
OVERSIZE_IMAGE_ACTIONS = ["ignore", "downscale"]
DEFAULT_INPUT_DIR = "tests/fixtures/us1/coco_minimal"
DEFAULT_OUTPUT_DIR = "/tmp/label_master_gui_output"
DEFAULT_MAPPING_ROWS: list[dict[str, str]] = []
GUI_STATE_FILE_NAME = "gui_state.json"
PREVIEW_MAX_IMAGE_DIMENSION = 1600
DEFAULT_PREVIEW_SCAN_LIMIT = 0
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
_PREVIEW_KEYBOARD_NAV_COMPONENT = components.declare_component(
    "preview_keyboard_nav",
    path=Path(__file__).resolve().parent / "components" / "preview_keyboard_nav",
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
    correct_out_of_frame_bboxes: bool,
) -> float:
    tolerance_px = max(0.0, _coerce_float(raw_value, DEFAULT_OUT_OF_FRAME_TOLERANCE_PX))
    if correct_out_of_frame_bboxes and tolerance_px <= 0.0:
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
    correct_out_of_frame_bboxes = payload.get("last_correct_out_of_frame_bboxes")
    if not isinstance(correct_out_of_frame_bboxes, bool):
        correct_out_of_frame_bboxes = DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES
    out_of_frame_tolerance_px = _coerce_out_of_frame_tolerance_px(
        payload.get("last_out_of_frame_tolerance_px"),
        correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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

    inference_payload = payload.get("last_inference_payload")
    if not isinstance(inference_payload, dict):
        inference_payload = None

    mapping_seed_signature = _coerce_text(payload.get("last_mapping_seed_signature")) or None
    last_input_dir = _coerce_text(payload.get("last_input_dir"))

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
        "gui_correct_out_of_frame_bboxes": correct_out_of_frame_bboxes,
        "gui_out_of_frame_tolerance_px": out_of_frame_tolerance_px,
        "gui_min_image_longest_edge_px": min_image_longest_edge_px,
        "gui_max_image_longest_edge_px": max_image_longest_edge_px,
        "gui_preview_scan_limit": preview_scan_limit,
        "gui_oversize_image_action": oversize_image_action,
        "gui_inference_payload": inference_payload,
        "gui_mapping_rows": normalize_mapping_rows(payload.get("last_mapping_rows")),
        "gui_mapping_seed_signature": mapping_seed_signature,
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


@lru_cache(maxsize=256)
def _render_preview_overlay_cached(
    *,
    dataset_root: str,
    image_rel_path: str,
    bboxes: tuple[tuple[float, float, float, float, str], ...],
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
    label_height = 16
    for x, y, w, h, label in bboxes:
        scaled_x = x * scale
        scaled_y = y * scale
        scaled_w = w * scale
        scaled_h = h * scale
        x2 = scaled_x + scaled_w
        y2 = scaled_y + scaled_h
        draw.rectangle((scaled_x, scaled_y, x2, y2), outline="red", width=line_width)
        label_x = max(0, int(scaled_x))
        label_y = max(0, int(scaled_y) - label_height)
        label_width = max(40, int(len(label) * 8) + 8)
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height), fill="red")
        draw.text((label_x + 4, label_y + 2), label, fill="white")

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue(), ()


def render_preview_overlay(
    *,
    dataset_root: Path,
    image_rel_path: str,
    bboxes: list[tuple[float, float, float, float, str]],
) -> tuple[Image.Image | None, list[str]]:
    overlay_bytes, warnings = _render_preview_overlay_cached(
        dataset_root=str(dataset_root.expanduser().resolve()),
        image_rel_path=image_rel_path,
        bboxes=tuple((float(x), float(y), float(w), float(h), str(label)) for x, y, w, h, label in bboxes),
        image_cache_token=_preview_image_cache_token(dataset_root, image_rel_path),
    )
    if overlay_bytes is None:
        return None, list(warnings)

    with Image.open(BytesIO(overlay_bytes)) as opened:
        return opened.copy(), list(warnings)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


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


def _format_payload_yaml(payload: dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=False)
    return str(dumped).strip()


def format_details_yaml(
    source_format: str | None,
    *,
    dataset_root: Path | None,
    inference_payload: dict[str, Any] | None = None,
) -> str | None:
    normalized_format = _coerce_text(source_format)
    spec = None

    if normalized_format == "custom" and dataset_root is not None:
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
    if src in {"coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"}:
        return src
    if inferred_format in {"coco", "custom", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"}:
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
    correct_out_of_frame_bboxes = bool(
        st.session_state.get("gui_correct_out_of_frame_bboxes", DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES)
    )

    if destination_format not in DESTINATION_FORMATS:
        destination_format = DEFAULT_DESTINATION_FORMAT
    if not isinstance(inference_payload, dict):
        inference_payload = None

    return {
        "last_input_dir": validated_input_dir or persisted_input_dir or DEFAULT_INPUT_DIR,
        "last_output_dir": output_dir,
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
        "last_correct_out_of_frame_bboxes": correct_out_of_frame_bboxes,
        "last_out_of_frame_tolerance_px": _coerce_out_of_frame_tolerance_px(
            st.session_state.get("gui_out_of_frame_tolerance_px"),
            correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
        "last_oversize_image_action": (
            _coerce_text(st.session_state.get("gui_oversize_image_action"))
            if _coerce_text(st.session_state.get("gui_oversize_image_action")) in OVERSIZE_IMAGE_ACTIONS
            else "ignore"
        ),
        "last_inference_payload": inference_payload,
        "last_mapping_rows": normalize_mapping_rows(st.session_state.get("gui_mapping_rows")),
        "last_mapping_seed_signature": mapping_seed_signature,
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
    if "gui_correct_out_of_frame_bboxes" not in st.session_state:
        st.session_state["gui_correct_out_of_frame_bboxes"] = persisted_preferences["gui_correct_out_of_frame_bboxes"]
    if "gui_out_of_frame_tolerance_px" not in st.session_state:
        st.session_state["gui_out_of_frame_tolerance_px"] = persisted_preferences["gui_out_of_frame_tolerance_px"]
    elif bool(st.session_state.get("gui_correct_out_of_frame_bboxes", DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES)):
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
        "4. Label Mapping",
        "5. Review & Run",
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
        input_path_include_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_include_substring"))
        )
        input_path_exclude_substring = normalize_input_path_filter_substring(
            _coerce_text(st.session_state.get("gui_input_path_exclude_substring"))
        )
        correct_out_of_frame_bboxes = bool(
            st.session_state.get("gui_correct_out_of_frame_bboxes", DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES)
        )
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
        else:
            try:
                preview_vm = preview_dataset_view(
                    input_path,
                    source_format=preview_source_format,
                    correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
                    out_of_frame_tolerance_px=out_of_frame_tolerance_px,
                    input_path_include_substring=input_path_include_substring,
                    input_path_exclude_substring=input_path_exclude_substring,
                    preview_scan_limit=preview_scan_limit,
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
        st.checkbox(
            "Correct out-of-frame bboxes",
            key="gui_correct_out_of_frame_bboxes",
        )
        st.number_input(
            "Out-of-frame correction tolerance (px)",
            key="gui_out_of_frame_tolerance_px",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            disabled=not bool(
                st.session_state.get("gui_correct_out_of_frame_bboxes", DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES)
            ),
        )
        if bool(st.session_state.get("gui_correct_out_of_frame_bboxes", DEFAULT_CORRECT_OUT_OF_FRAME_BBOXES)):
            tolerance_preview = _coerce_out_of_frame_tolerance_px(
                st.session_state.get("gui_out_of_frame_tolerance_px"),
                correct_out_of_frame_bboxes=True,
            )
            st.caption(
                f"Near-edge boxes are clipped to image bounds when they exceed the frame by at most `{tolerance_preview:g}` px."
            )
        else:
            st.caption("Out-of-frame boxes remain invalid when automatic correction is disabled.")

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

    with tabs[3]:
        st.subheader("Step 4: Label Mapping")
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

    with tabs[4]:
        st.subheader("Step 5: Review & Run")
        st.caption("Runs are blocked until required fields and mapping rows validate.")
        st.checkbox("Dry run", key="gui_dry_run")
        st.checkbox("Copy images to output", key="gui_copy_images")
        st.checkbox("Allow overwriting existing output files", key="gui_allow_overwrite")

        input_dir_raw = _coerce_text(st.session_state["gui_input_dir"])
        output_dir_raw = _coerce_text(st.session_state["gui_output_dir"])
        src = _coerce_text(st.session_state["gui_src"])
        dst = _coerce_text(st.session_state["gui_dst"])
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
        correct_out_of_frame_bboxes = bool(st.session_state["gui_correct_out_of_frame_bboxes"])
        out_of_frame_tolerance_px = _coerce_out_of_frame_tolerance_px(
            st.session_state["gui_out_of_frame_tolerance_px"],
            correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
                    "dst": dst,
                    "validation_mode": validation_mode,
                    "unmapped_policy": unmapped_policy,
                    "allow_shared_output_dir": allow_shared_output_dir,
                    "prefix_output_filenames": prefix_output_filenames,
                    "output_filename_prefix": output_filename_prefix,
                    "output_file_stem_prefix": output_file_stem_prefix,
                    "output_file_stem_suffix": output_file_stem_suffix,
                    "correct_out_of_frame_bboxes": correct_out_of_frame_bboxes,
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
                    correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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
                    map_path=final_map_path,
                    unmapped_policy=unmapped_policy,
                    dry_run=dry_run,
                    allow_overwrite=allow_overwrite,
                    input_path_include_substring=input_path_include_substring,
                    input_path_exclude_substring=input_path_exclude_substring,
                    validation_mode=validation_mode,
                    permissive_invalid_annotation_action=permissive_invalid_annotation_action,
                    correct_out_of_frame_bboxes=correct_out_of_frame_bboxes,
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

    _remember_gui_preferences()


def main() -> None:
    render()


if __name__ == "__main__":
    main()
