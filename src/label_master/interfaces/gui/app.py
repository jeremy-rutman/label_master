from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
import yaml
from PIL import Image, ImageDraw

from label_master.infra.filesystem import atomic_write_json, ensure_directory
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
SOURCE_FORMATS = ["auto", "coco", "yolo"]
DESTINATION_FORMATS = ["coco", "yolo"]
UNMAPPED_POLICIES = ["error", "drop", "identity"]
MAPPING_ACTIONS = ["map", "drop"]
DEFAULT_INPUT_DIR = "tests/fixtures/us1/coco_minimal"
DEFAULT_OUTPUT_DIR = "/tmp/label_master_gui_output"
DEFAULT_MAPPING_ROWS: list[dict[str, str]] = []
RUN_STATUSES = {"idle", "running", "completed", "failed"}
RUN_EVENTS = {"start", "complete", "fail", "reset"}
RUN_PROGRESS_BY_STATUS = {
    "idle": 0,
    "running": 15,
    "completed": 100,
    "failed": 100,
}


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
class RunSummaryMetrics:
    images_processed: int
    annotations_converted: int
    warning_count: int
    error_count: int


def is_localhost_binding(address: str | None) -> bool:
    if address is None:
        return True
    return address in LOCALHOST_VALUES


def export_run_config(payload: dict[str, Any], output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    atomic_write_json(output_path, payload)
    return output_path


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


def render_preview_overlay(
    *,
    dataset_root: Path,
    image_rel_path: str,
    bboxes: list[tuple[float, float, float, float, str]],
) -> tuple[Image.Image | None, list[str]]:
    image_path = dataset_root / image_rel_path
    if not image_path.exists():
        return None, [f"Preview image not found: {image_rel_path}"]

    try:
        with Image.open(image_path) as opened:
            canvas = opened.convert("RGB").copy()
    except OSError:
        return None, [f"Preview image could not be opened: {image_rel_path}"]

    draw = ImageDraw.Draw(canvas)
    for x, y, w, h, label in bboxes:
        x2 = x + w
        y2 = y + h
        draw.rectangle((x, y, x2, y2), outline="red", width=3)
        label_x = max(0, int(x))
        label_y = max(0, int(y) - 16)
        label_width = max(40, int(len(label) * 8) + 8)
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + 16), fill="red")
        draw.text((label_x + 4, label_y + 2), label, fill="white")

    return canvas, []


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


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

        if normalized_action == "drop":
            destination_class_id = ""

        normalized.append(
            {
                "source_class_id": source_class_id,
                "action": normalized_action,
                "destination_class_id": destination_class_id,
            }
        )

    return normalized


_normalize_mapping_rows = normalize_mapping_rows


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
        destination_id = _parse_int_or_none(row["destination_class_id"])
        with_labels.append(
            {
                **row,
                "source_label": class_labels.get(source_id, "") if source_id is not None else "",
                "destination_label": class_labels.get(destination_id, "")
                if destination_id is not None
                else "",
            }
        )
    return with_labels


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


def _format_payload_yaml(payload: dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=False)
    return str(dumped).strip()


def _maybe_auto_infer_for_preview(input_path: Path) -> None:
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
        return

    st.session_state["gui_inference_payload"] = {
        "predicted_format": infer_vm.predicted_format,
        "confidence": infer_vm.confidence,
        "candidates": infer_vm.candidates,
        "warnings": infer_vm.warnings,
    }
    st.session_state["gui_inference_error"] = None


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


def describe_class_label_source(
    *,
    input_path: Path,
    source_format: str | None,
    class_labels: dict[int, str],
) -> str:
    if not class_labels:
        return "Class labels are not available yet. Load preview data first."

    if source_format == "yolo":
        classes_path = input_path / "classes.txt"
        if classes_path.exists():
            try:
                lines = [
                    line.strip()
                    for line in classes_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                lines = []

            provided_ids = set(range(len(lines)))
            observed_ids = set(class_labels)
            fallback_ids = sorted(observed_ids - provided_ids)
            if fallback_ids:
                fallback_text = ", ".join(str(class_id) for class_id in fallback_ids)
                return (
                    "YOLO labels source: classes.txt. "
                    f"Missing class IDs ({fallback_text}) use fallback names class_<id>."
                )
            return "YOLO labels source: classes.txt."

        return "YOLO labels source: classes.txt not found; using fallback names class_<id>."

    if source_format == "coco":
        return "COCO labels source: categories from annotations.json."

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
    if src in {"coco", "yolo"}:
        return src
    if inferred_format in {"coco", "yolo"}:
        return inferred_format
    return None


def attempt_input_directory_browse(current_input_dir: str) -> InputDirectoryBrowseState:
    current_value = current_input_dir.strip()
    initial_directory = Path(current_value).expanduser() if current_value else None
    if initial_directory and not initial_directory.exists():
        initial_directory = None

    result = system_actions.browse_for_directory(initial_directory=initial_directory)
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
    st.session_state["browse_available"] = browse_state.browse_available
    st.session_state["browse_message"] = browse_state.browse_message


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


def attempt_output_directory_access(output_dir_raw: str) -> OutputDirectoryOpenResult:
    output_value = output_dir_raw.strip()
    if not output_value:
        return OutputDirectoryOpenResult(
            requested_path=Path.cwd().resolve(),
            opened=False,
            message="Output directory is empty. Provide a directory path.",
        )

    return system_actions.open_output_directory(Path(output_value).expanduser())


def _initialize_state() -> None:
    if "gui_input_dir" not in st.session_state:
        st.session_state["gui_input_dir"] = DEFAULT_INPUT_DIR
    if "gui_output_dir" not in st.session_state:
        st.session_state["gui_output_dir"] = DEFAULT_OUTPUT_DIR
    if "gui_src" not in st.session_state:
        st.session_state["gui_src"] = "auto"
    if "gui_dst" not in st.session_state:
        st.session_state["gui_dst"] = "yolo"
    if "gui_unmapped_policy" not in st.session_state:
        st.session_state["gui_unmapped_policy"] = "error"
    if "gui_dry_run" not in st.session_state:
        st.session_state["gui_dry_run"] = False
    if "gui_copy_images" not in st.session_state:
        st.session_state["gui_copy_images"] = True
    if "gui_inference_payload" not in st.session_state:
        st.session_state["gui_inference_payload"] = None
    if "gui_inference_error" not in st.session_state:
        st.session_state["gui_inference_error"] = None
    if "gui_preview_index" not in st.session_state:
        st.session_state["gui_preview_index"] = 0
    if "gui_preview_key" not in st.session_state:
        st.session_state["gui_preview_key"] = ""
    if "gui_mapping_rows" not in st.session_state:
        st.session_state["gui_mapping_rows"] = [dict(row) for row in DEFAULT_MAPPING_ROWS]
    if "gui_class_labels" not in st.session_state:
        st.session_state["gui_class_labels"] = {}
    if "gui_mapping_seed_signature" not in st.session_state:
        st.session_state["gui_mapping_seed_signature"] = None
    if "gui_last_run" not in st.session_state:
        st.session_state["gui_last_run"] = None
    if "browse_available" not in st.session_state:
        st.session_state["browse_available"] = True
    if "browse_message" not in st.session_state:
        st.session_state["browse_message"] = None
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
    if "gui_output_action_message" not in st.session_state:
        st.session_state["gui_output_action_message"] = None


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

        browse_message = st.session_state.get("browse_message")
        if isinstance(browse_message, str) and browse_message:
            if bool(st.session_state.get("browse_available", True)):
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
            st.success(f"Using dataset directory: {directory_validation.resolved_path}")

    with tabs[1]:
        st.subheader("Step 2: Format & Preview")
        st.selectbox("Source format", SOURCE_FORMATS, key="gui_src")
        _inject_preview_table_alignment_css()

        input_dir_raw = _coerce_text(st.session_state["gui_input_dir"])
        input_path = Path(input_dir_raw).expanduser() if input_dir_raw else Path(".")

        if st.button("Infer format", key="gui_infer_button"):
            try:
                infer_vm = infer_view(input_path)
                st.session_state["gui_inference_payload"] = {
                    "predicted_format": infer_vm.predicted_format,
                    "confidence": infer_vm.confidence,
                    "candidates": infer_vm.candidates,
                    "warnings": infer_vm.warnings,
                }
                st.session_state["gui_inference_error"] = None
            except Exception as exc:
                st.session_state["gui_inference_payload"] = None
                st.session_state["gui_inference_error"] = str(exc)

        if input_dir_raw and input_path.exists() and input_path.is_dir():
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
        preview_vm = None
        if not input_dir_raw or not input_path.exists() or not input_path.is_dir():
            _store_class_labels({})
            st.info("Set a valid input directory to load preview.")
        elif preview_source_format is None:
            st.info("Choose a source format or run inference to load preview.")
        else:
            try:
                preview_vm = preview_dataset_view(input_path, source_format=preview_source_format)
            except Exception as exc:
                st.error(f"Unable to load preview dataset: {exc}")
                preview_vm = None

            if preview_vm:
                preview_key = f"{input_path.resolve()}::{preview_source_format}"
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
                else:
                    max_index = len(preview_vm.images) - 1
                    current_index = int(st.session_state["gui_preview_index"])
                    current_index = min(max(current_index, 0), max_index)
                    st.session_state["gui_preview_index"] = current_index

                    nav_cols = st.columns([1, 3, 1])
                    with nav_cols[0]:
                        if st.button("Previous", disabled=current_index == 0, key="gui_preview_prev"):
                            st.session_state["gui_preview_index"] = current_index - 1
                            st.rerun()
                    with nav_cols[1]:
                        st.markdown(f"**Image {current_index + 1} / {len(preview_vm.images)}**")
                    with nav_cols[2]:
                        if st.button("Next", disabled=current_index == max_index, key="gui_preview_next"):
                            st.session_state["gui_preview_index"] = current_index + 1
                            st.rerun()

                    current_image = preview_vm.images[current_index]
                    st.caption(current_image.file_name)

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
                    if overlay is None:
                        for warning in overlay_warnings:
                            st.warning(warning)
                    else:
                        st.image(overlay, use_container_width=True)

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
                    if bbox_rows:
                        st.dataframe(bbox_rows, use_container_width=True, hide_index=True)
                    else:
                        st.info("No bounding boxes for this image.")

        if inference_error:
            st.error(inference_error)
        if isinstance(inference_payload, dict):
            st.caption("Format details (YAML)")
            st.code(_format_payload_yaml(inference_payload), language="yaml")

    with tabs[2]:
        st.subheader("Step 3: Output")
        st.text_input("Output directory", key="gui_output_dir")
        st.selectbox("Destination format", DESTINATION_FORMATS, key="gui_dst")
        st.selectbox("Unmapped policy", UNMAPPED_POLICIES, key="gui_unmapped_policy")

    with tabs[3]:
        st.subheader("Step 4: Label Mapping")
        st.caption("Define source-to-destination mappings. Set action to 'drop' to remove a class.")
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
            st.caption("Default rows are identity mappings derived from detected classes.")
        else:
            st.caption(
                "Class labels will appear once preview data is available for the selected dataset."
            )

        _inject_mapping_table_css()
        current_rows = normalize_mapping_rows(st.session_state["gui_mapping_rows"])
        editor_rows = attach_mapping_labels(current_rows, class_labels)

        with st.container(border=True):
            editor_value = st.data_editor(
                editor_rows,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="gui_mapping_editor",
                column_config={
                    "source_class_id": st.column_config.TextColumn("source_class_id"),
                    "source_label": st.column_config.TextColumn("source_label", disabled=True),
                    "action": st.column_config.SelectboxColumn("action", options=MAPPING_ACTIONS),
                    "destination_class_id": st.column_config.TextColumn("destination_class_id"),
                    "destination_label": st.column_config.TextColumn(
                        "destination_label",
                        disabled=True,
                    ),
                },
            )

        normalized_rows = normalize_mapping_rows(editor_value)
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

        input_dir_raw = _coerce_text(st.session_state["gui_input_dir"])
        output_dir_raw = _coerce_text(st.session_state["gui_output_dir"])
        src = _coerce_text(st.session_state["gui_src"])
        dst = _coerce_text(st.session_state["gui_dst"])
        unmapped_policy = _coerce_text(st.session_state["gui_unmapped_policy"])
        dry_run = bool(st.session_state["gui_dry_run"])
        copy_images = bool(st.session_state["gui_copy_images"])

        review_rows = materialize_mapping_rows(st.session_state["gui_mapping_rows"])
        parsed_mappings = parse_mapping_rows(mapping_rows_to_viewmodels(review_rows))

        blocking_errors = run_blocking_errors(
            input_dir_raw=input_dir_raw,
            output_dir_raw=output_dir_raw,
            src=src,
            dst=dst,
            mapping_errors=parsed_mappings.errors,
        )

        status = _coerce_text(st.session_state["gui_run_status"]) or "idle"
        progress = int(st.session_state["gui_run_progress"])
        st.markdown(f"**Run status:** `{status}`")
        st.progress(min(max(progress, 0), 100) / 100.0)

        run_error = st.session_state.get("gui_run_error")
        if isinstance(run_error, str) and run_error:
            st.error(run_error)

        st.json(
            {
                "input_dir": input_dir_raw,
                "output_dir": output_dir_raw,
                "src": src,
                "dst": dst,
                "unmapped_policy": unmapped_policy,
                "dry_run": dry_run,
                "copy_images": copy_images,
                "mapping_rows": review_rows,
                "parsed_class_map_size": len(parsed_mappings.class_map),
            }
        )

        if blocking_errors:
            for error in blocking_errors:
                st.error(error)

        run_disabled = bool(blocking_errors) or status == "running"
        if st.button("Run conversion", disabled=run_disabled, type="primary"):
            started_status, started_progress = transition_run_state(status, "start")
            st.session_state["gui_run_status"] = started_status
            st.session_state["gui_run_progress"] = started_progress
            st.session_state["gui_run_error"] = None
            st.session_state["gui_output_action_message"] = None

            input_path = Path(input_dir_raw).expanduser()
            output_path = Path(output_dir_raw).expanduser()
            pending_map_path: Path | None = None
            final_map_path: Path | None = None

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
                    copy_images=copy_images,
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
                )
                config_path = export_run_config(
                    config.model_dump(mode="json"),
                    output_path / f"{convert_vm.run_id}.gui.config.json",
                )

                st.session_state["gui_last_run"] = {
                    "run_id": convert_vm.run_id,
                    "report": result.report.model_dump(mode="json"),
                    "config_path": str(config_path),
                    "mapping_path": str(final_map_path) if final_map_path else None,
                    "output_path": str(output_path.resolve()),
                }
                completed_status, completed_progress = transition_run_state(started_status, "complete")
                st.session_state["gui_run_status"] = completed_status
                st.session_state["gui_run_progress"] = completed_progress
            except Exception as exc:
                failed_status, failed_progress = transition_run_state(started_status, "fail")
                st.session_state["gui_run_status"] = failed_status
                st.session_state["gui_run_progress"] = failed_progress
                st.session_state["gui_run_error"] = str(exc)
                st.error(str(exc))

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

            st.json(last_run["report"])
            st.code(
                json.dumps(
                    {
                        "config_export": last_run["config_path"],
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


def main() -> None:
    render()


if __name__ == "__main__":
    main()
