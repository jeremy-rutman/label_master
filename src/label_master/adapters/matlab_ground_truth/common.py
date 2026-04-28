from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.io.matlab._mio5 import MatFile5Reader

from label_master.adapters.video_bbox.common import resolve_video_file
from label_master.core.domain.value_objects import ValidationError

_GROUND_TRUTH_FIELDS = ("DataSource", "LabelDefinitions", "LabelData", "Version")


@dataclass(frozen=True)
class MatlabGroundTruthRow:
    timestamp_ms: float | None
    bboxes_by_label: dict[str, list[tuple[float, float, float, float]]]


@dataclass(frozen=True)
class MatlabGroundTruthPayload:
    source_video_path: str
    label_names: tuple[str, ...]
    rows: tuple[MatlabGroundTruthRow, ...]


def peek_matlab_opaque_class(mat_path: Path) -> str | None:
    try:
        with mat_path.open("rb") as handle:
            reader = MatFile5Reader(handle, struct_as_record=True)
            outer = reader.get_variables()
    except Exception:
        return None

    return _extract_outer_opaque_class_name(outer)


def find_matlab_ground_truth_files(
    dataset_root: Path,
    *,
    max_files: int | None = None,
) -> list[Path]:
    matches: list[Path] = []
    for mat_path in sorted(path for path in dataset_root.rglob("*.mat") if path.is_file()):
        if peek_matlab_opaque_class(mat_path) != "groundTruth":
            continue
        matches.append(mat_path)
        if max_files is not None and len(matches) >= max_files:
            break
    return matches


def load_matlab_ground_truth_payload(mat_path: Path) -> MatlabGroundTruthPayload:
    try:
        with mat_path.open("rb") as handle:
            reader = MatFile5Reader(handle, struct_as_record=True)
            outer = reader.get_variables()
            opaque_class = _extract_outer_opaque_class_name(outer)
            if opaque_class != "groundTruth":
                raise ValidationError(f"MATLAB annotation file is not a groundTruth object: {mat_path.name}")
            workspace = _read_workspace_vars(reader, outer)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Unable to read MATLAB annotation file: {mat_path.name}") from exc

    wrapper_array = _find_workspace_wrapper_array(workspace)
    ground_truth_record = _find_ground_truth_record(wrapper_array)
    label_data = ground_truth_record["LabelData"]
    row_count = int(getattr(label_data, "shape", (0,))[0] or 0)
    timestamps = _find_timestamp_series(wrapper_array, expected_length=row_count)
    label_definitions = ground_truth_record["LabelDefinitions"]

    label_names = tuple(
        name
        for name in (_extract_label_name(label_definitions, index) for index in range(int(label_definitions.shape[0])))
        if name
    )
    if not label_names:
        raise ValidationError(f"MATLAB groundTruth object has no label definitions: {mat_path.name}")

    rows: list[MatlabGroundTruthRow] = []
    label_fields = tuple(field for field in getattr(label_data.dtype, "names", ()) if field != "Time")
    for row_index in range(row_count):
        row_record = label_data[row_index, 0]
        bboxes_by_label: dict[str, list[tuple[float, float, float, float]]] = {}
        for label_name in label_fields:
            bboxes = _extract_bbox_rows(row_record[label_name])
            if bboxes:
                bboxes_by_label[label_name] = bboxes
        timestamp_ms = timestamps[row_index] if timestamps and row_index < len(timestamps) else None
        rows.append(
            MatlabGroundTruthRow(
                timestamp_ms=timestamp_ms,
                bboxes_by_label=bboxes_by_label,
            )
        )

    source_video_path = _extract_string_scalar(ground_truth_record["DataSource"][0, 0]["Source"])
    if not source_video_path:
        raise ValidationError(f"MATLAB groundTruth source video path is missing: {mat_path.name}")

    return MatlabGroundTruthPayload(
        source_video_path=source_video_path,
        label_names=label_names,
        rows=tuple(rows),
    )


def resolve_matlab_ground_truth_video_path(
    dataset_root: Path,
    *,
    source_video_path: str,
    annotation_path: Path | None = None,
) -> Path | None:
    normalized = source_video_path.replace("\\", "/").strip()
    if not normalized:
        return None

    source_ref = Path(normalized)
    direct_by_stem = resolve_video_file(dataset_root, source_ref.stem)
    if direct_by_stem is not None:
        return direct_by_stem

    candidates: list[Path] = [
        dataset_root / source_ref.name,
        dataset_root / source_ref,
    ]
    if annotation_path is not None:
        candidates.append(annotation_path.parent / source_ref.name)
        candidates.append(annotation_path.parent / source_ref)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = list(dataset_root.rglob(source_ref.name))
    if matches:
        return matches[0]
    return None


def _extract_outer_opaque_class_name(outer: dict[str, Any]) -> str | None:
    for name, value in outer.items():
        if name.startswith("__"):
            continue
        dtype_names = getattr(getattr(value, "dtype", None), "names", None)
        if not dtype_names or "s2" not in dtype_names:
            continue
        try:
            record = value.reshape(-1)[0]
        except Exception:
            continue
        class_name = _extract_string_scalar(record["s2"])
        if class_name:
            return class_name
    return None


def _read_workspace_vars(reader: MatFile5Reader, outer: dict[str, Any]) -> dict[str, Any]:
    workspace = outer.get("__function_workspace__")
    if workspace is None or not hasattr(workspace, "tobytes"):
        raise ValidationError("MATLAB object is missing __function_workspace__")

    workspace_stream = io.BytesIO(workspace.tobytes())
    workspace_stream.seek(2)
    reader.mat_stream = workspace_stream
    endian_test = reader.mat_stream.read(2)
    reader.byte_order = "<" if endian_test == b"IM" else ">"
    reader.mat_stream.read(4)
    reader.initialize_read()

    variables: dict[str, Any] = {}
    counter = 0
    while not reader.end_of_stream():
        header, next_position = reader.read_var_header()
        name = "None" if header.name is None else header.name.decode("latin1")
        if name == "":
            name = f"var_{counter}"
            counter += 1
        variables[name] = reader.read_var_array(header, process=False)
        reader.mat_stream.seek(next_position)
    return variables


def _find_workspace_wrapper_array(workspace: dict[str, Any]) -> Any:
    for value in workspace.values():
        dtype_names = getattr(getattr(value, "dtype", None), "names", None)
        if dtype_names != ("MCOS",):
            continue
        try:
            wrapper = value[0, 0]["MCOS"][0]["arr"]
        except Exception:
            continue
        if getattr(wrapper, "shape", ()) and len(wrapper.shape) == 2 and wrapper.shape[1] == 1:
            return wrapper
    raise ValidationError("MATLAB workspace wrapper payload is unavailable")


def _find_ground_truth_record(wrapper_array: Any) -> Any:
    for index in range(int(wrapper_array.shape[0])):
        candidate = wrapper_array[index, 0]
        dtype_names = getattr(getattr(candidate, "dtype", None), "names", None)
        if dtype_names and tuple(dtype_names) == _GROUND_TRUTH_FIELDS:
            return candidate[0, 0]
    raise ValidationError("MATLAB groundTruth payload could not be located in the workspace")


def _extract_label_name(label_definitions: Any, index: int) -> str | None:
    label_definition = label_definitions[index, 0]
    return _extract_string_scalar(label_definition["Name"])


def _find_timestamp_series(wrapper_array: Any, *, expected_length: int) -> list[float]:
    for index in range(int(wrapper_array.shape[0])):
        candidate = wrapper_array[index, 0]
        if getattr(getattr(candidate, "dtype", None), "kind", None) != "f":
            continue
        shape = getattr(candidate, "shape", ())
        if shape not in {(expected_length, 1), (1, expected_length)}:
            continue
        values = [float(value) for value in candidate.reshape(-1).tolist()]
        if len(values) != expected_length:
            continue
        if any(values[position] > values[position + 1] for position in range(len(values) - 1)):
            continue
        return values
    return []


def _extract_bbox_rows(value: Any) -> list[tuple[float, float, float, float]]:
    dtype_names = getattr(getattr(value, "dtype", None), "names", None)
    if dtype_names and "Position" in dtype_names:
        bboxes: list[tuple[float, float, float, float]] = []
        for item in value.reshape(-1):
            bboxes.extend(_extract_bbox_rows(item["Position"]))
        return bboxes

    if getattr(value, "size", 0) == 0:
        return []

    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if dtype_kind == "O":
        bboxes: list[tuple[float, float, float, float]] = []
        for item in value.reshape(-1).tolist():
            bboxes.extend(_extract_bbox_rows(item))
        return bboxes

    shape = getattr(value, "shape", ())
    if len(shape) == 1:
        rows = [value.tolist()]
    elif len(shape) == 2 and shape[1] == 4:
        rows = value.tolist()
    elif len(shape) == 2 and shape[0] == 4:
        rows = value.T.tolist()
    else:
        raise ValidationError("MATLAB groundTruth bbox arrays must have four columns")

    normalized: list[tuple[float, float, float, float]] = []
    for row in rows:
        if len(row) != 4:
            raise ValidationError("MATLAB groundTruth bbox rows must contain four values")
        x, y, width, height = (float(component) for component in row)
        if width <= 0 or height <= 0:
            raise ValidationError("MATLAB groundTruth bbox width/height must be positive")
        normalized.append((x, y, width, height))
    return normalized


def _extract_string_scalar(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bytes):
        decoded = value.decode("utf-8", errors="replace").strip()
        return decoded or None

    if hasattr(value, "dtype"):
        dtype_kind = getattr(value.dtype, "kind", None)
        if dtype_kind in {"U", "S"} and getattr(value, "size", 0) > 0:
            text = str(value.reshape(-1)[0]).strip()
            return text or None
        if dtype_kind == "O" and getattr(value, "size", 0) > 0:
            return _extract_string_scalar(value.reshape(-1)[0])

    return None
