from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from label_master.format_specs.registry import (
    CsvBracketBBoxDatasetParserSpec,
    resolve_builtin_format_spec,
)

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class KitwareBBoxColumn:
    header_name: str
    class_name: str


@dataclass(frozen=True)
class KitwareCsvLayout:
    csv_path: Path
    image_field: str
    bbox_columns: tuple[KitwareBBoxColumn, ...]


@lru_cache(maxsize=1)
def _kitware_parser() -> CsvBracketBBoxDatasetParserSpec:
    spec = resolve_builtin_format_spec("kitware")
    if spec is None or not isinstance(spec.parser, CsvBracketBBoxDatasetParserSpec):
        raise ValueError("Built-in Kitware format spec is unavailable")
    return spec.parser


def _normalize_header_name(value: str) -> str:
    return value.strip().lstrip("\ufeff").lower().replace("-", "_").replace(" ", "_")


def normalize_kitware_label_name(header_name: str) -> str:
    parser = _kitware_parser()
    normalized = _normalize_header_name(header_name)
    return parser.bbox_column_class_map.get(normalized, normalized)


def _is_bbox_column(header_name: str) -> bool:
    parser = _kitware_parser()
    normalized = _normalize_header_name(header_name)
    return normalized in parser.bbox_column_class_map


def parse_kitware_csv_layout(csv_path: Path) -> KitwareCsvLayout | None:
    parser = _kitware_parser()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header_row = next(reader, None)
    except OSError:
        return None

    if not header_row:
        return None

    image_field = None
    bbox_columns: list[KitwareBBoxColumn] = []
    for field_name in header_row:
        normalized = _normalize_header_name(field_name)
        if normalized in {_normalize_header_name(value) for value in parser.image_field_aliases}:
            image_field = field_name
            continue
        if not _is_bbox_column(field_name):
            continue
        bbox_columns.append(
            KitwareBBoxColumn(
                header_name=field_name,
                class_name=normalize_kitware_label_name(field_name),
            )
        )

    if image_field is None or not bbox_columns:
        return None

    return KitwareCsvLayout(
        csv_path=csv_path,
        image_field=image_field,
        bbox_columns=tuple(bbox_columns),
    )


def discover_kitware_csv_layouts(
    dataset_root: Path,
    *,
    max_layouts: int | None = None,
) -> list[KitwareCsvLayout]:
    parser = _kitware_parser()
    layouts: list[KitwareCsvLayout] = []
    csv_paths: Iterable[Path]
    seen: set[Path] = set()
    discovered_paths: list[Path] = []
    for pattern in parser.csv_globs:
        for path in sorted(dataset_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            discovered_paths.append(path)
    if max_layouts is None:
        csv_paths = discovered_paths
    else:
        csv_paths = discovered_paths

    for csv_path in csv_paths:
        layout = parse_kitware_csv_layout(csv_path)
        if layout is None:
            continue
        layouts.append(layout)
        if max_layouts is not None and len(layouts) >= max_layouts:
            break
    return layouts


def parse_kitware_bboxes(
    value: str,
    *,
    parser: CsvBracketBBoxDatasetParserSpec | None = None,
) -> list[tuple[float, float, float, float]]:
    parser = parser or _kitware_parser()
    text = value.strip()
    if not text or text == "[]":
        return []
    if not (text.startswith(parser.bbox_enclosure[0]) and text.endswith(parser.bbox_enclosure[1])):
        raise ValueError("Kitware bbox values must use bracketed xmin/ymin/width/height notation")

    body = text[1:-1].strip()
    if not body:
        return []

    bboxes: list[tuple[float, float, float, float]] = []
    for raw_bbox in body.split(parser.box_separator):
        bbox_text = raw_bbox.strip()
        if not bbox_text:
            continue
        numeric_values = [float(token) for token in _NUMBER_PATTERN.findall(bbox_text)]
        bbox_positions = parser.bbox_fields
        if len(numeric_values) < max(
            bbox_positions.xmin,
            bbox_positions.ymin,
            bbox_positions.width,
            bbox_positions.height,
        ):
            raise ValueError("Kitware bbox values must provide all mapped xmin/ymin/width/height fields")

        xmin = numeric_values[bbox_positions.xmin - 1]
        ymin = numeric_values[bbox_positions.ymin - 1]
        width = numeric_values[bbox_positions.width - 1]
        height = numeric_values[bbox_positions.height - 1]
        if xmin < 0 or ymin < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Kitware bbox values must use non-negative xmin/ymin and positive width/height"
            )
        bboxes.append((xmin, ymin, width, height))

    return bboxes


def parse_kitware_bbox(value: str) -> tuple[float, float, float, float] | None:
    bboxes = parse_kitware_bboxes(value)
    if not bboxes:
        return None
    if len(bboxes) != 1:
        raise ValueError("Kitware bbox value contains multiple boxes; use parse_kitware_bboxes")
    return bboxes[0]


def resolve_kitware_image_path(
    dataset_root: Path,
    csv_path: Path,
    raw_image_ref: str,
) -> Path | None:
    normalized = raw_image_ref.strip().replace("\\", "/")
    if not normalized:
        return None

    reference = Path(normalized)
    csv_dir = csv_path.parent
    candidates: list[Path] = [csv_dir / reference.name]

    if reference.parts:
        candidates.append(csv_dir / reference)
        if csv_dir.name in reference.parts:
            csv_dir_index = reference.parts.index(csv_dir.name)
            tail_parts = reference.parts[csv_dir_index + 1 :]
            if tail_parts:
                candidates.append(csv_dir / Path(*tail_parts))
        candidates.append(dataset_root / reference)
    candidates.append(dataset_root / reference.name)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
