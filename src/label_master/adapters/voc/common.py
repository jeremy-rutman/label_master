from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from xml.etree import ElementTree as ET

from label_master.format_specs.registry import (
    XmlAnnotationDatasetParserSpec,
    resolve_builtin_format_spec,
)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_IMAGE_DIR_HINTS = ("img", "image", "images", "jpeg", "jpegimages", "jpg")


@dataclass(frozen=True)
class VOCObjectBox:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class VOCAnnotationFile:
    filename: str
    path_hint: str | None
    width: int | None
    height: int | None
    objects: list[VOCObjectBox]


@dataclass(frozen=True)
class VOCImageIndex:
    by_name: dict[str, list[Path]]
    by_rel_path: dict[str, Path]


@lru_cache(maxsize=1)
def _voc_parser() -> XmlAnnotationDatasetParserSpec:
    spec = resolve_builtin_format_spec("voc")
    if spec is None or not isinstance(spec.parser, XmlAnnotationDatasetParserSpec):
        raise ValueError("Built-in VOC format spec is unavailable")
    return spec.parser


def discover_voc_xml_files(dataset_root: Path, *, sample_limit: int | None = None) -> list[Path]:
    parser = _voc_parser()
    xml_files: list[Path] = []
    seen: set[Path] = set()
    for pattern in parser.annotation_globs:
        for file in sorted(dataset_root.glob(pattern)):
            if not file.is_file() or file in seen:
                continue
            seen.add(file)
            xml_files.append(file)
    if sample_limit is None:
        return xml_files
    return list(islice(xml_files, sample_limit))


def build_voc_image_index(dataset_root: Path) -> VOCImageIndex:
    by_name: dict[str, list[Path]] = {}
    by_rel_path: dict[str, Path] = {}

    for file_path in sorted(file for file in dataset_root.rglob("*") if file.is_file()):
        if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        rel_path = file_path.relative_to(dataset_root).as_posix()
        by_rel_path[rel_path] = file_path
        by_name.setdefault(file_path.name, []).append(file_path)

    return VOCImageIndex(by_name=by_name, by_rel_path=by_rel_path)


def parse_voc_annotation_file(
    xml_path: Path,
    *,
    parser: XmlAnnotationDatasetParserSpec | None = None,
) -> VOCAnnotationFile:
    parser = parser or _voc_parser()
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"VOC XML could not be parsed: {xml_path}") from exc

    if _local_name(root.tag) != parser.root_tag:
        raise ValueError(f"XML is not a Pascal VOC annotation file: {xml_path}")

    path_hint = _clean_path_hint(root.findtext(parser.path_field)) if parser.path_field else None
    filename = _clean_path_hint(root.findtext(parser.filename_field)) or (Path(path_hint).name if path_hint else "")
    if not filename:
        raise ValueError(f"VOC annotation is missing filename: {xml_path}")

    width = _parse_optional_int(root.findtext(parser.size_width_field)) if parser.size_width_field else None
    height = _parse_optional_int(root.findtext(parser.size_height_field)) if parser.size_height_field else None
    objects: list[VOCObjectBox] = []

    for object_element in root.findall(parser.object_tag):
        name = (object_element.findtext(parser.object_name_field) or "").strip()
        bbox_element = object_element.find(parser.bbox_tag)
        if not name or bbox_element is None:
            raise ValueError(f"VOC object is incomplete: {xml_path}")

        xmin = _parse_required_float(
            bbox_element.findtext(parser.bbox_fields.xmin),
            field_name=parser.bbox_fields.xmin,
            xml_path=xml_path,
        )
        ymin = _parse_required_float(
            bbox_element.findtext(parser.bbox_fields.ymin),
            field_name=parser.bbox_fields.ymin,
            xml_path=xml_path,
        )
        xmax = _parse_required_float(
            bbox_element.findtext(parser.bbox_fields.xmax),
            field_name=parser.bbox_fields.xmax,
            xml_path=xml_path,
        )
        ymax = _parse_required_float(
            bbox_element.findtext(parser.bbox_fields.ymax),
            field_name=parser.bbox_fields.ymax,
            xml_path=xml_path,
        )
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"VOC bbox must have xmax > xmin and ymax > ymin: {xml_path}")

        objects.append(
            VOCObjectBox(
                name=name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return VOCAnnotationFile(
        filename=filename,
        path_hint=path_hint,
        width=width,
        height=height,
        objects=objects,
    )


def resolve_voc_image_path(
    dataset_root: Path,
    xml_path: Path,
    annotation: VOCAnnotationFile,
    *,
    image_index: VOCImageIndex | None = None,
) -> Path | None:
    candidate_paths: list[Path] = []
    candidate_paths.extend(_candidate_paths_for_hint(dataset_root, annotation.path_hint))
    candidate_paths.extend(_candidate_paths_for_hint(dataset_root, annotation.filename))

    image_name = Path(annotation.filename).name
    if image_name:
        for base_dir in _candidate_base_directories(dataset_root, xml_path):
            candidate_paths.append(base_dir / image_name)
            for child_dir in _child_image_directories(base_dir):
                candidate_paths.append(child_dir / image_name)

    for candidate in candidate_paths:
        resolved = _existing_path_within_root(dataset_root, candidate)
        if resolved is not None:
            return resolved

    if image_index is None:
        return None

    if annotation.path_hint:
        indexed = image_index.by_rel_path.get(annotation.path_hint)
        if indexed is not None:
            return indexed

    normalized_filename = annotation.filename.replace("\\", "/").lstrip("./")
    indexed = image_index.by_rel_path.get(normalized_filename)
    if indexed is not None:
        return indexed

    indexed_candidates = image_index.by_name.get(image_name, [])
    if not indexed_candidates:
        return None
    if len(indexed_candidates) == 1:
        return indexed_candidates[0]

    return min(
        indexed_candidates,
        key=lambda candidate: _candidate_rank(dataset_root, xml_path, candidate),
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _parse_optional_int(raw_value: str | None) -> int | None:
    text = (raw_value or "").strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Invalid integer value in VOC annotation: {text}") from exc
    return value if value > 0 else None


def _parse_required_float(raw_value: str | None, *, field_name: str, xml_path: Path) -> float:
    text = (raw_value or "").strip()
    if not text:
        raise ValueError(f"VOC bbox is missing {field_name}: {xml_path}")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"VOC bbox {field_name} is not numeric: {xml_path}") from exc


def _clean_path_hint(raw_value: str | None) -> str | None:
    text = (raw_value or "").strip().replace("\\", "/")
    if not text:
        return None
    return text.lstrip("./")


def _candidate_paths_for_hint(dataset_root: Path, raw_hint: str | None) -> list[Path]:
    if not raw_hint:
        return []

    hint = raw_hint.replace("\\", "/").lstrip("./")
    if not hint:
        return []

    candidate = Path(hint)
    if candidate.is_absolute():
        return [candidate]
    return [dataset_root / candidate]


def _candidate_base_directories(dataset_root: Path, xml_path: Path) -> list[Path]:
    bases = [xml_path.parent, xml_path.parent.parent, dataset_root]
    unique: list[Path] = []
    seen: set[Path] = set()
    for base_dir in bases:
        if base_dir in seen:
            continue
        seen.add(base_dir)
        unique.append(base_dir)
    return unique


@lru_cache(maxsize=64)
def _child_image_directories(base_dir: Path) -> list[Path]:
    try:
        children = sorted(item for item in base_dir.iterdir() if item.is_dir())
    except OSError:
        return []

    return [
        child
        for child in children
        if any(hint in child.name.lower() for hint in _IMAGE_DIR_HINTS)
    ]


def _existing_path_within_root(dataset_root: Path, candidate: Path) -> Path | None:
    try:
        if not candidate.exists() or not candidate.is_file():
            return None
        resolved_root = dataset_root.resolve()
        resolved_candidate = candidate.resolve()
    except OSError:
        return None

    if not resolved_candidate.is_relative_to(resolved_root):
        return None
    return resolved_candidate


def _candidate_rank(dataset_root: Path, xml_path: Path, candidate: Path) -> tuple[int, int, int, str]:
    xml_parts = xml_path.relative_to(dataset_root).parts[:-1]
    candidate_parts = candidate.relative_to(dataset_root).parts[:-1]
    shared_prefix = 0
    for xml_part, candidate_part in zip(xml_parts, candidate_parts, strict=False):
        if xml_part != candidate_part:
            break
        shared_prefix += 1

    return (
        -shared_prefix,
        abs(len(xml_parts) - len(candidate_parts)),
        len(candidate_parts),
        candidate.as_posix(),
    )
