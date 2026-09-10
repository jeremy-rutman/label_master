from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from label_master.adapters.video_bbox.common import VIDEO_EXTENSIONS, resolve_video_file
from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import (
    Bdd100kImageLabelsParserSpec,
    FormatSpec,
    TokenizedVideoParserSpec,
)
from label_master.infra.filesystem import safe_resolve


def annotation_files_for_spec(dataset_root: Path, spec: FormatSpec) -> list[Path]:
    parser = _require_tokenized_video_parser(spec)
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in parser.annotation_globs:
        for path in sorted(dataset_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def split_row_tokens(row: str, *, delimiter: str) -> list[str]:
    if delimiter == "comma":
        return [token.strip() for token in row.split(",")]
    return row.split()


def build_image_rel_path(spec: FormatSpec, *, video_stem: str, frame_index: int) -> str:
    parser = _require_tokenized_video_parser(spec)
    return parser.image_path_template.format(video_stem=video_stem, frame_index=frame_index)


def resolve_spec_video_file(dataset_root: Path, video_stem: str, spec: FormatSpec) -> Path | None:
    parser = _require_tokenized_video_parser(spec)
    for root_name in parser.video_roots:
        root = (dataset_root / root_name).resolve()
        if not root.is_dir():
            continue
        matches = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.stem == video_stem and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if matches:
            return matches[0]
    return resolve_video_file(dataset_root, video_stem)


def probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValidationError("Required binary is unavailable: ffprobe")

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValidationError(
            f"Unable to probe video metadata: {video_path.name}",
            context={"stderr": completed.stderr.strip()},
        )

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid ffprobe metadata payload for: {video_path.name}") from exc

    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ValidationError(f"Video metadata missing stream data: {video_path.name}")

    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValidationError(f"Video metadata stream must be an object: {video_path.name}")

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValidationError(f"Video dimensions must be positive: {video_path.name}")
    return width, height


def bdd100k_annotation_records(dataset_root: Path, spec: FormatSpec) -> list[dict[str, Any]]:
    parser = _require_bdd100k_image_labels_parser(spec)
    annotations_path = dataset_root / parser.annotations_file
    if not annotations_path.exists():
        raise ValidationError(f"BDD100K annotations file not found: {annotations_path}")

    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"BDD100K annotations payload is not valid JSON: {annotations_path.name}"
        ) from exc
    except OSError as exc:
        raise ValidationError(f"Unable to read BDD100K annotations file: {annotations_path}") from exc

    if parser.records_key is not None:
        if not isinstance(payload, dict):
            raise ValidationError(
                f"BDD100K annotations payload must be an object when records_key is set: {annotations_path.name}"
            )
        payload = payload.get(parser.records_key, [])

    if not isinstance(payload, list):
        raise ValidationError(
            f"BDD100K annotations payload must contain an array of image records: {annotations_path.name}"
        )

    records: list[dict[str, Any]] = []
    for raw_record in payload:
        if not isinstance(raw_record, dict):
            raise ValidationError(
                f"BDD100K image record must be an object: {annotations_path.name}"
            )
        records.append(raw_record)
    return records


def build_bdd100k_image_rel_path(spec: FormatSpec, *, raw_name: str) -> str:
    parser = _require_bdd100k_image_labels_parser(spec)
    candidate = raw_name.strip().replace("\\", "/")
    if not candidate:
        raise ValidationError("BDD100K image name field cannot be empty")

    image_root = parser.image_root.strip().strip("/")
    if image_root and candidate != image_root and not candidate.startswith(f"{image_root}/"):
        candidate = f"{image_root}/{candidate}"
    return candidate


def resolve_bdd100k_image_path(dataset_root: Path, spec: FormatSpec, *, raw_name: str) -> tuple[str, Path]:
    parser = _require_bdd100k_image_labels_parser(spec)
    image_rel = build_bdd100k_image_rel_path(spec, raw_name=raw_name)
    image_path = safe_resolve(dataset_root, image_rel)
    if image_path.is_file():
        return image_rel, image_path

    normalized_name = raw_name.strip().replace("\\", "/")
    suffix_parts = tuple(part for part in PurePosixPath(normalized_name).parts if part not in {"", "."})
    if not suffix_parts:
        return image_rel, image_path

    image_root = parser.image_root.strip().strip("/")
    search_root = safe_resolve(dataset_root, image_root) if image_root else dataset_root.expanduser().resolve()
    if not search_root.is_dir():
        return image_rel, image_path

    candidate_index = _bdd100k_image_candidates(search_root)
    matches = [
        relative_path
        for relative_path in candidate_index.get(suffix_parts[-1], ())
        if relative_path.parts[-len(suffix_parts) :] == suffix_parts
    ]
    if not matches:
        return image_rel, image_path
    if len(matches) > 1:
        raise ValidationError(
            f"BDD100K image name is ambiguous under image_root: {normalized_name}",
            context={
                "format_id": spec.format_id,
                "matches": [match.as_posix() for match in matches[:10]],
            },
        )

    resolved_rel = matches[0].as_posix()
    if image_root:
        resolved_rel = f"{image_root}/{resolved_rel}"
    resolved_path = safe_resolve(dataset_root, resolved_rel)
    return resolved_rel, resolved_path


def _bdd100k_image_candidates(search_root: Path) -> dict[str, tuple[PurePosixPath, ...]]:
    return _bdd100k_image_candidates_cached(str(search_root.resolve()))


@lru_cache(maxsize=32)
def _bdd100k_image_candidates_cached(search_root: str) -> dict[str, tuple[PurePosixPath, ...]]:
    root = Path(search_root)
    grouped: dict[str, list[PurePosixPath]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = PurePosixPath(path.relative_to(root).as_posix())
        grouped.setdefault(path.name, []).append(relative_path)

    return {
        basename: tuple(sorted(paths, key=lambda path: path.as_posix()))
        for basename, paths in grouped.items()
    }


def _require_tokenized_video_parser(spec: FormatSpec) -> TokenizedVideoParserSpec:
    parser = spec.parser
    if not isinstance(parser, TokenizedVideoParserSpec):
        raise TypeError("Format spec parser must be tokenized_video")
    return parser


def _require_bdd100k_image_labels_parser(spec: FormatSpec) -> Bdd100kImageLabelsParserSpec:
    parser = spec.parser
    if not isinstance(parser, Bdd100kImageLabelsParserSpec):
        raise TypeError("Format spec parser must be bdd100k_image_labels")
    return parser
