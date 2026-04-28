from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from label_master.adapters.video_bbox.common import VIDEO_EXTENSIONS, resolve_video_file
from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import FormatSpec, TokenizedVideoParserSpec


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
        return [token.strip() for token in row.split(",") if token.strip()]
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


def _require_tokenized_video_parser(spec: FormatSpec) -> TokenizedVideoParserSpec:
    parser = spec.parser
    if not isinstance(parser, TokenizedVideoParserSpec):
        raise TypeError("Format spec parser must be tokenized_video")
    return parser
