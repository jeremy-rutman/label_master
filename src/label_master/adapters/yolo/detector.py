from __future__ import annotations

from functools import lru_cache
from itertools import islice
from pathlib import Path

from label_master.format_specs.registry import (
    TokenizedImageLabelsParserSpec,
    resolve_builtin_format_spec,
)


@lru_cache(maxsize=1)
def _yolo_parser() -> TokenizedImageLabelsParserSpec | None:
    spec = resolve_builtin_format_spec("yolo")
    if spec is None or not isinstance(spec.parser, TokenizedImageLabelsParserSpec):
        return None
    return spec.parser


def _parents_within_root(file_path: Path, root: Path) -> list[Path]:
    parents: list[Path] = []
    for parent in file_path.parents:
        parents.append(parent)
        if parent == root:
            break
    return parents


def detect_yolo(path: Path, *, sample_limit: int = 500) -> float:
    parser = _yolo_parser()
    if parser is None:
        return 0.0

    max_txt_files = max(1, min(sample_limit, 50))
    txt_candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in parser.label_globs:
        for file in path.glob(pattern):
            if not file.is_file() or file in seen:
                continue
            seen.add(file)
            txt_candidates.append(file)
    txt_files = list(islice(iter(sorted(txt_candidates)), max_txt_files))
    if not txt_files:
        return 0.0

    yolo_like_rows = 0
    for txt_file in txt_files:
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if not row:
                continue
            tokens = row.split(",") if parser.row_format.delimiter == "comma" else row.split()
            if len(tokens) == 5:
                try:
                    int(tokens[parser.row_format.class_id_field - 1])
                    [
                        float(tokens[index - 1])
                        for index in (
                            parser.row_format.x_center_field,
                            parser.row_format.y_center_field,
                            parser.row_format.width_field,
                            parser.row_format.height_field,
                        )
                    ]
                except ValueError:
                    break
                yolo_like_rows += 1
            break

    if yolo_like_rows == 0:
        return 0.0

    score = 0.7
    parent_directories = [
        parent
        for file in txt_files
        for parent in _parents_within_root(file, path)
    ]
    has_label_named_directory = any("label" in parent.name.lower() for parent in parent_directories)
    if has_label_named_directory:
        score += 0.2

    try:
        top_level_directories = [item for item in path.iterdir() if item.is_dir()]
    except OSError:
        top_level_directories = []

    has_images_dir = any("image" in directory.name.lower() for directory in top_level_directories) or any(
        "image" in parent.name.lower() for parent in parent_directories
    )
    if has_images_dir:
        score += 0.1

    return min(score + parser.score_boost, 1.0)
