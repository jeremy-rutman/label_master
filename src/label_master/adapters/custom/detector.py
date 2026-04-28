from __future__ import annotations

from pathlib import Path

from label_master.adapters.custom.common import (
    annotation_files_for_spec,
    resolve_spec_video_file,
    split_row_tokens,
)
from label_master.format_specs.registry import (
    FormatSpec,
    TokenizedVideoParserSpec,
    custom_format_specs,
)


def _detect_tokenized_video_format(path: Path, spec: FormatSpec, *, sample_limit: int) -> float:
    parser = spec.parser
    if not isinstance(parser, TokenizedVideoParserSpec):
        return 0.0

    annotation_files = annotation_files_for_spec(path, spec)
    if not annotation_files:
        return 0.0

    max_rows = max(1, min(sample_limit, 50))
    max_files = max(1, min((sample_limit + 9) // 10, 20))
    sampled_rows = 0
    valid_rows = 0
    matched_videos = 0

    for annotation_file in annotation_files[:max_files]:
        if resolve_spec_video_file(path, annotation_file.stem, spec) is not None:
            matched_videos += 1

        try:
            lines = annotation_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in lines:
            row = line.strip()
            if not row:
                continue
            sampled_rows += 1
            if _is_valid_count_prefixed_row(row, parser):
                valid_rows += 1
            if sampled_rows >= max_rows:
                break
        if sampled_rows >= max_rows:
            break

    if sampled_rows == 0 or valid_rows == 0 or matched_videos == 0:
        return 0.0

    score = 0.6
    if valid_rows == sampled_rows:
        score += 0.2
    if matched_videos == min(len(annotation_files), max_files):
        score += 0.15
    if parser.video_roots:
        score += 0.05
    score += parser.score_boost
    return min(score, 1.0)


def detect_custom_format(path: Path, *, sample_limit: int = 500) -> tuple[float, str | None]:
    best_score = 0.0
    best_spec_id: str | None = None

    for spec in custom_format_specs(path):
        score = _detect_tokenized_video_format(path, spec, sample_limit=sample_limit)
        if score > best_score:
            best_score = score
            best_spec_id = spec.format_id

    return best_score, best_spec_id


def _is_valid_count_prefixed_row(row: str, parser: TokenizedVideoParserSpec) -> bool:
    tokens = split_row_tokens(row, delimiter=parser.row_format.delimiter)
    if len(tokens) < max(parser.row_format.frame_index_field, parser.row_format.object_count_field):
        return False

    try:
        int(tokens[parser.row_format.frame_index_field - 1])
        object_count = int(tokens[parser.row_format.object_count_field - 1])
    except ValueError:
        return False

    if object_count < 0:
        return False

    header_width = max(parser.row_format.frame_index_field, parser.row_format.object_count_field)
    expected_tokens = header_width + object_count * parser.row_format.object_group_size
    if len(tokens) != expected_tokens:
        return False

    cursor = header_width
    for _ in range(object_count):
        try:
            float(tokens[cursor + parser.row_format.object_fields.xmin - 1])
            float(tokens[cursor + parser.row_format.object_fields.ymin - 1])
            width = float(tokens[cursor + parser.row_format.object_fields.width - 1])
            height = float(tokens[cursor + parser.row_format.object_fields.height - 1])
        except ValueError:
            return False
        if width <= 0 or height <= 0:
            return False
        cursor += parser.row_format.object_group_size

    return True
