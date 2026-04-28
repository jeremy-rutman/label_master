from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from label_master.adapters.custom.common import annotation_files_for_spec, split_row_tokens
from label_master.adapters.video_bbox.common import (
    MOT_CHALLENGE_ANNOTATION_FORMAT,
    discover_paired_video_json_sources,
    discover_frame_sequence_layout,
    discover_video_files,
    is_valid_mot_ground_truth_row,
    is_valid_tracking_bbox_row,
    parse_mot_ground_truth_row,
    standard_annotation_files,
)
from label_master.format_specs.registry import (
    FormatSpec,
    TokenizedVideoParserSpec,
    resolve_builtin_format_spec,
)


def _video_files(path: Path) -> list[Path]:
    return list(discover_video_files(path, max_roots=6, max_files=500))


@lru_cache(maxsize=1)
def _video_bbox_spec() -> FormatSpec | None:
    return resolve_builtin_format_spec("video_bbox")


def _standard_annotation_files(path: Path) -> list[Path]:
    spec = _video_bbox_spec()
    if spec is not None and isinstance(spec.parser, TokenizedVideoParserSpec):
        return annotation_files_for_spec(path, spec)
    return standard_annotation_files(path)


def _is_valid_row(row: str, parser: TokenizedVideoParserSpec | None = None) -> bool:
    if parser is not None:
        return _is_valid_tokenized_row(row, parser)

    tokens = row.split()
    if len(tokens) < 2:
        return False

    try:
        int(tokens[0])
        instance_count = int(tokens[1])
    except ValueError:
        return False

    if instance_count < 0:
        return False

    expected_tokens = 2 + instance_count * 5
    if len(tokens) != expected_tokens:
        return False

    cursor = 2
    for _ in range(instance_count):
        try:
            float(tokens[cursor])
            float(tokens[cursor + 1])
            w = float(tokens[cursor + 2])
            h = float(tokens[cursor + 3])
        except ValueError:
            return False
        if w <= 0 or h <= 0:
            return False
        cursor += 5

    return True


def _is_valid_tokenized_row(row: str, parser: TokenizedVideoParserSpec) -> bool:
    tokens = split_row_tokens(row, delimiter=parser.row_format.delimiter)
    header_width = max(parser.row_format.frame_index_field, parser.row_format.object_count_field)
    if len(tokens) < header_width:
        return False

    try:
        frame_index = int(tokens[parser.row_format.frame_index_field - 1]) - parser.row_format.frame_index_base
        object_count = int(tokens[parser.row_format.object_count_field - 1])
    except ValueError:
        return False

    if frame_index < 0 or object_count < 0:
        return False

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
        if parser.row_format.object_fields.class_name is not None:
            if not tokens[cursor + parser.row_format.object_fields.class_name - 1].strip():
                return False
        if parser.row_format.object_fields.class_id is not None:
            try:
                int(tokens[cursor + parser.row_format.object_fields.class_id - 1])
            except ValueError:
                return False
        cursor += parser.row_format.object_group_size

    return True


def _mot_row_frame_number(row: str) -> int | None:
    parsed = parse_mot_ground_truth_row(row)
    if parsed is None:
        return None
    return parsed.frame_number


def _is_valid_paired_video_json_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False

    exist = payload.get("exist")
    gt_rect = payload.get("gt_rect")
    if not isinstance(exist, list) or not isinstance(gt_rect, list):
        return False
    if not exist or len(exist) != len(gt_rect):
        return False

    valid_rows = 0
    sampled_rows = 0
    for exists, rect in zip(exist, gt_rect):
        sampled_rows += 1
        if not isinstance(exists, (bool, int, float)):
            return False
        if not isinstance(rect, list):
            return False
        if len(rect) not in {0, 4}:
            return False
        if len(rect) == 4:
            try:
                for value in rect:
                    float(value)
            except (TypeError, ValueError):
                return False
            valid_rows += 1
        if sampled_rows >= 20:
            break

    return valid_rows > 0


def detect_video_bbox(path: Path, *, sample_limit: int = 500) -> float:
    paired_json_sources = discover_paired_video_json_sources(
        path,
        max_sources=max(1, min(sample_limit, 20)),
    )
    if paired_json_sources:
        valid_sources = 0
        sampled_sources = 0
        for source in paired_json_sources:
            sampled_sources += 1
            try:
                payload = json.loads(source.annotation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if _is_valid_paired_video_json_payload(payload):
                valid_sources += 1

        if valid_sources > 0:
            score = 0.82
            if valid_sources == sampled_sources:
                score += 0.13
            if len(paired_json_sources) >= min(4, sampled_sources):
                score += 0.05
            return min(score, 1.0)

    video_files = _video_files(path)
    video_stems = {file.stem for file in video_files}
    annotation_files = _standard_annotation_files(path)
    tokenized_parser = None
    spec = _video_bbox_spec()
    if spec is not None and isinstance(spec.parser, TokenizedVideoParserSpec):
        tokenized_parser = spec.parser

    if video_files and annotation_files:
        matching_annotations = [file for file in annotation_files if file.stem in video_stems]
        if matching_annotations:
            max_rows = max(1, min(sample_limit, 50))
            max_annotation_files = max(1, min((sample_limit + 9) // 10, 20))
            valid_rows = 0
            sampled_rows = 0
            for annotation_file in matching_annotations[:max_annotation_files]:
                try:
                    lines = annotation_file.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue

                for line in lines:
                    row = line.strip()
                    if not row:
                        continue
                    sampled_rows += 1
                    if _is_valid_row(row, tokenized_parser):
                        valid_rows += 1
                    if sampled_rows >= max_rows:
                        break
                if sampled_rows >= max_rows:
                    break

            if valid_rows > 0:
                score = 0.6
                if len(matching_annotations) == len(annotation_files):
                    score += 0.15
                if valid_rows == sampled_rows:
                    score += 0.25
                return min(score, 1.0)

    sequence_layout = discover_frame_sequence_layout(path)
    if sequence_layout is None:
        return 0.0

    max_rows = max(1, min(sample_limit, 50))
    max_sequences = max(1, min((sample_limit + 9) // 10, 20))
    valid_rows = 0
    sampled_rows = 0
    valid_sequences = 0
    for source in sequence_layout.sources[:max_sequences]:
        try:
            lines = source.ground_truth_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        nonblank_rows = [line.strip() for line in lines if line.strip()]
        sequence_is_valid = bool(source.frame_files)

        for row in nonblank_rows:
            sampled_rows += 1
            if source.ground_truth_format == MOT_CHALLENGE_ANNOTATION_FORMAT:
                try:
                    frame_number = _mot_row_frame_number(row)
                except ValueError:
                    sequence_is_valid = False
                else:
                    if frame_number is not None and frame_number > len(source.frame_files):
                        sequence_is_valid = False
                    if is_valid_mot_ground_truth_row(row):
                        valid_rows += 1
            elif is_valid_tracking_bbox_row(row):
                valid_rows += 1
            else:
                sequence_is_valid = False
            if sampled_rows >= max_rows:
                break
        if (
            source.ground_truth_format != MOT_CHALLENGE_ANNOTATION_FORMAT
            and len(source.frame_files) != len(nonblank_rows)
        ):
            sequence_is_valid = False
        if sequence_is_valid:
            valid_sequences += 1
        if sampled_rows >= max_rows:
            break

    if valid_rows == 0:
        return 0.0

    frame_names = set(sequence_layout.frame_directory_names)
    ground_truth_names = set(sequence_layout.ground_truth_sequence_names)
    score = 0.7
    if frame_names == ground_truth_names:
        score += 0.15
    if valid_sequences == len(sequence_layout.sources):
        score += 0.05
    if valid_rows == sampled_rows:
        score += 0.1
    return min(score, 1.0)
