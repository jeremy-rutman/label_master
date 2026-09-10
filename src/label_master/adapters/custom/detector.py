from __future__ import annotations

from pathlib import Path

from label_master.adapters.custom.common import (
    annotation_files_for_spec,
    bdd100k_annotation_records,
    resolve_bdd100k_image_path,
    resolve_spec_video_file,
    split_row_tokens,
)
from label_master.adapters.custom.waymo_parquet import (
    WAYMO_PARQUET_FORMAT_ID,
    detect_waymo_parquet_dataset,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import (
    Bdd100kImageLabelsParserSpec,
    CountPrefixedObjectsRowFormatSpec,
    FormatSpec,
    SingleObjectVideoRowFormatSpec,
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

        for line in lines[parser.skip_rows :]:
            row = line.strip()
            if not row:
                continue
            sampled_rows += 1
            if _is_valid_tokenized_video_row(row, parser):
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
    best_score = detect_waymo_parquet_dataset(path, sample_limit=sample_limit)
    best_spec_id: str | None = WAYMO_PARQUET_FORMAT_ID if best_score > 0 else None

    for spec in custom_format_specs(path):
        score = max(
            _detect_bdd100k_image_labels_format(path, spec, sample_limit=sample_limit),
            _detect_tokenized_video_format(path, spec, sample_limit=sample_limit),
        )
        if score > best_score:
            best_score = score
            best_spec_id = spec.format_id

    return best_score, best_spec_id


def _is_valid_tokenized_video_row(row: str, parser: TokenizedVideoParserSpec) -> bool:
    if isinstance(parser.row_format, CountPrefixedObjectsRowFormatSpec):
        return _is_valid_count_prefixed_row(row, parser)
    if isinstance(parser.row_format, SingleObjectVideoRowFormatSpec):
        return _is_valid_single_object_row(row, parser)
    return False


def _is_valid_count_prefixed_row(row: str, parser: TokenizedVideoParserSpec) -> bool:
    if not isinstance(parser.row_format, CountPrefixedObjectsRowFormatSpec):
        return False
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


def _is_valid_single_object_row(row: str, parser: TokenizedVideoParserSpec) -> bool:
    if not isinstance(parser.row_format, SingleObjectVideoRowFormatSpec):
        return False

    row_format = parser.row_format
    tokens = split_row_tokens(row, delimiter=row_format.delimiter)
    if len(tokens) < _single_object_required_token_count(row_format):
        return False

    try:
        frame_index = int(tokens[row_format.frame_index_field - 1]) - row_format.frame_index_base
    except ValueError:
        return False
    if frame_index < 0:
        return False

    if row_format.class_name_field is not None and not tokens[row_format.class_name_field - 1].strip():
        return False
    if row_format.class_id_field is not None:
        try:
            int(tokens[row_format.class_id_field - 1])
        except ValueError:
            return False

    try:
        if row_format.bbox_fields is not None:
            width = float(tokens[row_format.bbox_fields.width - 1])
            height = float(tokens[row_format.bbox_fields.height - 1])
            return width > 0 and height > 0

        quad = row_format.quadrilateral_fields
        if quad is None:
            return False
        xs = [
            float(tokens[quad.x1 - 1]),
            float(tokens[quad.x2 - 1]),
            float(tokens[quad.x3 - 1]),
            float(tokens[quad.x4 - 1]),
        ]
        ys = [
            float(tokens[quad.y1 - 1]),
            float(tokens[quad.y2 - 1]),
            float(tokens[quad.y3 - 1]),
            float(tokens[quad.y4 - 1]),
        ]
    except ValueError:
        return False

    return max(xs) > min(xs) and max(ys) > min(ys)


def _single_object_required_token_count(row_format: SingleObjectVideoRowFormatSpec) -> int:
    positions = [row_format.frame_index_field]
    if row_format.class_name_field is not None:
        positions.append(row_format.class_name_field)
    if row_format.class_id_field is not None:
        positions.append(row_format.class_id_field)
    if row_format.bbox_fields is not None:
        positions.extend(
            [
                row_format.bbox_fields.xmin,
                row_format.bbox_fields.ymin,
                row_format.bbox_fields.width,
                row_format.bbox_fields.height,
            ]
        )
    if row_format.quadrilateral_fields is not None:
        positions.extend(
            [
                row_format.quadrilateral_fields.x1,
                row_format.quadrilateral_fields.y1,
                row_format.quadrilateral_fields.x2,
                row_format.quadrilateral_fields.y2,
                row_format.quadrilateral_fields.x3,
                row_format.quadrilateral_fields.y3,
                row_format.quadrilateral_fields.x4,
                row_format.quadrilateral_fields.y4,
            ]
        )
    return max(positions)


def _detect_bdd100k_image_labels_format(path: Path, spec: FormatSpec, *, sample_limit: int) -> float:
    parser = spec.parser
    if not isinstance(parser, Bdd100kImageLabelsParserSpec):
        return 0.0

    try:
        records = bdd100k_annotation_records(path, spec)
    except ValidationError:
        return 0.0

    if not records:
        return 0.0

    max_records = max(1, min(sample_limit, 50))
    sampled_records = 0
    valid_records = 0
    matched_images = 0
    sampled_box_labels = 0
    valid_box_labels = 0

    for record in records[:max_records]:
        image_name = str(record.get(parser.image_name_field, "")).strip()
        labels = record.get(parser.labels_field)
        if not image_name or not isinstance(labels, list):
            continue

        sampled_records += 1
        valid_records += 1
        try:
            _image_rel, image_path = resolve_bdd100k_image_path(path, spec, raw_name=image_name)
        except ValidationError:
            image_path = path / "__missing__"
        if image_path.is_file():
            matched_images += 1

        for label in labels:
            if not isinstance(label, dict):
                continue
            if parser.bbox_field not in label:
                continue
            sampled_box_labels += 1
            if _is_valid_bdd100k_box_label(label, parser):
                valid_box_labels += 1

    if sampled_records == 0 or valid_records == 0:
        return 0.0

    score = 0.55
    if valid_records == sampled_records:
        score += 0.15
    elif valid_records > 0:
        score += 0.05

    if sampled_box_labels > 0:
        if valid_box_labels == sampled_box_labels:
            score += 0.15
        elif valid_box_labels > 0:
            score += 0.08

    if matched_images == sampled_records:
        score += 0.1
    elif matched_images > 0:
        score += 0.05

    score += parser.score_boost
    return min(score, 1.0)


def _is_valid_bdd100k_box_label(label: dict[str, object], parser: Bdd100kImageLabelsParserSpec) -> bool:
    category = str(label.get(parser.category_field, "")).strip()
    bbox = label.get(parser.bbox_field)
    if not category or not isinstance(bbox, dict):
        return False

    try:
        x1 = float(bbox[parser.x1_field])
        y1 = float(bbox[parser.y1_field])
        x2 = float(bbox[parser.x2_field])
        y2 = float(bbox[parser.y2_field])
    except (KeyError, TypeError, ValueError):
        return False

    return x2 > x1 and y2 > y1
