from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image

from label_master.adapters.custom.common import (
    annotation_files_for_spec,
    bdd100k_annotation_records,
    build_image_rel_path,
    probe_video_dimensions,
    resolve_bdd100k_image_path,
    resolve_spec_video_file,
    split_row_tokens,
)
from label_master.adapters.custom.waymo_parquet import (
    WAYMO_PARQUET_FORMAT_ID,
    detect_waymo_parquet_dataset,
    read_waymo_parquet_dataset,
)
from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.value_objects import ValidationError
from label_master.format_specs.registry import (
    Bdd100kImageLabelsParserSpec,
    CountPrefixedObjectsRowFormatSpec,
    FormatSpec,
    SingleObjectVideoRowFormatSpec,
    TokenizedVideoParserSpec,
    load_custom_format_spec_from_path,
    resolve_custom_format_spec,
)
from label_master.infra.filesystem import (
    InputPathFilter,
    relative_path_matches_input_filter,
)


def read_custom_dataset(
    dataset_root: Path,
    *,
    format_id: str | None = None,
    format_path: Path | None = None,
    input_path_filter: InputPathFilter | None = None,
    max_records: int | None = None,
) -> AnnotationDataset:
    if format_path is None and format_id == WAYMO_PARQUET_FORMAT_ID:
        return read_waymo_parquet_dataset(
            dataset_root,
            input_path_filter=input_path_filter,
            max_records=max_records,
        )

    if format_path is None and format_id is None:
        waymo_score = detect_waymo_parquet_dataset(dataset_root)
        if waymo_score > 0:
            return read_waymo_parquet_dataset(
                dataset_root,
                input_path_filter=input_path_filter,
                max_records=max_records,
            )

    spec = _resolve_spec(dataset_root, format_id=format_id, format_path=format_path)
    parser = spec.parser
    if isinstance(parser, Bdd100kImageLabelsParserSpec):
        return _read_bdd100k_image_labels_dataset(
            dataset_root,
            spec=spec,
            input_path_filter=input_path_filter,
            max_records=max_records,
        )
    if not isinstance(parser, TokenizedVideoParserSpec):
        raise ValidationError(f"Unsupported custom format parser for: {spec.format_id}")

    annotation_files = annotation_files_for_spec(dataset_root, spec)
    if not annotation_files:
        raise ValidationError(f"No annotation files found for custom format: {spec.format_id}")

    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    categories_by_name: dict[str, int] = {}
    categories: dict[int, CategoryRecord] = {}

    for annotation_file in annotation_files:
        video_stem = annotation_file.stem
        video_path = resolve_spec_video_file(dataset_root, video_stem, spec)
        if video_path is None:
            raise ValidationError(
                f"No source video found for custom format annotation file stem: {video_stem}",
                context={"format_id": spec.format_id},
            )
        width, height = probe_video_dimensions(video_path)

        with annotation_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line_no <= parser.skip_rows:
                    continue
                row = line.strip()
                if not row:
                    continue

                parsed_image, parsed_annotations = _parse_tokenized_video_row(
                    row,
                    parser=parser,
                    video_stem=video_stem,
                    annotation_rel=annotation_file.relative_to(dataset_root).as_posix(),
                    line_no=line_no,
                    width=width,
                    height=height,
                    spec=spec,
                )
                if not relative_path_matches_input_filter(
                    parsed_image.file_name,
                    input_path_filter=input_path_filter,
                ):
                    continue

                existing = images_by_id.get(parsed_image.image_id)
                if existing is not None and existing.file_name != parsed_image.file_name:
                    raise ValidationError(
                        f"Duplicate custom-format image_id with conflicting paths: {parsed_image.image_id}"
                    )
                images_by_id[parsed_image.image_id] = parsed_image

                for annotation in parsed_annotations:
                    category_key = f"id:{annotation.class_id}" if annotation.class_id is not None else f"name:{annotation.class_name}"
                    if category_key not in categories_by_name:
                        class_id = (
                            annotation.class_id
                            if annotation.class_id is not None
                            else len(categories_by_name)
                        )
                        while class_id in categories:
                            class_id += 1
                        categories_by_name[category_key] = class_id
                        categories[class_id] = CategoryRecord(
                            class_id=class_id,
                            name=annotation.class_name or f"class_{class_id}",
                        )

                    resolved_class_id = categories_by_name[category_key]
                    annotations.append(
                        AnnotationRecord(
                            annotation_id=annotation.annotation_id,
                            image_id=annotation.image_id,
                            class_id=resolved_class_id,
                            bbox_xywh_abs=annotation.bbox_xywh_abs,
                        )
                    )

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.CUSTOM,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda ann: ann.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="custom_format_reader",
            details={
                "format_id": spec.format_id,
                "display_name": spec.display_name,
                "media_kind": "video_collection",
            },
        ),
    )


def _read_bdd100k_image_labels_dataset(
    dataset_root: Path,
    *,
    spec: FormatSpec,
    input_path_filter: InputPathFilter | None = None,
    max_records: int | None = None,
) -> AnnotationDataset:
    parser = spec.parser
    if not isinstance(parser, Bdd100kImageLabelsParserSpec):
        raise TypeError("Format spec parser must be bdd100k_image_labels")

    resolved_root = dataset_root.expanduser().resolve()
    all_records = bdd100k_annotation_records(resolved_root, spec)
    records_total = len(all_records)
    records = all_records[:max_records] if max_records is not None and max_records > 0 else all_records

    images_by_id: dict[str, ImageRecord] = {}
    annotations: list[AnnotationRecord] = []
    categories_by_name: dict[str, int] = {}
    categories: dict[int, CategoryRecord] = {}

    for record in records:
        image_name = str(record.get(parser.image_name_field, "")).strip()
        if not image_name:
            raise ValidationError(f"BDD100K image record missing {parser.image_name_field}")

        image_rel, image_path = resolve_bdd100k_image_path(resolved_root, spec, raw_name=image_name)
        if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
            continue
        if not image_path.is_file():
            raise ValidationError(
                f"BDD100K image file not found for record: {image_rel}",
                context={"format_id": spec.format_id},
            )
        width, height = _probe_image_size(image_path)

        image_id = image_rel
        images_by_id.setdefault(
            image_id,
            ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
            ),
        )

        labels = record.get(parser.labels_field, [])
        if not isinstance(labels, list):
            raise ValidationError(
                f"BDD100K image record labels field must be a list: {image_rel}",
                context={"format_id": spec.format_id},
            )

        for label_index, raw_label in enumerate(labels, start=1):
            if not isinstance(raw_label, dict):
                raise ValidationError(
                    f"BDD100K label entry must be an object: {image_rel}",
                    context={"format_id": spec.format_id},
                )

            bbox = raw_label.get(parser.bbox_field)
            if bbox is None:
                continue
            if not isinstance(bbox, dict):
                raise ValidationError(
                    f"BDD100K bbox field must be an object: {image_rel}",
                    context={"format_id": spec.format_id},
                )

            category_name = str(raw_label.get(parser.category_field, "")).strip()
            if not category_name:
                raise ValidationError(
                    f"BDD100K label category is required: {image_rel}",
                    context={"format_id": spec.format_id},
                )

            try:
                x1 = float(bbox[parser.x1_field])
                y1 = float(bbox[parser.y1_field])
                x2 = float(bbox[parser.x2_field])
                y2 = float(bbox[parser.y2_field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"BDD100K bbox coordinates must be numeric: {image_rel}",
                    context={"format_id": spec.format_id},
                ) from exc

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w <= 0 or bbox_h <= 0:
                raise ValidationError(
                    f"BDD100K bbox width/height must be positive: {image_rel}",
                    context={"format_id": spec.format_id},
                )

            class_id = categories_by_name.setdefault(category_name, len(categories_by_name))
            categories.setdefault(
                class_id,
                CategoryRecord(
                    class_id=class_id,
                    name=category_name,
                ),
            )

            label_id = (
                str(raw_label.get(parser.label_id_field, "")).strip()
                if parser.label_id_field is not None
                else ""
            )
            annotations.append(
                AnnotationRecord(
                    annotation_id=label_id or f"{image_rel}:{label_index:03d}",
                    image_id=image_id,
                    class_id=class_id,
                    bbox_xywh_abs=(x1, y1, bbox_w, bbox_h),
                )
            )

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.CUSTOM,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda ann: ann.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(resolved_root),
            loader="custom_format_reader",
            details={
                "format_id": spec.format_id,
                "display_name": spec.display_name,
                "media_kind": "image_collection",
                "annotations_file": parser.annotations_file,
                "image_root": parser.image_root,
                "records_loaded": str(len(records)),
                "records_total": str(records_total),
            },
        ),
    )


@lru_cache(maxsize=512)
def _probe_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as opened:
        return opened.size


class _ParsedPendingAnnotation:
    def __init__(
        self,
        *,
        annotation_id: str,
        image_id: str,
        bbox_xywh_abs: tuple[float, float, float, float],
        class_name: str | None,
        class_id: int | None,
    ) -> None:
        self.annotation_id = annotation_id
        self.image_id = image_id
        self.bbox_xywh_abs = bbox_xywh_abs
        self.class_name = class_name
        self.class_id = class_id


def _parse_tokenized_video_row(
    row: str,
    *,
    parser: TokenizedVideoParserSpec,
    video_stem: str,
    annotation_rel: str,
    line_no: int,
    width: int,
    height: int,
    spec: FormatSpec,
) -> tuple[ImageRecord, list[_ParsedPendingAnnotation]]:
    if isinstance(parser.row_format, CountPrefixedObjectsRowFormatSpec):
        return _parse_count_prefixed_row(
            row,
            parser=parser,
            video_stem=video_stem,
            annotation_rel=annotation_rel,
            line_no=line_no,
            width=width,
            height=height,
            spec=spec,
        )
    if isinstance(parser.row_format, SingleObjectVideoRowFormatSpec):
        return _parse_single_object_row(
            row,
            parser=parser,
            video_stem=video_stem,
            annotation_rel=annotation_rel,
            line_no=line_no,
            width=width,
            height=height,
            spec=spec,
        )
    raise ValidationError(
        f"Unsupported custom tokenized video row format for: {spec.format_id}",
        context={"format_id": spec.format_id},
    )


def _parse_count_prefixed_row(
    row: str,
    *,
    parser: TokenizedVideoParserSpec,
    video_stem: str,
    annotation_rel: str,
    line_no: int,
    width: int,
    height: int,
    spec: FormatSpec,
) -> tuple[ImageRecord, list[_ParsedPendingAnnotation]]:
    if not isinstance(parser.row_format, CountPrefixedObjectsRowFormatSpec):
        raise ValidationError(
            f"Custom format row parser mismatch for: {spec.format_id}",
            context={"format_id": spec.format_id},
        )
    tokens = split_row_tokens(row, delimiter=parser.row_format.delimiter)
    header_width = max(parser.row_format.frame_index_field, parser.row_format.object_count_field)
    if len(tokens) < header_width:
        raise ValidationError(
            f"Custom format row is too short at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id, "row": row},
        )

    try:
        frame_index = int(tokens[parser.row_format.frame_index_field - 1]) - parser.row_format.frame_index_base
        object_count = int(tokens[parser.row_format.object_count_field - 1])
    except ValueError as exc:
        raise ValidationError(
            f"Custom format row must start with numeric frame/object counts at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id},
        ) from exc

    if frame_index < 0 or object_count < 0:
        raise ValidationError(
            f"Custom format row cannot use negative frame/object counts at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id},
        )

    expected_tokens = header_width + object_count * parser.row_format.object_group_size
    if len(tokens) != expected_tokens:
        raise ValidationError(
            f"Custom format row token count mismatch at {annotation_rel}:{line_no}",
            context={
                "format_id": spec.format_id,
                "expected_tokens": str(expected_tokens),
                "actual_tokens": str(len(tokens)),
            },
        )

    image_id = f"{video_stem}:{frame_index:06d}"
    image = ImageRecord(
        image_id=image_id,
        file_name=build_image_rel_path(spec, video_stem=video_stem, frame_index=frame_index),
        width=width,
        height=height,
    )

    parsed: list[_ParsedPendingAnnotation] = []
    cursor = header_width
    for object_index in range(object_count):
        try:
            xmin = float(tokens[cursor + parser.row_format.object_fields.xmin - 1])
            ymin = float(tokens[cursor + parser.row_format.object_fields.ymin - 1])
            bbox_w = float(tokens[cursor + parser.row_format.object_fields.width - 1])
            bbox_h = float(tokens[cursor + parser.row_format.object_fields.height - 1])
        except ValueError as exc:
            raise ValidationError(
                f"Custom format bbox fields must be numeric at {annotation_rel}:{line_no}",
                context={"format_id": spec.format_id},
            ) from exc

        if bbox_w <= 0 or bbox_h <= 0:
            raise ValidationError(
                f"Custom format bbox width/height must be positive at {annotation_rel}:{line_no}",
                context={"format_id": spec.format_id},
            )

        class_name: str | None = None
        class_id: int | None = None
        if parser.row_format.object_fields.class_name is not None:
            class_name = tokens[cursor + parser.row_format.object_fields.class_name - 1].strip()
            if not class_name:
                raise ValidationError(
                    f"Custom format class_name is required at {annotation_rel}:{line_no}",
                    context={"format_id": spec.format_id},
                )
        if parser.row_format.object_fields.class_id is not None:
            try:
                class_id = int(tokens[cursor + parser.row_format.object_fields.class_id - 1])
            except ValueError as exc:
                raise ValidationError(
                    f"Custom format class_id must be an integer at {annotation_rel}:{line_no}",
                    context={"format_id": spec.format_id},
                ) from exc

        parsed.append(
            _ParsedPendingAnnotation(
                annotation_id=f"{annotation_rel}:{line_no:06d}:{object_index + 1:03d}",
                image_id=image_id,
                bbox_xywh_abs=(xmin, ymin, bbox_w, bbox_h),
                class_name=class_name,
                class_id=class_id,
            )
        )
        cursor += parser.row_format.object_group_size

    return image, parsed


def _parse_single_object_row(
    row: str,
    *,
    parser: TokenizedVideoParserSpec,
    video_stem: str,
    annotation_rel: str,
    line_no: int,
    width: int,
    height: int,
    spec: FormatSpec,
) -> tuple[ImageRecord, list[_ParsedPendingAnnotation]]:
    if not isinstance(parser.row_format, SingleObjectVideoRowFormatSpec):
        raise ValidationError(
            f"Custom format row parser mismatch for: {spec.format_id}",
            context={"format_id": spec.format_id},
        )

    row_format = parser.row_format
    tokens = split_row_tokens(row, delimiter=row_format.delimiter)
    required_tokens = _single_object_required_token_count(row_format)
    if len(tokens) < required_tokens:
        raise ValidationError(
            f"Custom format single-object row is too short at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id, "row": row},
        )

    try:
        frame_index = int(tokens[row_format.frame_index_field - 1]) - row_format.frame_index_base
    except ValueError as exc:
        raise ValidationError(
            f"Custom format frame index must be an integer at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id},
        ) from exc

    if frame_index < 0:
        raise ValidationError(
            f"Custom format frame index cannot be negative at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id},
        )

    class_name: str | None = None
    class_id: int | None = None
    if row_format.class_name_field is not None:
        class_name = tokens[row_format.class_name_field - 1].strip()
        if not class_name:
            if row_format.skip_empty_class:
                image_id = f"{video_stem}:{frame_index:06d}"
                image_path = parser.image_path_template.format(
                    video_stem=video_stem, frame_index=frame_index
                )
                image = ImageRecord(
                    image_id=image_id,
                    file_name=image_path,
                    width=width,
                    height=height,
                )
                return image, []
            raise ValidationError(
                f"Custom format class_name is required at {annotation_rel}:{line_no}",
                context={"format_id": spec.format_id},
            )
    if row_format.class_id_field is not None:
        try:
            class_id = int(tokens[row_format.class_id_field - 1])
        except ValueError as exc:
            raise ValidationError(
                f"Custom format class_id must be an integer at {annotation_rel}:{line_no}",
                context={"format_id": spec.format_id},
            ) from exc

    bbox_xywh_abs = _parse_single_object_bbox(
        tokens,
        row_format=row_format,
        annotation_rel=annotation_rel,
        line_no=line_no,
        spec=spec,
    )
    if bbox_xywh_abs is None:
        image_id = f"{video_stem}:{frame_index:06d}"
        return ImageRecord(
            image_id=image_id,
            file_name=build_image_rel_path(spec, video_stem=video_stem, frame_index=frame_index),
            width=width,
            height=height,
        ), []

    image_id = f"{video_stem}:{frame_index:06d}"
    image = ImageRecord(
        image_id=image_id,
        file_name=build_image_rel_path(spec, video_stem=video_stem, frame_index=frame_index),
        width=width,
        height=height,
    )

    return image, [
        _ParsedPendingAnnotation(
            annotation_id=f"{annotation_rel}:{line_no:06d}:001",
            image_id=image_id,
            bbox_xywh_abs=bbox_xywh_abs,
            class_name=class_name,
            class_id=class_id,
        )
    ]


def _parse_single_object_bbox(
    tokens: list[str],
    *,
    row_format: SingleObjectVideoRowFormatSpec,
    annotation_rel: str,
    line_no: int,
    spec: FormatSpec,
) -> tuple[float, float, float, float]:
    try:
        if row_format.bbox_fields is not None:
            xmin = float(tokens[row_format.bbox_fields.xmin - 1])
            ymin = float(tokens[row_format.bbox_fields.ymin - 1])
            bbox_w = float(tokens[row_format.bbox_fields.width - 1])
            bbox_h = float(tokens[row_format.bbox_fields.height - 1])
            if bbox_w <= 0 or bbox_h <= 0:
                raise ValidationError(
                    f"Custom format bbox width/height must be positive at {annotation_rel}:{line_no}",
                    context={"format_id": spec.format_id},
                )
            return xmin, ymin, bbox_w, bbox_h

        quad = row_format.quadrilateral_fields
        if quad is None:
            raise ValidationError(
                f"Custom format single-object row is missing bbox fields at {annotation_rel}:{line_no}",
                context={"format_id": spec.format_id},
            )

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
    except ValueError as exc:
        raise ValidationError(
            f"Custom format bbox fields must be numeric at {annotation_rel}:{line_no}",
            context={"format_id": spec.format_id},
        ) from exc

    xmin = min(xs)
    ymin = min(ys)
    bbox_w = max(xs) - xmin
    bbox_h = max(ys) - ymin
    if bbox_w <= 0 or bbox_h <= 0:
        return None
    return xmin, ymin, bbox_w, bbox_h


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


def _resolve_spec(dataset_root: Path, *, format_id: str | None, format_path: Path | None) -> FormatSpec:
    if format_path is not None:
        spec = load_custom_format_spec_from_path(format_path)
        if format_id is not None and spec.format_id != format_id:
            raise ValidationError(
                "Explicit custom format YAML does not match the selected format id",
                context={
                    "requested_format_id": format_id,
                    "resolved_format_id": spec.format_id,
                    "format_path": str(format_path),
                },
            )
        return spec

    if format_id is not None:
        spec = resolve_custom_format_spec(format_id, dataset_root)
        if spec is None:
            raise ValidationError(f"Custom format spec not found: {format_id}")
        return spec

    from label_master.adapters.custom.detector import detect_custom_format

    score, best_id = detect_custom_format(dataset_root, sample_limit=200)
    if best_id is None or score <= 0:
        raise ValidationError("Unable to resolve matching custom format spec")
    spec = resolve_custom_format_spec(best_id, dataset_root)
    if spec is None:
        raise ValidationError(f"Custom format spec not found: {best_id}")
    return spec
