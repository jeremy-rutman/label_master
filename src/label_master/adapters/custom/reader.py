from __future__ import annotations

from pathlib import Path

from label_master.adapters.custom.common import (
    annotation_files_for_spec,
    build_image_rel_path,
    probe_video_dimensions,
    resolve_spec_video_file,
    split_row_tokens,
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
    FormatSpec,
    TokenizedVideoParserSpec,
    resolve_custom_format_spec,
)
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter


def read_custom_dataset(
    dataset_root: Path,
    *,
    format_id: str | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    spec = _resolve_spec(dataset_root, format_id=format_id)
    parser = spec.parser
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
                row = line.strip()
                if not row:
                    continue

                parsed_image, parsed_annotations = _parse_count_prefixed_row(
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


def _resolve_spec(dataset_root: Path, *, format_id: str | None) -> FormatSpec:
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
