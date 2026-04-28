from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from label_master.core.domain.value_objects import ConfigurationError


class BuiltInParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["built_in"]
    builtin_format: Literal["coco", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]


class JsonImageFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    width: str = Field(min_length=1)
    height: str = Field(min_length=1)


class JsonCategoryFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    supercategory: str | None = None


class JsonAnnotationFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    class_id: str = Field(min_length=1)
    bbox: str = Field(min_length=1)
    iscrowd: str | None = None


class XYWHBBoxFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    xmin: int = Field(ge=1, validation_alias=AliasChoices("xmin", "x"))
    ymin: int = Field(ge=1, validation_alias=AliasChoices("ymin", "y"))
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_unique_positions(self) -> "XYWHBBoxFieldSpec":
        positions = [self.xmin, self.ymin, self.width, self.height]
        if len(set(positions)) != len(positions):
            raise ValueError("bbox field positions must be unique")
        return self


class JsonObjectDatasetParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["json_object_dataset"]
    annotations_file: str = Field(min_length=1)
    images_key: str = Field(min_length=1)
    annotations_key: str = Field(min_length=1)
    categories_key: str = Field(min_length=1)
    image_fields: JsonImageFieldSpec
    category_fields: JsonCategoryFieldSpec
    annotation_fields: JsonAnnotationFieldSpec
    bbox_fields: XYWHBBoxFieldSpec = Field(
        default_factory=lambda: XYWHBBoxFieldSpec(xmin=1, ymin=2, width=3, height=4)
    )
    bbox_format: Literal["xywh_list"] | None = None
    score_boost: float = Field(default=0.0, ge=0.0, le=0.2)


class XmlBBoxFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xmin: str = Field(min_length=1)
    ymin: str = Field(min_length=1)
    xmax: str = Field(min_length=1)
    ymax: str = Field(min_length=1)


class XmlAnnotationDatasetParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["xml_annotation_dataset"]
    annotation_globs: list[str] = Field(min_length=1)
    root_tag: str = Field(min_length=1)
    filename_field: str = Field(min_length=1)
    path_field: str | None = None
    size_width_field: str | None = None
    size_height_field: str | None = None
    object_tag: str = Field(min_length=1)
    object_name_field: str = Field(min_length=1)
    bbox_tag: str = Field(min_length=1)
    bbox_fields: XmlBBoxFieldSpec
    score_boost: float = Field(default=0.0, ge=0.0, le=0.2)


class CsvBracketBBoxDatasetParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["csv_bracket_bbox_dataset"]
    csv_globs: list[str] = Field(min_length=1)
    image_field_aliases: list[str] = Field(min_length=1)
    bbox_column_class_map: dict[str, str] = Field(min_length=1)
    bbox_fields: XYWHBBoxFieldSpec = Field(
        default_factory=lambda: XYWHBBoxFieldSpec(xmin=1, ymin=2, width=3, height=4)
    )
    bbox_enclosure: str = "[]"
    box_separator: str = ";"
    score_boost: float = Field(default=0.0, ge=0.0, le=0.2)

    @model_validator(mode="after")
    def _validate_bbox_enclosure(self) -> "CsvBracketBBoxDatasetParserSpec":
        if len(self.bbox_enclosure) != 2:
            raise ValueError("bbox_enclosure must contain exactly two characters")
        return self


class TokenizedObjectFieldSpec(XYWHBBoxFieldSpec):
    class_name: int | None = Field(default=None, ge=1)
    class_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_class_field(self) -> "TokenizedObjectFieldSpec":
        if self.class_name is None and self.class_id is None:
            raise ValueError("object_fields must define either class_name or class_id")
        return self


class CountPrefixedObjectsRowFormatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["count_prefixed_objects"]
    delimiter: Literal["whitespace", "comma"] = "whitespace"
    frame_index_field: int = Field(ge=1)
    object_count_field: int = Field(ge=1)
    frame_index_base: int = 0
    object_group_size: int = Field(ge=1)
    object_fields: TokenizedObjectFieldSpec

    @model_validator(mode="after")
    def _validate_group_fields(self) -> "CountPrefixedObjectsRowFormatSpec":
        positions = [
            self.object_fields.xmin,
            self.object_fields.ymin,
            self.object_fields.width,
            self.object_fields.height,
        ]
        if self.object_fields.class_name is not None:
            positions.append(self.object_fields.class_name)
        if self.object_fields.class_id is not None:
            positions.append(self.object_fields.class_id)
        if max(positions) > self.object_group_size:
            raise ValueError("object_fields positions must not exceed object_group_size")
        return self


class TokenizedVideoParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["tokenized_video"]
    annotation_globs: list[str] = Field(min_length=1)
    video_roots: list[str] = Field(default_factory=list)
    row_format: CountPrefixedObjectsRowFormatSpec
    image_path_template: str = "images/{video_stem}/frame_{frame_index:06d}.jpg"
    score_boost: float = Field(default=0.0, ge=0.0, le=0.2)


class TokenizedImageLabelRowFormatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["single_object"]
    delimiter: Literal["whitespace", "comma"] = "whitespace"
    class_id_field: int = Field(ge=1)
    x_center_field: int = Field(ge=1)
    y_center_field: int = Field(ge=1)
    width_field: int = Field(ge=1)
    height_field: int = Field(ge=1)
    normalized_coordinates: bool = True


class PathRewriteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_text: str = Field(min_length=1, alias="from")
    to_text: str = Field(alias="to")


class TokenizedImageLabelsParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["tokenized_image_labels"]
    label_globs: list[str] = Field(min_length=1)
    classes_file_name: str = Field(min_length=1)
    image_sizes_file_name: str = Field(min_length=1)
    image_extensions: list[str] = Field(min_length=1)
    path_rewrites: list[PathRewriteSpec] = Field(default_factory=list)
    row_format: TokenizedImageLabelRowFormatSpec
    score_boost: float = Field(default=0.0, ge=0.0, le=0.2)


ParserSpec: TypeAdapter[
    BuiltInParserSpec
    | JsonObjectDatasetParserSpec
    | XmlAnnotationDatasetParserSpec
    | CsvBracketBBoxDatasetParserSpec
    | TokenizedVideoParserSpec
    | TokenizedImageLabelsParserSpec
] = TypeAdapter(
    BuiltInParserSpec
    | JsonObjectDatasetParserSpec
    | XmlAnnotationDatasetParserSpec
    | CsvBracketBBoxDatasetParserSpec
    | TokenizedVideoParserSpec
    | TokenizedImageLabelsParserSpec
)


class FormatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str | None = None
    parser: (
        BuiltInParserSpec
        | JsonObjectDatasetParserSpec
        | XmlAnnotationDatasetParserSpec
        | CsvBracketBBoxDatasetParserSpec
        | TokenizedVideoParserSpec
        | TokenizedImageLabelsParserSpec
    )


def _load_yaml_payload(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Unable to read format spec: {path}") from exc

    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Format spec must contain a top-level object: {path}",
            context={"path": str(path)},
        )
    return payload


def _load_spec_from_path(path: Path) -> FormatSpec:
    payload = _load_yaml_payload(path)
    parser_payload = payload.get("parser")
    if not isinstance(parser_payload, dict):
        raise ConfigurationError(
            f"Format spec parser section must be an object: {path}",
            context={"path": str(path)},
        )

    normalized = dict(payload)
    normalized["parser"] = ParserSpec.validate_python(parser_payload)
    try:
        return FormatSpec.model_validate(normalized)
    except Exception as exc:  # pragma: no cover - defensive conversion wrapper
        raise ConfigurationError(
            f"Invalid format spec: {path.name}",
            context={"path": str(path), "reason": str(exc)},
        ) from exc


@lru_cache(maxsize=1)
def load_builtin_format_specs() -> dict[str, FormatSpec]:
    spec_dir = files("label_master.format_specs").joinpath("builtins")
    specs: dict[str, FormatSpec] = {}
    for resource in sorted(spec_dir.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith((".yaml", ".yml")):
            continue
        with as_file(resource) as resource_path:
            spec = _load_spec_from_path(resource_path)
        specs[spec.format_id] = spec
    return specs


def resolve_builtin_format_spec(format_id: str) -> FormatSpec | None:
    return load_builtin_format_specs().get(format_id)


def _candidate_custom_spec_directories(dataset_root: Path | None) -> list[Path]:
    candidates = [Path.home() / ".label_master" / "formats"]
    if dataset_root is not None:
        resolved = dataset_root.expanduser().resolve()
        candidates.extend(
            [
                resolved / "format_specs",
                resolved / ".label_master" / "formats",
            ]
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def custom_format_specs(dataset_root: Path | None = None) -> list[FormatSpec]:
    specs: list[FormatSpec] = []
    for directory in _candidate_custom_spec_directories(dataset_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.y*ml")):
            spec = _load_spec_from_path(path)
            if not isinstance(spec.parser, TokenizedVideoParserSpec):
                continue
            if spec.format_id in load_builtin_format_specs():
                raise ConfigurationError(
                    f"Custom format spec cannot override built-in format id: {spec.format_id}",
                    context={"path": str(path)},
                )
            specs.append(spec)
    return specs


def resolve_custom_format_spec(format_id: str, dataset_root: Path | None = None) -> FormatSpec | None:
    for spec in custom_format_specs(dataset_root):
        if spec.format_id == format_id:
            return spec
    return None
