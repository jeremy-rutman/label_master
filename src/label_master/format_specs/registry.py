from __future__ import annotations

from dataclasses import dataclass
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
    builtin_format: Literal["cityscapes", "coco", "kitware", "matlab_ground_truth", "voc", "video_bbox", "yolo"]


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


class Bdd100kImageLabelsParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["bdd100k_image_labels"]
    annotations_file: str = Field(min_length=1)
    records_key: str | None = None
    image_root: str = ""
    image_name_field: str = Field(default="name", min_length=1)
    labels_field: str = Field(default="labels", min_length=1)
    label_id_field: str | None = "id"
    category_field: str = Field(default="category", min_length=1)
    bbox_field: str = Field(default="box2d", min_length=1)
    x1_field: str = Field(default="x1", min_length=1)
    y1_field: str = Field(default="y1", min_length=1)
    x2_field: str = Field(default="x2", min_length=1)
    y2_field: str = Field(default="y2", min_length=1)
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


class QuadrilateralBBoxFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: int = Field(ge=1)
    y1: int = Field(ge=1)
    x2: int = Field(ge=1)
    y2: int = Field(ge=1)
    x3: int = Field(ge=1)
    y3: int = Field(ge=1)
    x4: int = Field(ge=1)
    y4: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_unique_positions(self) -> "QuadrilateralBBoxFieldSpec":
        positions = [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]
        if len(set(positions)) != len(positions):
            raise ValueError("quadrilateral field positions must be unique")
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


class SingleObjectVideoRowFormatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["single_object"]
    delimiter: Literal["whitespace", "comma"] = "whitespace"
    frame_index_field: int = Field(ge=1)
    frame_index_base: int = 0
    class_name_field: int | None = Field(default=None, ge=1)
    class_id_field: int | None = Field(default=None, ge=1)
    skip_empty_class: bool = False
    bbox_fields: XYWHBBoxFieldSpec | None = None
    quadrilateral_fields: QuadrilateralBBoxFieldSpec | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "SingleObjectVideoRowFormatSpec":
        if self.class_name_field is None and self.class_id_field is None:
            raise ValueError("single_object row format must define either class_name_field or class_id_field")
        if self.bbox_fields is None and self.quadrilateral_fields is None:
            raise ValueError("single_object row format must define bbox_fields or quadrilateral_fields")
        if self.bbox_fields is not None and self.quadrilateral_fields is not None:
            raise ValueError("single_object row format cannot define both bbox_fields and quadrilateral_fields")

        positions = [self.frame_index_field]
        if self.class_name_field is not None:
            positions.append(self.class_name_field)
        if self.class_id_field is not None:
            positions.append(self.class_id_field)
        if self.bbox_fields is not None:
            positions.extend(
                [
                    self.bbox_fields.xmin,
                    self.bbox_fields.ymin,
                    self.bbox_fields.width,
                    self.bbox_fields.height,
                ]
            )
        if self.quadrilateral_fields is not None:
            positions.extend(
                [
                    self.quadrilateral_fields.x1,
                    self.quadrilateral_fields.y1,
                    self.quadrilateral_fields.x2,
                    self.quadrilateral_fields.y2,
                    self.quadrilateral_fields.x3,
                    self.quadrilateral_fields.y3,
                    self.quadrilateral_fields.x4,
                    self.quadrilateral_fields.y4,
                ]
            )
        if len(set(positions)) != len(positions):
            raise ValueError("single_object row field positions must be unique")
        return self


class TokenizedVideoParserSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["tokenized_video"]
    annotation_globs: list[str] = Field(min_length=1)
    video_roots: list[str] = Field(default_factory=list)
    skip_rows: int = Field(default=0, ge=0)
    row_format: CountPrefixedObjectsRowFormatSpec | SingleObjectVideoRowFormatSpec
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
    | Bdd100kImageLabelsParserSpec
    | JsonObjectDatasetParserSpec
    | XmlAnnotationDatasetParserSpec
    | CsvBracketBBoxDatasetParserSpec
    | TokenizedVideoParserSpec
    | TokenizedImageLabelsParserSpec
] = TypeAdapter(
    BuiltInParserSpec
    | Bdd100kImageLabelsParserSpec
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
        | Bdd100kImageLabelsParserSpec
        | JsonObjectDatasetParserSpec
        | XmlAnnotationDatasetParserSpec
        | CsvBracketBBoxDatasetParserSpec
        | TokenizedVideoParserSpec
        | TokenizedImageLabelsParserSpec
    )


@dataclass(frozen=True)
class CustomFormatSpecEntry:
    path: Path
    spec: FormatSpec


DATASET_ROOT_CUSTOM_SPEC_FILENAMES = (
    "custom_format.yaml",
    "custom_format.yml",
    "data_format.yaml",
    "data_format.yml",
    "label_format.yaml",
    "label_format.yml",
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


def load_custom_format_spec_from_path(path: Path) -> FormatSpec:
    return _load_spec_from_path(path.expanduser().resolve())


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


def _candidate_custom_spec_paths(
    dataset_root: Path | None,
    *,
    extra_spec_paths: tuple[Path, ...] = (),
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _append(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for directory in _candidate_custom_spec_directories(dataset_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.y*ml")):
            if path.is_file():
                _append(path)

    if dataset_root is not None:
        resolved_root = dataset_root.expanduser().resolve()
        for file_name in DATASET_ROOT_CUSTOM_SPEC_FILENAMES:
            candidate = resolved_root / file_name
            if candidate.is_file():
                _append(candidate)

    for path in extra_spec_paths:
        _append(path)

    return candidates


def custom_format_spec_entries(
    dataset_root: Path | None = None,
    *,
    extra_spec_paths: tuple[Path, ...] = (),
) -> list[CustomFormatSpecEntry]:
    entries: list[CustomFormatSpecEntry] = []
    builtin_specs = load_builtin_format_specs()
    seen_ids: dict[str, Path] = {}
    for path in _candidate_custom_spec_paths(dataset_root, extra_spec_paths=extra_spec_paths):
        spec = _load_spec_from_path(path)
        if not isinstance(spec.parser, (Bdd100kImageLabelsParserSpec, TokenizedVideoParserSpec)):
            continue
        if spec.format_id in builtin_specs:
            raise ConfigurationError(
                f"Custom format spec cannot override built-in format id: {spec.format_id}",
                context={"path": str(path)},
            )
        if spec.format_id in seen_ids:
            raise ConfigurationError(
                f"Duplicate custom format spec id discovered: {spec.format_id}",
                context={
                    "path": str(path),
                    "existing_path": str(seen_ids[spec.format_id]),
                },
            )
        seen_ids[spec.format_id] = path
        entries.append(CustomFormatSpecEntry(path=path, spec=spec))
    return entries


def custom_format_specs(
    dataset_root: Path | None = None,
    *,
    extra_spec_paths: tuple[Path, ...] = (),
) -> list[FormatSpec]:
    return [entry.spec for entry in custom_format_spec_entries(dataset_root, extra_spec_paths=extra_spec_paths)]


def resolve_custom_format_spec(
    format_id: str,
    dataset_root: Path | None = None,
    *,
    extra_spec_paths: tuple[Path, ...] = (),
) -> FormatSpec | None:
    for spec in custom_format_specs(dataset_root, extra_spec_paths=extra_spec_paths):
        if spec.format_id == format_id:
            return spec
    return None
