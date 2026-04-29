from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

from label_master.core.domain.entities import (
    AnnotationDataset,
    AnnotationRecord,
    CategoryRecord,
    ImageRecord,
    SourceFormat,
    SourceMetadata,
)
from label_master.core.domain.value_objects import BBoxCXCYWHNormalized, BBoxXYWH, ValidationError
from label_master.format_specs.registry import (
    TokenizedImageLabelsParserSpec,
    resolve_builtin_format_spec,
)
from label_master.infra.filesystem import InputPathFilter, relative_path_matches_input_filter


@lru_cache(maxsize=1)
def _yolo_parser() -> TokenizedImageLabelsParserSpec:
    spec = resolve_builtin_format_spec("yolo")
    if spec is None or not isinstance(spec.parser, TokenizedImageLabelsParserSpec):
        raise ValidationError("Built-in YOLO format spec is unavailable")
    return spec.parser


def _load_classes(dataset_root: Path) -> dict[int, CategoryRecord]:
    parser = _yolo_parser()
    candidate_files = [dataset_root / parser.classes_file_name, dataset_root / "obj.names"]
    candidate_files.extend(sorted(dataset_root.glob("**/obj.names")))
    classes_file = next((path for path in candidate_files if path.exists() and path.is_file()), None)
    if classes_file is None:
        return {}

    categories: dict[int, CategoryRecord] = {}
    with classes_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            name = line.strip()
            if not name:
                continue
            categories[idx] = CategoryRecord(class_id=idx, name=name)
    return categories


def _load_image_sizes(dataset_root: Path) -> dict[str, tuple[int, int]]:
    parser = _yolo_parser()
    sizes_file = dataset_root / parser.image_sizes_file_name
    if not sizes_file.exists():
        return {}
    with sizes_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, tuple[int, int]] = {}
    for key, value in raw.items():
        if isinstance(value, list) and len(value) == 2:
            parsed[str(key)] = (int(value[0]), int(value[1]))
    return parsed


def _discover_label_files(dataset_root: Path) -> list[Path]:
    parser = _yolo_parser()
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in parser.label_globs:
        for path in sorted(dataset_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def _sample_evenly(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[0]]

    selected_indices: list[int] = []
    seen: set[int] = set()
    last_index = len(paths) - 1
    for position in range(limit):
        index = round(position * last_index / (limit - 1))
        if index in seen:
            continue
        seen.add(index)
        selected_indices.append(index)

    if len(selected_indices) < limit:
        for index in range(len(paths)):
            if index in seen:
                continue
            seen.add(index)
            selected_indices.append(index)
            if len(selected_indices) == limit:
                break

    return [paths[index] for index in selected_indices]


def _guess_image_rel_from_label_rel(label_rel: Path) -> str:
    parser = _yolo_parser()
    image_rel = label_rel.with_suffix("").as_posix()
    for rewrite in parser.path_rewrites:
        image_rel = image_rel.replace(rewrite.from_text, rewrite.to_text)
    return image_rel


def _resolve_image_rel(dataset_root: Path, label_file: Path) -> str:
    label_rel = label_file.relative_to(dataset_root)
    stem_guess = _guess_image_rel_from_label_rel(label_rel)

    for extension in _yolo_parser().image_extensions:
        candidate = dataset_root / f"{stem_guess}{extension}"
        if candidate.exists():
            return f"{stem_guess}{extension}"

    return f"{stem_guess}.jpg"


def _image_dimensions(
    dataset_root: Path,
    image_rel: str,
    image_sizes: dict[str, tuple[int, int]],
) -> tuple[int, int, bool]:
    if image_rel in image_sizes:
        width, height = image_sizes[image_rel]
        return width, height, False

    image_name = Path(image_rel).name
    if image_name in image_sizes:
        width, height = image_sizes[image_name]
        return width, height, False

    image_path = dataset_root / image_rel
    if image_path.exists():
        try:
            with Image.open(image_path) as opened:
                return int(opened.width), int(opened.height), False
        except OSError:
            pass

    return 1, 1, True


def _normalized_bbox_to_absolute(
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> BBoxXYWH:
    try:
        return BBoxCXCYWHNormalized(cx=cx, cy=cy, w=w, h=h).to_absolute(image_width, image_height)
    except ValueError:
        # Preserve slightly out-of-range YOLO rows so validation can clip or report them.
        w_abs = w * image_width
        h_abs = h * image_height
        return BBoxXYWH(
            x=(cx * image_width) - (w_abs / 2.0),
            y=(cy * image_height) - (h_abs / 2.0),
            w=w_abs,
            h=h_abs,
        )


def read_yolo_dataset(
    dataset_root: Path,
    *,
    max_label_files: int | None = None,
    input_path_filter: InputPathFilter | None = None,
) -> AnnotationDataset:
    parser = _yolo_parser()
    all_label_files = _discover_label_files(dataset_root)
    if not all_label_files:
        raise ValidationError(f"No YOLO label files found under: {dataset_root}")
    label_files = (
        _sample_evenly(all_label_files, max_label_files)
        if max_label_files is not None and input_path_filter is None
        else all_label_files
    )

    class_map = _load_classes(dataset_root)
    image_sizes = _load_image_sizes(dataset_root)

    annotations: list[AnnotationRecord] = []
    images_by_id: dict[str, ImageRecord] = {}
    seen_classes: set[int] = set()
    label_files_seen = 0

    for label_file in label_files:
        if (
            max_label_files is not None
            and input_path_filter is not None
            and len(images_by_id) >= max_label_files
        ):
            break
        label_files_seen += 1
        label_rel_from_dataset = label_file.relative_to(dataset_root)
        image_rel = _resolve_image_rel(dataset_root, label_file)
        if not relative_path_matches_input_filter(image_rel, input_path_filter=input_path_filter):
            continue
        image_id = str(Path(image_rel).with_suffix(""))
        width, height, size_unknown = _image_dimensions(dataset_root, image_rel, image_sizes)

        if image_id not in images_by_id:
            images_by_id[image_id] = ImageRecord(
                image_id=image_id,
                file_name=image_rel,
                width=width,
                height=height,
                checksum="unknown_size" if size_unknown else None,
            )

        with label_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                row = line.strip()
                if not row:
                    continue
                tokens = row.split(",") if parser.row_format.delimiter == "comma" else row.split()
                if len(tokens) != 5:
                    continue

                try:
                    class_id = int(tokens[parser.row_format.class_id_field - 1])
                    cx = float(tokens[parser.row_format.x_center_field - 1])
                    cy = float(tokens[parser.row_format.y_center_field - 1])
                    w = float(tokens[parser.row_format.width_field - 1])
                    h = float(tokens[parser.row_format.height_field - 1])
                except ValueError:
                    continue

                absolute = _normalized_bbox_to_absolute(
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    image_width=width,
                    image_height=height,
                )
                seen_classes.add(class_id)
                annotations.append(
                    AnnotationRecord(
                        annotation_id=f"{label_rel_from_dataset.as_posix()}:{line_no}",
                        image_id=image_id,
                        class_id=class_id,
                        bbox_xywh_abs=(absolute.x, absolute.y, absolute.w, absolute.h),
                    )
                )

    categories = dict(class_map)
    if not categories:
        categories = {
            class_id: CategoryRecord(class_id=class_id, name=f"class_{class_id}")
            for class_id in sorted(seen_classes)
        }
    else:
        # Some datasets provide classes.txt but omit class IDs that still appear in labels.
        # Fill those gaps with deterministic fallback names instead of failing preview/load.
        for class_id in sorted(seen_classes):
            categories.setdefault(
                class_id,
                CategoryRecord(class_id=class_id, name=f"class_{class_id}"),
            )

    return AnnotationDataset(
        dataset_id=dataset_root.name,
        source_format=SourceFormat.YOLO,
        images=sorted(images_by_id.values(), key=lambda image: image.image_id),
        annotations=sorted(annotations, key=lambda ann: ann.annotation_id),
        categories=categories,
        source_metadata=SourceMetadata(
            dataset_root=str(dataset_root.resolve()),
            loader="yolo_reader",
            details={
                "label_files_loaded": str(len(images_by_id)),
                "label_files_considered": str(label_files_seen),
                "label_files_total": str(len(all_label_files)),
                **({"label_files_limit": str(max_label_files)} if max_label_files is not None else {}),
            },
        ),
    )
