from __future__ import annotations

import json
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
from label_master.core.domain.value_objects import BBoxCXCYWHNormalized, ValidationError

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_classes(dataset_root: Path) -> dict[int, CategoryRecord]:
    classes_file = dataset_root / "classes.txt"
    if not classes_file.exists():
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
    sizes_file = dataset_root / "image_sizes.json"
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
    explicit_labels_dir = dataset_root / "labels"
    if explicit_labels_dir.is_dir():
        files = sorted(explicit_labels_dir.rglob("*.txt"))
        if files:
            return files

    candidates: list[Path] = []
    for txt_file in dataset_root.rglob("*.txt"):
        if not txt_file.is_file():
            continue
        parent_parts = [part.lower() for part in txt_file.relative_to(dataset_root).parts[:-1]]
        if any("label" in part for part in parent_parts):
            candidates.append(txt_file)

    if candidates:
        return sorted(candidates)

    return []


def _guess_image_rel_from_label_rel(label_rel: Path) -> str:
    image_rel = label_rel.with_suffix("").as_posix()
    image_rel = image_rel.replace("/labels/", "/images/")
    image_rel = image_rel.replace("labels/", "images/")
    image_rel = image_rel.replace("_labels/", "_images/")
    image_rel = image_rel.replace("_labels", "_images")
    return image_rel


def _resolve_image_rel(dataset_root: Path, label_file: Path) -> str:
    label_rel = label_file.relative_to(dataset_root)
    stem_guess = _guess_image_rel_from_label_rel(label_rel)

    for extension in _IMAGE_EXTENSIONS:
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


def read_yolo_dataset(dataset_root: Path) -> AnnotationDataset:
    label_files = _discover_label_files(dataset_root)
    if not label_files:
        raise ValidationError(f"No YOLO label files found under: {dataset_root}")

    class_map = _load_classes(dataset_root)
    image_sizes = _load_image_sizes(dataset_root)

    annotations: list[AnnotationRecord] = []
    images_by_id: dict[str, ImageRecord] = {}
    seen_classes: set[int] = set()

    for label_file in label_files:
        label_rel_from_dataset = label_file.relative_to(dataset_root)
        image_rel = _resolve_image_rel(dataset_root, label_file)
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
                tokens = row.split()
                if len(tokens) != 5:
                    continue

                try:
                    class_id = int(tokens[0])
                    cx, cy, w, h = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))
                except ValueError:
                    continue

                bbox = BBoxCXCYWHNormalized(cx=cx, cy=cy, w=w, h=h)
                absolute = bbox.to_absolute(width, height)
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
        source_metadata=SourceMetadata(dataset_root=str(dataset_root.resolve()), loader="yolo_reader"),
    )
