from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from label_master.core.domain.value_objects import ValidationError

CityscapesLabelMode = Literal["instances", "all"]


@dataclass(frozen=True)
class CityscapesLabel:
    name: str
    category: str
    has_instances: bool
    ignore_in_eval: bool


@dataclass(frozen=True)
class NormalizedLabel:
    name: str
    is_group: bool


CITYSCAPES_LABELS: tuple[CityscapesLabel, ...] = (
    CityscapesLabel("unlabeled", "void", False, True),
    CityscapesLabel("ego vehicle", "void", False, True),
    CityscapesLabel("rectification border", "void", False, True),
    CityscapesLabel("out of roi", "void", False, True),
    CityscapesLabel("static", "void", False, True),
    CityscapesLabel("dynamic", "void", False, True),
    CityscapesLabel("ground", "void", False, True),
    CityscapesLabel("road", "flat", False, False),
    CityscapesLabel("sidewalk", "flat", False, False),
    CityscapesLabel("parking", "flat", False, True),
    CityscapesLabel("rail track", "flat", False, True),
    CityscapesLabel("building", "construction", False, False),
    CityscapesLabel("wall", "construction", False, False),
    CityscapesLabel("fence", "construction", False, False),
    CityscapesLabel("guard rail", "construction", False, True),
    CityscapesLabel("bridge", "construction", False, True),
    CityscapesLabel("tunnel", "construction", False, True),
    CityscapesLabel("pole", "object", False, False),
    CityscapesLabel("polegroup", "object", False, True),
    CityscapesLabel("traffic light", "object", False, False),
    CityscapesLabel("traffic sign", "object", False, False),
    CityscapesLabel("vegetation", "nature", False, False),
    CityscapesLabel("terrain", "nature", False, False),
    CityscapesLabel("sky", "sky", False, False),
    CityscapesLabel("person", "human", True, False),
    CityscapesLabel("rider", "human", True, False),
    CityscapesLabel("car", "vehicle", True, False),
    CityscapesLabel("truck", "vehicle", True, False),
    CityscapesLabel("bus", "vehicle", True, False),
    CityscapesLabel("caravan", "vehicle", True, True),
    CityscapesLabel("trailer", "vehicle", True, True),
    CityscapesLabel("train", "vehicle", True, False),
    CityscapesLabel("motorcycle", "vehicle", True, False),
    CityscapesLabel("bicycle", "vehicle", True, False),
    CityscapesLabel("license plate", "vehicle", False, True),
)

LABELS_BY_NAME = {label.name: label for label in CITYSCAPES_LABELS}


def resolve_labels_root(dataset_root: Path) -> tuple[str, Path]:
    for variant in ("gtFine", "gtCoarse"):
        candidate = dataset_root / variant / variant
        if candidate.exists():
            return variant, candidate
    raise ValidationError(
        "Cityscapes labels root not found. Expected gtFine/gtFine or gtCoarse/gtCoarse under the dataset root.",
        context={"dataset_root": str(dataset_root)},
    )


def image_roots_for_dataset(dataset_root: Path) -> tuple[Path, ...]:
    candidates = (
        dataset_root / "leftImg8bit_trainvaltest" / "leftImg8bit",
        dataset_root / "leftImg8bit_trainextra" / "leftImg8bit",
    )
    return tuple(candidate for candidate in candidates if candidate.exists())


def resolve_image_path(dataset_root: Path, split_name: str, city_name: str, stem: str) -> Path | None:
    for image_root in image_roots_for_dataset(dataset_root):
        candidate = image_root / split_name / city_name / f"{stem}_leftImg8bit.png"
        if candidate.exists():
            return candidate
    return None


def annotation_file_stem(annotation_file: Path, *, labels_variant: str) -> str | None:
    suffix = f"_{labels_variant}_polygons.json"
    if not annotation_file.name.endswith(suffix):
        return None
    return annotation_file.name[: -len(suffix)]


def annotation_files_for_split(labels_root: Path, split_name: str) -> list[Path]:
    split_root = labels_root / split_name
    if not split_root.exists():
        return []
    return sorted(split_root.glob("*/*_polygons.json"))


def available_split_names(labels_root: Path, dataset_root: Path) -> list[str]:
    image_splits: set[str] = set()
    for image_root in image_roots_for_dataset(dataset_root):
        for child in image_root.iterdir():
            if child.is_dir():
                image_splits.add(child.name)

    split_names: list[str] = []
    for child in sorted(labels_root.iterdir()):
        if child.is_dir() and child.name in image_splits:
            split_names.append(child.name)
    return split_names


def normalize_label_name(raw_label: str) -> NormalizedLabel | None:
    label_name = raw_label.strip()
    if label_name in LABELS_BY_NAME:
        return NormalizedLabel(name=label_name, is_group=False)

    if not label_name.endswith("group"):
        return None

    base_name = label_name[: -len("group")]
    base_label = LABELS_BY_NAME.get(base_name)
    if base_label is None or not base_label.has_instances:
        return None
    return NormalizedLabel(name=base_name, is_group=True)


def bbox_from_polygon(
    polygon: list[list[int | float]],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None

    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]

    x_min = max(0, min(int(value) for value in xs))
    y_min = max(0, min(int(value) for value in ys))
    x_max = min(image_width - 1, max(int(value) for value in xs))
    y_max = min(image_height - 1, max(int(value) for value in ys))

    if x_max < x_min or y_max < y_min:
        return None

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        return None
    return (float(x_min), float(y_min), float(width), float(height))


def category_table(label_mode: CityscapesLabelMode) -> tuple[dict[str, int], list[CityscapesLabel]]:
    if label_mode == "instances":
        filtered = [label for label in CITYSCAPES_LABELS if label.has_instances and not label.ignore_in_eval]
    else:
        filtered = [label for label in CITYSCAPES_LABELS if not label.ignore_in_eval]

    category_ids = {label.name: index for index, label in enumerate(filtered)}
    return category_ids, filtered