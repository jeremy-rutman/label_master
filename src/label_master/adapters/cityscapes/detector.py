from __future__ import annotations

import json
from pathlib import Path

from label_master.adapters.cityscapes.common import (
    annotation_file_stem,
    annotation_files_for_split,
    available_split_names,
    resolve_image_path,
    resolve_labels_root,
)
from label_master.core.domain.value_objects import ValidationError


def detect_cityscapes(path: Path, *, sample_limit: int = 500) -> float:
    try:
        labels_variant, labels_root = resolve_labels_root(path)
    except ValidationError:
        return 0.0

    split_names = available_split_names(labels_root, path)
    if not split_names:
        return 0.0

    annotation_files: list[Path] = []
    max_files = max(1, min(sample_limit, 50))
    for split_name in split_names:
        for annotation_file in annotation_files_for_split(labels_root, split_name):
            annotation_files.append(annotation_file)
            if len(annotation_files) >= max_files:
                break
        if len(annotation_files) >= max_files:
            break

    if not annotation_files:
        return 0.0

    valid_files = 0
    matched_images = 0
    files_with_objects = 0

    for annotation_file in annotation_files:
        stem = annotation_file_stem(annotation_file, labels_variant=labels_variant)
        if stem is None:
            continue

        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue
        if "imgWidth" not in payload or "imgHeight" not in payload:
            continue
        objects = payload.get("objects")
        if not isinstance(objects, list):
            continue

        valid_files += 1
        if objects:
            files_with_objects += 1

        split_name = annotation_file.parent.parent.name
        city_name = annotation_file.parent.name
        if resolve_image_path(path, split_name, city_name, stem) is not None:
            matched_images += 1

    if valid_files == 0 or matched_images == 0:
        return 0.0

    score = 0.82
    if matched_images == valid_files:
        score += 0.1
    elif matched_images > 0:
        score += 0.05

    if files_with_objects == valid_files:
        score += 0.05
    elif files_with_objects > 0:
        score += 0.02

    if labels_variant == "gtFine":
        score += 0.03

    return min(score, 1.0)