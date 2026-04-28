from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from label_master.core.domain.entities import AnnotationDataset
from label_master.core.domain.value_objects import BBoxXYWH
from label_master.infra.filesystem import ensure_directory

LabelWriteProgressCallback = Callable[[int, int], None]


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _sanitize_flattened_name_prefix(raw_value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip()).strip("._-")


def _sanitize_stem_affix(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip())


def _flatten_rel_path(rel_path: Path, fallback_name: str) -> Path:
    candidate_name = rel_path.name or fallback_name
    if rel_path.parent != Path("."):
        parent_prefix_parts = [
            _sanitize_flattened_name_prefix(part)
            for part in rel_path.parent.parts
            if part not in {"", "."}
        ]
        parent_prefix = "_".join(part for part in parent_prefix_parts if part)
        if parent_prefix:
            return Path(f"{parent_prefix}_{candidate_name}")
    return Path(candidate_name)


def _with_stem_affixes(
    rel_path: Path,
    *,
    fallback_name: str,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
) -> Path:
    stem_prefix = _sanitize_stem_affix(output_file_stem_prefix)
    stem_suffix = _sanitize_stem_affix(output_file_stem_suffix)
    if not stem_prefix and not stem_suffix:
        return rel_path

    file_name = rel_path.name or fallback_name
    file_path = Path(file_name)
    stem = file_path.stem or file_path.name
    affixed_name = f"{stem_prefix}{stem}{stem_suffix}{file_path.suffix}"
    if rel_path.parent == Path("."):
        return Path(affixed_name)
    return rel_path.parent / affixed_name


def _label_rel_path_for_image(
    image_file_name: str,
    image_id: str,
    *,
    flatten_output_layout: bool = False,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
) -> Path:
    image_rel = Path(image_file_name)
    if image_rel.is_absolute():
        image_rel = Path(image_rel.name)

    label_rel = image_rel.with_suffix(".txt")
    parts = list(label_rel.parts)
    if parts and parts[0].lower() == "images":
        parts = parts[1:]
        if parts:
            label_rel = Path(*parts)
        else:
            label_rel = Path(label_rel.name)

    if label_rel == Path(".") or not str(label_rel).strip():
        label_rel = Path(f"{image_id}.txt")

    if flatten_output_layout:
        label_rel = _flatten_rel_path(label_rel, f"{image_id}.txt")

    return _with_stem_affixes(
        label_rel,
        fallback_name=f"{image_id}.txt",
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    )


def image_output_rel_path_for_image(
    image_file_name: str,
    image_id: str,
    *,
    flatten_output_layout: bool = False,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
) -> Path:
    image_rel = Path(image_file_name)
    suffix = image_rel.suffix or ".jpg"
    return Path("images") / _label_rel_path_for_image(
        image_file_name,
        image_id,
        flatten_output_layout=flatten_output_layout,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    ).with_suffix(suffix)


def label_output_rel_path_for_image(
    image_file_name: str,
    image_id: str,
    *,
    flatten_output_layout: bool = False,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
) -> Path:
    return Path("labels") / _label_rel_path_for_image(
        image_file_name,
        image_id,
        flatten_output_layout=flatten_output_layout,
        output_file_stem_prefix=output_file_stem_prefix,
        output_file_stem_suffix=output_file_stem_suffix,
    )


def write_yolo_dataset(
    dataset: AnnotationDataset,
    output_root: Path,
    *,
    classes_file_name: str = "classes.txt",
    flatten_output_layout: bool = False,
    output_file_stem_prefix: str | None = None,
    output_file_stem_suffix: str | None = None,
    progress_callback: LabelWriteProgressCallback | None = None,
) -> Path:
    labels_dir = ensure_directory(output_root / "labels")
    ensure_directory(output_root / "images")

    image_lookup = {image.image_id: image for image in dataset.images}
    grouped: dict[str, list[str]] = {
        _label_rel_path_for_image(
            image.file_name,
            image.image_id,
            flatten_output_layout=flatten_output_layout,
            output_file_stem_prefix=output_file_stem_prefix,
            output_file_stem_suffix=output_file_stem_suffix,
        ).as_posix(): []
        for image in dataset.images
    }

    for annotation in dataset.deterministic_annotations():
        image = image_lookup.get(annotation.image_id)
        if image is None:
            continue

        normalized = BBoxXYWH(*annotation.bbox_xywh_abs).to_normalized(image.width, image.height)
        line = " ".join(
            [
                str(annotation.class_id),
                _format_float(normalized.cx),
                _format_float(normalized.cy),
                _format_float(normalized.w),
                _format_float(normalized.h),
            ]
        )
        label_rel = _label_rel_path_for_image(
            image.file_name,
            image.image_id,
            flatten_output_layout=flatten_output_layout,
            output_file_stem_prefix=output_file_stem_prefix,
            output_file_stem_suffix=output_file_stem_suffix,
        ).as_posix()
        grouped.setdefault(label_rel, []).append(line)

    total_label_files = len(grouped)
    update_every = max(total_label_files // 50, 1) if total_label_files else 1
    for index, label_rel in enumerate(sorted(grouped), start=1):
        lines = grouped[label_rel]
        label_path = labels_dir / Path(label_rel)
        ensure_directory(label_path.parent)
        content = "\n".join(lines) + ("\n" if lines else "")
        label_path.write_text(content, encoding="utf-8")
        if progress_callback is not None and (index % update_every == 0 or index == total_label_files):
            progress_callback(index, total_label_files)

    classes_path = output_root / classes_file_name
    class_lines = [
        category.name for _, category in sorted(dataset.categories.items(), key=lambda item: item[0])
    ]
    classes_path.write_text("\n".join(class_lines) + ("\n" if class_lines else ""), encoding="utf-8")

    return labels_dir
