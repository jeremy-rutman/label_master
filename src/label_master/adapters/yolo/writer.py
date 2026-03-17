from __future__ import annotations

from pathlib import Path

from label_master.core.domain.entities import AnnotationDataset
from label_master.core.domain.value_objects import BBoxXYWH
from label_master.infra.filesystem import ensure_directory


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _label_rel_path_for_image(image_file_name: str, image_id: str) -> Path:
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

    return label_rel


def write_yolo_dataset(dataset: AnnotationDataset, output_root: Path) -> Path:
    labels_dir = ensure_directory(output_root / "labels")
    ensure_directory(output_root / "images")

    image_lookup = {image.image_id: image for image in dataset.images}
    grouped: dict[str, list[str]] = {
        _label_rel_path_for_image(image.file_name, image.image_id).as_posix(): []
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
        label_rel = _label_rel_path_for_image(image.file_name, image.image_id).as_posix()
        grouped.setdefault(label_rel, []).append(line)

    for label_rel in sorted(grouped):
        lines = grouped[label_rel]
        label_path = labels_dir / Path(label_rel)
        ensure_directory(label_path.parent)
        content = "\n".join(lines) + ("\n" if lines else "")
        label_path.write_text(content, encoding="utf-8")

    classes_path = output_root / "classes.txt"
    class_lines = [
        category.name for _, category in sorted(dataset.categories.items(), key=lambda item: item[0])
    ]
    classes_path.write_text("\n".join(class_lines) + ("\n" if class_lines else ""), encoding="utf-8")

    return labels_dir
