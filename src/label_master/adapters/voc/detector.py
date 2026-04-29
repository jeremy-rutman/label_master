from __future__ import annotations

from pathlib import Path

from label_master.adapters.voc.common import (
    _voc_parser,
    build_voc_image_index,
    discover_voc_xml_files,
    parse_voc_annotation_file,
    resolve_voc_image_path,
)


def detect_voc(path: Path, *, sample_limit: int = 500) -> float:
    parser = _voc_parser()
    max_xml_files = max(1, min(sample_limit, 50))
    xml_files = discover_voc_xml_files(path, sample_limit=max_xml_files)
    if not xml_files:
        return 0.0

    image_index = None
    valid_files = 0
    files_with_boxes = 0
    resolved_images = 0

    for xml_file in xml_files:
        try:
            annotation = parse_voc_annotation_file(xml_file)
        except ValueError:
            continue

        valid_files += 1
        if annotation.objects:
            files_with_boxes += 1
        image_path = resolve_voc_image_path(path, xml_file, annotation, image_index=image_index)
        if image_path is None and image_index is None:
            image_index = build_voc_image_index(path)
            image_path = resolve_voc_image_path(path, xml_file, annotation, image_index=image_index)
        if image_path is not None:
            resolved_images += 1

    if valid_files == 0 or resolved_images == 0:
        return 0.0

    score = 0.75
    if files_with_boxes == valid_files:
        score += 0.1
    elif files_with_boxes > 0:
        score += 0.05

    if resolved_images == valid_files:
        score += 0.1
    elif resolved_images > 0:
        score += 0.05

    if any(
        any(part.lower() in {"annotation", "annotations", "xml"} for part in xml_file.relative_to(path).parts[:-1])
        for xml_file in xml_files
    ):
        score += 0.05

    return min(score + parser.score_boost, 1.0)
