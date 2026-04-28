from __future__ import annotations

import csv
from pathlib import Path

from label_master.adapters.kitware.common import (
    _kitware_parser,
    discover_kitware_csv_layouts,
    parse_kitware_bboxes,
    resolve_kitware_image_path,
)


def detect_kitware(path: Path, *, sample_limit: int = 500) -> float:
    parser = _kitware_parser()
    max_rows = max(1, min(sample_limit, 50))
    max_layouts = max(1, min((sample_limit + 24) // 25, 5))
    layouts = discover_kitware_csv_layouts(path, max_layouts=max_layouts)
    if not layouts:
        return 0.0

    sampled_rows = 0
    valid_rows = 0
    rows_with_boxes = 0

    for layout in layouts[:20]:
        try:
            with layout.csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row is None:
                        continue
                    raw_image_ref = str(row.get(layout.image_field, "")).strip()
                    if not raw_image_ref:
                        sampled_rows += 1
                        if sampled_rows >= max_rows:
                            break
                        continue

                    image_path = resolve_kitware_image_path(path, layout.csv_path, raw_image_ref)
                    row_valid = image_path is not None
                    row_has_box = False

                    for bbox_column in layout.bbox_columns:
                        raw_value = str(row.get(bbox_column.header_name, "")).strip()
                        try:
                            bboxes = parse_kitware_bboxes(raw_value, parser=parser)
                        except ValueError:
                            row_valid = False
                            break
                        if bboxes:
                            row_has_box = True

                    sampled_rows += 1
                    if row_valid:
                        valid_rows += 1
                    if row_valid and row_has_box:
                        rows_with_boxes += 1

                    if sampled_rows >= max_rows:
                        break
        except OSError:
            continue

        if sampled_rows >= max_rows:
            break

    if sampled_rows == 0 or rows_with_boxes == 0:
        return 0.0

    score = 0.65
    if valid_rows == sampled_rows:
        score += 0.2
    if rows_with_boxes == sampled_rows:
        score += 0.1
    if len(layouts) > 1:
        score += 0.05
    return min(score + parser.score_boost, 1.0)
