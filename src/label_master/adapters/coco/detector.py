from __future__ import annotations

import json
from pathlib import Path


def detect_coco(path: Path) -> float:
    """Return confidence score that dataset at path is COCO-like."""
    annotations_file = path / "annotations.json"
    if not annotations_file.exists():
        return 0.0

    try:
        with annotations_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0.0

    if not isinstance(payload, dict):
        return 0.0

    keys = {"images", "annotations", "categories"}
    present = sum(1 for key in keys if key in payload)
    return present / len(keys)
