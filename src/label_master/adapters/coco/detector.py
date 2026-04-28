from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from label_master.format_specs.registry import (
    JsonObjectDatasetParserSpec,
    resolve_builtin_format_spec,
)


@lru_cache(maxsize=1)
def _coco_parser() -> JsonObjectDatasetParserSpec | None:
    spec = resolve_builtin_format_spec("coco")
    if spec is None or not isinstance(spec.parser, JsonObjectDatasetParserSpec):
        return None
    return spec.parser


def detect_coco(path: Path, *, sample_limit: int = 500) -> float:
    """Return confidence score that dataset at path is COCO-like."""
    del sample_limit
    parser = _coco_parser()
    if parser is None:
        return 0.0

    annotations_file = path / parser.annotations_file
    if not annotations_file.exists():
        return 0.0

    try:
        with annotations_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0.0

    if not isinstance(payload, dict):
        return 0.0

    keys = {parser.images_key, parser.annotations_key, parser.categories_key}
    present = sum(1 for key in keys if key in payload)
    return min((present / len(keys)) + parser.score_boost, 1.0)
