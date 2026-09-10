from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from label_master.adapters.cityscapes.common import (
    available_split_names,
    resolve_labels_root,
)
from label_master.adapters.cityscapes.reader import read_cityscapes_dataset
from label_master.adapters.coco.writer import write_coco_dataset
from label_master.core.domain.entities import AnnotationDataset


@dataclass(frozen=True)
class ExportSummary:
    split_name: str
    images: int
    annotations: int
    skipped_annotation_files: int
    skipped_objects: int
    output_path: Path


def _int_detail(dataset: AnnotationDataset, key: str) -> int:
    return int(dataset.source_metadata.details.get(key, "0"))


def export_cityscapes_split_to_coco(
    *,
    dataset_root: Path,
    split_name: str,
    output_path: Path,
    label_mode: Literal["instances", "all"] = "instances",
    include_groups: bool = False,
) -> ExportSummary:
    dataset = read_cityscapes_dataset(
        dataset_root,
        split_names=[split_name],
        label_mode=label_mode,
        include_groups=include_groups,
    )
    written_output = write_coco_dataset(
        dataset,
        output_path.parent,
        annotations_file_name=output_path.name,
    )

    return ExportSummary(
        split_name=split_name,
        images=len(dataset.images),
        annotations=len(dataset.annotations),
        skipped_annotation_files=_int_detail(dataset, "annotation_files_skipped"),
        skipped_objects=_int_detail(dataset, "skipped_objects"),
        output_path=written_output,
    )


def export_cityscapes_dataset(
    *,
    dataset_root: Path,
    output_dir: Path,
    split_names: list[str] | None = None,
    label_mode: Literal["instances", "all"] = "instances",
    include_groups: bool = False,
) -> list[ExportSummary]:
    _, labels_root = resolve_labels_root(dataset_root)
    resolved_splits = split_names or available_split_names(labels_root, dataset_root)
    if not resolved_splits:
        raise ValueError(
            "No Cityscapes splits have both polygon annotations and matching leftImg8bit images."
        )

    summaries: list[ExportSummary] = []
    for split_name in resolved_splits:
        output_path = output_dir / f"cityscapes_bbox_{label_mode}_{split_name}.json"
        summaries.append(
            export_cityscapes_split_to_coco(
                dataset_root=dataset_root,
                split_name=split_name,
                output_path=output_path,
                label_mode=label_mode,
                include_groups=include_groups,
            )
        )
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes polygon annotations into COCO bbox annotations."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Cityscapes dataset root containing gtFine/ or gtCoarse/ plus leftImg8bit_*/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where COCO JSON files will be written.",
    )
    parser.add_argument(
        "--set-name",
        dest="set_names",
        action="append",
        default=None,
        help="Split to export. Repeat for multiple splits. Defaults to the splits that have both labels and images.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("instances", "all"),
        default="instances",
        help="Export only instance-capable classes or all non-ignored Cityscapes labels.",
    )
    parser.add_argument(
        "--include-groups",
        action="store_true",
        help="Include labels such as cargroup as crowd boxes mapped to their base class.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summaries = export_cityscapes_dataset(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        split_names=args.set_names,
        label_mode=args.label_mode,
        include_groups=args.include_groups,
    )

    totals = defaultdict(int)
    for summary in summaries:
        totals["images"] += summary.images
        totals["annotations"] += summary.annotations
        totals["skipped_annotation_files"] += summary.skipped_annotation_files
        totals["skipped_objects"] += summary.skipped_objects
        print(
            f"{summary.split_name}: images={summary.images} annotations={summary.annotations} "
            f"skipped_annotation_files={summary.skipped_annotation_files} "
            f"skipped_objects={summary.skipped_objects} output={summary.output_path}"
        )

    print(
        "total: "
        f"images={totals['images']} annotations={totals['annotations']} "
        f"skipped_annotation_files={totals['skipped_annotation_files']} "
        f"skipped_objects={totals['skipped_objects']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())