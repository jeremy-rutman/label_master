from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui.app import render_preview_overlay
from label_master.interfaces.gui.viewmodels import preview_dataset_view


def test_render_preview_overlay_success() -> None:
    dataset_root = Path("tests/fixtures/us1/coco_minimal")
    preview = preview_dataset_view(dataset_root, source_format="coco")
    image = preview.images[0]

    overlay, warnings = render_preview_overlay(
        dataset_root=dataset_root,
        image_rel_path=image.file_name,
        bboxes=[
            (
                bbox.bbox_xywh_abs[0],
                bbox.bbox_xywh_abs[1],
                bbox.bbox_xywh_abs[2],
                bbox.bbox_xywh_abs[3],
                f"{bbox.class_id}:{bbox.class_name}",
            )
            for bbox in image.bboxes
        ],
    )

    assert overlay is not None
    assert warnings == []
    assert overlay.size[0] > 0
    assert overlay.size[1] > 0


def test_render_preview_overlay_missing_image_warning() -> None:
    overlay, warnings = render_preview_overlay(
        dataset_root=Path("tests/fixtures/us1/coco_minimal"),
        image_rel_path="images/does_not_exist.jpg",
        bboxes=[],
    )

    assert overlay is None
    assert len(warnings) == 1
    assert "not found" in warnings[0]
