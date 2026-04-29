from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from label_master.interfaces.gui.app import (
    PREVIEW_MAX_IMAGE_DIMENSION,
    bbox_size_histogram_spec,
    build_annotation_distribution_rows,
    build_class_example_groups,
    class_occurrence_chart_spec,
    render_preview_overlay,
)
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


def test_render_preview_overlay_video_bbox_sequence_success() -> None:
    dataset_root = Path("tests/fixtures/us5")
    preview = preview_dataset_view(dataset_root, source_format="video_bbox")
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
    assert overlay.size == (1280, 720)


def test_render_preview_overlay_voc_success() -> None:
    dataset_root = Path("tests/fixtures/us6")
    preview = preview_dataset_view(dataset_root, source_format="voc")
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
    assert overlay.size == (1280, 720)


def test_render_preview_overlay_matlab_ground_truth_video_success() -> None:
    dataset_root = Path("tests/fixtures/us9")
    preview = preview_dataset_view(dataset_root, source_format="matlab_ground_truth")
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


def test_render_preview_overlay_downscales_large_images(tmp_path) -> None:  # type: ignore[no-untyped-def]
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (4000, 3000), color="black").save(images_dir / "large.jpg")

    overlay, warnings = render_preview_overlay(
        dataset_root=tmp_path,
        image_rel_path="images/large.jpg",
        bboxes=[(100.0, 120.0, 600.0, 500.0, "1:test")],
    )

    assert warnings == []
    assert overlay is not None
    assert max(overlay.size) == PREVIEW_MAX_IMAGE_DIMENSION


def test_build_annotation_distribution_rows_collects_class_and_bbox_sizes() -> None:
    dataset = SimpleNamespace(
        annotations=[
            SimpleNamespace(class_id=2, bbox_xywh_abs=(0, 0, 10, 20)),
            SimpleNamespace(class_id=2, bbox_xywh_abs=(3, 4, 5, 6)),
            SimpleNamespace(class_id=4, bbox_xywh_abs=(1, 1, 7, 8)),
        ]
    )

    rows = build_annotation_distribution_rows(dataset)

    assert rows == [
        {"class_id": "2", "bbox_width": 10.0, "bbox_height": 20.0},
        {"class_id": "2", "bbox_width": 5.0, "bbox_height": 6.0},
        {"class_id": "4", "bbox_width": 7.0, "bbox_height": 8.0},
    ]


def test_build_class_example_groups_collects_examples_per_class() -> None:
    preview = SimpleNamespace(
        images=[
            SimpleNamespace(
                file_name="images/a.jpg",
                bboxes=[
                    SimpleNamespace(class_id=1, class_name="truck", bbox_xywh_abs=(1, 2, 3, 4)),
                    SimpleNamespace(class_id=1, class_name="truck", bbox_xywh_abs=(5, 6, 7, 8)),
                    SimpleNamespace(class_id=2, class_name="person", bbox_xywh_abs=(9, 10, 11, 12)),
                ],
            ),
            SimpleNamespace(
                file_name="images/b.jpg",
                bboxes=[
                    SimpleNamespace(class_id=2, class_name="person", bbox_xywh_abs=(2, 3, 4, 5)),
                ],
            ),
            SimpleNamespace(
                file_name="images/c.jpg",
                bboxes=[
                    SimpleNamespace(class_id=1, class_name="truck", bbox_xywh_abs=(3, 4, 5, 6)),
                ],
            ),
        ]
    )

    groups = build_class_example_groups(preview, examples_per_class=2)

    assert [(group.class_id, group.class_name, group.image_count, len(group.examples)) for group in groups] == [
        (1, "truck", 2, 2),
        (2, "person", 2, 2),
    ]
    assert groups[0].examples[0].file_name == "images/a.jpg"
    assert groups[0].examples[0].annotation_count == 2
    assert groups[0].examples[0].overlay_labels == (
        (1.0, 2.0, 3.0, 4.0, "1:truck"),
        (5.0, 6.0, 7.0, 8.0, "1:truck"),
    )
def test_chart_specs_reference_expected_fields() -> None:
    rows = [{"class_id": "2", "bbox_width": 10.0, "bbox_height": 20.0}]

    class_spec = class_occurrence_chart_spec(rows)
    bbox_spec = bbox_size_histogram_spec(rows)

    assert class_spec["data"]["values"] == rows
    assert class_spec["encoding"]["x"]["field"] == "class_id"
    assert class_spec["encoding"]["y"]["aggregate"] == "count"

    assert bbox_spec["data"]["values"] == rows
    assert bbox_spec["encoding"]["x"]["field"] == "bbox_width"
    assert bbox_spec["encoding"]["y"]["field"] == "bbox_height"
    assert bbox_spec["encoding"]["color"]["aggregate"] == "count"
