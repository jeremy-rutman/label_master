from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.domain.value_objects import ConversionError
from label_master.core.services.convert_service import ConvertRequest, execute_conversion


def _image_files(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])


def _write_coco_dataset_with_named_images(
    dataset_root: Path,
    *,
    image_names: list[str],
) -> None:
    images_dir = dataset_root / "images"
    payload_images = []
    payload_annotations = []
    for index, image_name in enumerate(image_names, start=1):
        image_path = images_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color="black").save(image_path)
        image_id = f"img-{index}"
        annotation_id = f"ann-{index}"
        payload_images.append(
            {
                "id": image_id,
                "file_name": f"images/{image_name}",
                "width": 100,
                "height": 100,
            }
        )
        payload_annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
            }
        )

    payload = {
        "images": payload_images,
        "annotations": payload_annotations,
        "categories": [{"id": 0, "name": "object"}],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def test_convert_copy_images_enabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_copy"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-enabled",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        )
    )

    assert result.report.status == "completed"
    output_images = _image_files(output / "images")
    source_images = _image_files(source / "images")
    assert output_images
    assert [path.name for path in output_images] == [path.name for path in source_images]
    assert all((output / "images") in path.parents for path in output_images)


def test_convert_copy_images_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_no_copy"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-disabled",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=False,
        )
    )

    assert result.report.status == "completed"
    output_images = _image_files(output / "images")
    assert output_images == []


def test_convert_copy_images_with_output_filename_prefix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "shared_out"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-prefixed",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            output_file_name_prefix="coco_minimal",
        )
    )

    assert result.report.status == "completed"
    assert (output / "coco_minimal_classes.txt").exists()
    label_files = sorted((output / "labels").rglob("*.txt"))
    image_files = _image_files(output / "images")
    assert [path.name for path in label_files] == [
        "coco_minimal_46c32d34-example_drone_picture_bw.txt",
        "coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025.txt",
    ]
    assert [path.name for path in image_files] == [
        "coco_minimal_46c32d34-example_drone_picture_bw.jpg",
        "coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025.png",
    ]


def test_convert_filters_input_paths_by_include_and_exclude_substrings(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "filtered_out"

    result = execute_conversion(
        ConvertRequest(
            run_id="filtered-input-paths",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            input_path_include_substring="drone",
            input_path_exclude_substring="screenshot",
        )
    )

    assert result.report.status == "completed"
    assert [path.name for path in (output / "labels").rglob("*.txt")] == [
        "46c32d34-example_drone_picture_bw.txt",
    ]
    assert [path.name for path in _image_files(output / "images")] == [
        "46c32d34-example_drone_picture_bw.jpg",
    ]
    filter_warnings = [warning for warning in result.report.warnings if warning.code == "input_path_filter_applied"]
    assert len(filter_warnings) == 1
    assert filter_warnings[0].context["include_substring"] == "drone"
    assert filter_warnings[0].context["exclude_substring"] == "screenshot"


def test_convert_yolo_shared_output_layout_is_flat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "shared_flat"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-flat-shared",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            flatten_output_layout=True,
            output_file_name_prefix="coco_minimal",
        )
    )

    assert result.report.status == "completed"
    assert (output / "coco_minimal_classes.txt").exists()
    label_files = sorted((output / "labels").glob("*.txt"))
    image_files = _image_files(output / "images")
    assert label_files
    assert image_files
    assert sorted(path.name for path in label_files) == [
        "coco_minimal_46c32d34-example_drone_picture_bw.txt",
        "coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025.txt",
    ]
    assert sorted(path.name for path in image_files) == [
        "coco_minimal_46c32d34-example_drone_picture_bw.jpg",
        "coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025.png",
    ]
    assert not any(path.is_dir() for path in (output / "labels").iterdir())
    assert not any(path.is_dir() for path in (output / "images").iterdir())


def test_convert_detects_flattened_output_collisions_before_writing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "colliding_coco"
    output = tmp_path / "out_collision"
    _write_coco_dataset_with_named_images(
        source,
        image_names=[
            "train img/example.jpg",
            "train_img/example.jpg",
        ],
    )

    with pytest.raises(ConversionError) as exc_info:
        execute_conversion(
            ConvertRequest(
                run_id="copy-images-flat-collision",
                input_path=source,
                output_path=output,
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.YOLO,
                unmapped_policy=UnmappedPolicy.ERROR,
                copy_images=True,
                flatten_output_layout=True,
            )
        )

    assert "Output path collision detected" in str(exc_info.value)
    assert exc_info.value.context["first_collision_kind"] == "label"
    assert exc_info.value.context["first_collision_path"] == "labels/train_img_example.txt"
    assert "images/train img/example.jpg" in exc_info.value.context["first_collision_sources"]
    assert "images/train_img/example.jpg" in exc_info.value.context["first_collision_sources"]


def test_convert_detects_existing_destination_label_before_writing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_existing_label"
    existing_label = output / "labels" / "46c32d34-example_drone_picture_bw.txt"
    existing_label.parent.mkdir(parents=True, exist_ok=True)
    existing_label.write_text("stale\n", encoding="utf-8")

    with pytest.raises(ConversionError) as exc_info:
        execute_conversion(
            ConvertRequest(
                run_id="copy-images-existing-label",
                input_path=source,
                output_path=output,
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.YOLO,
                unmapped_policy=UnmappedPolicy.ERROR,
                copy_images=True,
            )
        )

    assert "Existing destination path detected" in str(exc_info.value)
    assert exc_info.value.context["first_existing_kind"] == "label"
    assert exc_info.value.context["first_existing_path"] == "labels/46c32d34-example_drone_picture_bw.txt"


def test_convert_allow_overwrite_replaces_existing_outputs_and_logs_warning(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_allow_overwrite"
    existing_label = output / "labels" / "46c32d34-example_drone_picture_bw.txt"
    existing_image = output / "images" / "46c32d34-example_drone_picture_bw.jpg"
    existing_label.parent.mkdir(parents=True, exist_ok=True)
    existing_image.parent.mkdir(parents=True, exist_ok=True)
    existing_label.write_text("stale-label\n", encoding="utf-8")
    existing_image.write_text("stale-image\n", encoding="utf-8")

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-allow-overwrite",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            allow_overwrite=True,
        )
    )

    assert result.report.status == "completed"
    overwrite_warnings = [warning for warning in result.report.warnings if warning.code == "output_files_overwritten"]
    assert len(overwrite_warnings) == 1
    assert overwrite_warnings[0].context["overwritten_count"] == "2"
    assert "labels/46c32d34-example_drone_picture_bw.txt" in overwrite_warnings[0].context["overwritten_paths_json"]
    assert "images/46c32d34-example_drone_picture_bw.jpg" in overwrite_warnings[0].context["overwritten_paths_json"]
    assert existing_label.read_text(encoding="utf-8") != "stale-label\n"
    assert existing_image.read_bytes() != b"stale-image\n"


def test_convert_detects_existing_destination_parent_path_before_writing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_existing_parent"
    blocked_parent = output / "labels"
    blocked_parent.parent.mkdir(parents=True, exist_ok=True)
    blocked_parent.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ConversionError) as exc_info:
        execute_conversion(
            ConvertRequest(
                run_id="copy-images-existing-parent",
                input_path=source,
                output_path=output,
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.YOLO,
                unmapped_policy=UnmappedPolicy.ERROR,
                copy_images=False,
            )
        )

    assert "Existing destination path detected" in str(exc_info.value)
    assert exc_info.value.context["first_existing_kind"] == "label_parent_path"
    assert exc_info.value.context["first_existing_path"] == "labels"


def test_convert_allow_overwrite_still_rejects_existing_destination_parent_path(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_existing_parent_allow_overwrite"
    blocked_parent = output / "labels"
    blocked_parent.parent.mkdir(parents=True, exist_ok=True)
    blocked_parent.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ConversionError) as exc_info:
        execute_conversion(
            ConvertRequest(
                run_id="copy-images-existing-parent-allow-overwrite",
                input_path=source,
                output_path=output,
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.YOLO,
                unmapped_policy=UnmappedPolicy.ERROR,
                copy_images=False,
                allow_overwrite=True,
            )
        )

    assert "Existing destination path detected" in str(exc_info.value)
    assert exc_info.value.context["first_existing_kind"] == "label_parent_path"
    assert exc_info.value.context["first_existing_path"] == "labels"


def test_convert_yolo_output_stem_affixes_apply_to_artifacts_images_and_labels(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "stem_affixes"

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-stem-affixes",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            output_file_name_prefix="coco_minimal",
            output_file_stem_prefix="batchA_",
            output_file_stem_suffix="_fold1",
        )
    )

    assert result.report.status == "completed"
    assert (output / "batchA_coco_minimal_classes_fold1.txt").exists()
    assert sorted(path.name for path in (output / "labels").rglob("*.txt")) == [
        "batchA_coco_minimal_46c32d34-example_drone_picture_bw_fold1.txt",
        "batchA_coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025_fold1.txt",
    ]
    assert sorted(path.name for path in _image_files(output / "images")) == [
        "batchA_coco_minimal_46c32d34-example_drone_picture_bw_fold1.jpg",
        "batchA_coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025_fold1.png",
    ]


def test_convert_coco_output_with_prefix_uses_prefixed_annotations_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "shared_out_coco"

    result = execute_conversion(
        ConvertRequest(
            run_id="coco-prefixed",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.COCO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=False,
            output_file_name_prefix="coco_minimal",
            output_file_stem_prefix="batchA_",
            output_file_stem_suffix="_fold1",
        )
    )

    assert result.report.status == "completed"
    annotations_path = output / "batchA_coco_minimal_annotations_fold1.json"
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    assert sorted(Path(item["file_name"]).name for item in payload["images"]) == [
        "batchA_coco_minimal_46c32d34-example_drone_picture_bw_fold1.jpg",
        "batchA_coco_minimal_90f04c9f-Drone_Detection_screenshot_05.12.2025_fold1.png",
    ]


def test_convert_emits_progress_updates_for_longer_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_progress"
    progress_updates: list[tuple[str, int]] = []

    result = execute_conversion(
        ConvertRequest(
            run_id="copy-images-progress",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
        ),
        progress_callback=lambda message, percent: progress_updates.append((message, percent)),
    )

    assert result.report.status == "completed"
    assert progress_updates
    assert [percent for _, percent in progress_updates] == sorted(
        percent for _, percent in progress_updates
    )
    assert progress_updates[-1] == ("Conversion complete.", 100)
    assert any("Validating dataset" in message for message, _ in progress_updates)
    assert any("Validating annotations... (" in message for message, _ in progress_updates)
    assert any("Writing YOLO output" in message for message, _ in progress_updates)
    assert any("Copying images" in message for message, _ in progress_updates)


def test_convert_size_gate_can_drop_small_images_from_output(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_size_gate_drop"

    result = execute_conversion(
        ConvertRequest(
            run_id="size-gate-drop-small",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.COCO,
            unmapped_policy=UnmappedPolicy.ERROR,
            min_image_longest_edge_px=800,
        )
    )

    payload = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    assert result.report.status == "completed"
    assert result.report.summary_counts.images == 1
    assert result.report.summary_counts.annotations_out == 1
    assert result.report.summary_counts.dropped == 6
    assert result.report.summary_counts.skipped == 1
    assert [warning.code for warning in result.report.warnings] == ["size_gate_small_images_dropped"]
    assert len(result.dropped_annotations) == 6
    assert all(item.stage == "size_gate" for item in result.dropped_annotations)
    assert all(item.reason_code == "image_below_min_longest_edge" for item in result.dropped_annotations)
    assert [image["file_name"] for image in payload["images"]] == [
        "images/90f04c9f-Drone_Detection_screenshot_05.12.2025.png"
    ]
    assert len(payload["annotations"]) == 1


def test_convert_size_gate_can_downscale_large_images_when_copying(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")
    output = tmp_path / "out_size_gate_downscale"

    result = execute_conversion(
        ConvertRequest(
            run_id="size-gate-downscale-large",
            input_path=source,
            output_path=output,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.COCO,
            unmapped_policy=UnmappedPolicy.ERROR,
            copy_images=True,
            max_image_longest_edge_px=1000,
            oversize_image_action="downscale",
        )
    )

    payload = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
    large_image = next(
        item
        for item in payload["images"]
        if item["file_name"] == "images/90f04c9f-Drone_Detection_screenshot_05.12.2025.png"
    )
    large_annotation = next(
        item
        for item in payload["annotations"]
        if item["image_id"] == large_image["id"]
    )

    assert result.report.status == "completed"
    assert result.report.summary_counts.images == 2
    assert result.report.summary_counts.annotations_out == 7
    assert [warning.code for warning in result.report.warnings] == ["size_gate_large_images_downscaled"]
    assert large_image["width"] == 1000
    assert large_image["height"] == 562
    assert large_annotation["bbox"] == pytest.approx(
        [915.6818181818185, 250.68606060606058, 84.31818181818161, 114.44363636363639]
    )
    with Image.open(output / "images" / "90f04c9f-Drone_Detection_screenshot_05.12.2025.png") as opened:
        assert opened.size == (1000, 562)
    with Image.open(output / "images" / "46c32d34-example_drone_picture_bw.jpg") as opened:
        assert opened.size == (720, 400)


def test_convert_size_gate_rejects_downscale_without_copy_images(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = Path("tests/fixtures/us1/coco_minimal")

    with pytest.raises(ConversionError, match="Downscaling oversized images requires 'copy_images'"):
        execute_conversion(
            ConvertRequest(
                run_id="size-gate-downscale-no-copy",
                input_path=source,
                output_path=tmp_path / "out_invalid",
                src_format=SourceFormat.COCO,
                dst_format=SourceFormat.COCO,
                unmapped_policy=UnmappedPolicy.ERROR,
                max_image_longest_edge_px=1000,
                oversize_image_action="downscale",
            )
        )
