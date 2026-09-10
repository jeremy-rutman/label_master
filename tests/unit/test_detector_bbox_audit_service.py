from __future__ import annotations

from pathlib import Path

from PIL import Image

import label_master.core.services.missing_label_hint_service as hint_service
from label_master.core.services.missing_label_hint_service import (
    HintDetection,
    apply_yolo_bbox_audit,
    discover_yolo_review_image_paths,
    generate_yolo_bbox_audit,
    generate_yolo_detector_review,
    generate_yolo_detector_review_for_image,
)


def _write_yolo_dataset(dataset_root: Path) -> Path:
    image_path = dataset_root / "train" / "images" / "example.jpg"
    label_path = dataset_root / "train" / "labels" / "example.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color="black").save(image_path)
    label_path.write_text(
        "0 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.200000 0.200000 0.100000 0.100000\n",
        encoding="utf-8",
    )
    (dataset_root / "classes.txt").write_text("drone\nbird\nplane\n", encoding="utf-8")
    return label_path


def _write_unlabeled_image(dataset_root: Path, *, image_rel: str = "train/images/unlabeled.jpg") -> None:
    image_path = dataset_root / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color="black").save(image_path)


def test_generate_yolo_bbox_audit_proposes_add_remove_and_adjust(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path)

    def _predict(_image_path: Path) -> list[HintDetection]:
        return [
            HintDetection(
                class_id=0,
                class_name="drone",
                confidence=0.93,
                bbox_xywh_normalized=(0.52, 0.52, 0.20, 0.20),
            ),
            HintDetection(
                class_id=2,
                class_name="plane",
                confidence=0.88,
                bbox_xywh_normalized=(0.80, 0.80, 0.10, 0.10),
            ),
        ]

    result = generate_yolo_bbox_audit(
        dataset_root=tmp_path,
        detector_model_path=tmp_path / "model.pt",
        report_output_dir=tmp_path / "audit",
        max_labeled_images=10,
        predictor=_predict,
    )

    assert result.scanned_labeled_images == 1
    assert result.images_with_proposals == 1
    assert result.add_proposals == 1
    assert result.remove_proposals == 1
    assert result.adjust_proposals == 1
    assert len(result.items) == 1
    assert [proposal.action for proposal in result.items[0].proposals] == ["remove", "adjust", "add"]


def test_apply_yolo_bbox_audit_rewrites_label_file_with_selected_proposals(tmp_path: Path) -> None:
    label_path = _write_yolo_dataset(tmp_path)

    def _predict(_image_path: Path) -> list[HintDetection]:
        return [
            HintDetection(
                class_id=0,
                class_name="drone",
                confidence=0.93,
                bbox_xywh_normalized=(0.52, 0.52, 0.20, 0.20),
            ),
            HintDetection(
                class_id=2,
                class_name="plane",
                confidence=0.88,
                bbox_xywh_normalized=(0.80, 0.80, 0.10, 0.10),
            ),
        ]

    audit = generate_yolo_bbox_audit(
        dataset_root=tmp_path,
        detector_model_path=tmp_path / "model.pt",
        report_output_dir=tmp_path / "audit",
        max_labeled_images=10,
        predictor=_predict,
    )

    approval = apply_yolo_bbox_audit(
        dataset_root=tmp_path,
        approved_items=audit.items,
    )

    assert approval.approved_label_files == 1
    assert approval.applied_proposals == 3
    assert approval.label_paths == [str(label_path)]
    assert label_path.read_text(encoding="utf-8") == (
        "0 0.520000 0.520000 0.200000 0.200000\n"
        "2 0.800000 0.800000 0.100000 0.100000\n"
    )


def test_generate_yolo_detector_review_reuses_one_predictor(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _write_yolo_dataset(tmp_path)
    _write_unlabeled_image(tmp_path)

    build_calls = 0
    prediction_calls: list[str] = []

    def _fake_build_predictor(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal build_calls
        build_calls += 1

        def _predict(image_path: Path) -> list[HintDetection]:
            prediction_calls.append(image_path.name)
            return [
                HintDetection(
                    class_id=0,
                    class_name="drone",
                    confidence=0.93,
                    bbox_xywh_normalized=(0.52, 0.52, 0.20, 0.20),
                ),
                HintDetection(
                    class_id=2,
                    class_name="plane",
                    confidence=0.88,
                    bbox_xywh_normalized=(0.80, 0.80, 0.10, 0.10),
                ),
            ]

        return _predict

    monkeypatch.setattr(hint_service, "_build_ultralytics_predictor", _fake_build_predictor)

    result = generate_yolo_detector_review(
        dataset_root=tmp_path,
        detector_model_path=tmp_path / "model.pt",
        hints_output_dir=tmp_path / "hints",
        report_output_dir=tmp_path / "audit",
    )

    assert build_calls == 1
    assert sorted(prediction_calls) == ["example.jpg", "unlabeled.jpg"]
    assert result.missing_label_hints.hinted_images == 1
    assert result.missing_label_hints.hint_files_written == 1
    assert result.bbox_audit.images_with_proposals == 1
    assert result.bbox_audit.add_proposals == 1


def test_discover_yolo_review_image_paths_returns_sorted_filtered_images(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path)
    _write_unlabeled_image(tmp_path)
    _write_unlabeled_image(tmp_path, image_rel="train/images/subset/second.jpg")

    image_paths = discover_yolo_review_image_paths(
        dataset_root=tmp_path,
        input_path_include_substring="train/images",
        input_path_exclude_substring="subset",
    )

    assert image_paths == [
        "train/images/example.jpg",
        "train/images/unlabeled.jpg",
    ]


def test_generate_yolo_detector_review_for_image_builds_single_missing_label_item(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path)
    _write_unlabeled_image(tmp_path)

    def _predict(_image_path: Path) -> list[HintDetection]:
        return [
            HintDetection(
                class_id=2,
                class_name="plane",
                confidence=0.88,
                bbox_xywh_normalized=(0.80, 0.80, 0.10, 0.10),
            )
        ]

    review = generate_yolo_detector_review_for_image(
        dataset_root=tmp_path,
        image_rel_path="train/images/unlabeled.jpg",
        detector_model_path=tmp_path / "model.pt",
        predictor=_predict,
    )

    assert review.image_rel_path == "train/images/unlabeled.jpg"
    assert review.label_rel_path == "train/labels/unlabeled.txt"
    assert review.has_existing_label_file is False
    assert review.existing_label_count == 0
    assert [proposal.action for proposal in review.proposals] == ["add"]


def test_generate_yolo_detector_review_for_image_builds_single_bbox_audit_item(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path)

    def _predict(_image_path: Path) -> list[HintDetection]:
        return [
            HintDetection(
                class_id=0,
                class_name="drone",
                confidence=0.93,
                bbox_xywh_normalized=(0.52, 0.52, 0.20, 0.20),
            ),
            HintDetection(
                class_id=2,
                class_name="plane",
                confidence=0.88,
                bbox_xywh_normalized=(0.80, 0.80, 0.10, 0.10),
            ),
        ]

    review = generate_yolo_detector_review_for_image(
        dataset_root=tmp_path,
        image_rel_path="train/images/example.jpg",
        detector_model_path=tmp_path / "model.pt",
        predictor=_predict,
    )

    assert review.image_rel_path == "train/images/example.jpg"
    assert review.label_rel_path == "train/labels/example.txt"
    assert review.has_existing_label_file is True
    assert review.existing_label_count == 2
    assert [proposal.action for proposal in review.proposals] == ["remove", "adjust", "add"]