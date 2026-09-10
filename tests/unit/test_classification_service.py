from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from label_master.core.domain.value_objects import ConfigurationError, PathTraversalError
from label_master.core.services.classification_service import (
    ClassificationClass,
    ClassificationConfig,
    apply_classification_label,
    clear_classification_label,
    discover_classification_image_paths,
    find_default_classification_config_path,
    image_labels,
    load_classification_config,
    load_classification_labels,
    next_unlabeled_index,
    parse_classification_config,
    resolve_classification_labels_path,
    save_classification_labels,
    summarize_classification_labels,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def test_parse_config_with_explicit_keys_and_options() -> None:
    config = parse_classification_config(
        {
            "classes": [
                {"key": "1", "name": "helicopter"},
                {"key": "A", "name": "airplane"},
            ],
            "multi_label": True,
            "labels_file": "cls/labels.json",
        }
    )
    assert config.classes == (
        ClassificationClass(key="1", name="helicopter"),
        ClassificationClass(key="a", name="airplane"),
    )
    assert config.multi_label is True
    assert config.labels_file == "cls/labels.json"
    assert config.class_for_key("A") == ClassificationClass(key="a", name="airplane")
    assert config.class_for_key("9") is None


def test_parse_config_assigns_keys_for_plain_name_list() -> None:
    names = [f"class_{index}" for index in range(12)]
    config = parse_classification_config({"classes": names})
    assert config.keys == list("1234567890ab")
    assert config.class_names == names
    assert config.multi_label is False
    assert config.labels_file == "classification_labels.json"


def test_parse_config_auto_keys_skip_explicit_ones() -> None:
    config = parse_classification_config(
        {"classes": [{"key": "1", "name": "one"}, "two", {"key": "2", "name": "three"}]}
    )
    assert config.keys == ["1", "3", "2"]


def test_parse_config_accepts_yolo_names_mapping() -> None:
    config = parse_classification_config({"names": {1: "truck", 0: "car"}})
    assert config.class_names == ["car", "truck"]
    assert config.keys == ["1", "2"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"classes": []},
        {"classes": "helicopter"},
        {"classes": [{"key": "1", "name": ""}]},
        {"classes": [{"key": "12", "name": "a"}]},
        {"classes": [{"key": "1", "name": "a"}, {"key": "1", "name": "b"}]},
        {"classes": ["a", "a"]},
        {"classes": ["a"], "multi_label": "yes"},
    ],
)
def test_parse_config_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        parse_classification_config(payload)


def test_parse_config_rejects_too_many_auto_keys() -> None:
    with pytest.raises(ConfigurationError):
        parse_classification_config({"classes": [f"c{index}" for index in range(40)]})


def test_load_config_from_yaml_and_json(tmp_path: Path) -> None:
    yaml_path = tmp_path / "classification.yaml"
    yaml_path.write_text("classes:\n  - key: '1'\n    name: cat\n  - key: '2'\n    name: dog\n", encoding="utf-8")
    json_path = tmp_path / "classification.json"
    json_path.write_text(json.dumps({"classes": ["cat", "dog"]}), encoding="utf-8")

    yaml_config = load_classification_config(yaml_path)
    json_config = load_classification_config(json_path)
    assert yaml_config.class_names == ["cat", "dog"]
    assert json_config.class_names == ["cat", "dog"]
    assert yaml_config.source_path == yaml_path.resolve()

    with pytest.raises(ConfigurationError):
        load_classification_config(tmp_path / "missing.yaml")


def test_find_default_config_prefers_yaml(tmp_path: Path) -> None:
    assert find_default_classification_config_path(tmp_path) is None
    (tmp_path / "classification.json").write_text("{}", encoding="utf-8")
    assert find_default_classification_config_path(tmp_path) == tmp_path / "classification.json"
    (tmp_path / "classification.yaml").write_text("", encoding="utf-8")
    assert find_default_classification_config_path(tmp_path) == tmp_path / "classification.yaml"


def test_discover_images_respects_filters_and_sorting(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "b.jpg")
    _write_image(tmp_path / "images" / "a.png")
    _write_image(tmp_path / "val" / "c.jpeg")
    (tmp_path / "images" / "a.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

    assert discover_classification_image_paths(tmp_path) == ["images/a.png", "images/b.jpg", "val/c.jpeg"]
    assert discover_classification_image_paths(tmp_path, input_path_include_substring="images/") == [
        "images/a.png",
        "images/b.jpg",
    ]
    assert discover_classification_image_paths(tmp_path, input_path_exclude_substring="val") == [
        "images/a.png",
        "images/b.jpg",
    ]
    with pytest.raises(FileNotFoundError):
        discover_classification_image_paths(tmp_path / "nope")


def test_labels_round_trip_and_manifest_shape(tmp_path: Path) -> None:
    config = parse_classification_config({"classes": ["cat", "dog"]})
    labels_path = resolve_classification_labels_path(tmp_path, config)
    assert labels_path == (tmp_path / "classification_labels.json").resolve()
    assert load_classification_labels(labels_path) == {}

    labels = apply_classification_label({}, "images/a.png", "cat", multi_label=False)
    labels = apply_classification_label(labels, "images/b.jpg", "dog", multi_label=False)
    labels["images/empty.jpg"] = []
    save_classification_labels(labels_path, labels, config=config)

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["classes"] == ["cat", "dog"]
    assert payload["multi_label"] is False
    assert payload["labels"] == {"images/a.png": ["cat"], "images/b.jpg": ["dog"]}

    assert load_classification_labels(labels_path) == {
        "images/a.png": ["cat"],
        "images/b.jpg": ["dog"],
    }


def test_resolve_labels_path_accepts_absolute_and_rejects_escape(tmp_path: Path) -> None:
    absolute_target = tmp_path / "elsewhere" / "labels.json"
    config = ClassificationConfig(
        classes=(ClassificationClass(key="1", name="cat"),),
        labels_file=str(absolute_target),
    )
    assert resolve_classification_labels_path(tmp_path / "dataset", config) == absolute_target

    escaping = ClassificationConfig(
        classes=(ClassificationClass(key="1", name="cat"),),
        labels_file="../outside.json",
    )
    with pytest.raises(PathTraversalError):
        resolve_classification_labels_path(tmp_path / "dataset", escaping)


def test_load_labels_accepts_legacy_plain_mapping(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"a.jpg": "cat", "b.jpg": ["cat", "dog"], "c.jpg": None}), encoding="utf-8")
    assert load_classification_labels(labels_path) == {"a.jpg": ["cat"], "b.jpg": ["cat", "dog"], "c.jpg": []}

    labels_path.write_text(json.dumps({"labels": {"a.jpg": 3}}), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_classification_labels(labels_path)


def test_apply_label_single_replaces_and_multi_toggles() -> None:
    labels = apply_classification_label({}, "a.jpg", "cat", multi_label=False)
    labels = apply_classification_label(labels, "a.jpg", "dog", multi_label=False)
    assert labels == {"a.jpg": ["dog"]}

    labels = apply_classification_label(labels, "a.jpg", "cat", multi_label=True)
    assert labels == {"a.jpg": ["dog", "cat"]}
    labels = apply_classification_label(labels, "a.jpg", "dog", multi_label=True)
    assert labels == {"a.jpg": ["cat"]}
    labels = apply_classification_label(labels, "a.jpg", "cat", multi_label=True)
    assert labels == {}

    assert image_labels({"x/y.jpg": ["cat"]}, "x\\y.jpg") == ["cat"]
    assert clear_classification_label({"a.jpg": ["cat"], "b.jpg": ["dog"]}, "a.jpg") == {"b.jpg": ["dog"]}


def test_next_unlabeled_index_wraps_and_summary_counts() -> None:
    config = parse_classification_config({"classes": ["cat", "dog"]})
    images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    labels = {"a.jpg": ["cat"], "c.jpg": ["cat", "dog"], "zzz_missing.jpg": ["dog"]}

    assert next_unlabeled_index(images, labels, start_index=0) == 1
    assert next_unlabeled_index(images, labels, start_index=1) == 3
    assert next_unlabeled_index(images, labels, start_index=3) == 1
    assert next_unlabeled_index(images, {path: ["cat"] for path in images}, start_index=0) is None
    assert next_unlabeled_index([], labels, start_index=0) is None

    summary = summarize_classification_labels(images, labels, config=config)
    assert summary.image_count == 4
    assert summary.labeled_count == 2
    assert summary.unlabeled_count == 2
    assert summary.class_counts == {"cat": 2, "dog": 1}
