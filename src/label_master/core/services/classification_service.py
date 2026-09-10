"""Per-image classification labels driven by a small config file.

The config file (YAML or JSON) declares the classes and the single keystroke that
assigns each one:

    classes:
      - key: "1"
        name: helicopter
      - key: "2"
        name: airplane
    multi_label: false
    labels_file: classification_labels.json

``classes`` may also be a plain list of names (``[helicopter, airplane]``); keys
are then assigned automatically as ``1``-``9``, ``0``, then ``a``-``z``. A YOLO style
``names`` list or ``{id: name}`` mapping is accepted as an alias for ``classes``.

Labels are stored in one JSON manifest (``labels_file``, relative to the dataset
root unless absolute) mapping the image's dataset-relative path to a list of class
names. The manifest is independent of any bounding-box label files, so it can be
used alongside them or on a directory of bare images.
"""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from label_master.core.domain.value_objects import ConfigurationError
from label_master.infra.filesystem import (
    atomic_write_json,
    build_input_path_filter,
    iter_files,
    read_json,
    read_yaml,
    relative_path_matches_input_filter,
    safe_resolve,
)

CLASSIFICATION_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
CLASSIFICATION_CONFIG_FILE_CANDIDATES: tuple[str, ...] = (
    "classification.yaml",
    "classification.yml",
    "classification.json",
    "classification_config.yaml",
    "classification_config.yml",
    "classification_config.json",
)
DEFAULT_CLASSIFICATION_LABELS_FILE = "classification_labels.json"
CLASSIFICATION_LABELS_MANIFEST_VERSION = 1
AUTO_CLASS_KEYS: tuple[str, ...] = tuple("1234567890") + tuple(string.ascii_lowercase)


@dataclass(frozen=True)
class ClassificationClass:
    key: str
    name: str


@dataclass(frozen=True)
class ClassificationConfig:
    classes: tuple[ClassificationClass, ...]
    multi_label: bool = False
    labels_file: str = DEFAULT_CLASSIFICATION_LABELS_FILE
    source_path: Path | None = None

    @property
    def class_names(self) -> list[str]:
        return [entry.name for entry in self.classes]

    @property
    def keys(self) -> list[str]:
        return [entry.key for entry in self.classes]

    def class_for_key(self, key: str | None) -> ClassificationClass | None:
        normalized = _normalize_key(key)
        if normalized is None:
            return None
        for entry in self.classes:
            if entry.key == normalized:
                return entry
        return None


@dataclass(frozen=True)
class ClassificationSummary:
    image_count: int
    labeled_count: int
    class_counts: dict[str, int]

    @property
    def unlabeled_count(self) -> int:
        return self.image_count - self.labeled_count


def _normalize_key(key: Any) -> str | None:
    if key is None:
        return None
    text = str(key).strip().lower()
    if not text:
        return None
    return text


def _normalize_image_rel_path(image_rel_path: str | Path) -> str:
    return str(image_rel_path).replace("\\", "/").strip().lstrip("./")


def find_default_classification_config_path(dataset_root: Path) -> Path | None:
    resolved_root = dataset_root.expanduser()
    for candidate_name in CLASSIFICATION_CONFIG_FILE_CANDIDATES:
        candidate = resolved_root / candidate_name
        if candidate.is_file():
            return candidate
    return None


def _load_config_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Classification config does not exist: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Classification config is not a file: {path}")
    try:
        if path.suffix.lower() == ".json":
            payload = read_json(path)
        else:
            payload = read_yaml(path)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Unable to parse classification config {path}: {exc}") from exc
    return {str(key): value for key, value in payload.items()}


def _raw_class_entries(payload: Mapping[str, Any], *, path: Path) -> list[Any]:
    raw = payload.get("classes")
    if raw is None:
        raw = payload.get("names")
    if raw is None:
        raise ConfigurationError(
            f"Classification config {path} must define a 'classes' list (or a YOLO style 'names' list)."
        )
    if isinstance(raw, Mapping):
        # YOLO data.yaml style ``names: {0: car, 1: truck}`` keeps numeric order.
        def _sort_key(item: tuple[Any, Any]) -> tuple[int, Any]:
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        return [value for _, value in sorted(raw.items(), key=_sort_key)]
    if isinstance(raw, list):
        return list(raw)
    raise ConfigurationError(f"Classification config {path}: 'classes' must be a list.")


def parse_classification_config(payload: Mapping[str, Any], *, source_path: Path | None = None) -> ClassificationConfig:
    display_path = source_path if source_path is not None else Path("<inline>")
    raw_entries = _raw_class_entries(payload, path=display_path)
    if not raw_entries:
        raise ConfigurationError(f"Classification config {display_path}: 'classes' must not be empty.")

    parsed: list[ClassificationClass] = []
    explicit_keys: set[str] = set()
    pending_auto: list[int] = []
    for index, entry in enumerate(raw_entries):
        if isinstance(entry, Mapping):
            name = str(entry.get("name", "")).strip()
            key = _normalize_key(entry.get("key"))
        else:
            name = str(entry).strip()
            key = None
        if not name:
            raise ConfigurationError(
                f"Classification config {display_path}: class entry {index + 1} has an empty name."
            )
        if key is not None:
            if len(key) != 1:
                raise ConfigurationError(
                    f"Classification config {display_path}: key for class '{name}' must be a single character, got {key!r}."
                )
            if key in explicit_keys:
                raise ConfigurationError(
                    f"Classification config {display_path}: key {key!r} is assigned to more than one class."
                )
            explicit_keys.add(key)
        else:
            pending_auto.append(index)
        parsed.append(ClassificationClass(key=key or "", name=name))

    auto_keys = iter(candidate for candidate in AUTO_CLASS_KEYS if candidate not in explicit_keys)
    for index in pending_auto:
        try:
            auto_key = next(auto_keys)
        except StopIteration as exc:
            raise ConfigurationError(
                f"Classification config {display_path}: too many classes to assign single-key shortcuts automatically; set explicit keys."
            ) from exc
        parsed[index] = ClassificationClass(key=auto_key, name=parsed[index].name)

    names = [entry.name for entry in parsed]
    duplicate_names = sorted({name for name, count in Counter(names).items() if count > 1})
    if duplicate_names:
        raise ConfigurationError(
            f"Classification config {display_path}: duplicate class names {duplicate_names}."
        )

    multi_label_raw = payload.get("multi_label", False)
    if not isinstance(multi_label_raw, bool):
        raise ConfigurationError(f"Classification config {display_path}: 'multi_label' must be true or false.")

    labels_file_raw = payload.get("labels_file", DEFAULT_CLASSIFICATION_LABELS_FILE)
    labels_file = str(labels_file_raw).strip() if labels_file_raw is not None else ""
    if not labels_file:
        labels_file = DEFAULT_CLASSIFICATION_LABELS_FILE

    return ClassificationConfig(
        classes=tuple(parsed),
        multi_label=multi_label_raw,
        labels_file=labels_file,
        source_path=source_path,
    )


def load_classification_config(path: Path) -> ClassificationConfig:
    resolved = path.expanduser()
    payload = _load_config_payload(resolved)
    return parse_classification_config(payload, source_path=resolved.resolve())


def discover_classification_image_paths(
    dataset_root: Path,
    *,
    input_path_include_substring: str | None = None,
    input_path_exclude_substring: str | None = None,
) -> list[str]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {resolved_root}")

    input_path_filter = build_input_path_filter(
        include_substring=input_path_include_substring,
        exclude_substring=input_path_exclude_substring,
    )
    rel_paths: list[str] = []
    for file_path in iter_files(resolved_root, suffixes=CLASSIFICATION_IMAGE_EXTENSIONS):
        rel_path = file_path.relative_to(resolved_root).as_posix()
        if not relative_path_matches_input_filter(rel_path, input_path_filter=input_path_filter):
            continue
        rel_paths.append(rel_path)
    return sorted(rel_paths)


def resolve_classification_labels_path(dataset_root: Path, config: ClassificationConfig) -> Path:
    labels_file = Path(config.labels_file).expanduser()
    if labels_file.is_absolute():
        return labels_file
    return safe_resolve(dataset_root.expanduser().resolve(), labels_file)


def load_classification_labels(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    raw_labels: Any = payload.get("labels", payload)
    if not isinstance(raw_labels, Mapping):
        raise ConfigurationError(f"Classification labels file {path} must contain a 'labels' mapping.")

    labels: dict[str, list[str]] = {}
    for image_rel_path, value in raw_labels.items():
        normalized_path = _normalize_image_rel_path(str(image_rel_path))
        if not normalized_path:
            continue
        if value is None:
            names: list[str] = []
        elif isinstance(value, str):
            names = [value]
        elif isinstance(value, list):
            names = [str(item) for item in value if str(item).strip()]
        else:
            raise ConfigurationError(
                f"Classification labels file {path}: label for {image_rel_path!r} must be a string or list."
            )
        labels[normalized_path] = names
    return labels


def save_classification_labels(
    path: Path,
    labels: Mapping[str, list[str]],
    *,
    config: ClassificationConfig,
) -> Path:
    payload = {
        "version": CLASSIFICATION_LABELS_MANIFEST_VERSION,
        "classes": config.class_names,
        "multi_label": config.multi_label,
        "labels": {key: list(value) for key, value in sorted(labels.items()) if value},
    }
    atomic_write_json(path, payload)
    return path


def apply_classification_label(
    labels: Mapping[str, list[str]],
    image_rel_path: str,
    class_name: str,
    *,
    multi_label: bool,
) -> dict[str, list[str]]:
    """Return a new labels mapping with ``class_name`` applied to ``image_rel_path``.

    Single-label mode replaces the image's label. Multi-label mode toggles the class
    in the image's label list.
    """

    updated = {key: list(value) for key, value in labels.items()}
    normalized_path = _normalize_image_rel_path(image_rel_path)
    current = updated.get(normalized_path, [])
    if multi_label:
        if class_name in current:
            current = [name for name in current if name != class_name]
        else:
            current = [*current, class_name]
    else:
        current = [class_name]
    if current:
        updated[normalized_path] = current
    else:
        updated.pop(normalized_path, None)
    return updated


def clear_classification_label(
    labels: Mapping[str, list[str]],
    image_rel_path: str,
) -> dict[str, list[str]]:
    updated = {key: list(value) for key, value in labels.items()}
    updated.pop(_normalize_image_rel_path(image_rel_path), None)
    return updated


def image_labels(labels: Mapping[str, list[str]], image_rel_path: str) -> list[str]:
    return list(labels.get(_normalize_image_rel_path(image_rel_path), []))


def next_unlabeled_index(
    image_rel_paths: list[str],
    labels: Mapping[str, list[str]],
    *,
    start_index: int,
) -> int | None:
    """Index of the first unlabeled image after ``start_index``, wrapping around."""

    count = len(image_rel_paths)
    if count == 0:
        return None
    for offset in range(1, count + 1):
        candidate = (start_index + offset) % count
        if not labels.get(_normalize_image_rel_path(image_rel_paths[candidate])):
            return candidate
    return None


def summarize_classification_labels(
    image_rel_paths: list[str],
    labels: Mapping[str, list[str]],
    *,
    config: ClassificationConfig,
) -> ClassificationSummary:
    class_counts: Counter[str] = Counter({name: 0 for name in config.class_names})
    labeled_count = 0
    for rel_path in image_rel_paths:
        names = labels.get(_normalize_image_rel_path(rel_path), [])
        if not names:
            continue
        labeled_count += 1
        class_counts.update(names)
    return ClassificationSummary(
        image_count=len(image_rel_paths),
        labeled_count=labeled_count,
        class_counts=dict(class_counts),
    )


def classification_labels_to_csv_rows(labels: Mapping[str, list[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_rel_path, names in sorted(labels.items()):
        if not names:
            continue
        rows.append({"image": image_rel_path, "classes": ";".join(names)})
    return rows

