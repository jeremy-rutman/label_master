from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import yaml

from label_master.core.domain.value_objects import PathTraversalError


@dataclass(frozen=True)
class InputPathFilter:
    include_substring: str | None = None
    exclude_substring: str | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.include_substring or self.exclude_substring)


def normalize_input_path_filter_substring(raw_value: str | None) -> str | None:
    normalized = (raw_value or "").strip()
    return normalized or None


def build_input_path_filter(
    *,
    include_substring: str | None = None,
    exclude_substring: str | None = None,
) -> InputPathFilter | None:
    normalized_include = normalize_input_path_filter_substring(include_substring)
    normalized_exclude = normalize_input_path_filter_substring(exclude_substring)
    if not normalized_include and not normalized_exclude:
        return None
    return InputPathFilter(
        include_substring=normalized_include,
        exclude_substring=normalized_exclude,
    )


def relative_path_matches_input_filter(
    candidate: str | Path,
    *,
    input_path_filter: InputPathFilter | None,
) -> bool:
    if input_path_filter is None or not input_path_filter.is_active:
        return True

    candidate_text = str(candidate).replace("\\", "/").strip().lower()
    include_substring = (
        input_path_filter.include_substring.lower()
        if input_path_filter.include_substring
        else None
    )
    exclude_substring = (
        input_path_filter.exclude_substring.lower()
        if input_path_filter.exclude_substring
        else None
    )

    if include_substring and include_substring not in candidate_text:
        return False
    if exclude_substring and exclude_substring in candidate_text:
        return False
    return True


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_resolve(base_dir: Path, candidate: str | Path) -> Path:
    base = base_dir.resolve()
    cand_path = Path(candidate)
    resolved = (base / cand_path).resolve() if not cand_path.is_absolute() else cand_path.resolve()
    if not is_relative_to(resolved, base):
        raise PathTraversalError(
            "Path escapes allowed base directory",
            context={"base": str(base), "candidate": str(candidate)},
        )
    return resolved


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("uri must use file:// scheme")
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        raise ValueError("file:// URI netloc must be empty or localhost")
    return Path(unquote(parsed.path)).resolve()


def safe_file_uri_to_path(uri: str, allowed_root: Path) -> Path:
    resolved = file_uri_to_path(uri)
    root = allowed_root.resolve()
    if not is_relative_to(resolved, root):
        raise PathTraversalError(
            "file:// URI points outside allowed root",
            context={"allowed_root": str(root), "candidate": str(resolved)},
        )
    return resolved


def iter_files(root: Path, suffixes: Iterable[str] | None = None) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    if suffixes is None:
        return sorted(files)
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(p for p in files if p.suffix.lower() in suffix_set)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def atomic_write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
