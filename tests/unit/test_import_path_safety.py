from __future__ import annotations

import pytest

from label_master.core.domain.value_objects import PathTraversalError
from label_master.infra.filesystem import safe_file_uri_to_path


def test_file_uri_path_safety_enforced(tmp_path) -> None:  # type: ignore[no-untyped-def]
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    good = allowed / "safe.zip"
    good.write_text("data", encoding="utf-8")
    resolved = safe_file_uri_to_path(f"file://{good}", allowed)
    assert resolved == good.resolve()

    bad = tmp_path / "outside.zip"
    bad.write_text("data", encoding="utf-8")

    with pytest.raises(PathTraversalError):
        safe_file_uri_to_path(f"file://{bad}", allowed)
