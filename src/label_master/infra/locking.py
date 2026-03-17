from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from label_master.core.domain.entities import ContentionEvent
from label_master.infra.filesystem import ensure_directory


class OutputPathLockManager:
    """Simple filesystem-backed lock metadata manager with last-write-wins semantics."""

    def __init__(self, lock_root: Path | None = None) -> None:
        self.lock_root = ensure_directory((lock_root or Path("/tmp/label_master_locks")).resolve())

    def _lock_file_path(self, output_path: Path) -> Path:
        normalized = str(output_path.resolve())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.lock_root / f"{digest}.json"

    def _read_metadata(self, lock_file: Path) -> dict[str, str]:
        if not lock_file.exists():
            return {}
        with lock_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def _write_metadata(self, lock_file: Path, metadata: dict[str, str]) -> None:
        with lock_file.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def acquire(self, output_path: Path, run_id: str) -> list[ContentionEvent]:
        lock_file = self._lock_file_path(output_path)
        existing = self._read_metadata(lock_file)
        previous_owner = existing.get("owner_run_id")

        now = datetime.now(UTC).isoformat()
        metadata = {
            "output_path": str(output_path.resolve()),
            "owner_run_id": run_id,
            "acquired_at": now,
            "previous_owner_run_id": previous_owner or "",
        }
        self._write_metadata(lock_file, metadata)

        if previous_owner and previous_owner != run_id:
            return [
                ContentionEvent(
                    output_path=str(output_path.resolve()),
                    run_id=run_id,
                    competing_run_id=previous_owner,
                )
            ]
        return []

    def mark_completed(self, output_path: Path, run_id: str) -> None:
        lock_file = self._lock_file_path(output_path)
        metadata = self._read_metadata(lock_file)
        metadata["last_completed_run_id"] = run_id
        metadata["completed_at"] = datetime.now(UTC).isoformat()
        self._write_metadata(lock_file, metadata)

    def get_owner(self, output_path: Path) -> str | None:
        lock_file = self._lock_file_path(output_path)
        metadata = self._read_metadata(lock_file)
        owner = metadata.get("owner_run_id")
        return owner if owner else None

    def read_metadata(self, output_path: Path) -> dict[str, str]:
        lock_file = self._lock_file_path(output_path)
        return self._read_metadata(lock_file)
