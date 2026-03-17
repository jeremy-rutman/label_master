from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx

from label_master.adapters.providers.common import ProviderFetchResult
from label_master.core.domain.value_objects import ImportError
from label_master.infra.filesystem import safe_file_uri_to_path


def fetch_direct_url(
    source_ref: str,
    output_path: Path,
    *,
    allowed_file_root: Path,
    timeout_seconds: float = 30.0,
) -> ProviderFetchResult:
    output_path.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(source_ref)
    scheme = parsed.scheme.lower()
    warnings: list[str] = []

    if scheme not in {"https", "http", "file"}:
        raise ImportError(f"Unsupported direct_url protocol: {scheme or 'none'}")

    if scheme in {"http", "file"}:
        warnings.append(f"direct_url protocol '{scheme}' is allowed but considered unsafe")

    if scheme == "file":
        source_path = safe_file_uri_to_path(source_ref, allowed_file_root)
        target = output_path / source_path.name
        if source_path.is_dir():
            import shutil

            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
        else:
            import shutil

            shutil.copy2(source_path, target)
        return ProviderFetchResult(local_path=target, protocol=scheme, warnings=warnings)

    filename = Path(parsed.path).name or "downloaded_artifact.bin"
    target = output_path / filename

    try:
        with httpx.stream("GET", source_ref, timeout=timeout_seconds, follow_redirects=True) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        raise ImportError(f"Failed to fetch URL: {source_ref}") from exc

    return ProviderFetchResult(local_path=target, protocol=scheme, warnings=warnings)
