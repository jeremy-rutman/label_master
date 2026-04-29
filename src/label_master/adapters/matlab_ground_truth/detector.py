from __future__ import annotations

from pathlib import Path

from label_master.adapters.matlab_ground_truth.common import (
    find_matlab_ground_truth_files,
    load_matlab_ground_truth_payload,
    resolve_matlab_ground_truth_video_path,
)


def detect_matlab_ground_truth(path: Path, *, sample_limit: int = 500) -> float:
    max_files = max(1, min(sample_limit, 8))
    mat_files = find_matlab_ground_truth_files(path, max_files=max_files)
    if not mat_files:
        return 0.0

    matched_payloads = 0
    matched_videos = 0
    for mat_path in mat_files:
        try:
            payload = load_matlab_ground_truth_payload(mat_path)
        except Exception:
            continue
        matched_payloads += 1
        if resolve_matlab_ground_truth_video_path(
            path,
            source_video_path=payload.source_video_path,
            annotation_path=mat_path,
        ) is not None:
            matched_videos += 1

    if matched_payloads == 0 or matched_videos == 0:
        return 0.0

    score = 0.72
    if matched_payloads == len(mat_files):
        score += 0.13
    if matched_videos == matched_payloads:
        score += 0.1
    if len(mat_files) > 1:
        score += 0.05
    return min(score, 1.0)
