from __future__ import annotations

from pathlib import Path


def detect_yolo(path: Path) -> float:
    txt_files = [file for file in path.rglob("*.txt") if file.is_file()]
    if not txt_files:
        return 0.0

    yolo_like_rows = 0
    for txt_file in txt_files[:50]:
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if not row:
                continue
            tokens = row.split()
            if len(tokens) == 5:
                try:
                    int(tokens[0])
                    [float(token) for token in tokens[1:]]
                except ValueError:
                    break
                yolo_like_rows += 1
            break

    if yolo_like_rows == 0:
        return 0.0

    score = 0.7
    has_label_named_directory = any("label" in parent.name.lower() for file in txt_files for parent in file.parents)
    if has_label_named_directory:
        score += 0.2

    has_images_dir = any(directory.is_dir() and "image" in directory.name.lower() for directory in path.rglob("*"))
    if has_images_dir:
        score += 0.1

    return min(score, 1.0)
