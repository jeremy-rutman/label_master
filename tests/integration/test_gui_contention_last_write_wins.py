from __future__ import annotations

from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.domain.policies import UnmappedPolicy
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.infra.config import load_mapping_file
from label_master.infra.locking import OutputPathLockManager


def test_contention_last_write_wins(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_path = Path("tests/fixtures/us1/coco_minimal")
    output_path = tmp_path / "shared_output"
    lock_manager = OutputPathLockManager(lock_root=tmp_path / "locks")

    result_a = execute_conversion(
        ConvertRequest(
            run_id="run-a",
            input_path=input_path,
            output_path=output_path,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            class_map=load_mapping_file(Path("tests/fixtures/us2/map_a.yaml")),
            unmapped_policy=UnmappedPolicy.DROP,
            dry_run=False,
        ),
        lock_manager=lock_manager,
    )
    assert len(result_a.report.contention_events) == 0

    result_b = execute_conversion(
        ConvertRequest(
            run_id="run-b",
            input_path=input_path,
            output_path=output_path,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            class_map=load_mapping_file(Path("tests/fixtures/us2/map_b.yaml")),
            unmapped_policy=UnmappedPolicy.DROP,
            dry_run=False,
        ),
        lock_manager=lock_manager,
    )

    assert len(result_b.report.contention_events) == 1
    label_files = sorted((output_path / "labels").glob("*.txt"))
    assert label_files

    class_ids = {
        int(line.split()[0])
        for label_file in label_files
        for line in label_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert class_ids
    assert class_ids.issubset({30, 31})
    assert 30 in class_ids
