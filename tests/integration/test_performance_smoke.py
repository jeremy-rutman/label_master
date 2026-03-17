from __future__ import annotations

import time
from pathlib import Path

from label_master.core.domain.entities import SourceFormat
from label_master.core.services.convert_service import ConvertRequest, execute_conversion
from label_master.core.services.infer_service import infer_format


def test_performance_smoke_infer_and_convert(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_path = Path("tests/fixtures/us1/coco_minimal")
    output_path = tmp_path / "out"

    t0 = time.perf_counter()
    inference = infer_format(input_path, force=True)
    t1 = time.perf_counter()

    result = execute_conversion(
        ConvertRequest(
            run_id="perf-smoke",
            input_path=input_path,
            output_path=output_path,
            src_format=SourceFormat.COCO,
            dst_format=SourceFormat.YOLO,
            dry_run=False,
        )
    )
    t2 = time.perf_counter()

    assert inference.predicted_format in {SourceFormat.COCO, SourceFormat.AMBIGUOUS}
    assert result.report.summary_counts.annotations_out > 0
    assert (t1 - t0) < 5.0
    assert (t2 - t1) < 5.0
