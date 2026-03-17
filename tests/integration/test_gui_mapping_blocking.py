from __future__ import annotations

from label_master.interfaces.gui.app import run_blocking_errors
from label_master.interfaces.gui.viewmodels import MappingRowViewModel, parse_mapping_rows


def test_invalid_mapping_rows_disable_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_mapping_rows(
        [
            MappingRowViewModel(source_class_id="3", action="map", destination_class_id=""),
            MappingRowViewModel(source_class_id="3", action="drop", destination_class_id=""),
            MappingRowViewModel(source_class_id="x", action="drop", destination_class_id=""),
        ]
    )

    errors = run_blocking_errors(
        input_dir_raw="tests/fixtures/us1/coco_minimal",
        output_dir_raw=str(tmp_path / "out"),
        src="coco",
        dst="yolo",
        mapping_errors=parsed.errors,
    )

    assert parsed.errors
    assert any(error.startswith("Row ") for error in parsed.errors)
    assert any(error.startswith("Row ") for error in errors)
