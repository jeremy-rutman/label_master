from __future__ import annotations

from label_master.interfaces.gui.app import build_run_summary_metrics, transition_run_state


def test_run_status_transitions_idle_running_completed_failed() -> None:
    status, progress = transition_run_state("idle", "start")
    assert status == "running"
    assert progress == 15

    status, progress = transition_run_state(status, "complete")
    assert status == "completed"
    assert progress == 100

    status, progress = transition_run_state("idle", "start")
    assert status == "running"
    status, progress = transition_run_state(status, "fail")
    assert status == "failed"
    assert progress == 100


def test_run_summary_metrics_render_from_report_totals() -> None:
    report = {
        "summary_counts": {
            "images": 12,
            "annotations_out": 34,
        },
        "warnings": [
            {"severity": "warning", "message": "one"},
            {"severity": "error", "message": "two"},
            {"severity": "error", "message": "three"},
        ],
    }

    metrics = build_run_summary_metrics(report)
    assert metrics.images_processed == 12
    assert metrics.annotations_converted == 34
    assert metrics.warning_count == 1
    assert metrics.error_count == 2
