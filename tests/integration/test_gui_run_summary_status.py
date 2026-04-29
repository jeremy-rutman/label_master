from __future__ import annotations

from label_master.interfaces.gui.app import (
    build_run_summary_metrics,
    is_streamlit_control_flow_exception,
    reset_gui_run_state,
    transition_run_state,
)


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


def test_run_status_transition_reset_returns_idle() -> None:
    status, progress = transition_run_state("running", "reset")
    assert status == "idle"
    assert progress == 0


def test_reset_gui_run_state_clears_error_payload_and_marks_interrupt() -> None:
    session_state = {
        "gui_run_status": "running",
        "gui_run_progress": 72,
        "gui_run_error": "boom",
        "gui_run_error_details": ["detail"],
        "gui_run_error_issue_rows": [{"issue": "x"}],
        "gui_run_detail": "Processing...",
        "gui_run_interrupted_notice": False,
    }

    status, progress = reset_gui_run_state(
        session_state,
        detail="Conversion interrupted before completion. You can run conversion again.",
        interrupted=True,
    )

    assert status == "idle"
    assert progress == 0
    assert session_state["gui_run_status"] == "idle"
    assert session_state["gui_run_progress"] == 0
    assert session_state["gui_run_error"] is None
    assert session_state["gui_run_error_details"] == []
    assert session_state["gui_run_error_issue_rows"] == []
    assert (
        session_state["gui_run_detail"]
        == "Conversion interrupted before completion. You can run conversion again."
    )
    assert session_state["gui_run_interrupted_notice"] is True


def test_streamlit_control_flow_exception_detection_uses_class_name() -> None:
    class StopException(Exception):
        pass

    class SomethingElse(Exception):
        pass

    assert is_streamlit_control_flow_exception(StopException("stop"))
    assert is_streamlit_control_flow_exception(type("RerunException", (Exception,), {})("rerun"))
    assert is_streamlit_control_flow_exception(type("ScriptRunnerStopException", (Exception,), {})("nope")) is False
    assert is_streamlit_control_flow_exception(SomethingElse("boom")) is False


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
