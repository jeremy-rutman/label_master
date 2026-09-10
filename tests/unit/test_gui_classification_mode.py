from __future__ import annotations

from pathlib import Path

from label_master.interfaces.gui.app import (
    _coerce_classification_keyboard_action,
    classification_index_after_label,
    load_persisted_gui_preferences,
    resolve_classification_config_candidate,
)


def test_coerce_classification_keyboard_action_handles_nonce_and_actions() -> None:
    assert _coerce_classification_keyboard_action(None, last_nonce=3) == (None, None, 3)
    assert _coerce_classification_keyboard_action({"action": "class", "key": "1"}, last_nonce=3) == (None, None, 3)
    assert _coerce_classification_keyboard_action(
        {"action": "class", "key": "1", "nonce": 3}, last_nonce=3
    ) == (None, None, 3)
    assert _coerce_classification_keyboard_action(
        {"action": "class", "key": "A", "nonce": 4}, last_nonce=3
    ) == ("class", "a", 4)
    assert _coerce_classification_keyboard_action({"action": "class", "nonce": 5}, last_nonce=4) == (None, None, 5)
    assert _coerce_classification_keyboard_action({"action": "clear", "nonce": 6}, last_nonce=5) == ("clear", None, 6)
    assert _coerce_classification_keyboard_action({"action": "next", "nonce": 7}, last_nonce=6) == ("next", None, 7)
    assert _coerce_classification_keyboard_action({"action": "bogus", "nonce": 8}, last_nonce=7) == (None, None, 8)


def test_classification_index_after_label_advances_only_in_single_label_mode() -> None:
    assert classification_index_after_label(0, max_index=3, advance_on_label=True, multi_label=False) == 1
    assert classification_index_after_label(3, max_index=3, advance_on_label=True, multi_label=False) == 3
    assert classification_index_after_label(0, max_index=3, advance_on_label=False, multi_label=False) == 0
    assert classification_index_after_label(0, max_index=3, advance_on_label=True, multi_label=True) == 0


def test_resolve_classification_config_candidate(tmp_path: Path) -> None:
    assert resolve_classification_config_candidate("", dataset_root=None) is None
    assert resolve_classification_config_candidate("", dataset_root=tmp_path) is None
    explicit = tmp_path / "custom.yaml"
    assert resolve_classification_config_candidate(f" {explicit} ", dataset_root=tmp_path) == explicit
    default = tmp_path / "classification.yaml"
    default.write_text("classes: [a]\n", encoding="utf-8")
    assert resolve_classification_config_candidate("", dataset_root=tmp_path) == default


def test_persisted_preferences_include_classification_defaults(tmp_path: Path) -> None:
    state_path = tmp_path / "gui_state.json"
    preferences = load_persisted_gui_preferences(state_path)
    assert preferences["gui_classification_config_path"] == ""
    assert preferences["gui_classification_advance_on_label"] is True
    assert preferences["gui_classification_show_bboxes"] is True

    state_path.write_text(
        '{"last_classification_config_path": "/cfg.yaml", '
        '"last_classification_advance_on_label": false, '
        '"last_classification_show_bboxes": "nope"}',
        encoding="utf-8",
    )
    preferences = load_persisted_gui_preferences(state_path)
    assert preferences["gui_classification_config_path"] == "/cfg.yaml"
    assert preferences["gui_classification_advance_on_label"] is False
    assert preferences["gui_classification_show_bboxes"] is True
