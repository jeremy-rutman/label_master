from __future__ import annotations

from label_master.interfaces.gui.app import is_localhost_binding


def test_gui_binding_localhost_only() -> None:
    assert is_localhost_binding("127.0.0.1")
    assert is_localhost_binding("localhost")
    assert is_localhost_binding("::1")
    assert not is_localhost_binding("0.0.0.0")
    assert not is_localhost_binding("192.168.1.8")
