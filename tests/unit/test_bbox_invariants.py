from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from label_master.core.domain.value_objects import BBoxXYWH


@given(
    image_width=st.integers(min_value=32, max_value=4096),
    image_height=st.integers(min_value=32, max_value=4096),
    x=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_bbox_absolute_normalized_roundtrip(image_width: int, image_height: int, x: float, y: float, w: float, h: float) -> None:
    x = min(x, image_width - 1)
    y = min(y, image_height - 1)
    w = min(w, max(1.0, image_width - x))
    h = min(h, max(1.0, image_height - y))

    bbox = BBoxXYWH(x=x, y=y, w=w, h=h)
    normalized = bbox.to_normalized(image_width, image_height)
    roundtrip = normalized.to_absolute(image_width, image_height)

    assert abs(roundtrip.x - bbox.x) <= 1e-6
    assert abs(roundtrip.y - bbox.y) <= 1e-6
    assert abs(roundtrip.w - bbox.w) <= 1e-6
    assert abs(roundtrip.h - bbox.h) <= 1e-6
