from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBoxXYWH:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("BBox width and height must be positive")

    def to_normalized(self, image_width: int, image_height: int) -> "BBoxCXCYWHNormalized":
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        cx = (self.x + self.w / 2.0) / image_width
        cy = (self.y + self.h / 2.0) / image_height
        nw = self.w / image_width
        nh = self.h / image_height
        return BBoxCXCYWHNormalized(cx=cx, cy=cy, w=nw, h=nh)


@dataclass(frozen=True)
class BBoxCXCYWHNormalized:
    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name, value in (("cx", self.cx), ("cy", self.cy), ("w", self.w), ("h", self.h)):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("normalized width and height must be positive")

    def to_absolute(self, image_width: int, image_height: int) -> BBoxXYWH:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        w_abs = self.w * image_width
        h_abs = self.h * image_height
        x_abs = self.cx * image_width - w_abs / 2.0
        y_abs = self.cy * image_height - h_abs / 2.0
        return BBoxXYWH(x=x_abs, y=y_abs, w=w_abs, h=h_abs)


@dataclass(frozen=True)
class RunIdentifier:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("run identifier cannot be empty")


class LabelMasterError(Exception):
    def __init__(self, message: str, *, code: str = "label_master_error", context: dict[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class ConfigurationError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="configuration_error", context=context)


class InferenceError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="inference_error", context=context)


class ValidationError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="validation_error", context=context)


class ConversionError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="conversion_error", context=context)


class ImportError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="import_error", context=context)


class LockError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="lock_error", context=context)


class PathTraversalError(LabelMasterError):
    def __init__(self, message: str, *, context: dict[str, str] | None = None):
        super().__init__(message, code="path_traversal_error", context=context)


JSONDict = dict[str, Any]
PathLike = str | Path
