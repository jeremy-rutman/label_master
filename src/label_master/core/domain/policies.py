from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_OUT_OF_FRAME_TOLERANCE_PX = 20.0
DEFAULT_OUT_OF_FRAME_BBOX_POLICY = "correct"
DEFAULT_MIN_IMAGE_LONGEST_EDGE_PX = 0
DEFAULT_MAX_IMAGE_LONGEST_EDGE_PX = 0


class OutOfFrameBBoxPolicy(str, Enum):
    CORRECT = "correct"
    WARN = "warn"
    IGNORE = "ignore"
    DROP = "drop"


class UnmappedPolicy(str, Enum):
    ERROR = "error"
    DROP = "drop"
    IDENTITY = "identity"


class ValidationMode(str, Enum):
    STRICT = "strict"
    PERMISSIVE = "permissive"


class InvalidAnnotationAction(str, Enum):
    KEEP = "keep"
    DROP = "drop"


class OversizeImageAction(str, Enum):
    IGNORE = "ignore"
    DOWNSCALE = "downscale"


class InferencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_limit: int = Field(default=500, ge=1)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.15, ge=0.0, le=1.0)

    def is_ambiguous(self, top_score: float, second_score: float) -> bool:
        return abs(top_score - second_score) <= self.ambiguity_margin


class ValidationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ValidationMode = ValidationMode.STRICT
    max_invalid_annotations: int = Field(default=0, ge=0)
    invalid_annotation_action: InvalidAnnotationAction = InvalidAnnotationAction.KEEP
    out_of_frame_bbox_policy: OutOfFrameBBoxPolicy = OutOfFrameBBoxPolicy.CORRECT
    out_of_frame_tolerance_px: float = Field(default=DEFAULT_OUT_OF_FRAME_TOLERANCE_PX, ge=0.0)

    @classmethod
    def for_mode(
        cls,
        mode: ValidationMode,
        *,
        invalid_annotation_action: InvalidAnnotationAction = InvalidAnnotationAction.KEEP,
        out_of_frame_bbox_policy: OutOfFrameBBoxPolicy = OutOfFrameBBoxPolicy.CORRECT,
        out_of_frame_tolerance_px: float = DEFAULT_OUT_OF_FRAME_TOLERANCE_PX,
    ) -> "ValidationPolicy":
        if mode == ValidationMode.PERMISSIVE:
            return cls(
                mode=mode,
                max_invalid_annotations=10_000,
                invalid_annotation_action=invalid_annotation_action,
                out_of_frame_bbox_policy=out_of_frame_bbox_policy,
                out_of_frame_tolerance_px=out_of_frame_tolerance_px,
            )
        return cls(
            mode=mode,
            max_invalid_annotations=0,
            invalid_annotation_action=invalid_annotation_action,
            out_of_frame_bbox_policy=out_of_frame_bbox_policy,
            out_of_frame_tolerance_px=out_of_frame_tolerance_px,
        )


class RemapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unmapped_policy: UnmappedPolicy = UnmappedPolicy.ERROR

    def resolve_destination(self, source_class_id: int, class_map: dict[int, int | None]) -> int | None:
        if source_class_id in class_map:
            return class_map[source_class_id]

        if self.unmapped_policy == UnmappedPolicy.IDENTITY:
            return source_class_id
        if self.unmapped_policy == UnmappedPolicy.DROP:
            return None

        raise ValueError(f"unmapped class id encountered: {source_class_id}")
