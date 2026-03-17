from __future__ import annotations

from dataclasses import dataclass

from label_master.core.domain.entities import AnnotationDataset, AnnotationRecord, CategoryRecord
from label_master.core.domain.policies import RemapPolicy
from label_master.core.domain.value_objects import ValidationError


@dataclass(frozen=True)
class RemapResult:
    dataset: AnnotationDataset
    dropped: int
    unmapped: int
    dropped_class_ids: list[int]
    unmapped_class_ids: list[int]


def apply_class_remap(
    dataset: AnnotationDataset,
    class_map: dict[int, int | None],
    *,
    policy: RemapPolicy,
) -> RemapResult:
    remapped_annotations: list[AnnotationRecord] = []
    remapped_categories: dict[int, CategoryRecord] = {}
    dropped = 0
    unmapped = 0
    dropped_class_ids: set[int] = set()
    unmapped_class_ids: set[int] = set()

    for annotation in dataset.deterministic_annotations():
        try:
            destination = policy.resolve_destination(annotation.class_id, class_map)
        except ValueError as exc:
            unmapped += 1
            unmapped_class_ids.add(annotation.class_id)
            raise ValidationError(str(exc)) from exc

        if destination is None:
            dropped += 1
            dropped_class_ids.add(annotation.class_id)
            continue

        source_category = dataset.categories.get(annotation.class_id)
        source_name = source_category.name if source_category else f"class_{destination}"
        remapped_categories[destination] = CategoryRecord(class_id=destination, name=source_name)
        remapped_annotations.append(
            annotation.model_copy(update={"class_id": destination})
        )

    remapped_dataset = dataset.model_copy(
        update={
            "annotations": remapped_annotations,
            "categories": remapped_categories,
        }
    )

    return RemapResult(
        dataset=remapped_dataset,
        dropped=dropped,
        unmapped=unmapped,
        dropped_class_ids=sorted(dropped_class_ids),
        unmapped_class_ids=sorted(unmapped_class_ids),
    )
