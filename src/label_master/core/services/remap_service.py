from __future__ import annotations

from dataclasses import dataclass

from label_master.core.domain.entities import AnnotationDataset, AnnotationRecord, CategoryRecord
from label_master.core.domain.policies import RemapPolicy
from label_master.core.domain.value_objects import ValidationError
from label_master.reports.schemas import DroppedAnnotationModel


@dataclass(frozen=True)
class RemapResult:
    dataset: AnnotationDataset
    dropped: int
    unmapped: int
    dropped_class_ids: list[int]
    unmapped_class_ids: list[int]
    dropped_annotations: list[DroppedAnnotationModel]


def apply_class_remap(
    dataset: AnnotationDataset,
    class_map: dict[int, int | None],
    *,
    policy: RemapPolicy,
) -> RemapResult:
    remapped_annotations: list[AnnotationRecord] = []
    remapped_categories: dict[int, CategoryRecord] = {}
    images_by_id = {image.image_id: image for image in dataset.images}
    dropped = 0
    unmapped = 0
    dropped_class_ids: set[int] = set()
    unmapped_class_ids: set[int] = set()
    dropped_annotations: list[DroppedAnnotationModel] = []

    for annotation in dataset.deterministic_annotations():
        source_category = dataset.categories.get(annotation.class_id)
        try:
            destination = policy.resolve_destination(annotation.class_id, class_map)
        except ValueError as exc:
            unmapped += 1
            unmapped_class_ids.add(annotation.class_id)
            raise ValidationError(str(exc)) from exc

        if destination is None:
            dropped += 1
            dropped_class_ids.add(annotation.class_id)
            explicit_mapping_drop = annotation.class_id in class_map and class_map[annotation.class_id] is None
            dropped_annotations.append(
                DroppedAnnotationModel(
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    image_file=(
                        images_by_id[annotation.image_id].file_name
                        if annotation.image_id in images_by_id
                        else None
                    ),
                    class_id=annotation.class_id,
                    class_name=source_category.name if source_category else None,
                    bbox_xywh_abs=annotation.bbox_xywh_abs,
                    stage="remap",
                    reason_code=(
                        "class_dropped_by_mapping"
                        if explicit_mapping_drop
                        else "class_dropped_by_unmapped_policy"
                    ),
                    reason=(
                        "Dropped by class mapping."
                        if explicit_mapping_drop
                        else "Dropped by unmapped-policy=drop."
                    ),
                    context={"unmapped_policy": policy.unmapped_policy.value},
                )
            )
            continue

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
        dropped_annotations=dropped_annotations,
    )
