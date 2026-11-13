"""The materials scene of an AEDT preview: each native object coloured by the source material its receipt records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from scgsim.refusal import Declared, found_text
from scgsim.visualization.loaders.common import AEDT_RUN_FOLDER

from ..scene import ScenePart, color_for_identity
from .aedt_receipt import (
    mapping_record,
    material_records,
    nonempty_text,
)


def object_materials(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    source_conductors: tuple[Mapping[str, Any], ...],
) -> list[ScenePart]:
    """One scene part per native object, coloured by the source material the receipt records for it."""
    recorded = _ReceiptMaterials.read(receipt)
    parts: list[ScenePart] = []
    for object_name, dataset in sorted(datasets.items()):
        native_material = str(app.modeler[object_name].material_name).lower()
        material = recorded.object_material(
            object_name, native_material, source_conductors
        )
        parts.append(
            ScenePart(
                dataset,
                f"material:{material.material_id}",
                material.material_id,
                material.kind,
                color_for_identity(material.material_id),
                0.08 if material.kind == "vacuum" else 0.42,
                1,
            )
        )
    return parts


class _ObjectMaterial(NamedTuple):
    """The material a native object is drawn with: its source material id and that material's kind."""

    material_id: str
    kind: str


@dataclass(frozen=True)
class _ReceiptMaterials:
    """What a receipt records about materials: each object's binding, the PDK materials and the Region."""

    receipt: dict[str, Any] | None
    items: dict[str, Mapping[str, Any]]
    pdk_materials: Mapping[str, Any]
    region: Mapping[str, Any]

    @classmethod
    def read(cls, receipt: dict[str, Any] | None) -> _ReceiptMaterials:
        """Read the material records of a receipt; a raw project has none."""
        if receipt is None:
            return cls(None, {}, {}, {})
        items = {
            nonempty_text(
                item.get("object_name"), "receipt.materials.object_name"
            ): item
            for item in material_records(receipt)
        }
        pdk_materials = mapping_record(
            receipt.get("pdk_materials", {}), "receipt.pdk_materials"
        )
        region = mapping_record(receipt.get("region", {}), "receipt.region")
        return cls(receipt, items, pdk_materials, region)

    def object_material(
        self,
        object_name: str,
        native_material: str,
        source_conductors: tuple[Mapping[str, Any], ...],
    ) -> _ObjectMaterial:
        """The material of one object, checked against the native material AEDT reports for it."""
        item = self.items.get(object_name)
        if item is not None:
            if item.get("kind") == "conductor" and "physical_layer" in item:
                material_id = _conductor_material_id(item, source_conductors)
            else:
                material_id = nonempty_text(
                    item.get("material_id"), "receipt material_id"
                )
            kind = self._kind(material_id)
            observed = mapping_record(
                item.get("observed", {}), "receipt material observed"
            )
            observed_name = observed.get("native_material_name")
            if (
                observed_name is not None
                and str(observed_name).lower() != native_material
            ):
                raise ValueError(
                    Declared(
                        f"native AEDT material changed for {object_name}",
                        AEDT_RUN_FOLDER,
                    ).refusal(found_text(native_material, str(observed_name).lower()))
                )
            return _ObjectMaterial(material_id, kind)
        if object_name == "Region" and self.region:
            material_id = nonempty_text(
                self.region.get("material_id"), "receipt region material_id"
            )
            kind = self._kind(material_id)
            if (
                str(self.region.get("observed_material_name", "")).lower()
                != native_material
            ):
                raise ValueError(
                    Declared(
                        "native AEDT Region material changed", AEDT_RUN_FOLDER
                    ).refusal(
                        found_text(
                            self.region.get("observed_material_name"), native_material
                        )
                    )
                )
            return _ObjectMaterial(material_id, kind)
        if self.receipt is not None:
            raise ValueError(
                Declared(
                    f"AEDT receipt has no material identity for {object_name}",
                    AEDT_RUN_FOLDER,
                ).refusal(found_text(sorted(self.items)))
            )
        return _ObjectMaterial(native_material, "native AEDT material")

    def _kind(self, material_id: str) -> str:
        material = mapping_record(
            self.pdk_materials.get(material_id), "receipt PDK material"
        )
        return nonempty_text(material.get("kind"), "receipt material kind")


def _conductor_material_id(
    item: Mapping[str, Any], source_conductors: tuple[Mapping[str, Any], ...]
) -> str:
    """A conductor's source material: native PEC treatment is separate from the source material identity."""
    matches = [
        conductor
        for conductor in source_conductors
        if conductor["semantic_id"] == item["semantic_id"]
        and item["source_polygon_id"] in conductor["polygon_ids"]
        and conductor["physical_layer"] == item["physical_layer"]
    ]
    if len(matches) != 1:
        raise ValueError(
            Declared(
                "native conductor has no unique source material binding",
                AEDT_RUN_FOLDER,
            ).refusal(
                found_text(
                    {"semantic_id": item["semantic_id"], "matches": len(matches)}
                )
            )
        )
    return nonempty_text(matches[0].get("material_id"), "source conductor material_id")
