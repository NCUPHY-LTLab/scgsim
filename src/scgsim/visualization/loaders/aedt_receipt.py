"""Validate AEDT receipt assignments against read-only native observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry.models.refusals import Declared
from scgsim.refusal import found_text
from scgsim.visualization.loaders.common import AEDT_RUN_FOLDER

_WHERE_AEDT_RUN = "the AEDT run folder: fetch it again, or run the AEDT handoff again"


def mapping_record(value: Any, name: str) -> Mapping[str, Any]:
    """Require one decoded JSON object and name it in type errors."""
    if not isinstance(value, Mapping):
        raise TypeError(
            Declared(f"{name}", AEDT_RUN_FOLDER).refusal("must be a mapping")
        )
    return value


def nonempty_text(value: Any, name: str) -> str:
    """Require non-empty text and return its whitespace-trimmed value."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            Declared(f"{name}", AEDT_RUN_FOLDER).refusal("must be non-empty text")
        )
    return value.strip()


def material_records(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Use body-first object records; imported models retain their material rows."""
    geometry = receipt.get("geometry")
    if isinstance(geometry, Mapping):
        return tuple(geometry.get("objects", ()))
    return tuple(receipt.get("materials", ()))


def validate_native_receipt(
    app: Any, receipt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Compare native assignments with receipt authority before rendering it."""
    mode = receipt.get("mode")
    object_ids = {name: int(app.modeler[name].id) for name in app.modeler.object_names}
    native_boundaries: dict[str, dict[str, Any]] = {}
    if mode in {"terminal", "modal", "eigenmode"}:
        native_boundaries = {
            str(boundary.name): dict(boundary.props) for boundary in app.boundaries
        }
        _validate_hfss_receipt(app, receipt, mode, object_ids, native_boundaries)
    elif mode == "q3d":
        _validate_q3d_receipt(app, receipt, object_ids)
    elif mode == "q2d":
        _validate_q2d_receipt(app, receipt, object_ids)
    else:
        raise ValueError(
            Declared("unsupported AEDT receipt mode", AEDT_RUN_FOLDER).refusal(
                f"{mode!r}"
            )
        )
    return native_boundaries


def _validate_hfss_receipt(
    app: Any,
    receipt: Mapping[str, Any],
    mode: str,
    object_ids: Mapping[str, int],
    boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate HFSS PEC bodies and port assignments in receipt order."""
    _validate_hfss_pec_materials(receipt, object_ids, boundaries)
    _validate_hfss_geometry(app, receipt, object_ids, boundaries)
    _validate_hfss_ports(receipt, boundaries)
    if mode == "eigenmode" and receipt.get("ports"):
        raise ValueError(
            Declared(
                "AEDT Eigenmode receipt must not contain ports", AEDT_RUN_FOLDER
            ).refusal(found_text(receipt.get("ports")))
        )


def _validate_hfss_pec_materials(
    receipt: Mapping[str, Any],
    object_ids: Mapping[str, int],
    boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Check PEC boundary object assignments recorded for HFSS materials."""
    pec_records = {
        (
            str(item.get("observed", {}).get("native_pec_boundary")),
            tuple(item.get("observed", {}).get("native_pec_objects", ())),
        )
        for item in material_records(receipt)
        if item.get("observed", {}).get("native_pec_boundary")
    }
    for boundary_name, object_names in pec_records:
        props = boundaries.get(boundary_name)
        if props is None or props.get("BoundType") != "Perfect E":
            raise ValueError(
                Declared(
                    "native AEDT PEC boundary disagrees with the receipt",
                    AEDT_RUN_FOLDER,
                ).refusal(
                    found_text(
                        None if props is None else props.get("BoundType"), "Perfect E"
                    )
                )
            )
        expected_ids = sorted(object_ids[name] for name in object_names)
        if sorted(int(value) for value in props.get("Objects", ())) != expected_ids:
            raise ValueError(
                Declared(
                    "native AEDT PEC object assignment changed", AEDT_RUN_FOLDER
                ).refusal(
                    found_text(
                        sorted(int(value) for value in props.get("Objects", ())),
                        expected_ids,
                    )
                )
            )


def _validate_hfss_geometry(
    app: Any,
    receipt: Mapping[str, Any],
    object_ids: Mapping[str, int],
    boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Check HFSS native body identity, sheet PEC, and Solid PEC state."""
    geometry = receipt.get("geometry")
    if not isinstance(geometry, Mapping):
        return
    for item in geometry.get("objects", ()):
        name = nonempty_text(item.get("object_name"), "native object name")
        if object_ids.get(name) != item.get("native_object_id"):
            raise ValueError(
                Declared("native AEDT body identity changed", AEDT_RUN_FOLDER).refusal(
                    found_text(object_ids.get(name), item.get("native_object_id"))
                )
            )
        _validate_sheet_pec(item, boundaries)
        _validate_solid_pec(app, item, name)


def _validate_sheet_pec(
    item: Mapping[str, Any], boundaries: Mapping[str, Mapping[str, Any]]
) -> None:
    """Compare assigned sheet faces or body IDs with the native PEC boundary."""
    boundary_name = item.get("boundary_name")
    if not boundary_name:
        return
    props = boundaries.get(boundary_name)
    if props is None or props.get("BoundType") != "Perfect E":
        raise ValueError(
            Declared("native Sheet PEC boundary changed", AEDT_RUN_FOLDER).refusal(
                found_text(
                    None if props is None else props.get("BoundType"), "Perfect E"
                )
            )
        )
    faces = {int(value) for value in props.get("Faces", ())}
    objects = {int(value) for value in props.get("Objects", ())}
    if faces != set(item["native_face_ids"]) and objects != {item["native_object_id"]}:
        raise ValueError(
            Declared("native Sheet PEC assignment changed", AEDT_RUN_FOLDER).refusal(
                found_text(
                    {"faces": sorted(faces), "objects": sorted(objects)},
                    {
                        "faces": sorted(item["native_face_ids"]),
                        "objects": [item["native_object_id"]],
                    },
                )
            )
        )


def _validate_solid_pec(app: Any, item: Mapping[str, Any], name: str) -> None:
    """Check native material and Solve Inside state when receipt marks Solid PEC."""
    observed = item.get("observed", {})
    if observed.get("native_solve_inside") is False:
        obj = app.modeler[name]
        if str(obj.material_name).casefold() != "pec" or obj.solve_inside is not False:
            raise ValueError(
                Declared("native Solid PEC material changed", AEDT_RUN_FOLDER).refusal(
                    found_text(
                        {
                            "material": str(obj.material_name),
                            "solve_inside": obj.solve_inside,
                        },
                        {"material": "pec", "solve_inside": False},
                    )
                )
            )


def _validate_hfss_ports(
    receipt: Mapping[str, Any], boundaries: Mapping[str, Mapping[str, Any]]
) -> None:
    """Check HFSS port type and native face assignments."""
    for port in receipt.get("ports", ()):
        boundary_name = nonempty_text(port.get("boundary"), "receipt port boundary")
        props = boundaries.get(boundary_name)
        if props is None or props.get("BoundType") not in {"Wave Port", "Lumped Port"}:
            raise ValueError(
                Declared(
                    "native AEDT port boundary disagrees with the receipt",
                    AEDT_RUN_FOLDER,
                ).refusal(
                    found_text(
                        None if props is None else props.get("BoundType"),
                        "Wave Port or Lumped Port",
                    )
                )
            )
        if int(port["face_id"]) not in {int(value) for value in props.get("Faces", ())}:
            raise ValueError(
                Declared(
                    "native AEDT port face assignment changed", AEDT_RUN_FOLDER
                ).refusal(
                    found_text(
                        sorted(int(value) for value in props.get("Faces", ())),
                        int(port["face_id"]),
                    )
                )
            )


def _validate_q3d_receipt(
    app: Any, receipt: Mapping[str, Any], object_ids: Mapping[str, int]
) -> None:
    """Check Q3D net order, object sets, and source/sink face assignments."""
    if list(app.net_names) != [
        str(item.get("name")) for item in receipt.get("nets", ())
    ]:
        raise ValueError(
            Declared("native AEDT Q3D net order changed", AEDT_RUN_FOLDER).refusal(
                found_text(
                    list(app.net_names),
                    [str(item.get("name")) for item in receipt.get("nets", ())],
                )
            )
        )
    for net in receipt.get("nets", ()):
        names = tuple(net.get("object_names", ()))
        expected = sorted(object_ids[name] for name in names)
        native = sorted(
            int(value)
            for value in app.oboundary.GetExcitationAssignment(str(net.get("name")))
        )
        if native != expected or native != sorted(
            int(value) for value in net.get("native_object_ids", ())
        ):
            raise ValueError(
                Declared(
                    "native AEDT Q3D net object assignment changed", AEDT_RUN_FOLDER
                ).refusal(found_text(native, expected))
            )
        _validate_q3d_terminal(app, net, "source")
        _validate_q3d_terminal(app, net, "sink")


def _validate_q3d_terminal(app: Any, net: Mapping[str, Any], role: str) -> None:
    """Check one optional Q3D terminal face assignment."""
    record = net.get(role)
    if record is None:
        return
    record = mapping_record(record, f"receipt Q3D {role}")
    native_faces = sorted(
        int(value)
        for value in app.oboundary.GetExcitationAssignment(
            nonempty_text(record.get("name"), f"receipt Q3D {role} name")
        )
    )
    if native_faces != sorted(
        int(value) for value in record.get("native_face_ids", ())
    ):
        raise ValueError(
            Declared(f"native AEDT Q3D {role} assignment", _WHERE_AEDT_RUN).refusal(
                "changed"
            )
        )


def _validate_q2d_receipt(
    app: Any, receipt: Mapping[str, Any], object_ids: Mapping[str, int]
) -> None:
    """Check Q2D conductor object assignments against the receipt."""
    for conductor in receipt.get("conductors", ()):
        names = tuple(conductor.get("object_names", ()))
        expected = sorted(object_ids[name] for name in names)
        native = sorted(
            int(value)
            for value in app.oboundary.GetExcitationAssignment(
                str(conductor.get("name"))
            )
        )
        if native != expected or native != sorted(
            int(value) for value in conductor.get("native_object_ids", ())
        ):
            raise ValueError(
                Declared(
                    "native AEDT Q2D conductor assignment changed", AEDT_RUN_FOLDER
                ).refusal(found_text(native, expected))
            )
