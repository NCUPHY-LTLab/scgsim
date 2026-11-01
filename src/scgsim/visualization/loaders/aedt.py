"""Load AEDT-bound geometry and receipt facts into preview scenes.

Owns: opening one AEDT project headlessly, checking native readback against the run
receipt, and turning materials and boundaries into preview parts.
Does not own: rendering (``rendering.py``), the preview manifest (``preview.py``) or
any solve.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from scgsim.aedt.specs.common import LOCKED_PYAEDT, REQUIRED_AEDT_VERSION

from ..preview import GeometryPreview
from ..scene import FIXED_COLORS, Mode, Part, identity_color
from ._common import (
    confined,
    read_json_object,
    receipt_authority_sha256,
    sha256,
)

# Receipt modes that carry PEC (perfect electric conductor) boundaries and ports.
# The remaining supported modes are "q3d" and "q2d".
_HFSS_MODES = frozenset({"terminal", "modal", "eigenmode"})
_NATIVE_PORT_TYPES = frozenset({"Wave Port", "Lumped Port"})


@dataclass(frozen=True)
class _AedtSource:
    """Where the project lives and what the run folder says about it."""

    root: Path
    project_path: Path
    design: str
    receipt: dict[str, Any] | None
    receipt_path: Path | None
    source_conductors: tuple[Mapping[str, Any], ...]


def inspect_aedt_geometry(
    source: str | Path,
    *,
    design: str | None = None,
) -> GeometryPreview:
    """Open one AEDT project headlessly and bind native geometry readback."""
    resolved = _resolve_source(source, design)
    pv, desktop_class = _import_aedt_runtime()
    materials, boundaries = _read_native_scene(resolved, pv, desktop_class)
    return GeometryPreview(
        root=resolved.root,
        backend="aedt",
        source_hashes=_source_hashes(resolved),
        source_identity=_source_identity(resolved),
        bind_aedt_receipt=resolved.receipt is not None,
        modes={
            "materials": Mode(tuple(materials)),
            "boundaries": Mode(tuple(boundaries)),
            "surface_epr": Mode(
                unavailable_reason="AEDT V1 has no structured Surface-EPR assignment authority."
            ),
            "mesh": Mode(
                unavailable_reason="AEDT V1 has no readable native mesh artifact."
            ),
        },
    )


def _resolve_source(source: str | Path, design: str | None) -> _AedtSource:
    """Accept either a raw ``.aedt`` file or a run folder with spec and receipt."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        resolved = _resolve_project_file(source_path, design)
    elif source_path.is_dir():
        resolved = _resolve_run_folder(source_path, design)
    else:
        raise FileNotFoundError(f"AEDT geometry source does not exist: {source_path}")
    if not resolved.project_path.is_file():
        raise FileNotFoundError(
            f"AEDT project is missing: {resolved.project_path.name}"
        )
    return resolved


def _resolve_project_file(source_path: Path, design: str | None) -> _AedtSource:
    if source_path.suffix.lower() != ".aedt":
        raise ValueError("raw AEDT geometry source must be an .aedt project")
    if not design or not design.strip():
        raise ValueError("raw .aedt preview requires an explicit design")
    return _AedtSource(
        root=source_path.parent,
        project_path=source_path,
        design=design,
        receipt=None,
        receipt_path=None,
        source_conductors=(),
    )


def _resolve_run_folder(root: Path, design: str | None) -> _AedtSource:
    spec_path = root / "aedt_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError("AEDT run folder has no aedt_spec.json")
    spec = read_json_object(spec_path)
    project = _mapping_checked(spec.get("project"), "spec.project")
    recorded_design = _stripped_text_or_type_error(
        project.get("design"), "spec.project.design"
    )
    if design is not None and design != recorded_design:
        raise ValueError("explicit design disagrees with the AEDT run spec")
    project_name = _stripped_text_or_type_error(
        project.get("name"), "spec.project.name"
    )
    project_path = confined(root, f"{project_name}.aedt")
    receipt_path = root / "metadata/aedt_run_receipt.json"
    receipt = read_json_object(receipt_path) if receipt_path.is_file() else None
    source_conductors: tuple[Mapping[str, Any], ...] = ()
    if receipt is not None:
        source_conductors = _check_receipt_against_spec(
            root, spec, spec_path, receipt, project_path
        )
    return _AedtSource(
        root=root,
        project_path=project_path,
        design=recorded_design,
        receipt=receipt,
        receipt_path=receipt_path,
        source_conductors=source_conductors,
    )


def _check_receipt_against_spec(
    root: Path,
    spec: Mapping[str, Any],
    spec_path: Path,
    receipt: Mapping[str, Any],
    project_path: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Confirm the receipt, spec and project file describe the same run.

    Returns the source conductors recorded in the spec, which later bind native
    conductors back to their source material.
    """
    source_conductors: tuple[Mapping[str, Any], ...] = ()
    if receipt.get("status") == "native_preparation_only":
        from scgsim.aedt.results.prepared import resolve_prepared_model

        resolve_prepared_model(root)
    geometry = spec.get("geometry")
    if isinstance(geometry, Mapping):
        source_geometry = _mapping_checked(
            geometry.get("source"), "spec.geometry.source"
        )
        source_conductors = tuple(source_geometry.get("conductors", ()))
        if receipt.get("status") != "native_preparation_only":
            recorded_source = _mapping_checked(receipt.get("source"), "receipt.source")
            if recorded_source.get("spec_sha256") != sha256(spec_path):
                raise ValueError("AEDT source spec hash disagrees with the receipt")
    if receipt.get("project") != project_path.relative_to(root).as_posix():
        raise ValueError("AEDT receipt project identity disagrees with the spec")
    outputs = _mapping_checked(receipt.get("outputs", {}), "receipt.outputs")
    relative = project_path.relative_to(root).as_posix()
    expected = outputs.get(relative)
    if not isinstance(expected, str) or sha256(project_path) != expected:
        raise ValueError("AEDT project hash disagrees with the run receipt")
    return source_conductors


def _import_aedt_runtime() -> tuple[Any, Any]:
    """Import the optional PyVista and PyAEDT dependencies, pinned to the locked PyAEDT."""
    try:
        import pyvista as pv
        from ansys.aedt.core import Desktop
    except ImportError as exc:
        raise RuntimeError(
            "AEDT geometry preview requires scgsim[aedt,visualization]"
        ) from exc
    if version("pyaedt") != LOCKED_PYAEDT:
        raise RuntimeError(
            f"AEDT preview requires PyAEDT {LOCKED_PYAEDT}, got {version('pyaedt')}"
        )
    return pv, Desktop


def _read_native_scene(
    source: _AedtSource, pv: Any, desktop_class: Any
) -> tuple[list[Part], list[Part]]:
    """Open the design, read materials and boundaries, and always release AEDT."""
    desktop = None
    try:
        desktop = desktop_class(
            version=REQUIRED_AEDT_VERSION,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        if desktop.aedt_version_id != REQUIRED_AEDT_VERSION:
            raise RuntimeError("AEDT Desktop release does not match the locked runtime")
        app = _open_design(desktop, source)
        datasets = _export_object_meshes(app, pv)

        native_boundaries: Mapping[str, Mapping[str, Any]] = {}
        if source.receipt is not None:
            native_boundaries = _validate_native_receipt(app, source.receipt)
        materials = _materials(app, datasets, source.receipt, source.source_conductors)
        boundaries = _boundaries(app, datasets, source.receipt, pv, native_boundaries)
        return materials, boundaries
    finally:
        if desktop is not None:
            released = desktop.release_desktop(close_projects=True, close_on_exit=True)
            if released is False:
                raise RuntimeError("AEDT Desktop did not confirm release after preview")


def _open_design(desktop: Any, source: _AedtSource) -> Any:
    app = desktop.load_project(str(source.project_path), design_name=source.design)
    if app is False or app is None:
        raise RuntimeError("PyAEDT could not open the requested project/design")
    if str(app.design_name) != source.design:
        raise ValueError("native AEDT design readback disagrees with the request")
    return app


def _export_object_meshes(app: Any, pv: Any) -> dict[str, Any]:
    """Export every model object as OBJ and read each back as a PyVista dataset."""
    app.modeler.refresh_all_ids()
    object_names = tuple(str(name) for name in app.modeler.object_names)
    if not object_names:
        raise ValueError("AEDT design contains no model objects")
    with tempfile.TemporaryDirectory(prefix="scgsim-aedt-preview-") as raw:
        exported = app.post.export_model_obj(
            assignment=list(object_names),
            export_path=raw,
            export_as_multiple_objects=True,
            air_objects=True,
        )
        datasets: dict[str, Any] = {}
        for item in exported:
            path = Path(item[0])
            name = path.stem
            if name in datasets:
                raise ValueError(f"AEDT exported duplicate object {name}")
            datasets[name] = pv.read(path).copy(deep=True)
    missing = sorted(set(object_names) - set(datasets))
    if missing:
        raise ValueError(f"AEDT OBJ export omitted native objects: {missing!r}")
    return datasets


def _source_hashes(source: _AedtSource) -> dict[str, str]:
    hashes = {
        source.project_path.relative_to(source.root).as_posix(): sha256(
            source.project_path
        )
    }
    spec_path = source.root / "aedt_spec.json"
    if spec_path.is_file():
        hashes["aedt_spec.json"] = sha256(spec_path)
    return hashes


def _source_identity(source: _AedtSource) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "design": source.design,
        "aedt_version": REQUIRED_AEDT_VERSION,
        "pyaedt_version": LOCKED_PYAEDT,
    }
    if source.receipt_path is not None and source.receipt is not None:
        identity["receipt_authority_sha256"] = receipt_authority_sha256(
            source.receipt_path
        )
    return identity


def _validate_native_receipt(
    app: Any, receipt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Check native AEDT readback against the receipt; return native boundary props."""
    mode = receipt.get("mode")
    object_ids = {name: int(app.modeler[name].id) for name in app.modeler.object_names}
    if mode in _HFSS_MODES:
        return _check_hfss_receipt(app, receipt, object_ids)
    if mode == "q3d":
        _check_q3d_nets(app, receipt, object_ids)
    elif mode == "q2d":
        _check_q2d_conductors(app, receipt, object_ids)
    else:
        raise ValueError(f"unsupported AEDT receipt mode: {mode!r}")
    return {}


def _check_hfss_receipt(
    app: Any, receipt: Mapping[str, Any], object_ids: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    native_boundaries = {
        str(boundary.name): dict(boundary.props) for boundary in app.boundaries
    }
    _check_pec_boundaries(receipt, native_boundaries, object_ids)
    _check_geometry_objects(app, receipt, native_boundaries, object_ids)
    _check_ports(receipt, native_boundaries)
    if receipt.get("mode") == "eigenmode" and receipt.get("ports"):
        raise ValueError("AEDT Eigenmode receipt must not contain ports")
    return native_boundaries


def _check_pec_boundaries(
    receipt: Mapping[str, Any],
    native_boundaries: Mapping[str, Mapping[str, Any]],
    object_ids: Mapping[str, int],
) -> None:
    pec_records = {
        (
            str(item.get("observed", {}).get("native_pec_boundary")),
            tuple(item.get("observed", {}).get("native_pec_objects", ())),
        )
        for item in _material_records(receipt)
        if item.get("observed", {}).get("native_pec_boundary")
    }
    for boundary_name, object_names in pec_records:
        props = native_boundaries.get(boundary_name)
        if props is None or props.get("BoundType") != "Perfect E":
            raise ValueError("native AEDT PEC boundary disagrees with the receipt")
        expected_ids = sorted(object_ids[name] for name in object_names)
        if sorted(int(value) for value in props.get("Objects", ())) != expected_ids:
            raise ValueError("native AEDT PEC object assignment changed")


def _check_geometry_objects(
    app: Any,
    receipt: Mapping[str, Any],
    native_boundaries: Mapping[str, Mapping[str, Any]],
    object_ids: Mapping[str, int],
) -> None:
    geometry = receipt.get("geometry")
    if not isinstance(geometry, Mapping):
        return
    for item in geometry.get("objects", ()):
        name = _stripped_text_or_type_error(
            item.get("object_name"), "native object name"
        )
        if object_ids.get(name) != item.get("native_object_id"):
            raise ValueError("native AEDT body identity changed")
        boundary_name = item.get("boundary_name")
        if boundary_name:
            _check_sheet_pec(item, native_boundaries.get(boundary_name))
        if item.get("observed", {}).get("native_solve_inside") is False:
            obj = app.modeler[name]
            if (
                str(obj.material_name).casefold() != "pec"
                or obj.solve_inside is not False
            ):
                raise ValueError("native Solid PEC material changed")


def _check_sheet_pec(item: Mapping[str, Any], props: Mapping[str, Any] | None) -> None:
    if props is None or props.get("BoundType") != "Perfect E":
        raise ValueError("native Sheet PEC boundary changed")
    faces = {int(value) for value in props.get("Faces", ())}
    objects = {int(value) for value in props.get("Objects", ())}
    # The assignment passes when either the faces or the object match the receipt.
    if faces != set(item["native_face_ids"]) and objects != {item["native_object_id"]}:
        raise ValueError("native Sheet PEC assignment changed")


def _check_ports(
    receipt: Mapping[str, Any],
    native_boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    for port in receipt.get("ports", ()):
        boundary_name = _stripped_text_or_type_error(
            port.get("boundary"), "receipt port boundary"
        )
        props = native_boundaries.get(boundary_name)
        if props is None or props.get("BoundType") not in _NATIVE_PORT_TYPES:
            raise ValueError("native AEDT port boundary disagrees with the receipt")
        if int(port["face_id"]) not in {int(value) for value in props.get("Faces", ())}:
            raise ValueError("native AEDT port face assignment changed")


def _excitation_assignment(app: Any, name: str) -> list[int]:
    """Sorted native ids that AEDT assigns to one named excitation."""
    return sorted(int(value) for value in app.oboundary.GetExcitationAssignment(name))


def _check_q3d_nets(
    app: Any, receipt: Mapping[str, Any], object_ids: Mapping[str, int]
) -> None:
    if list(app.net_names) != [
        str(item.get("name")) for item in receipt.get("nets", ())
    ]:
        raise ValueError("native AEDT Q3D net order changed")
    for net in receipt.get("nets", ()):
        names = tuple(net.get("object_names", ()))
        expected = sorted(object_ids[name] for name in names)
        native = _excitation_assignment(app, str(net.get("name")))
        if native != expected or native != sorted(
            int(value) for value in net.get("native_object_ids", ())
        ):
            raise ValueError("native AEDT Q3D net object assignment changed")
        for role in ("source", "sink"):
            record = net.get(role)
            if record is None:
                continue
            record = _mapping_checked(record, f"receipt Q3D {role}")
            native_faces = _excitation_assignment(
                app,
                _stripped_text_or_type_error(
                    record.get("name"), f"receipt Q3D {role} name"
                ),
            )
            if native_faces != sorted(
                int(value) for value in record.get("native_face_ids", ())
            ):
                raise ValueError(f"native AEDT Q3D {role} assignment changed")


def _check_q2d_conductors(
    app: Any, receipt: Mapping[str, Any], object_ids: Mapping[str, int]
) -> None:
    for conductor in receipt.get("conductors", ()):
        names = tuple(conductor.get("object_names", ()))
        expected = sorted(object_ids[name] for name in names)
        native = _excitation_assignment(app, str(conductor.get("name")))
        if native != expected or native != sorted(
            int(value) for value in conductor.get("native_object_ids", ())
        ):
            raise ValueError("native AEDT Q2D conductor assignment changed")


def _material_records(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Use body-first object records; imported models retain their material rows."""
    geometry = receipt.get("geometry")
    if isinstance(geometry, Mapping):
        return tuple(geometry.get("objects", ()))
    return tuple(receipt.get("materials", ()))


def _materials(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    source_conductors: tuple[Mapping[str, Any], ...],
) -> list[Part]:
    receipt_items: dict[str, Mapping[str, Any]] = {}
    pdk_materials: Mapping[str, Any] = {}
    region: Mapping[str, Any] = {}
    if receipt is not None:
        receipt_items = {
            _stripped_text_or_type_error(
                item.get("object_name"), "receipt.materials.object_name"
            ): item
            for item in _material_records(receipt)
        }
        pdk_materials = _mapping_checked(
            receipt.get("pdk_materials", {}), "receipt.pdk_materials"
        )
        region = _mapping_checked(receipt.get("region", {}), "receipt.region")
    parts: list[Part] = []
    for object_name, dataset in sorted(datasets.items()):
        native_material = str(app.modeler[object_name].material_name).lower()
        item = receipt_items.get(object_name)
        if item is not None:
            material_id, kind = _receipt_object_material(
                item, object_name, native_material, pdk_materials, source_conductors
            )
        elif object_name == "Region" and region:
            material_id = _stripped_text_or_type_error(
                region.get("material_id"), "receipt region material_id"
            )
            kind = _pdk_material_kind(pdk_materials, material_id)
            if str(region.get("observed_material_name", "")).lower() != native_material:
                raise ValueError("native AEDT Region material changed")
        elif receipt is not None:
            raise ValueError(f"AEDT receipt has no material identity for {object_name}")
        else:
            material_id = native_material
            kind = "native AEDT material"
        parts.append(
            Part(
                dataset,
                f"material:{material_id}",
                material_id,
                kind,
                identity_color(material_id),
                0.08 if kind == "vacuum" else 0.42,
                1,
            )
        )
    return parts


def _pdk_material_kind(pdk_materials: Mapping[str, Any], material_id: str) -> str:
    material = _mapping_checked(pdk_materials.get(material_id), "receipt PDK material")
    return _stripped_text_or_type_error(material.get("kind"), "receipt material kind")


def _receipt_object_material(
    item: Mapping[str, Any],
    object_name: str,
    native_material: str,
    pdk_materials: Mapping[str, Any],
    source_conductors: tuple[Mapping[str, Any], ...],
) -> tuple[str, str]:
    """Return (material id, material kind) for one object the receipt describes."""
    if item.get("kind") == "conductor" and "physical_layer" in item:
        # Native PEC treatment is separate from the source material identity.
        matches = [
            conductor
            for conductor in source_conductors
            if conductor["semantic_id"] == item["semantic_id"]
            and item["source_polygon_id"] in conductor["polygon_ids"]
            and conductor["physical_layer"] == item["physical_layer"]
        ]
        if len(matches) != 1:
            raise ValueError("native conductor has no unique source material binding")
        material_id = _stripped_text_or_type_error(
            matches[0].get("material_id"), "source conductor material_id"
        )
    else:
        material_id = _stripped_text_or_type_error(
            item.get("material_id"), "receipt material_id"
        )
    kind = _pdk_material_kind(pdk_materials, material_id)
    observed = _mapping_checked(item.get("observed", {}), "receipt material observed")
    observed_name = observed.get("native_material_name")
    if observed_name is not None and str(observed_name).lower() != native_material:
        raise ValueError(f"native AEDT material changed for {object_name}")
    return material_id, kind


@dataclass
class _BoundaryScene:
    """Boundary parts collected so far, and which objects already have one."""

    datasets: Mapping[str, Any]
    parts: list[Part] = field(default_factory=list)
    assigned: set[str] = field(default_factory=set)

    def add_object(self, object_name: str, label: str, role: str) -> None:
        """Show a whole exported object as a boundary; its first assignment wins."""
        if object_name not in self.datasets:
            raise ValueError(f"AEDT boundary object was not exported: {object_name}")
        if object_name in self.assigned:
            return
        self.assigned.add(object_name)
        self.parts.append(
            Part(
                self.datasets[object_name],
                f"boundary:{label}:{object_name}",
                label,
                role,
                identity_color(role if role in FIXED_COLORS else label),
                0.95,
                1,
            )
        )


def _boundaries(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    pv: Any,
    native_boundaries: Mapping[str, Mapping[str, Any]],
) -> list[Part]:
    scene = _BoundaryScene(datasets)
    if receipt is not None:
        _add_receipt_boundaries(scene, app, receipt, pv, native_boundaries)
    else:
        _add_native_boundaries(scene, app, pv)
    _add_unassigned_objects(scene)
    return scene.parts


def _add_receipt_boundaries(
    scene: _BoundaryScene,
    app: Any,
    receipt: Mapping[str, Any],
    pv: Any,
    native_boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    mode = receipt.get("mode")
    if mode in _HFSS_MODES:
        _add_pec_objects(scene, receipt)
        _add_port_faces(scene, app, receipt, pv, native_boundaries)
    elif mode == "q3d":
        _add_q3d_nets(scene, receipt)
    elif mode == "q2d":
        _add_q2d_conductors(scene, receipt)
    else:
        raise ValueError(f"unsupported AEDT receipt mode: {mode!r}")


def _add_pec_objects(scene: _BoundaryScene, receipt: Mapping[str, Any]) -> None:
    for item in _material_records(receipt):
        observed = _mapping_checked(item.get("observed", {}), "receipt observed")
        native_objects = tuple(observed.get("native_pec_objects", ()))
        if native_objects:
            object_name = _stripped_text_or_type_error(
                item.get("object_name"), "receipt object_name"
            )
            if object_name not in native_objects:
                raise ValueError("AEDT PEC receipt object readback is inconsistent")
            scene.add_object(object_name, "PEC", "PEC")
        elif item.get("boundary_name") or observed.get("native_solve_inside") is False:
            object_name = _stripped_text_or_type_error(
                item.get("object_name"), "receipt object_name"
            )
            label = item.get("boundary_name") or "PEC material / SolveInside=false"
            scene.add_object(object_name, label, "PEC")


def _face_points(face: Any) -> list[tuple[float, ...]]:
    return [
        tuple(float(value) for value in vertex.position) for vertex in face.vertices
    ]


def _polygon(pv: Any, points: list[tuple[float, ...]]) -> Any:
    """One flat PyVista polygon through the points, in order."""
    return pv.PolyData(points, faces=[len(points), *range(len(points))])


def _add_port_faces(
    scene: _BoundaryScene,
    app: Any,
    receipt: Mapping[str, Any],
    pv: Any,
    native_boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    for port in receipt.get("ports", ()):
        face_id = int(port["face_id"])
        face = app.modeler.get_face_by_id(face_id)
        if face is None:
            raise ValueError(f"AEDT native port face {face_id} is unavailable")
        points = _face_points(face)
        if len(points) < 3:
            raise ValueError(f"AEDT native port face {face_id} is degenerate")
        dataset = _polygon(pv, points)
        label = _stripped_text_or_type_error(
            port.get("boundary"), "receipt port boundary"
        )
        boundary_type = str(native_boundaries[label]["BoundType"])
        scene.parts.append(
            Part(
                dataset,
                f"port:{label}",
                label,
                boundary_type,
                identity_color(boundary_type),
                1.0,
                1,
            )
        )


def _add_q3d_nets(scene: _BoundaryScene, receipt: Mapping[str, Any]) -> None:
    for net in receipt.get("nets", ()):
        role = _stripped_text_or_type_error(net.get("net_type"), "receipt net_type")
        name = _stripped_text_or_type_error(net.get("name"), "receipt net name")
        for object_name in net.get("object_names", ()):
            object_name = _stripped_text_or_type_error(
                object_name, "receipt net object"
            )
            scene.add_object(
                object_name, name, "Ground" if role == "Ground" else "Terminal"
            )


def _add_q2d_conductors(scene: _BoundaryScene, receipt: Mapping[str, Any]) -> None:
    for conductor in receipt.get("conductors", ()):
        role = _stripped_text_or_type_error(
            conductor.get("conductor_type"), "receipt conductor_type"
        )
        name = _stripped_text_or_type_error(
            conductor.get("name"), "receipt conductor name"
        )
        for object_name in conductor.get("object_names", ()):
            object_name = _stripped_text_or_type_error(
                object_name, "receipt conductor object"
            )
            scene.add_object(
                object_name, name, "Ground" if role == "ReferenceGround" else "Terminal"
            )


def _add_native_boundaries(scene: _BoundaryScene, app: Any, pv: Any) -> None:
    """Raw ``.aedt`` projects have no receipt, so read the boundaries from AEDT itself."""
    for boundary in app.boundaries:
        props = dict(boundary.props)
        boundary_type = str(props.get("BoundType") or getattr(boundary, "type", ""))
        name = str(getattr(boundary, "name", props.get("Name", boundary_type)))
        if boundary_type in {"Terminal"}:
            continue
        if boundary_type not in {"Perfect E", "Wave Port", "Lumped Port"}:
            raise ValueError(
                f"unsupported native AEDT boundary type: {boundary_type!r}"
            )
        role = "PEC" if boundary_type == "Perfect E" else boundary_type
        for object_name in _native_object_names(app, props.get("Objects", ())):
            scene.add_object(object_name, name, role)
        for face_id in props.get("Faces", ()):
            face = app.modeler.get_face_by_id(int(face_id))
            if face is None:
                raise ValueError(f"native AEDT boundary face {face_id} is unavailable")
            dataset = _polygon(pv, _face_points(face))
            scene.parts.append(
                Part(
                    dataset,
                    f"boundary:{name}",
                    name,
                    role,
                    identity_color(role),
                    1.0,
                    1,
                )
            )


def _add_unassigned_objects(scene: _BoundaryScene) -> None:
    """Show objects with no boundary in grey so a missing assignment is visible."""
    for object_name, dataset in sorted(scene.datasets.items()):
        if object_name not in scene.assigned:
            scene.parts.append(
                Part(
                    dataset,
                    f"unassigned:{object_name}",
                    f"{object_name}: no explicit boundary assignment",
                    "unassigned",
                    FIXED_COLORS["unassigned"],
                    0.35,
                    1,
                )
            )


def _native_object_names(app: Any, values: Any) -> tuple[str, ...]:
    by_id = {int(app.modeler[name].id): str(name) for name in app.modeler.object_names}
    names: list[str] = []
    for value in values:
        if str(value) in app.modeler.object_names:
            names.append(str(value))
            continue
        try:
            names.append(by_id[int(value)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"native AEDT boundary object is unavailable: {value!r}"
            ) from exc
    return tuple(names)


def _mapping_checked(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _stripped_text_or_type_error(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be non-empty text")
    return value.strip()
