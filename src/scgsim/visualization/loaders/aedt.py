"""Load AEDT-bound geometry and receipt facts into preview scenes."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.results.prepared import resolve_prepared_model
from scgsim.aedt.runtime.native.compatibility import native_compatibility
from scgsim.aedt.specs.common import LOCKED_PYAEDT
from scgsim.aedt.versions import AedtVersionProfile, resolve_aedt_version
from scgsim.refusal import refusal

from ..preview import GeometryPreview
from ..scene import FIXED_COLORS, SceneMode, ScenePart, color_for_identity
from .aedt_receipt import (
    mapping_record,
    material_records,
    nonempty_text,
    validate_native_receipt,
)
from .common import (
    confined_artifact_path,
    file_sha256,
    read_json_record,
    receipt_authority_sha256,
)


def inspect_aedt_geometry(
    source: str | Path,
    *,
    design: str | None = None,
    aedt_version: str | None = None,
) -> GeometryPreview:
    """Read an AEDT project into a ``GeometryPreview``: take its identity, inspect it in AEDT, release Desktop, then build the scenes."""
    source_request = _read_source(source, design=design, aedt_version=aedt_version)
    native_scene = _inspect_native_project(source_request)
    return _build_scene(source_request, native_scene)


@dataclass(frozen=True)
class _AedtSourceRequest:
    """Resolved project, version, receipt, and source-side geometry evidence."""

    root: Path
    project_path: Path
    design: str | None
    profile: AedtVersionProfile
    receipt: dict[str, Any] | None
    receipt_path: Path | None
    source_conductors: tuple[Mapping[str, Any], ...]
    receipt_connected_version: str | None


@dataclass(frozen=True)
class _NativeSceneData:
    """Detached native exports and semantic scene parts after Desktop release."""

    connected_version: str | None
    materials: tuple[ScenePart, ...]
    boundaries: tuple[ScenePart, ...]


def _read_source(
    source: str | Path,
    *,
    design: str | None,
    aedt_version: str | None,
) -> _AedtSourceRequest:
    """Resolve raw projects or run folders without opening AEDT."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        request = _read_raw_project(source_path, design, aedt_version)
    elif source_path.is_dir():
        request = _read_run_folder(source_path, design, aedt_version)
    else:
        raise FileNotFoundError(f"AEDT geometry source does not exist: {source_path}")
    if not request.project_path.is_file():
        raise FileNotFoundError(f"AEDT project is missing: {request.project_path.name}")
    return request


def _read_raw_project(
    project_path: Path, design: str | None, aedt_version: str | None
) -> _AedtSourceRequest:
    """Validate explicit identity for one raw project file."""
    if project_path.suffix.lower() != ".aedt":
        raise ValueError(
            refusal(
                "raw AEDT geometry source must be an .aedt project",
                found=str(project_path),
            )
        )
    if aedt_version is None:
        raise TypeError(
            refusal(
                "raw .aedt preview requires an explicit aedt_version",
                found=aedt_version,
            )
        )
    if not design or not design.strip():
        raise ValueError(
            refusal("raw .aedt preview requires an explicit design", found=design)
        )
    return _AedtSourceRequest(
        project_path.parent,
        project_path,
        design,
        resolve_aedt_version(aedt_version),
        None,
        None,
        (),
        None,
    )


def _read_run_folder(
    root: Path, design: str | None, aedt_version: str | None
) -> _AedtSourceRequest:
    """Validate run spec and receipt identity before native inspection."""
    spec_path = root / "aedt_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(
            refusal("AEDT run folder has no aedt_spec.json", found=str(spec_path))
        )
    spec = read_json_record(spec_path)
    aedt = mapping_record(spec.get("aedt"), "spec.aedt")
    profile = resolve_aedt_version(cast(str, aedt.get("requested_version")))
    if (
        aedt_version is not None
        and resolve_aedt_version(aedt_version).requested_version
        != profile.requested_version
    ):
        raise ValueError(
            refusal(
                "explicit AEDT version disagrees with the run spec",
                found=aedt_version,
                expected=profile.requested_version,
            )
        )
    project = mapping_record(spec.get("project"), "spec.project")
    recorded_design = nonempty_text(project.get("design"), "spec.project.design")
    if design is not None and design != recorded_design:
        raise ValueError(
            refusal(
                "explicit design disagrees with the AEDT run spec",
                found=design,
                expected=recorded_design,
            )
        )
    project_name = nonempty_text(project.get("name"), "spec.project.name")
    project_path = confined_artifact_path(root, f"{project_name}.aedt")
    receipt_path = root / "metadata/aedt_run_receipt.json"
    receipt = read_json_record(receipt_path) if receipt_path.is_file() else None
    source_conductors, receipt_version = _read_receipt_source(
        root, spec, spec_path, project_path, receipt
    )
    return _AedtSourceRequest(
        root,
        project_path,
        recorded_design,
        profile,
        receipt,
        receipt_path if receipt is not None else None,
        source_conductors,
        receipt_version,
    )


def _read_receipt_source(
    root: Path,
    spec: Mapping[str, Any],
    spec_path: Path,
    project_path: Path,
    receipt: dict[str, Any] | None,
) -> tuple[tuple[Mapping[str, Any], ...], str | None]:
    """Check receipt hashes and capture source conductor bindings."""
    source_conductors: tuple[Mapping[str, Any], ...] = ()
    receipt_version = None
    if receipt is None:
        return source_conductors, receipt_version
    connected = receipt.get("connected")
    if isinstance(connected, Mapping) and "aedt_version" in connected:
        receipt_version = resolve_aedt_version(
            connected["aedt_version"]
        ).requested_version
    if receipt.get("status") == "native_preparation_only":
        resolve_prepared_model(root)
    geometry = spec.get("geometry")
    if isinstance(geometry, Mapping):
        source_geometry = mapping_record(geometry.get("source"), "spec.geometry.source")
        source_conductors = tuple(source_geometry.get("conductors", ()))
        if receipt.get("status") != "native_preparation_only":
            recorded_source = mapping_record(receipt.get("source"), "receipt.source")
            if recorded_source.get("spec_sha256") != file_sha256(spec_path):
                raise ValueError(
                    refusal(
                        "AEDT source spec hash disagrees with the receipt",
                        found=file_sha256(spec_path),
                        expected=recorded_source.get("spec_sha256"),
                    )
                )
    if receipt.get("project") != project_path.relative_to(root).as_posix():
        raise ValueError(
            refusal(
                "AEDT receipt project identity disagrees with the spec",
                found=receipt.get("project"),
                expected=project_path.relative_to(root).as_posix(),
            )
        )
    outputs = mapping_record(receipt.get("outputs", {}), "receipt.outputs")
    relative = project_path.relative_to(root).as_posix()
    expected = outputs.get(relative)
    if not isinstance(expected, str) or file_sha256(project_path) != expected:
        raise ValueError(
            refusal(
                "AEDT project hash disagrees with the run receipt",
                found=file_sha256(project_path),
                expected=expected,
            )
        )
    return source_conductors, receipt_version


def _inspect_native_project(source: _AedtSourceRequest) -> _NativeSceneData:
    """Read native geometry and assignments while guaranteeing Desktop release."""
    try:
        import pyvista as pv  # noqa: PLC0415 -- optional preview mesh types load only during native geometry inspection.
        from ansys.aedt.core import Desktop  # noqa: PLC0415 -- optional PyAEDT loads only when opening the owned inspection Desktop.
    except ImportError as exc:
        raise RuntimeError(
            "AEDT geometry preview requires scgsim[aedt,visualization]"
        ) from exc
    if version("pyaedt") != LOCKED_PYAEDT:
        raise RuntimeError(
            f"AEDT preview requires PyAEDT {LOCKED_PYAEDT}, got {version('pyaedt')}"
        )

    return _inspect_desktop(source, pv, Desktop)


def _inspect_desktop(
    source: _AedtSourceRequest, pv: Any, desktop_type: Any
) -> _NativeSceneData:
    """Inspect the selected project and release the Desktop on every exit."""
    desktop = None
    app = None
    connected_aedt_version: str | None = None
    with native_compatibility(source.profile):
        try:
            desktop = desktop_type(
                version=source.profile.requested_version,
                non_graphical=True,
                new_desktop=True,
                close_on_exit=False,
            )
            connected_aedt_version = resolve_aedt_version(
                desktop.aedt_version_id
            ).requested_version
            if connected_aedt_version != source.profile.requested_version:
                raise RuntimeError(
                    refusal(
                        "AEDT Desktop release does not match the selected runtime",
                        found=connected_aedt_version,
                        expected=source.profile.requested_version,
                    )
                )
            if (
                source.receipt_connected_version is not None
                and connected_aedt_version != source.receipt_connected_version
            ):
                raise RuntimeError(
                    refusal(
                        "AEDT Desktop release disagrees with the run receipt",
                        found=connected_aedt_version,
                        expected=source.receipt_connected_version,
                    )
                )
            app = desktop.load_project(
                str(source.project_path), design_name=source.design
            )
            if app is False or app is None:
                raise RuntimeError(
                    refusal(
                        "PyAEDT could not open the requested project/design",
                        found={
                            "project": str(source.project_path),
                            "design": source.design,
                        },
                    )
                )
            if str(app.design_name) != source.design:
                raise ValueError(
                    refusal(
                        "native AEDT design readback disagrees with the request",
                        found=str(app.design_name),
                        expected=source.design,
                    )
                )
            app.modeler.refresh_all_ids()
            object_names = tuple(str(name) for name in app.modeler.object_names)
            if not object_names:
                raise ValueError(
                    refusal(
                        "AEDT design contains no model objects", found=source.design
                    )
                )
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

            native_boundaries: Mapping[str, Mapping[str, Any]] = {}
            if source.receipt is not None:
                native_boundaries = validate_native_receipt(app, source.receipt)
            materials = _materials(
                app, datasets, source.receipt, source.source_conductors
            )
            boundaries = _boundaries(
                app, datasets, source.receipt, pv, native_boundaries
            )
        finally:
            if desktop is not None:
                released = desktop.release_desktop(
                    close_projects=True, close_on_exit=True
                )
                if released is False:
                    raise RuntimeError(
                        refusal(
                            "AEDT Desktop did not confirm release after preview",
                            found=released,
                        )
                    )

    return _NativeSceneData(
        connected_aedt_version,
        tuple(materials),
        tuple(boundaries),
    )


def _build_scene(
    source: _AedtSourceRequest, native: _NativeSceneData
) -> GeometryPreview:
    """Create the solver-neutral preview after all native resources are released."""
    source_hashes = {
        source.project_path.relative_to(source.root).as_posix(): file_sha256(
            source.project_path
        )
    }
    if (
        source.root != source.project_path.parent
        or (source.root / "aedt_spec.json").is_file()
    ):
        for relative in ("aedt_spec.json",):
            path = source.root / relative
            if path.is_file():
                source_hashes[relative] = file_sha256(path)
    source_identity: dict[str, Any] = {
        "design": source.design,
        "aedt_version": native.connected_version,
        "requested_aedt_version": source.profile.requested_version,
        "connected_aedt_version": native.connected_version,
        "pyaedt_version": LOCKED_PYAEDT,
    }
    if source.receipt_path is not None and source.receipt is not None:
        source_identity["receipt_authority_sha256"] = receipt_authority_sha256(
            source.receipt_path
        )
    return GeometryPreview(
        root=source.root,
        backend="aedt",
        source_hashes=source_hashes,
        source_identity=source_identity,
        bind_aedt_receipt=source.receipt is not None,
        modes={
            "materials": SceneMode(native.materials),
            "boundaries": SceneMode(native.boundaries),
            "surface_epr": SceneMode(
                unavailable_reason="AEDT V1 has no structured Surface-EPR assignment authority."
            ),
            "mesh": SceneMode(
                unavailable_reason="AEDT V1 has no readable native mesh artifact."
            ),
        },
    )


def _materials(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    source_conductors: tuple[Mapping[str, Any], ...],
) -> list[ScenePart]:
    receipt_items: dict[str, Mapping[str, Any]] = {}
    pdk_materials: Mapping[str, Any] = {}
    region: Mapping[str, Any] = {}
    if receipt is not None:
        receipt_items = {
            nonempty_text(
                item.get("object_name"), "receipt.materials.object_name"
            ): item
            for item in material_records(receipt)
        }
        pdk_materials = mapping_record(
            receipt.get("pdk_materials", {}), "receipt.pdk_materials"
        )
        region = mapping_record(receipt.get("region", {}), "receipt.region")
    parts: list[ScenePart] = []
    for object_name, dataset in sorted(datasets.items()):
        native_material = str(app.modeler[object_name].material_name).lower()
        item = receipt_items.get(object_name)
        if item is not None:
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
                    raise ValueError(
                        refusal(
                            "native conductor has no unique source material binding",
                            found={
                                "semantic_id": item["semantic_id"],
                                "matches": len(matches),
                            },
                        )
                    )
                material_id = nonempty_text(
                    matches[0].get("material_id"), "source conductor material_id"
                )
            else:
                material_id = nonempty_text(
                    item.get("material_id"), "receipt material_id"
                )
            material = mapping_record(
                pdk_materials.get(material_id), "receipt PDK material"
            )
            kind = nonempty_text(material.get("kind"), "receipt material kind")
            observed = mapping_record(
                item.get("observed", {}), "receipt material observed"
            )
            observed_name = observed.get("native_material_name")
            if (
                observed_name is not None
                and str(observed_name).lower() != native_material
            ):
                raise ValueError(f"native AEDT material changed for {object_name}")
        elif object_name == "Region" and region:
            material_id = nonempty_text(
                region.get("material_id"), "receipt region material_id"
            )
            material = mapping_record(
                pdk_materials.get(material_id), "receipt PDK material"
            )
            kind = nonempty_text(material.get("kind"), "receipt material kind")
            if str(region.get("observed_material_name", "")).lower() != native_material:
                raise ValueError(
                    refusal(
                        "native AEDT Region material changed",
                        found=region.get("observed_material_name"),
                        expected=native_material,
                    )
                )
        elif receipt is not None:
            raise ValueError(f"AEDT receipt has no material identity for {object_name}")
        else:
            material_id = native_material
            kind = "native AEDT material"
        parts.append(
            ScenePart(
                dataset,
                f"material:{material_id}",
                material_id,
                kind,
                color_for_identity(material_id),
                0.08 if kind == "vacuum" else 0.42,
                1,
            )
        )
    return parts


def _boundaries(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    pv: Any,
    native_boundaries: Mapping[str, Mapping[str, Any]],
) -> list[ScenePart]:
    scene = _BoundaryScene(app, datasets, pv, native_boundaries)
    if receipt is None:
        _read_native_boundaries(scene)
    else:
        _read_receipt_boundaries(scene, receipt)
    scene.add_unassigned_objects()
    return scene.parts


@dataclass
class _BoundaryScene:
    """Collect boundary parts while tracking native object assignments."""

    app: Any
    datasets: Mapping[str, Any]
    pv: Any
    native_boundaries: Mapping[str, Mapping[str, Any]]
    parts: list[ScenePart] = field(default_factory=list)
    assigned: set[str] = field(default_factory=set)

    def add_object(self, object_name: str, label: str, role: str) -> None:
        """Add one uniquely assigned exported body to the boundary display."""
        if object_name not in self.datasets:
            raise ValueError(f"AEDT boundary object was not exported: {object_name}")
        if object_name in self.assigned:
            return
        self.assigned.add(object_name)
        self.parts.append(
            ScenePart(
                self.datasets[object_name],
                f"boundary:{label}:{object_name}",
                label,
                role,
                color_for_identity(role if role in FIXED_COLORS else label),
                0.95,
                1,
            )
        )

    def add_receipt_port(self, port: Mapping[str, Any]) -> None:
        """Build a displayed port face from the receipt's native face ID."""
        face_id = int(port["face_id"])
        face = self.app.modeler.get_face_by_id(face_id)
        if face is None:
            raise ValueError(f"AEDT native port face {face_id} is unavailable")
        points = [
            tuple(float(value) for value in vertex.position) for vertex in face.vertices
        ]
        if len(points) < 3:
            raise ValueError(f"AEDT native port face {face_id} is degenerate")
        dataset = self.pv.PolyData(points, faces=[len(points), *range(len(points))])
        label = nonempty_text(port.get("boundary"), "receipt port boundary")
        boundary_type = str(self.native_boundaries[label]["BoundType"])
        self.parts.append(
            ScenePart(
                dataset,
                f"port:{label}",
                label,
                boundary_type,
                color_for_identity(boundary_type),
                1.0,
                1,
            )
        )

    def add_native_face(self, face_id: Any, name: str, role: str) -> None:
        """Build a boundary display face from a native face assignment."""
        face = self.app.modeler.get_face_by_id(int(face_id))
        if face is None:
            raise ValueError(f"native AEDT boundary face {face_id} is unavailable")
        points = [
            tuple(float(value) for value in vertex.position) for vertex in face.vertices
        ]
        dataset = self.pv.PolyData(points, faces=[len(points), *range(len(points))])
        self.parts.append(
            ScenePart(
                dataset,
                f"boundary:{name}",
                name,
                role,
                color_for_identity(role),
                1.0,
                1,
            )
        )

    def add_unassigned_objects(self) -> None:
        """Show exported model bodies that have no boundary assignment."""
        for object_name, dataset in sorted(self.datasets.items()):
            if object_name not in self.assigned:
                self.parts.append(
                    ScenePart(
                        dataset,
                        f"unassigned:{object_name}",
                        f"{object_name}: no explicit boundary assignment",
                        "unassigned",
                        FIXED_COLORS["unassigned"],
                        0.35,
                        1,
                    )
                )


def _read_receipt_boundaries(scene: _BoundaryScene, receipt: Mapping[str, Any]) -> None:
    """Dispatch receipt assignments by AEDT solution family."""
    mode = receipt.get("mode")
    if mode in {"terminal", "modal", "eigenmode"}:
        _read_receipt_hfss_boundaries(scene, receipt)
    elif mode == "q3d":
        _read_receipt_q3d_boundaries(scene, receipt)
    elif mode == "q2d":
        _read_receipt_q2d_boundaries(scene, receipt)
    else:
        raise ValueError(f"unsupported AEDT receipt mode: {mode!r}")


def _read_receipt_hfss_boundaries(
    scene: _BoundaryScene, receipt: Mapping[str, Any]
) -> None:
    """Render PEC assignments and port faces from an HFSS receipt."""
    for item in material_records(receipt):
        observed = mapping_record(item.get("observed", {}), "receipt observed")
        native_objects = tuple(observed.get("native_pec_objects", ()))
        if native_objects:
            object_name = nonempty_text(item.get("object_name"), "receipt object_name")
            if object_name not in native_objects:
                raise ValueError(
                    refusal(
                        "AEDT PEC receipt object readback is inconsistent",
                        found=object_name,
                        expected=sorted(native_objects),
                    )
                )
            scene.add_object(object_name, "PEC", "PEC")
        elif item.get("boundary_name") or observed.get("native_solve_inside") is False:
            object_name = nonempty_text(item.get("object_name"), "receipt object_name")
            label = item.get("boundary_name") or "PEC material / SolveInside=false"
            scene.add_object(object_name, label, "PEC")
    for port in receipt.get("ports", ()):
        scene.add_receipt_port(port)


def _read_receipt_q3d_boundaries(
    scene: _BoundaryScene, receipt: Mapping[str, Any]
) -> None:
    """Render Q3D net object assignments."""
    for net in receipt.get("nets", ()):
        role = nonempty_text(net.get("net_type"), "receipt net_type")
        name = nonempty_text(net.get("name"), "receipt net name")
        for object_name in net.get("object_names", ()):
            scene.add_object(
                nonempty_text(object_name, "receipt net object"),
                name,
                "Ground" if role == "Ground" else "Terminal",
            )


def _read_receipt_q2d_boundaries(
    scene: _BoundaryScene, receipt: Mapping[str, Any]
) -> None:
    """Render Q2D conductor object assignments."""
    for conductor in receipt.get("conductors", ()):
        role = nonempty_text(conductor.get("conductor_type"), "receipt conductor_type")
        name = nonempty_text(conductor.get("name"), "receipt conductor name")
        for object_name in conductor.get("object_names", ()):
            scene.add_object(
                nonempty_text(object_name, "receipt conductor object"),
                name,
                "Ground" if role == "ReferenceGround" else "Terminal",
            )


def _read_native_boundaries(scene: _BoundaryScene) -> None:
    """Render supported native assignments when no receipt is available."""
    for boundary in scene.app.boundaries:
        props = dict(boundary.props)
        boundary_type = str(props.get("BoundType") or getattr(boundary, "type", ""))
        name = str(getattr(boundary, "name", props.get("Name", boundary_type)))
        if boundary_type == "Terminal":
            continue
        if boundary_type not in {"Perfect E", "Wave Port", "Lumped Port"}:
            raise ValueError(
                f"unsupported native AEDT boundary type: {boundary_type!r}"
            )
        role = "PEC" if boundary_type == "Perfect E" else boundary_type
        for object_name in _native_object_names(scene.app, props.get("Objects", ())):
            scene.add_object(object_name, name, role)
        for face_id in props.get("Faces", ()):
            scene.add_native_face(face_id, name, role)


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
