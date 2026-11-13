"""Load Palace mesh and metadata into detached semantic preview modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

# The loader reads the run-folder contracts, not Palace: AEDT's runtime source closure reaches this module
# through ``scgsim.visualization``, and AEDT must not ship Palace or SGB; the contracts import only the
# standard library.
from scgsim.contracts.palace_index_map import (
    DIELECTRIC_SECTION,
    LUMPED_PORT_SECTION,
    TERMINAL_SECTION,
)
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    INDEX_MAP,
    INDEX_MAP_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    MESH,
    MESH_MANIFEST,
)
from scgsim.refusal import refusal

from ..preview import GeometryPreview
from ..scene import FIXED_COLORS, SceneMode, ScenePart, color_for_identity
from .common import file_sha256, read_json_record


def inspect_palace_geometry(run_dir: str | Path) -> GeometryPreview:
    """Read a Palace run folder's mesh with its config, manifest and index map into a ``GeometryPreview``."""
    root = Path(run_dir).expanduser().resolve()
    paths = {name: root / name for name in (MESH, CONFIG, MESH_MANIFEST, INDEX_MAP)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Palace preview inputs are missing: {missing!r}")

    try:
        import meshio  # noqa: PLC0415 - optional visualization dependency.
        import pyvista as pv  # noqa: PLC0415 - optional visualization dependency.
    except ImportError as exc:
        raise RuntimeError(
            "Palace geometry preview requires scgsim[visualization]"
        ) from exc

    mesh_manifest = read_json_record(paths[MESH_MANIFEST])
    index_map = read_json_record(paths[INDEX_MAP])
    _require_schema(mesh_manifest, "mesh manifest", MANIFEST_SCHEMA_VERSION)
    _require_schema(index_map, "palace index map", INDEX_MAP_SCHEMA_VERSION)
    config = read_json_record(paths[CONFIG])
    mesh = meshio.read(paths[MESH])
    grid = _high_order_grid(mesh, mesh_manifest, pv)
    selection = _bind_mesh_selection(mesh, grid, mesh_manifest)
    materials = _palace_materials(config, selection.records, selection.cells)
    boundaries = _palace_boundaries(
        config, index_map, selection.records, selection.cells
    )
    surface_epr = _palace_surface_epr(
        config, index_map, selection.records, selection.cells
    )
    mesh_parts = selection.mesh_parts()
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    return GeometryPreview(
        root=root,
        backend="palace",
        source_hashes=source_hashes,
        modes={
            "materials": SceneMode(tuple(materials)),
            "boundaries": SceneMode(tuple(boundaries)),
            "surface_epr": (
                SceneMode(tuple(surface_epr))
                if surface_epr
                else SceneMode(
                    unavailable_reason="No structured Surface-EPR assignment is present."
                )
            ),
            "mesh": SceneMode(tuple(mesh_parts)),
        },
    )


def _require_schema(record: Mapping[str, Any], name: str, expected: int) -> None:
    """Refuse a run folder of a schema other than the contract's; the message names the recorded one."""
    recorded = record.get("schema_version")
    if type(recorded) is not int or recorded != expected:
        raise ValueError(
            f"{name} schema_version {recorded!r} is not readable: only schema_version "
            f"{expected} is read, and run folders of older schemas are not."
        )


@dataclass(frozen=True)
class _MeshSelection:
    """One mesh and its detached physical-tag arrays shared by preview modes."""

    grid: Any
    dimensions: Any
    physical: Any
    cell_types: Any
    records: dict[tuple[int, int], Mapping[str, Any]]
    field_names: dict[tuple[int, int], list[str]]
    actual: set[tuple[int, int]]

    def cells(self, dimension: int, attribute: int) -> Any:
        """Extract cells for one exact physical identity."""
        import numpy as np  # noqa: PLC0415 -- Defer NumPy array loading until preview geometry is requested.

        ids = np.flatnonzero(
            (self.dimensions == dimension) & (self.physical == attribute)
        )
        if not len(ids):
            raise ValueError(
                f"Palace physical group {(dimension, attribute)} has no cells"
            )
        return self.grid.extract_cells(ids)

    def mesh_parts(self) -> list[ScenePart]:
        """Build the mesh mode using complete element-type labels and tag counts."""
        mesh_parts: list[ScenePart] = []
        for dimension, attribute in sorted(self.actual):
            types = sorted(
                set(
                    self.cell_types[
                        (self.dimensions == dimension) & (self.physical == attribute)
                    ]
                )
            )
            name = " / ".join(sorted(self.field_names[(dimension, attribute)]))
            mesh_parts.append(
                ScenePart(
                    dataset=self.cells(dimension, attribute),
                    semantic_id=f"mesh:{dimension}:{attribute}",
                    label=f"{name} · dim {dimension} · {'/'.join(types)}",
                    role="solution physical group",
                    color=color_for_identity(name),
                    opacity=0.35 if dimension == 3 else 0.85,
                    count=int(
                        (
                            (self.dimensions == dimension)
                            & (self.physical == attribute)
                        ).sum()
                    ),
                    show_edges=True,
                )
            )
        return mesh_parts


def _bind_mesh_selection(
    mesh: Any, grid: Any, mesh_manifest: Mapping[str, Any]
) -> _MeshSelection:
    """Validate physical tags and bind their exact structured manifest records."""
    import numpy as np  # noqa: PLC0415 -- Defer NumPy array loading until preview geometry is requested.

    tags = _cell_tags(mesh, grid)
    records = _manifest_records(mesh_manifest)
    actual = {
        (int(dim), int(attribute))
        for dim, attribute in zip(tags.dimensions, tags.physical, strict=True)
    }
    if actual != set(records):
        raise ValueError(
            refusal(
                "Palace mesh physical tags do not match mesh manifest",
                found=sorted(actual),
                expected=sorted(records),
            )
        )
    field_names = _field_names(mesh)
    if actual != set(field_names):
        raise ValueError(
            refusal(
                "Palace mesh field names do not cover every physical tag",
                found=sorted(field_names),
                expected=sorted(actual),
            )
        )
    return _MeshSelection(
        grid,
        np.asarray(tags.dimensions),
        np.asarray(tags.physical),
        np.asarray(tags.cell_types),
        records,
        field_names,
        actual,
    )


class _CellTags(NamedTuple):
    """For each cell of the mesh, in order: its dimension, its meshio cell type and its physical tag."""

    dimensions: list[int]
    cell_types: list[str]
    physical: list[int]


def _cell_tags(mesh: Any, grid: Any) -> _CellTags:
    """Read every cell's dimension, type and ``gmsh:physical`` tag, one tag per cell of the grid."""
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None or len(physical_blocks) != len(mesh.cells):
        raise ValueError(
            refusal(
                "Palace mesh has no complete gmsh:physical cell tags",
                found=None if physical_blocks is None else len(physical_blocks),
                expected=len(mesh.cells),
            )
        )
    tags = _CellTags([], [], [])
    for block, block_tags in zip(mesh.cells, physical_blocks, strict=True):
        tags.dimensions.extend([block.dim] * len(block.data))
        tags.cell_types.extend([block.type] * len(block.data))
        tags.physical.extend(int(value) for value in block_tags)
    if len(tags.physical) != grid.n_cells:
        raise ValueError(
            refusal(
                "Palace mesh cell-tag cardinality mismatch",
                found=len(tags.physical),
                expected=grid.n_cells,
            )
        )
    return tags


def _manifest_records(
    mesh_manifest: Mapping[str, Any],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    """The manifest's group record of each (dimension, attribute); it lists each solver attribute once."""
    records: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in mesh_manifest.get("groups", ()):
        for attribute in record["attributes"]:
            key = (int(record["dimension"]), int(attribute))
            if key in records:
                raise ValueError(f"Palace mesh manifest lists attribute {key!r} twice")
            records[key] = record
    return records


def _field_names(mesh: Any) -> dict[tuple[int, int], list[str]]:
    """The physical names Gmsh wrote for each (dimension, tag) of the mesh."""
    field_names: dict[tuple[int, int], list[str]] = {}
    for name, value in mesh.field_data.items():
        key = (int(value[1]), int(value[0]))
        field_names.setdefault(key, []).append(str(name))
    return field_names


def _palace_materials(
    config: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[ScenePart]:
    assignments: set[int] = set()
    for material in config.get("Domains", {}).get("Materials", ()):
        for attribute in material.get("Attributes", ()):
            attribute = int(attribute)
            if attribute in assignments:
                raise ValueError(f"Palace volume {attribute} has multiple materials")
            assignments.add(attribute)
    parts: list[ScenePart] = []
    volume_attributes = {
        attribute for dimension, attribute in records if dimension == 3
    }
    if volume_attributes != assignments:
        raise ValueError(
            refusal(
                "Palace volume groups and config materials do not match",
                found=sorted(assignments),
                expected=sorted(volume_attributes),
            )
        )
    for attribute in sorted(assignments):
        record = records[(3, attribute)]
        material_id = str(record["material"])
        kind = str(record["kind"])
        dataset = cells(3, attribute)
        parts.append(
            ScenePart(
                dataset,
                material_id,
                material_id,
                kind,
                color_for_identity(material_id),
                0.08 if kind == "vacuum" else 0.42,
                dataset.n_cells,
            )
        )
    return parts


def _palace_boundaries(
    config: Mapping[str, Any],
    index_map: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[ScenePart]:
    """One scene part per boundary attribute: its Ground, PEC, Terminal or LumpedPort assignment, or none."""
    boundary_root = config.get("Boundaries", {})
    indexed = _boundary_index(index_map)
    assignments: dict[int, _Assignment] = {}
    for section, role in (("Ground", "Ground"), ("PEC", "PEC")):
        for attribute in boundary_root.get(section, {}).get("Attributes", ()):
            _assign(assignments, int(attribute), _Assignment(role, role, role))
    for config_key, section in (
        ("Terminal", TERMINAL_SECTION),
        ("LumpedPort", LUMPED_PORT_SECTION),
    ):
        _assign_indexed(
            assignments, boundary_root.get(config_key, ()), indexed, config_key, section
        )
    return _boundary_parts(assignments, records, cells)


class _Assignment(NamedTuple):
    """What a boundary attribute is assigned to: its semantic id, the label a person reads and its role."""

    semantic_id: str
    label: str
    role: str


def _assign(
    assignments: dict[int, _Assignment], attribute: int, value: _Assignment
) -> None:
    """Record one attribute's assignment, refusing a second, different one."""
    previous = assignments.get(attribute)
    if previous is not None and previous != value:
        raise ValueError(
            refusal(
                f"Palace boundary {attribute} has conflicting assignments",
                found=value,
                expected=previous,
            )
        )
    assignments[attribute] = value


def _assign_indexed(
    assignments: dict[int, _Assignment],
    items: Any,
    indexed: Mapping[tuple[str, int], Mapping[str, Any]],
    config_key: str,
    section: str,
) -> None:
    """Assign the attributes of each configured Terminal or LumpedPort row, bound to its index-map entry."""
    configured: set[int] = set()
    for item in items:
        index = int(item["Index"])
        if index in configured:
            raise ValueError(f"Palace {config_key} index {index} is duplicated")
        configured.add(index)
        record = indexed.get((section, index))
        if record is None:
            raise ValueError(
                f"Palace {config_key} index {index} lacks index-map identity"
            )
        label = str(
            record.get("entry_name")
            or record.get("terminal_name")
            or record.get("net_id")
            or f"{config_key} {index}"
        )
        attributes = tuple(int(value) for value in item.get("Attributes", ()))
        recorded = tuple(int(value) for value in record.get("attributes", ()))
        if attributes != recorded:
            raise ValueError(
                refusal(
                    f"Palace {config_key} {index} config/index-map mismatch",
                    found=list(attributes),
                    expected=list(recorded),
                )
            )
        for attribute in attributes:
            _assign(
                assignments,
                int(attribute),
                _Assignment(f"{config_key}:{index}", label, config_key),
            )
    mapped = {index for (mapped_section, index) in indexed if mapped_section == section}
    if mapped != configured:
        raise ValueError(
            refusal(
                f"Palace {config_key} config/index-map identities differ",
                found=sorted(configured),
                expected=sorted(mapped),
            )
        )


def _boundary_parts(
    assignments: dict[int, _Assignment],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[ScenePart]:
    """One part per surface attribute of the mesh; an attribute nothing assigns is drawn as unassigned."""
    surface_attributes = {
        attribute for dimension, attribute in records if dimension == 2
    }
    missing = sorted(set(assignments) - surface_attributes)
    if missing:
        raise ValueError(
            f"Palace configured boundary attributes are absent: {missing!r}"
        )
    parts: list[ScenePart] = []
    for attribute in sorted(surface_attributes):
        semantic_id, label, role = assignments.get(
            attribute,
            _Assignment(
                f"unassigned:{attribute}",
                "No explicit boundary assignment",
                "unassigned",
            ),
        )
        dataset = cells(2, attribute)
        parts.append(
            ScenePart(
                dataset,
                semantic_id,
                label,
                role,
                color_for_identity(role if role in FIXED_COLORS else semantic_id),
                1.0 if role != "unassigned" else 0.5,
                dataset.n_cells,
            )
        )
    return parts


def _palace_surface_epr(
    config: Mapping[str, Any],
    index_map: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[ScenePart]:
    """One scene part per Surface-EPR attribute, named by the whole surfaces whose rows integrate it."""
    entries = _dielectric_entries(index_map)
    grouped = _entries_by_attribute(config, entries)
    parts: list[ScenePart] = []
    for attribute, mappings in sorted(grouped.items()):
        if (2, attribute) not in records:
            raise ValueError(f"Surface-EPR attribute {attribute} is absent from mesh")
        role = _surface_epr_role(mappings)
        # A margin row only re-selects attributes of its whole surface, which names them.
        labels = sorted(
            {str(item["entry_name"]) for item in mappings if "inset" not in item}
        )
        dataset = cells(2, attribute)
        parts.append(
            ScenePart(
                dataset,
                f"surface-epr:{attribute}:{role}",
                " / ".join(labels),
                role,
                FIXED_COLORS[role],
                0.95,
                dataset.n_cells,
            )
        )
    return parts


def _dielectric_entries(index_map: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """The index map's Surface-EPR entries by index, refusing a repeated index."""
    entries: dict[int, Mapping[str, Any]] = {}
    for item in index_map.get("entries", ()):
        if item.get("section") != DIELECTRIC_SECTION:
            continue
        index = int(item["index"])
        if index in entries:
            raise ValueError(f"Palace Surface-EPR index {index} is duplicated")
        entries[index] = item
    return entries


def _entries_by_attribute(
    config: Mapping[str, Any], entries: dict[int, Mapping[str, Any]]
) -> dict[int, list[Mapping[str, Any]]]:
    """For each attribute a configured Dielectric row covers, the index entries of the rows that cover it."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    configured: set[int] = set()
    for item in (
        config.get("Boundaries", {}).get("Postprocessing", {}).get("Dielectric", ())
    ):
        index = int(item["Index"])
        if index in configured:
            raise ValueError(f"Palace Surface-EPR index {index} is duplicated")
        configured.add(index)
        record = entries.get(index)
        if record is None or tuple(item.get("Attributes", ())) != tuple(
            record.get("attributes", ())
        ):
            raise ValueError(
                refusal(
                    "Palace Surface-EPR config/index-map mismatch",
                    found={
                        "index": index,
                        "config Attributes": list(item.get("Attributes", ())),
                    },
                    expected=None
                    if record is None
                    else list(record.get("attributes", ())),
                )
            )
        for attribute in item.get("Attributes", ()):
            grouped.setdefault(int(attribute), []).append(record)
    if set(entries) != configured:
        raise ValueError(
            refusal(
                "Palace Surface-EPR config/index-map identities differ",
                found=sorted(configured),
                expected=sorted(entries),
            )
        )
    return grouped


def _surface_epr_role(mappings: list[Mapping[str, Any]]) -> str:
    """The interface role of one attribute: MA, MS or SA, or MS_MA for a sheet face integrated from both sides."""
    kinds = {
        row["interface_type"] for item in mappings for row in item["metadata"]["rows"]
    }
    role = (
        "MS_MA"
        if kinds == {"MA", "MS"}
        else next(iter(kinds))
        if len(kinds) == 1 and next(iter(kinds)) in {"MA", "MS", "SA"}
        else ""
    )
    if not role:
        raise ValueError(f"unsupported Surface-EPR mapping: {sorted(kinds)!r}")
    return role


def _high_order_grid(mesh: Any, manifest: Mapping[str, Any], pv: Any) -> Any:
    """Retain complete Gmsh p3+ nodes in native VTK Lagrange cell order.

    meshio already maps quadratic cells to VTK order. Complete higher-order
    simplices retain Gmsh order, so native reference lattices supply the exact
    permutation instead of dropping their non-corner nodes.
    """
    import numpy as np  # noqa: PLC0415 -- Defer NumPy array loading until preview geometry is requested.
    from meshio._vtk_common import meshio_to_vtk_type  # noqa: PLC0415 -- meshio is optional and needed only for VTK previews.  # private-import: meshio exposes its VTK cell-code mapping only through this module.

    cells = _lagrange_cells(mesh, manifest)
    # Supply each cell's complete connectivity length directly: the fixed-node
    # lookup in PyVista's meshio reader does not cover native Lagrange types.
    connectivity = [
        np.column_stack((np.full(len(data), data.shape[1]), data)).ravel()
        for _, data in cells
    ]
    cell_types = np.concatenate(
        [np.full(len(data), meshio_to_vtk_type[kind]) for kind, data in cells]
    )
    grid = pv.UnstructuredGrid(
        np.concatenate(connectivity).astype(np.int64, copy=False),
        cell_types,
        np.asarray(mesh.points, dtype=np.float64),
    )
    grid.point_data.update(
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in mesh.point_data.items()
        }
    )
    grid.cell_data.update(
        {key: np.concatenate(values) for key, values in mesh.cell_data.items()}
    )
    return grid


class _CellBlock(NamedTuple):
    """One block of cells for the VTK grid: its meshio or VTK Lagrange type name and its connectivity."""

    kind: str
    data: Any


def _lagrange_cells(mesh: Any, manifest: Mapping[str, Any]) -> list[_CellBlock]:
    """Each cell block as a (VTK or meshio type, connectivity) pair, p3+ simplices in VTK Lagrange order."""
    import numpy as np  # noqa: PLC0415 -- Defer NumPy array loading until preview geometry is requested.
    from vtkmodules.vtkCommonDataModel import (  # noqa: PLC0415 - optional VTK runtime.
        vtkLagrangeCurve,
        vtkLagrangeTetra,
        vtkLagrangeTriangle,
    )

    records = manifest.get("meshing", {}).get("statistics", {}).get("element_types", ())
    families = {
        "line": (vtkLagrangeCurve, "VTK_LAGRANGE_CURVE"),
        "triangle": (vtkLagrangeTriangle, "VTK_LAGRANGE_TRIANGLE"),
        "tetra": (vtkLagrangeTetra, "VTK_LAGRANGE_TETRAHEDRON"),
    }
    cells: list[_CellBlock] = []
    for block in mesh.cells:
        record = next(
            (
                r
                for r in records
                if r["dimension"] == block.dim
                and r["nodes_per_element"] == block.data.shape[1]
            ),
            None,
        )
        family = next((name for name in families if block.type.startswith(name)), None)
        if record is None or record["order"] <= 2 or family is None:
            cells.append(_CellBlock(block.type, block.data))
            continue
        cls, vtk_type = families[family]
        order, count, dimension = (
            record["order"],
            record["nodes_per_element"],
            block.dim,
        )
        native = np.asarray(record["reference_coordinates"]).reshape(count, dimension)
        if dimension == 1:
            native = (native + 1.0) / 2.0
        source_lattice = {
            tuple(round(float(x) * order) for x in point): i
            for i, point in enumerate(native)
        }
        cell = cls()
        cell.GetPointIds().SetNumberOfIds(count)
        cell.GetPoints().SetNumberOfPoints(count)
        cell.Initialize()
        target = np.asarray(cell.GetParametricCoords()).reshape(count, 3)[:, :dimension]
        permutation = [
            source_lattice[tuple(round(float(x) * order) for x in point)]
            for point in target
        ]
        cells.append(_CellBlock(vtk_type, block.data[:, permutation]))
    return cells


def _boundary_index(
    index_map: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Require unique section/index identities before boundary assignments."""
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in index_map.get("entries", ()):
        if item.get("index") is None:
            continue
        key = (str(item.get("section")), int(item["index"]))
        if key in indexed:
            raise ValueError(f"Palace index map has duplicate identity {key!r}")
        indexed[key] = item

    return indexed
