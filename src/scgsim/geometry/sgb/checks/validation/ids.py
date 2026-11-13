"""Invariants of the atomic ids: one id names one entity, in every dimension (design brief section 7).

Ids are 64-bit truncations of a content hash, so a collision is astronomically unlikely, but the
containers that hold the entities (``dict`` by id) would merge two colliding entities *silently*: a
face would vanish, a shell would stay closed around the wrong geometry and every later stage would
agree with the loss.  These checks look at the layers' id lists, which are still positional, so a
repeated id means two distinct entities (every position of an arrangement is a distinct entity) --
and then at the topology built from them.

* every vertex, edge and face (or slab cell) id is used by one entity of one arrangement, and the
  topology is keyed by the id each record carries;
* every labelled slab cell belongs to exactly one volume, so no volume id swallowed another's cells.

Emission checks the same property for the numbered shapes (``output.emission.plan.check_one_tag_per_id``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

from scgsim.geometry.sgb.arrangement.derivation.layer import Layer
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.topology.assembly.records import Topology

SHOWN_IDS = 3  # repeated ids named in a failure message


def check_atomic_ids_unique(
    report: ValidationReport, layers: list[Layer], topology: Topology
) -> None:
    """No two distinct entities of one dimension share an id; the topology keeps every labelled slab cell once."""
    dimensions: dict[str, Callable[[Layer], Iterable[str]]] = {
        "vertex": lambda layer: layer.vertex_ids,
        "edge": lambda layer: layer.edge_ids,
        "face": lambda layer: layer.face_ids[
            1:
        ],  # index 0 is the unbounded face, which has no id,
    }
    for name, ids_of in dimensions.items():
        repeated = _repeated(i for layer in layers for i in ids_of(layer))
        report.add(
            f"{name}_ids_unique",
            not repeated,
            f"{len(repeated)} {name} ids name more than one entity: {repeated[:SHOWN_IDS]}"
            if repeated
            else f"every {name} id names one entity",
        )
    _check_topology_keys(report, topology)
    _check_volume_cells(report, layers, topology)


def _repeated(ids: Iterable[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def _check_topology_keys(report: ValidationReport, topology: Topology) -> None:
    """Each topology table is keyed by the id its record carries."""
    tables = {
        "vertices": topology.vertices,
        "edges": topology.edges,
        "faces": topology.faces,
        "volumes": topology.volumes,
    }
    mismatched = {
        name: [key for key, record in table.items() if record.id != key]
        for name, table in tables.items()
    }
    bad = {name: keys for name, keys in mismatched.items() if keys}
    report.add(
        "topology_keyed_by_entity_id",
        not bad,
        f"records stored under another entity's id: { {n: k[:SHOWN_IDS] for n, k in bad.items()} }"
        if bad
        else "every vertex, edge, face and volume is stored under its own id",
    )


def _check_volume_cells(
    report: ValidationReport, layers: list[Layer], topology: Topology
) -> None:
    labelled_cells = [
        layer.face_ids[face]
        for layer in layers
        if layer.kind == "slab"
        for face in range(1, len(layer.arr.faces))
        if layer.labels[face] is not None
    ]
    in_volumes = Counter(cell for v in topology.volumes.values() for cell in v.cells)
    lost = sorted(set(labelled_cells) - in_volumes.keys())
    twice = sorted(cell for cell, n in in_volumes.items() if n > 1)
    report.add(
        "every_cell_in_exactly_one_volume",
        not lost and not twice,
        f"{len(lost)} cells in no volume, {len(twice)} in more than one: {(lost + twice)[:SHOWN_IDS]}"
        if lost or twice
        else f"{len(labelled_cells)} labelled cells, each in one of {len(topology.volumes)} volumes",
    )
