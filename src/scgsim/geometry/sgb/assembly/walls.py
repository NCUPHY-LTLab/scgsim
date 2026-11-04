"""Wall runs: the vertical faces of the model and the coalescing rule set (design brief sections 2 and 14).

Owns finding, for every maximal edge chain of the slab arrangements whose two sides carry
different labels, the *runs* of stacked slabs that form one wall face.  A level cuts a
wall (so the stacked walls of the slabs below and above stay two faces) where

* (a) an edge or a vertex of that level lies on the wall's horizontal line -- a port sheet
  edge touching a metal sidewall, another chain meeting it, or different edges of the
  chain below and above; or
* (b) a sheet face of that level lies along the wall chain.

Otherwise the stacked walls merge into one face.  The bottom boundary of a run is the chain
of level edges covering the slab edge at the lowest level of the run, the top boundary the
chain at the highest.  Levels are global in Z but their effect is local in XY.

Runs of different wall keys (the end points of the slab edge) are independent, so they are
computed as one parallel stage, each owned by the tile of its first end point.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.labelling.linking import TAG, walk_chain
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.labels import Domain


@dataclass(frozen=True)
class WallRun:
    """Stacked slabs ``k0 .. k1`` of one slab-edge chain that form one wall face."""

    u: Point
    v: Point
    k0: int
    k1: int
    dom_l: Domain | None  # domain on the left of u -> v
    dom_r: Domain | None
    left: int  # slab cell on the left in slab k0 (and in slab k1: left_top)
    right: int
    left_top: int
    right_top: int
    bottom: tuple[int, ...]  # half-edges of level k0 from u to v
    top: tuple[int, ...]  # half-edges of level k1 + 1 from u to v


def find_wall_runs(
    slabs: list[Layer], level_layers: list[Layer], rt: Runtime
) -> list[WallRun]:
    """Every wall run of the model (a parallel stage, one item per wall key)."""
    with rt.serial("wall_prep"):
        members = [_tag_members(layer) for layer in level_layers]
        keys: dict[tuple[Point, Point], list[tuple[int, int]]] = defaultdict(list)
        for k, slab in enumerate(slabs):
            for si, edge in enumerate(slab.arr.edges):
                left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
                if slab.labels[left] != slab.labels[right]:
                    keys[(edge.a, edge.b)].append((k, si))
        key_list = sorted(keys)
        owners = [rt.owner(key[0]) for key in key_list]
    shared = (slabs, level_layers, members, [keys[k] for k in key_list], key_list)
    out = rt.run("wall_runs", _wall_run_job, shared, owners)
    return [run for runs in out for run in runs]


def _tag_members(layer: Layer) -> dict[str, Any]:
    """Level edges carrying each slab-edge tag, and the number of edges at each vertex."""
    tags: dict[str, list[int]] = defaultdict(list)
    degree = [0] * len(layer.arr.vertices)
    for ei, edge in enumerate(layer.arr.edges):
        a, b = layer.arr.edge_ends[ei]
        degree[a] += 1
        degree[b] += 1
        for carrier in edge.carriers:
            if carrier.startswith(TAG):
                tags[carrier].append(ei)
    return {"tags": tags, "degree": degree}


def _level_cuts_wall(
    layer: Layer, degree: list[int], below: list[int], above: list[int]
) -> bool:
    """True when ``layer`` must cut the wall whose chains just below and above it are given."""
    if {h >> 1 for h in below} != {h >> 1 for h in above}:
        return True  # rule (a): the chains differ
    arr = layer.arr
    last = len(below) - 1
    for i, half_edge in enumerate(below):
        head = arr.edge_ends[half_edge >> 1][0 if half_edge & 1 else 1]
        if i < last and degree[head] != 2:
            return True  # rule (a): something else meets the wall line at this vertex
        sides = (
            layer.labels[arr.he_face[half_edge]],
            layer.labels[arr.he_face[half_edge ^ 1]],
        )
        if any(label is not None and label.needs_face() for label in sides):
            return True  # rule (b): a sheet face lies along the wall line
    return False


def _wall_run_job(shared: Any, items: list[int]) -> list[list[WallRun]]:
    slabs, level_layers, members, entries_by_key, key_list = shared
    return [
        _runs_of_key(
            slabs, level_layers, members, key_list[i], sorted(entries_by_key[i])
        )
        for i in items
    ]


def _runs_of_key(
    slabs: list[Layer],
    level_layers: list[Layer],
    members: list[dict[str, Any]],
    key: tuple[Point, Point],
    entries: list[tuple[int, int]],
) -> list[WallRun]:
    u, v = key
    runs: list[WallRun] = []
    current: dict[str, Any] | None = None
    for k, si in entries:
        slab = slabs[k]
        tag = TAG + slab.edge_ids[si]
        label = slab.edge_ids[si]
        bottom = walk_chain(level_layers[k].arr, members[k]["tags"][tag], u, v, label)
        top = walk_chain(
            level_layers[k + 1].arr, members[k + 1]["tags"][tag], u, v, label
        )
        left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
        domains = (slab.labels[left], slab.labels[right])
        if (
            current is not None
            and current["k1"] == k - 1
            and current["domains"] == domains
            and not _level_cuts_wall(
                level_layers[k], members[k]["degree"], current["top"], bottom
            )
        ):
            current.update(k1=k, top=top, left_top=left, right_top=right)
            continue
        if current is not None:
            runs.append(_finish(current, u, v))
        current = {
            "k0": k,
            "k1": k,
            "domains": domains,
            "bottom": bottom,
            "top": top,
            "left": left,
            "right": right,
            "left_top": left,
            "right_top": right,
        }
    if current is not None:
        runs.append(_finish(current, u, v))
    return runs


def _finish(run: dict[str, Any], u: Point, v: Point) -> WallRun:
    dom_l, dom_r = run["domains"]
    return WallRun(
        u,
        v,
        run["k0"],
        run["k1"],
        dom_l,
        dom_r,
        run["left"],
        run["right"],
        run["left_top"],
        run["right_top"],
        tuple(run["bottom"]),
        tuple(run["top"]),
    )
