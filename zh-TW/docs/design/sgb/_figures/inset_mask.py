"""The mask-against-imprint illustration and the Route A and Route B cross-section of the inset-imprint page, with the mesh checks behind them."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import matplotlib.pyplot as plt
import numpy as np  # noqa: E402
from inset_figures import LEVEL_Z0
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from sgb_gmsh_oracle import (  # noqa: E402
    band_mesh_report,
    check_mesh_conformity,
    gmsh_session,
    mesh_coarsely,
    refine_near,
)

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import verify_loaded  # noqa: E402

# --------------------------------------------------------------------------- 5. mask vs imprint (2-D gmsh illustration)

RADIUS = 0.5  # mask boundary: circle of this radius; masked region = the square minus the disc


def disc_mesh(h: float, aligned: bool):
    """Triangles of the square [-1, 1]^2 with element size ``h``; ``aligned`` makes the circle a mesh boundary."""

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mask")
        geo = gmsh.model.geo
        corners = [
            geo.addPoint(x, y, 0) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        outer = geo.addCurveLoop(
            [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        )
        inside_tag = None
        if aligned:
            c = geo.addPoint(0, 0, 0)
            q = [
                geo.addPoint(RADIUS * math.cos(a), RADIUS * math.sin(a), 0)
                for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
            ]
            arcs = [geo.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
            ring = geo.addCurveLoop(arcs)
            outside_tag = geo.addPlaneSurface([outer, ring])
            inside_tag = geo.addPlaneSurface([ring])
        else:
            outside_tag = geo.addPlaneSurface([outer])
        geo.synchronize()
        for key, value in (
            ("Mesh.MeshSizeMax", h),
            ("Mesh.MeshSizeMin", 0.0),
            ("Mesh.MeshSizeFromPoints", 0),
            ("Mesh.MeshSizeFromCurvature", 0),
            ("Mesh.MeshSizeExtendFromBoundary", 0),
            ("Mesh.Algorithm", 6),
        ):
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(node_tags)}
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]

        def triangles(tag):
            _, _, nodes = gmsh.model.mesh.getElements(2, tag)
            return (
                np.array(
                    [
                        [index[int(n)] for n in tri]
                        for tri in np.asarray(nodes[0]).reshape(-1, 3)
                    ]
                )
                if len(nodes)
                else np.zeros((0, 3), int)
            )

        return (
            xy,
            triangles(outside_tag),
            (triangles(inside_tag) if inside_tag else None),
        )
    finally:
        gmsh.finalize()


def _tri_areas(xy, tri):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


SHIFTS = [
    (0.0, 0.0),
    (0.011, 0.007),
    (-0.017, 0.013),
    (0.023, -0.019),
    (-0.007, -0.029),
    (0.031, 0.002),
    (-0.013, 0.021),
    (0.005, -0.011),
]


def sampled_mask_area(xy, tri, area, centre) -> tuple[float, float]:
    """Masked area (square minus disc about ``centre``) when the mask is sampled at 1 and at 3 points per element."""
    cx, cy = centre
    mid = (
        xy[tri[:, [0, 1]]].mean(axis=1),
        xy[tri[:, [1, 2]]].mean(axis=1),
        xy[tri[:, [2, 0]]].mean(axis=1),
    )
    centroid = xy[tri].mean(axis=1)
    one = (np.hypot(centroid[:, 0] - cx, centroid[:, 1] - cy) >= RADIUS).astype(float)
    three = (
        sum((np.hypot(m[:, 0] - cx, m[:, 1] - cy) >= RADIUS).astype(float) for m in mid)
        / 3.0
    )
    return float((area * one).sum()), float((area * three).sum())


def mask_metrics(h: float) -> dict:
    """Masked area (square minus disc) from a mesh the mask cuts (several circle positions), and from a mesh aligned to the contour."""
    exact = 4.0 - math.pi * RADIUS**2
    xy, tri, _ = disc_mesh(h, aligned=False)
    area = _tri_areas(xy, tri)
    radial = np.hypot(xy[:, 0], xy[:, 1])
    side = radial[tri] - RADIUS
    cut = int(np.sum((side.min(axis=1) < -1e-9) & (side.max(axis=1) > 1e-9)))
    one, three = zip(
        *(sampled_mask_area(xy, tri, area, shift) for shift in SHIFTS), strict=True
    )
    err1 = [abs(v - exact) / exact for v in one]
    err3 = [abs(v - exact) / exact for v in three]
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    radial2 = np.hypot(xy2[:, 0], xy2[:, 1])
    side2 = radial2[tri_out] - RADIUS
    side_in = radial2[tri_in] - RADIUS
    crossing = int(
        np.sum((side2.min(axis=1) < -1e-6) & (side2.max(axis=1) > 1e-6))
    ) + int(np.sum((side_in.max(axis=1) > 1e-6)))
    aligned_area = float(_tri_areas(xy2, tri_out).sum())
    return {
        "h": h,
        "exact": exact,
        "cut_triangles": cut,
        "triangles_unaligned": int(len(tri)),
        "err_one_point_median": float(np.median(err1)),
        "err_one_point_max": max(err1),
        "err_one_point_min": min(err1),
        "err_three_point_median": float(np.median(err3)),
        "err_three_point_max": max(err3),
        "err_three_point_min": min(err3),
        "crossing_aligned": crossing,
        "triangles_aligned": int(len(tri_out) + len(tri_in)),
        "err_aligned": abs(aligned_area - exact) / exact,
    }


def figure_mask_vs_imprint() -> list[dict]:
    """A 2-D illustration of a mask against a mesh aligned to the contour (``inset-mask-vs-imprint``); returns the measured rows."""
    h = 0.16
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )
    fig.patch.set_facecolor("white")
    xy, tri, _ = disc_mesh(h, aligned=False)
    radial = np.hypot(xy[:, 0], xy[:, 1]) - RADIUS
    side = radial[tri]
    is_cut = (side.min(axis=1) < 0) & (side.max(axis=1) > 0)
    centroid = xy[tri].mean(axis=1)
    kept = np.hypot(centroid[:, 0], centroid[:, 1]) >= RADIUS
    colours = np.where(is_cut, kit.ACCENT, np.where(kept, kit.PAD_BANDS[3], "#ffffff"))
    ax = axes[0]
    ax.add_collection(
        PolyCollection(xy[tri], facecolors=colours, edgecolors="#555", linewidths=0.4)
    )
    t = np.linspace(0, 2 * math.pi, 400)
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(
        f"mask evaluated on a mesh that does not know it:\n{int(is_cut.sum())} of {len(tri)} triangles straddle the contour (orange)",
        fontsize=9,
    )
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    ax = axes[1]
    ax.add_collection(
        PolyCollection(
            xy2[tri_out], facecolors=kit.PAD_BANDS[3], edgecolors="#555", linewidths=0.4
        )
    )
    ax.add_collection(
        PolyCollection(
            xy2[tri_in], facecolors="#ffffff", edgecolors="#555", linewidths=0.4
        )
    )
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(
        f"contour is a mesh boundary (an imprinted band edge):\n0 of {len(tri_out) + len(tri_in)} triangles straddle it",
        fontsize=9,
    )
    for ax in axes[:2]:
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    rows = [mask_metrics(x) for x in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)]
    ax = axes[2]
    hs = np.array([r["h"] for r in rows])
    for key, colour, marker, label in (
        ("one_point", kit.ACCENT, "o", "mask sampled at 1 point per element"),
        ("three_point", kit.OTHER, "s", "mask sampled at 3 points per element"),
    ):
        ax.loglog(
            hs,
            [r[f"err_{key}_median"] for r in rows],
            marker + "-",
            color=colour,
            label=label + " (median of 8 contour positions)",
        )
        ax.fill_between(
            hs,
            [r[f"err_{key}_min"] for r in rows],
            [r[f"err_{key}_max"] for r in rows],
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
    ax.loglog(
        hs,
        [r["err_aligned"] for r in rows],
        "^-",
        color=kit.EXACT,
        lw=2,
        label="contour is a mesh boundary (sum over band-group faces)",
    )
    ax.loglog(hs, 0.03 * (hs / 0.4) ** 2, color="#777", ls=":", lw=1, label="slope 2")
    ax.loglog(hs, 0.03 * (hs / 0.4), color="#777", ls="--", lw=1, label="slope 1")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{x:g}" for x in hs], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("element size h", fontsize=8)
    ax.set_ylabel("relative error of the masked area", fontsize=8)
    ax.set_title(
        "masked area of a surface against a circular contour\n(first-order elements; exact value 4 - pi r^2, r = 0.5)",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3, which="major")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    kit.save_svg(fig, "inset-mask-vs-imprint")
    (kit.FIGURES / "inset-mask-vs-imprint.json").write_text(json.dumps(rows, indent=1))
    print("wrote inset-mask-vs-imprint")
    return rows


# --------------------------------------------------------------------------- 6. conceptual cross-section of the routes


def figure_routes_section() -> None:
    """A conceptual cross-section of Route A and Route B (``inset-routes-section``)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.2, 3.4)
        ax.axis("off")
        ax.add_patch(Rectangle((0, -2), 10, 2, facecolor="#e5e5e5", edgecolor="none"))
        ax.text(0.15, -1.2, "substrate", fontsize=8)
        ax.text(0.15, 2.7, "vacuum", fontsize=8)
    # Route A: a sheet; bands lie on the sheet; one group serves the MA and the MS row
    ax = axes[0]
    ax.set_title("Route A: zero-thickness sheet", fontsize=9)
    edges = [2, 2.5, 3.5, 6.5, 7.5, 8]
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot(
            [edges[i], edges[i + 1]],
            [0, 0],
            color=kit.PAD_BANDS[k],
            lw=7,
            solid_capstyle="butt",
        )
    ax.text(5, 0.65, "one face per band: pad::band=k", fontsize=8, ha="center")
    ax.text(
        5,
        0.2,
        "(the sheet is metal-vacuum and metal-substrate at once)",
        fontsize=7,
        ha="center",
        color="#555",
    )
    ax.annotate(
        "MA row and MS row both\nread these band groups",
        (2.2, 0),
        (0.3, 1.5),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    # Route B: a finite prism; the top face and bottom face each carry bands
    ax = axes[1]
    ax.set_title(
        "Route B: finite metal (excluded volume), top and bottom faces banded separately",
        fontsize=9,
    )
    ax.add_patch(
        Rectangle(
            (2, 0), 6, 0.6, facecolor="white", edgecolor=kit.INK, lw=0.8, hatch="//"
        )
    )
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot(
            [edges[i], edges[i + 1]],
            [0.6, 0.6],
            color=kit.PAD_BANDS[k],
            lw=6,
            solid_capstyle="butt",
        )
        ax.plot(
            [edges[i], edges[i + 1]],
            [0.0, 0.0],
            color=kit.PAD_BANDS[k],
            lw=6,
            solid_capstyle="butt",
        )
    ax.text(5, 1.2, "top face: MA::pad::top::band=k", fontsize=8, ha="center")
    ax.text(5, -0.55, "bottom face: MS::pad::bottom::band=k", fontsize=8, ha="center")
    ax.text(
        8.25, 0.3, "side walls:\nnot banded in v1", fontsize=7, ha="left", color="#555"
    )
    fig.tight_layout()
    kit.save_svg(fig, "inset-routes-section")
    print("wrote inset-routes-section")


# --------------------------------------------------------------------------- the mesh check (Gmsh only; no Palace is run)


def mesh_check(model, size: float) -> dict:
    """Write a model, mesh it with a size field on the band curves, and check every band group's triangles.

    The check of the production test ``test_band_faces_through_the_direct_writer``: each triangle of a band group is a
    face of one tetrahedron on each side and its centroid lies in the band; the whole mesh is conformal.  Mesh size
    ``size``, 1 um on the band curves growing to ``size`` over 1 to 20 um.
    """

    with tempfile.TemporaryDirectory(prefix="inset-mesh-") as scratch:
        artifacts = emit_model(model, Path(scratch))
        sidecar = read_sidecar(artifacts.sidecar)
        plan = plan_brep(model.topology, model.routed)
        tags = plan.tags()
        supports = {
            f.family_id: f.support
            for f in model.labelled.catalogue.families
            if f.support
        }
        with gmsh_session(artifacts.xao):
            loaded = verify_loaded(sidecar)
            assert loaded.ok, loaded
            for name, group in sidecar[
                "physical_groups"
            ].items():  # the loaded area of every band group equals the plan
                if "::band=" in name:
                    area = sum(
                        gmsh.model.occ.getMass(2, t) for t in group["native_tags"]
                    )
                    assert abs(area - group["area_um2"]) <= 1e-9 * group["area_um2"], (
                        name
                    )
            refine_near(
                model,
                tags,
                ("band_pad_top", "band_pad_bottom", "band_gap"),
                size,
                1.0,
                (1.0, 20.0),
            )
            mesh_coarsely(size)
            bands = band_mesh_report(model, tags, supports)
            conformity = check_mesh_conformity(model, tags)
    return {
        "size_um": size,
        "nodes": conformity["nodes"],
        "tets": conformity["tets"],
        "duplicate_nodes": conformity["duplicate_nodes"],
        "conformal": conformity["conformal"],
        "bands": bands,
    }


# --------------------------------------------------------------------------- tables for the page


def group_face_areas(model, level_index: int = LEVEL_Z0) -> dict[str, list[float]]:
    """Area (um^2) of every face of the level arrangement, by primary group (the faces of a group, largest first)."""
    layer = model.labelled.level_layers[level_index]
    out: dict[str, list[float]] = {}
    for fi in range(1, len(layer.arr.faces)):
        group = model.routed.face_group.get(layer.face_ids[fi])
        if group:
            out.setdefault(group, []).append(
                face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6
            )
    return {g: sorted(v, reverse=True) for g, v in out.items()}
