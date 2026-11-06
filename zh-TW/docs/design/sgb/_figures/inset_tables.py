"""The tables and numbers the inset-imprint page quotes."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import numpy as np  # noqa: E402
from group_facts import banded_groups
from inset_figures import LEVEL_Z0
from inset_kit import MARGINS  # noqa: E402
from inset_mask import group_face_areas
from sgb_extreme_cases import FIXTURES, build_model  # noqa: E402
from sgb_klayout import comparison_rows  # noqa: E402
from sgb_scenes import exact_band_areas  # noqa: E402

from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402


def build_tables(
    model_a, topology_change: dict, mask_rows: list[dict], meshes: dict[str, dict]
) -> None:
    """Write ``inset-tables.md``: band areas, the KLayout cross-check, the mask comparison and the mesh check."""
    areas = group_face_areas(model_a)
    out = _band_areas_section(areas)
    out += _family_areas_section(model_a)
    out += _klayout_section(model_a)
    out += _route_a_mesh_section(meshes)
    out += _every_route_mesh_section(meshes)
    out += _mask_section(mask_rows)
    out += _route_b_section(model_a)
    out += _port_invariance_section(model_a)
    out += _topology_section(topology_change)
    out += _sagitta_section()
    (kit.FIGURES / "inset-tables.md").write_text("\n".join(out) + "\n")
    print("wrote inset-tables.md")


def _band_areas_section(areas):
    """Band areas per group and per band, and the port face."""
    out = [
        "## Band areas (computed by inset.py from the production Route A build, margins 0.1, 0.5, 2 um; areas of the faces of each band group)",
        "",
        "| support surface | band | group | faces | area (um^2) | R_m = bands >= k (um^2) |",
        "|---|---|---|---|---|---|",
    ]
    for prefix, label in (
        ("pad::band=", "pad top"),
        ("SA::substrate|vacuum__interface::band=", "gap"),
    ):
        per_band = {k: areas[f"{prefix}{k}"] for k in range(4)}
        total = {k: sum(v) for k, v in per_band.items()}
        for k in range(4):
            out.append(
                f"| {label} | B{k} | `{prefix}{k}` | {len(per_band[k])} | {total[k]:.4f} | {sum(a for b, a in total.items() if b >= k):.4f} |"
            )
    port = areas["PORT::p1"]
    out += [
        "",
        f"PORT::p1: {len(port)} faces, {sum(port):.4f} um^2 (the port footprint takes this area out of the gap bands)",
    ]
    return out


def _family_areas_section(model_a):
    """The exact band areas by family: the cells the band groups carry, the port face left out."""
    out = []
    # the exact band areas by family (cells the band groups carry; the port face is no band's surface)
    out += [
        "",
        "## Band areas by family (cells the band groups carry, the port face left out)",
        "",
        "| family | band | area (um^2) |",
        "|---|---|---|",
    ]
    for family in ("band_pad_top", "band_gap"):
        for k, a in sorted(
            exact_band_areas(model_a, family, include_ports=False).items()
        ):
            out.append(f"| {family} | B{k} | {a * 1e-6:.4f} |")
    return out


def _klayout_section(model_a):
    """The KLayout cross-check of the band areas (also written as JSON)."""
    out = []
    out += [
        "",
        "## KLayout cross-check (tests/sgb_klayout.py::comparison_rows on the Route A build)",
        "",
        "| region | margin (um) | components (KLayout) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for family, label in (("band_pad_top", "pad top"), ("band_gap", "gap")):
        for r in comparison_rows(model_a, family):
            rows.append({**r, "label": label})
            out.append(
                f"| {label} | {r['margin_units'] / 1000:g} | {r['components_klayout']} | {r['exact_um2']:.4f} | {r['klayout_um2']:.4f} | "
                f"{r['difference_um2']:+.5f} | {r['relative']:+.1e} | {r['oracle_rounding_bound_um2']:.3f} |"
            )
    (kit.FIGURES / "inset-klayout-comparison.json").write_text(
        json.dumps(rows, indent=1)
    )
    return out


def _route_a_mesh_section(meshes):
    """The mesh check of the Route A band groups."""
    out = []
    mesh = meshes["A"]
    out += [
        "",
        "## Mesh check of the Route A band groups (production writer, Gmsh; no Palace is run)",
        "",
        f"mesh size {mesh['size_um']:g} um, 1 um on the band curves; nodes: {mesh['nodes']}, tetrahedra: {mesh['tets']}, duplicate nodes: {mesh['duplicate_nodes']}, "
        f"conformal: {mesh['conformal']}",
        "",
        "| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in sorted(mesh["bands"].items()):
        sides = sorted(row["tets_per_side"].values())
        out.append(
            f"| `{name}` | {row['faces']} | {row['triangles']} | {sides[0]} + {sides[1]} | {row['all_shared_by_tets']} | {row['centroids_in_band']} |"
        )
    return out


def _every_route_mesh_section(meshes):
    """The same mesh check under every route."""
    out = []
    out += [
        "",
        "## The same mesh check under every route",
        "",
        "| route | mesh size (um) | band groups | nodes | tetrahedra | duplicate nodes | conformal | every band group passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for route, m in meshes.items():
        ok = all(
            r["triangles"] > 0
            and r["all_shared_by_tets"]
            and r["centroids_in_band"]
            and len(set(r["tets_per_side"].values())) == 1
            for r in m["bands"].values()
        )
        out.append(
            f"| {route} | {m['size_um']:g} | {len(m['bands'])} | {m['nodes']} | {m['tets']} | {m['duplicate_nodes']} | {m['conformal']} | {ok} |"
        )
    return out


def _mask_section(mask_rows):
    """Mask against imprint."""
    out = []
    out += [
        "",
        "## Mask against imprint (inset-mask-vs-imprint.json)",
        "",
        "| h | triangles | straddling the contour | error, 1-point mask (median / max) | error, 3-point mask (median / max) | triangles (aligned) | straddling (aligned) | error (aligned) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in mask_rows:
        out.append(
            f"| {r['h']} | {r['triangles_unaligned']} | {r['cut_triangles']} | {r['err_one_point_median']:.1e} / {r['err_one_point_max']:.1e} | {r['err_three_point_median']:.1e} / {r['err_three_point_max']:.1e} | "
            f"{r['triangles_aligned']} | {r['crossing_aligned']} | {r['err_aligned']:.2e} |"
        )
    return out


def _route_b_section(model_a):
    """Route B with its top and bottom faces banded."""
    out = []
    # Route B: the example bands the top face (MA) and the bottom face (MS)
    model_b = kit.pad_model("B", MARGINS)
    unions = model_b.routed.unions
    pad_surfaces = {*unions["MA"], *unions["MS"]} & set(unions["NET::pad"])
    groups = sorted(n for n in banded_groups(model_b) if n in pad_surfaces)
    counts = model_b.topology.counts()
    union = model_b.routed.unions["NET::pad"]
    out += [
        "",
        "## Route B (top and bottom faces banded)",
        "",
        "groups: " + ", ".join(groups),
        f"validation passed: {not model_b.report.failures} ({len(model_b.report.checks)} checks)",
        f"counts: {counts['faces']} faces, {counts['edges']} edges, {counts['vertices']} vertices, {len(model_b.routed.groups)} groups",
        "unions NET::pad: " + json.dumps(union),
        "surface_epr_rows (Route A): " + json.dumps(model_a.routed.surface_epr_rows),
    ]
    return out


def _port_invariance_section(model_a):
    """Route A with and without the lumped port."""
    out = []
    # the port moves no band: same supports, and it only takes its footprint out of the gap bands
    without = kit.pad_model("A", MARGINS, with_port=False)
    same = [f.support for f in model_a.labelled.catalogue.families] == [
        f.support for f in without.labelled.catalogue.families
    ]
    increase, removed, pad_top_equal = 0.0, 0.0, True
    for family in ("band_pad_top", "band_gap"):
        a, b = (
            exact_band_areas(model_a, family, include_ports=False),
            exact_band_areas(without, family),
        )
        increase = max(increase, max(a[k] - b[k] for k in a))
        if family == "band_gap":
            removed = (sum(b.values()) - sum(a.values())) * 1e-6
        else:
            pad_top_equal = a == exact_band_areas(without, family)
    out += [
        "",
        "## Port invariance (Route A, with and without the lumped port)",
        "",
        f"supports of the families equal: {same}",
        f"largest increase of a band area: {increase:.3g} grid unit^2",
        f"gap area taken out by the port: {removed:.4f} um^2",
        f"pad-top band areas equal: {pad_top_equal}",
    ]
    return out


def _topology_section(topology_change):
    """The margins at which the pad top changes topology."""
    out = []
    out += [
        "",
        "## Topology change (margins 4, 9, 30 um on the pad top)",
        "",
        json.dumps(topology_change),
    ]
    return out


def _sagitta_section():
    """The sagitta of a 128-gon disc."""
    out = []
    out += [
        "",
        "## Sagitta of a 128-gon disc, m (1 - cos(pi / 128)), in nm",
        "",
        "| margin (um) | sagitta (nm) |",
        "|---|---|",
    ] + [f"| {m} | {m * 1000 * (1 - math.cos(math.pi / 128)):.3f} |" for m in MARGINS]
    return out


def arc_deviation(model_a) -> dict:
    """Deviation of the arc edges of the assembled topology from their exact circles in the Route A model with bands (grid units)."""
    deviations = [
        e.curve.deviation
        for e in model_a.topology.edges.values()
        if e.curve is not None
    ]
    return {
        "arc_edges": len(deviations),
        "mean": float(np.mean(deviations)),
        "max": float(np.max(deviations)),
        "snap_bound": math.sqrt(2) / 2,
    }


def route_b_comparison() -> dict:
    """Route B's top and bottom faces against the oracle, and against the Route A pad sheet (same support, same numbers)."""
    model_b = kit.pad_model("B", MARGINS)
    rows_a = comparison_rows(kit.pad_model("A", MARGINS), "band_pad_top")
    out = {}
    for family in ("band_pad_top", "band_pad_bottom"):
        rows = comparison_rows(model_b, family)
        out[family] = {
            "max_relative": max(abs(r["relative"]) for r in rows),
            "same_exact_areas_as_route_a_pad": all(
                abs(r["exact_um2"] - a["exact_um2"]) < 1e-9
                for r, a in zip(rows, rows_a, strict=True)
            ),
            "within_oracle_noise": all(
                abs(r["difference_um2"]) <= r["oracle_rounding_bound_um2"] for r in rows
            ),
        }
    return out


def extreme_band_comparisons() -> dict:
    """The band fixtures of the extreme-case catalogue against KLayout: every family of every route, every margin."""
    rows = []
    for name in (
        "neck_split_by_inset",
        "band_vanishes",
        "reflex_small_angle",
        "refinement_crossing_band_edge",
    ):
        for route in "ABC":
            model = build_model(FIXTURES[name], route)
            for family in (
                f.family_id for f in model.labelled.catalogue.families if f.support
            ):
                for r in comparison_rows(model, family):
                    rows.append(
                        {
                            "fixture": name,
                            "route": route,
                            "family": family,
                            "margin_units": r["margin_units"],
                            "components_klayout": r["components_klayout"],
                            "difference_um2": r["difference_um2"],
                            "relative": r["relative"],
                            "within_oracle_noise": abs(r["difference_um2"])
                            <= r["oracle_rounding_bound_um2"],
                        }
                    )
    return {
        "rows": len(rows),
        "families": len({(r["fixture"], r["route"], r["family"]) for r in rows}),
        "all_within_oracle_noise": all(r["within_oracle_noise"] for r in rows),
        "max_relative": max(abs(r["relative"]) for r in rows),
    }


def noded_edges() -> dict:
    """The z = 0 level of the Route A example before and after dissolution (the witness chords make the first number large)."""
    catalogue = kit.pad_catalogue("A")
    model = build_model_from_catalogue(
        catalogue, "A", insets=kit.pad_requests("A"), keep_raw=True
    )
    layer = model.labelled.level_layers[LEVEL_Z0]
    raw = layer.raw_arrangement
    return {
        "noded_edges_before_dissolution": len(raw.edges),
        "hot_pixels": len(raw.vertices),
        "edges_after_dissolution": len(layer.arr.edges),
        "faces": len(layer.arr.faces) - 1,
    }
