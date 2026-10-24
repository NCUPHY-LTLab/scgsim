from pathlib import Path

from scgsim.palace.preparation.surface_epr import build_surface_epr_postprocessing
from scgsim.palace.presentation.epr import _show_view, _surface_label
from scgsim.palace.results.epr import epr_result
from scgsim.palace.results.resolve import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
    ResolvedPalaceResult,
)


def _band(index: int, lower: float, upper: float | None) -> dict[str, object]:
    return {
        "parent_surface_id": "pad_top",
        "source_entity_id": "entity_1",
        "source_polygon_id": "polygon_1",
        "origin_segments": [
            {"ring": "exterior", "segment_index": 0, "hole_index": None}
        ],
        "lower_um": lower,
        "upper_um": upper,
        "band_index": index,
        "component_index": 0,
        "precision_um": 0.001,
    }


def _surface_info(
    name: str, phys_group: int, band: dict[str, object] | None = None
) -> dict[str, object]:
    info: dict[str, object] = {
        "phys_group": [phys_group],
        "surface_epr": True,
        "interface_type": "MA",
        "surface_id": name,
        "face_kind": "top",
        "owner_semantic_ids": ["pad"],
        "source_provenance": {"surface_id": name},
        "physical_attribute": {"group": phys_group},
    }
    if band is not None:
        info["surface_distance_band"] = band
    return info


def test_surface_bands_use_native_attributes_without_fork_masks() -> None:
    groups = {
        "boundary_surfaces": {
            "pad_parent": _surface_info("pad_parent", 10),
            "pad_band_0": _surface_info("pad_band_0", 11, _band(0, 0.0, 0.5)),
            "pad_band_1": _surface_info("pad_band_1", 12, _band(1, 0.5, 1.0)),
            "pad_band_2": _surface_info("pad_band_2", 13, _band(2, 1.0, None)),
        }
    }
    specs = {
        "MA": {
            "thickness": 0.02,
            "permittivity": 4.0,
            "loss_tangent": 0.01,
            "required": True,
        }
    }

    rows, index_map = build_surface_epr_postprocessing(groups, specs)

    band_entries = [
        entry for entry in index_map if "surface_distance_band" in entry["metadata"]
    ]
    assert [entry["attributes"] for entry in band_entries] == [[11], [12], [13]]
    assert [row["Attributes"] for row in rows[:3]] == [[11], [12], [13]]
    assert all("Mask" not in rows[entry["index"] - 1] for entry in band_entries)
    assert [entry["metadata"]["surface_distance_band"] for entry in band_entries] == [
        _band(0, 0.0, 0.5),
        _band(1, 0.5, 1.0),
        _band(2, 1.0, None),
    ]
    assert all("Mask" not in row for row in rows)


def _resolved_band_run() -> ResolvedPalaceResult:
    entries = []
    groups = [("pad_parent", None)] + [
        (f"pad_band_{index}", _band(index, lower, upper))
        for index, lower, upper in (
            (0, 0.0, 0.5),
            (1, 0.5, 1.0),
            (2, 1.0, None),
        )
    ]
    for index, (surface_id, band) in enumerate(groups, start=1):
        metadata = {
            "surface_id": surface_id,
            "interface_type": "MA",
            "face_kind": "top",
            "owner_semantic_ids": ["pad"],
            "source_provenance": {"surface_id": surface_id},
            "physical_attribute": {"group": 10 + index},
        }
        if band is not None:
            metadata["surface_distance_band"] = band
        entries.append(
            {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": index,
                "entry_name": surface_id,
                "metadata": metadata,
                "epr_spec": {
                    "schema_version": "scgsim.palace.surface-film.v2",
                    "thickness": 0.02,
                    "film_thickness_m": 2e-8,
                    "model_l0_m": 1e-6,
                    "permittivity": 4.0,
                    "loss_tangent": 0.01,
                    "source": "explicit",
                    "preset": None,
                },
            }
        )
    index_map = {
        "entries": entries,
        "epr_request": {
            "schema_version": "scgsim.palace.epr-selection.v1",
            "surface_interfaces": None,
            "bulk_domain_ids": None,
            "port_names": None,
        },
    }
    config = {"Model": {"L0": 1e-6}}
    receipt = PalaceReturnedReceipt(
        schema="palace-returned-run-receipt.v1",
        schema_version=1,
        handoff_id="handoff-1",
        status="completed",
        exit_code=0,
        solver_exit_code=0,
        tee_exit_code=0,
        route="Route-A",
        problem="Eigenmode",
        identity_verified=True,
        timestamp_utc="2026-10-23T00:00:00Z",
        job_identity={},
        input_hashes=(),
        output_files=(),
        log=None,
        solver_identity={},
    )
    provenance = PalaceProvenance(
        handoff_metadata={"hashes": []},
        run_metadata={},
        resource_record={},
        handoff_archive_manifest={},
        index_map=index_map,
        mesh_manifest={},
        config=config,
        palace_json={},
        returned_receipt={},
    )
    surface_row: dict[str, object] = {"m": 1}
    for index, participation in enumerate((0.25, 0.1, 0.2, 0.3), start=1):
        surface_row[f"p_surf[{index}]"] = participation
        surface_row[f"Q_surf[{index}]"] = 1.0 / (participation * 0.01)
    tables = {
        "eig": ParsedTable(
            "eig",
            Path("eig.csv"),
            ("m", "Re{f} (GHz)"),
            ({"m": 1, "Re{f} (GHz)": 5.0},),
        ),
        "surface-Q": ParsedTable(
            "surface-Q", Path("surface-Q.csv"), tuple(surface_row), (surface_row,)
        ),
        "domain-E": ParsedTable(
            "domain-E",
            Path("domain-E.csv"),
            ("m", "E_elec (J)", "E_cap (J)"),
            ({"m": 1, "E_elec (J)": 1.0, "E_cap (J)": 0.0},),
        ),
    }
    return ResolvedPalaceResult(
        run_dir=Path("run"),
        problem="Eigenmode",
        route="Route-A",
        has_returned_outputs=True,
        status="completed",
        performance=PalacePerformance({}, {}),
        cost=PalaceCost(None, None, None, None, {}, {}, {}, None),
        tables=tables,
        returned_receipt=receipt,
        provenance=provenance,
    )


def test_epr_result_keeps_bands_distinct_from_parent_baseline() -> None:
    result = epr_result(_resolved_band_run())

    surfaces = result.rows[0]["surface"]
    assert [item["evaluation_kind"] for item in surfaces] == [
        "unmasked_baseline",
        "geometric_band",
        "geometric_band",
        "geometric_band",
    ]
    assert [item["surface_distance_band"]["band_index"] for item in surfaces[1:]] == [
        0,
        1,
        2,
    ]
    assert all(item["margin_um"] is None for item in surfaces[1:])

    view = _show_view(result, mode=1)
    assert view["coverage"]["geometric_bands"] == 3
    assert view["coverage"]["known_baseline_subtotal"] == "0.0025"
    assert "[0, 0.5) µm" in _surface_label(surfaces[1])


def _producer_band_fixture(
    *,
    cuts=(0.5, 1.0),
    small=False,
    wall=True,
    constructed_sa=False,
    interface_kind="MA",
):
    """Real source partition -> export metadata -> mesh-info -> Palace preparation.

    Physical attribute numbers are authored transport fixtures, not native mesh
    observations. No OCC/Gmsh or Palace process is created here.
    """
    from scgsim.geometry import (
        GeometryBuildInput,
        LayoutPolygonSpec,
        SemanticEntitySpec,
        SurfaceDistanceBandSpec,
        with_surface_distance_bands,
    )
    from scgsim.geometry.models.topology import SurfacePlanRecord
    from scgsim.geometry.planning.surface_bands import partition_surface_distance_bands
    from scgsim.geometry.construction.export import _surface_band_metadata
    from scgsim.palace.mesh.build import _structured_surface_info

    if constructed_sa:
        specs, children, summaries = _constructed_sa_partition(cuts)
    else:
        outer = (
            ((0, 0), (1, 0), (1, 1), (0, 1))
            if small
            else (
                (0, 0),
                (4, 0),
                (4, 1),
                (8, 1),
                (8, 0),
                (12, 0),
                (12, 4),
                (8, 4),
                (8, 3),
                (4, 3),
                (4, 4),
                (0, 4),
            )
        )
        polygon = LayoutPolygonSpec("polygon_1", "layer", outer, ())
        entity = SemanticEntitySpec(
            "entity_1",
            "substrate",
            "Si",
            "dielectric",
            0,
            "polygon_1",
            polygon_ids=("polygon_1",),
        )
        build = GeometryBuildInput((polygon,), (entity,))
        parents = []
        for parent_id, face_kind in (
            ("pad_top", "top"),
            *((("pad_wall", "wall"),) if wall else ()),
        ):
            ref = {
                "outer_loop": outer,
                "hole_loops": (),
                "z_um": 0,
                "source_polygon_ids": ("polygon_1",),
            }
            if face_kind == "wall":
                ref = {
                    "quad_points": ((0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)),
                    "source_polygon_ids": ("polygon_1",),
                }
            parents.append(
                SurfacePlanRecord(
                    parent_id,
                    "entity_1",
                    "domain_boundary",
                    ref,
                    interface_id="MA__entity_1__vacuum",
                    metadata={
                        "face_kind": face_kind,
                        "source_provenance": {"surface_id": parent_id},
                        "owner_semantic_ids": ("entity_1",),
                    },
                )
            )
        specs = tuple(
            SurfaceDistanceBandSpec(
                parent.surface_id, "entity_1", "polygon_1", cuts, 0.001
            )
            for parent in parents
        )
        build = with_surface_distance_bands(build, specs)
        children, _, summaries = partition_surface_distance_bands(
            build, surfaces=tuple(parents)
        )
    groups = {}
    for attribute, child in enumerate(children, 1):
        record = {
            "surface_id": child.surface_id,
            "metadata": _surface_band_metadata((child,)),
            "face_kind": child.metadata["face_kind"],
            "owner_semantic_ids": child.metadata["owner_semantic_ids"],
            "source_provenance": child.metadata["source_provenance"],
            "physical_attribute": {"group": attribute},
            "route": "B",
            "representation": "solid",
        }
        info = _structured_surface_info(
            name=child.surface_id,
            phys_group=attribute,
            entity_tags=(attribute,),
            record=record,
            interface_type="SA" if constructed_sa else interface_kind,
            surface_epr=True,
        )
        assert "surface_distance_band" not in info
        assert "surface_distance_band" in info["metadata"]
        groups[child.surface_id] = info
    rows, entries = build_surface_epr_postprocessing(
        {"boundary_surfaces": groups},
        {
            "SA" if constructed_sa else interface_kind: {
                "thickness": 0.5,
                "permittivity": 4,
                "loss_tangent": 0.125,
                "required": True,
            }
        },
    )
    return specs, children, summaries, rows, entries


def _resolved_producer_band_run(
    *,
    cuts=(0.5, 1.0),
    small=False,
    wall=True,
    constructed_sa=False,
    interface_kind="MA",
):
    from dataclasses import replace

    specs, children, summaries, rows, entries = _producer_band_fixture(
        cuts=cuts,
        small=small,
        wall=wall,
        constructed_sa=constructed_sa,
        interface_kind=interface_kind,
    )
    original = _resolved_band_run()
    index_map = {
        "entries": entries,
        "epr_request": original.provenance.index_map["epr_request"],
        "surface_distance_bands": list(summaries),
    }
    # Independent exact powers of two. The correction expectation is derived
    # from these authored raw component energies, never the consumer's totals.
    raw_energy = {
        entry["metadata"]["surface_id"]: 2.0 ** (-entry["index"]) for entry in entries
    }
    surface_row = {"m": 1}
    for entry in entries:
        number = entry["index"]
        energy = raw_energy[entry["metadata"]["surface_id"]]
        surface_row[f"p_surf[{number}]"] = energy / 16.0
        surface_row[f"Q_surf[{number}]"] = 1.0 / ((energy / 16.0) * 0.125)
    tables = {
        "eig": ParsedTable(
            "eig",
            Path("eig.csv"),
            ("m", "Re{f} (GHz)"),
            ({"m": 1, "Re{f} (GHz)": 4.0},),
        ),
        "surface-Q": ParsedTable(
            "surface-Q", Path("surface-Q.csv"), tuple(surface_row), (surface_row,)
        ),
        "domain-E": ParsedTable(
            "domain-E",
            Path("domain-E.csv"),
            ("m", "E_elec (J)", "E_cap (J)"),
            ({"m": 1, "E_elec (J)": 8.0, "E_cap (J)": 8.0},),
        ),
    }
    output = {
        "path": "results/palace/palace.json",
        "sha256": "a" * 64,
        "bytes": 0,
        "present": True,
    }
    # Typed table-reader fixture only; no strict completed-run/native proof is asserted.
    resolved = replace(
        original,
        tables=tables,
        returned_receipt=replace(original.returned_receipt, output_files=(output,)),
        provenance=replace(
            original.provenance,
            index_map=index_map,
            palace_json={"Problem": {"Iteration": 2}},
        ),
    )
    return resolved, specs, children, summaries, rows, entries, raw_energy


def test_real_partition_export_metadata_reaches_palace_without_native_masks():
    specs, children, summaries, rows, entries = _producer_band_fixture()
    assert len(specs) == 2
    assert len(entries) == len(children)
    assert all("Mask" not in row for row in rows)
    assert [entry["metadata"]["surface_distance_band"] for entry in entries] == [
        child.metadata["surface_distance_band"]
        for child in sorted(children, key=lambda item: item.surface_id)
    ]
    assert any(summary["component_count"] > 1 for summary in summaries)
    assert {entry["metadata"]["face_kind"] for entry in entries} == {"top", "wall"}


def test_real_producer_bands_reach_table_reader_as_distinct_components():
    resolved, _, children, _, _, _, _ = _resolved_producer_band_run()
    parsed = epr_result(resolved)
    surfaces = parsed.rows[0]["surface"]
    assert len(surfaces) == len(children)
    assert all(surface["evaluation_kind"] == "geometric_band" for surface in surfaces)
    assert all(surface["margin_um"] is None for surface in surfaces)
    assert len({surface["surface_id"] for surface in surfaces}) == len(children)
    assert parsed.rows[0]["normalization_energy_j"] == 16


def _constructed_sa_partition(cuts):
    """Public solid-domain planning; no GDS polygon or native mesh operation."""
    from scgsim.geometry import (
        ApproximationProfile,
        GeometryBuildInput,
        SemanticEntitySpec,
        SurfaceDistanceBandSpec,
        interpret_physical_geometry,
        surface_distance_band_catalogue,
        with_surface_distance_bands,
    )
    from scgsim.geometry.compiler import build_route_construction_plan

    def domain(name, material, kind, lower, upper):
        return SemanticEntitySpec(
            name,
            "solution_region",
            material,
            kind,
            0,
            "polygon",
            geometry={
                "outer_loop": ((-2, -2), (6, -2), (6, 6), (-2, 6)),
                "z_min_um": lower,
                "z_max_um": upper,
                "thickness_um": upper - lower,
                "domain_bounds_um": {
                    "x_min_um": -2,
                    "x_max_um": 6,
                    "y_min_um": -2,
                    "y_max_um": 6,
                    "z_min_um": lower,
                    "z_max_um": upper,
                },
            },
        )

    entities = (
        domain("substrate", "Si", "dielectric", -2, 0),
        domain("vacuum", "vacuum_material", "vacuum", 0, 6),
    )
    source = GeometryBuildInput((), entities)
    stack = {
        "layers": [],
        "metadata": {},
        "materials": {
            "Si": {"kind": "dielectric", "permittivity": 4},
            "vacuum_material": {"kind": "vacuum", "permittivity": 1},
        },
        "solution_regions": {
            entity.semantic_id: {
                "semantic_id": entity.semantic_id,
                "geometry": dict(entity.geometry),
                "metadata": {},
            }
            for entity in entities
        },
    }
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    plan = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    catalogue = surface_distance_band_catalogue(plan, build_input=effective)
    parent = next(
        row for row in catalogue if row["source_entity_ids"] == ("substrate", "vacuum")
    )
    assert parent["source_polygon_ids"] == ()
    assert {tuple(binding.items()) for binding in parent["source_bindings"]} == {
        (("source_entity_id", "substrate"), ("source_polygon_id", None)),
        (("source_entity_id", "vacuum"), ("source_polygon_id", None)),
    }
    spec = SurfaceDistanceBandSpec(
        parent["parent_surface_id"], "substrate", None, cuts, 0.001
    )
    result = build_route_construction_plan(
        with_surface_distance_bands(effective, (spec,)),
        route="B",
        representation_context=physical.context,
    )
    children = tuple(
        surface
        for surface in result.surfaces
        if surface.metadata.get("surface_distance_band")
    )
    return (spec,), children, result.metadata["surface_distance_bands"]


def test_constructed_sa_null_polygon_survives_producer_and_table_parser():
    import pytest
    from dataclasses import replace
    from scgsim.palace.results.epr import PalaceEprResult

    resolved, specs, children, summaries, _, entries, _ = _resolved_producer_band_run(
        constructed_sa=True
    )
    assert specs[0].source_polygon_id is None
    expected_owners = ("substrate", "vacuum")
    assert all(
        tuple(child.metadata["owner_semantic_ids"]) == expected_owners
        for child in children
    )
    assert all(
        tuple(entry["metadata"]["owner_semantic_ids"]) == expected_owners
        for entry in entries
    )
    assert all(
        "source_polygon_id" in row and row["source_polygon_id"] is None
        for row in summaries
    )
    assert all(
        child.metadata["surface_distance_band"]["source_polygon_id"] is None
        for child in children
    )
    parsed = epr_result(resolved)
    assert all(row["interface_kind"] == "SA" for row in parsed.rows[0]["surface"])
    assert all(
        tuple(row["owner_semantic_ids"]) == expected_owners
        for row in parsed.rows[0]["surface"]
    )
    assert all(
        row["surface_distance_band"]["source_polygon_id"] is None
        for row in parsed.rows[0]["surface"]
    )
    assert (
        PalaceEprResult.from_payload(parsed.to_payload()).to_payload()
        == parsed.to_payload()
    )
    # A missing binding is distinct from an explicit null, at the real table decoder.
    from copy import deepcopy

    broken = deepcopy(entries)
    del broken[0]["metadata"]["surface_distance_band"]["source_polygon_id"]
    with pytest.raises(ValueError, match="missing fields.*source_polygon_id"):
        epr_result(
            replace(
                resolved,
                provenance=replace(
                    resolved.provenance,
                    index_map={**resolved.provenance.index_map, "entries": broken},
                ),
            )
        )
