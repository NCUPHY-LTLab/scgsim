"""Index map v3 and mesh manifest v2: producer round trip through the readers, and rejection of older schemas.

Everything here is file-only: the sidecars are SGB's, the group dict, manifest, config and index map are the
producers' own, and the readers see them as JSON files.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from palace_sgb_support import (
    MATERIALS,
    effective_input,
    sidecar_of,
    surface_epr_specs,
)

from scgsim.palace.mesh.insets import inset_requests
from scgsim.palace.mesh.manifest import MANIFEST_SCHEMA_VERSION, build_mesh_manifest
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.preparation.config import (
    LayoutPortBinding,
    TerminalBinding,
    build_eigenmode_config,
    build_electrostatic_config,
    configure_numerical_controls,
)
from scgsim.palace.presentation.reports import inspect_run_trustworthiness
from scgsim.palace.presentation.surface_figures import (
    surface_margin_participation_figure,
    surface_ranking_figure,
)
from scgsim.palace.results.artifacts import read_json
from scgsim.palace.results.epr import epr_result, reanalyze_epr
from scgsim.palace.results.epr_model import PalaceEprResult
from scgsim.palace.results.index_contract import (
    INDEX_MAP_SCHEMA_VERSION,
    RouteConsistency,
    require_index_map_schema,
    require_manifest_schema,
    validate_config_index_correspondence,
)
from scgsim.palace.results.models import ParsedTable, ResolvedRunData
from scgsim.palace.results.surface_readers import (
    SurfaceEprReader,
    SurfaceMarginEprReader,
    read_surface_bindings,
    read_surface_margin_bindings,
)
from scgsim.palace.results.tables import count_index_entries

NETS = {"A": ("left", "right"), "B": ("P", "G")}
PORTS = {
    "A": LayoutPortBinding(
        1, "chip/o_junction_lumped", "metal", "port", (1.0, 0.0, 0.0), 1e-9
    ),
    "B": LayoutPortBinding(1, "P1", "metal", "9/0", (1.0, 0.0, 0.0), 1e-9),
}
THIN_FILM = {
    "schema_version": 1,
    "variant": "substrate_face",
    "display_label": "A_PRIME",
}
MESHING = {"recipe": {"name": "feature", "version": "1"}, "statistics": {}}


class Run:
    """The artifacts of one file-only run: group dict, manifest, config and index map as read from disk."""

    def __init__(self, route: str, problem: str, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        specs = surface_epr_specs(MA=[0.5, 1.0], MS=[0.5], SA=None)
        effective = effective_input(route, directory)
        unbanded = sidecar_of(effective, route, directory)
        self.sidecar = sidecar_of(
            effective, route, directory / "banded", inset_requests(specs, unbanded)
        )
        self.groups = groups_from_sidecar(self.sidecar)
        self.route, self.problem = route, problem
        thin_film = THIN_FILM if route == "A" else None
        self.thin_film = thin_film
        built = (
            self._eigenmode(specs)
            if problem == "Eigenmode"
            else self._electrostatic(specs)
        )
        config, entries, extra = built
        manifest = build_mesh_manifest(
            groups=self.groups,
            sidecar=self.sidecar,
            sidecar_file={"path": "geometry/sidecar.json", "sha256": "0" * 64},
            mesh_name="palace.msh",
            route=route,
            route_a_thin_film=thin_film,
            meshing=MESHING,
        )
        files = {
            "metadata/mesh_manifest.json": manifest,
            "metadata/palace_index_map.json": {
                "schema_version": INDEX_MAP_SCHEMA_VERSION,
                "entries": entries,
                **extra,
            },
            "config.json": config,
        }
        for name, payload in files.items():
            path = directory / "run" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        root = directory / "run"
        self.root = root
        self.manifest = read_json(root / "metadata/mesh_manifest.json")
        self.index = read_json(root / "metadata/palace_index_map.json")
        self.config = read_json(root / "config.json")

    def _electrostatic(self, specs):
        first, second = NETS[self.route]
        result = build_electrostatic_config(
            groups=self.groups,
            terminals=[TerminalBinding("t", first)],
            ground_net_ids=[second],
            materials=MATERIALS,
            save_fields=0,
            unassigned_conductor_policy="error",
            exterior_boundary_policy="none",
            mesh_path=Path("palace.msh"),
            surface_epr_specs=specs,
            numerical=configure_numerical_controls(),
        )
        return (
            result.config,
            [
                *result.terminal_index_map,
                *result.domain_energy_index_map,
                *result.surface_epr_index_map,
            ],
            {"ground_boundary": result.ground_boundary_resolution},
        )

    def _eigenmode(self, specs):
        result = build_eigenmode_config(
            groups=self.groups,
            ports=[PORTS[self.route]],
            materials=MATERIALS,
            surface_epr_specs=specs,
            numerical=configure_numerical_controls(amr_max_passes=0),
            num_modes=3,
            target_hz=None,
            eigenmode_tolerance=1e-6,
            save_fields=0,
            mesh_path=Path("palace.msh"),
        )
        return result.config, result.index_entries, {}

    def consistency(self, **overrides) -> RouteConsistency:
        """The route consistency check over this run's metadata."""
        metadata = {
            "route": self.route,
            "problem": self.problem,
            "route_a_thin_film": self.thin_film,
        }
        fields = {
            "route": self.route,
            "run_metadata": metadata,
            "handoff_metadata": metadata,
            "mesh_manifest": self.manifest,
            "index_map": self.index,
            "index_counts": count_index_entries(self.index),
            "problem": self.problem,
        }
        return RouteConsistency(**{**fields, **overrides})


@pytest.fixture(
    params=[
        ("A", "Electrostatic"),
        ("B", "Electrostatic"),
        ("A", "Eigenmode"),
        ("B", "Eigenmode"),
    ],
    ids=str,
)
def run(request, tmp_path):
    """A producer-made run of each route and problem, read back from files."""
    route, problem = request.param
    return Run(route, problem, tmp_path)


def _surface_q(run: Run, path: Path) -> None:
    """A surface-Q table with one column pair for every Dielectric row, as Palace writes it."""
    count = count_index_entries(run.index)["surface"]
    name = "m" if run.problem == "Eigenmode" else "i"
    header = ",".join(
        [name, *(f"p_surf[{k}],Q_surf[{k}]" for k in range(1, count + 1))]
    )
    cells = ",".join(f"{0.01 * k:g},+inf" for k in range(1, count + 1))
    path.write_text(f"{header}\n1,{cells}\n", encoding="utf-8")


def test_a_producer_made_index_map_round_trips_through_the_readers(run, tmp_path):
    """Index map v3 and manifest v2 as the producers write them pass every reader contract."""
    assert run.index["schema_version"] == INDEX_MAP_SCHEMA_VERSION == 3
    assert run.manifest["schema_version"] == 2
    run.consistency().validate()
    validate_config_index_correspondence(
        run.config, run.index, count_index_entries(run.index), run.problem
    )
    wholes = read_surface_bindings(run.index)
    margins = read_surface_margin_bindings(run.index)
    assert wholes and margins
    assert {b["interface_type"] for b in wholes} == {"MA", "MS", "SA"}
    assert all(b["baseline_index"] in {w["index"] for w in wholes} for b in margins)
    assert {m["margin_um"] for m in margins} == {0.5, 1.0}
    assert all(m["bands"] for m in margins)

    table = tmp_path / "surface-Q.csv"
    _surface_q(run, table)
    kwargs = {
        "problem": run.problem,
        "pass_index": 0,
        "source": "final",
        "expected_rows": 1,
        "surface_count": len(wholes) + len(margins),
    }
    (series,) = SurfaceEprReader(
        table, frequencies_ghz=None, bindings=wholes, **kwargs
    ).read()
    (margin_series,) = SurfaceMarginEprReader(table, bindings=margins, **kwargs).read()
    assert [r.index for r in series.records] == [b["index"] for b in wholes]
    assert [r.index for r in margin_series.records] == [b["index"] for b in margins]
    assert all(r.entities and r.rows and r.attributes for r in series.records)
    assert all(r.participation == 0.01 * r.index for r in series.records)


def test_the_electrostatic_index_map_keeps_the_ground_evidence(tmp_path):
    """Physical and exterior Ground evidence is part of index map v3, and is checked against the config."""
    run = Run("B", "Electrostatic", tmp_path)
    ground = run.index["ground_boundary"]
    assert ground["physical_net_ids"] == ["G"] and ground["conductor_component_ids"]
    assert (
        ground["union_attributes"] == run.config["Boundaries"]["Ground"]["Attributes"]
    )
    run.config["Boundaries"]["Ground"]["Attributes"] = [999]
    with pytest.raises(ValueError, match="Ground"):
        validate_config_index_correspondence(
            run.config, run.index, count_index_entries(run.index), run.problem
        )


@pytest.mark.parametrize("recorded", [1, 2, None, "3"])
def test_old_index_maps_are_rejected_naming_the_recorded_schema(tmp_path, recorded):
    """Index maps of schema 1 and 2 (and anything else) are not read, and the message says which."""
    run = Run("B", "Electrostatic", tmp_path)
    old = {**run.index, "schema_version": recorded}
    with pytest.raises(
        ValueError, match=rf"schema_version {recorded!r} is not readable"
    ):
        require_index_map_schema(old)
    with pytest.raises(ValueError, match=rf"schema_version {recorded!r}"):
        run.consistency(index_map=old).validate()


@pytest.mark.parametrize("recorded", [1, 3, None])
def test_old_manifests_are_rejected_naming_the_recorded_schema(tmp_path, recorded):
    """A mesh manifest of schema 1 (or any other than 2) is not read, and the message says which."""
    run = Run("B", "Eigenmode", tmp_path)
    old = {**run.manifest, "schema_version": recorded}
    with pytest.raises(
        ValueError, match=rf"schema_version {recorded!r} is not readable"
    ):
        require_manifest_schema(old)
    with pytest.raises(ValueError, match=rf"schema_version {recorded!r}"):
        run.consistency(mesh_manifest=old).validate()


def test_a_run_folder_of_older_schemas_is_not_inspected(tmp_path):
    """The trust report reads the folder's own files and refuses a manifest or index map of an old schema."""
    run = Run("B", "Eigenmode", tmp_path)
    handoff = {"problem": "Eigenmode", "route": "B", "profile": "direct-local"}
    (run.root / "metadata/palace_handoff_metadata.json").write_text(json.dumps(handoff))
    (run.root / "metadata/palace_index_map.json").write_text(
        json.dumps({**run.index, "schema_version": 2})
    )
    with pytest.raises(ValueError, match="palace index map schema_version 2"):
        inspect_run_trustworthiness(run.root)
    (run.root / "metadata/palace_index_map.json").write_text(json.dumps(run.index))
    (run.root / "metadata/mesh_manifest.json").write_text(
        json.dumps({**run.manifest, "schema_version": 1})
    )
    with pytest.raises(ValueError, match="mesh manifest schema_version 1"):
        inspect_run_trustworthiness(run.root)


def test_an_old_detached_epr_result_is_rejected_naming_its_schema():
    """The detached EPR record has one schema too."""
    with pytest.raises(ValueError, match=r"epr-result\.v1"):
        PalaceEprResult.from_payload(
            {
                "schema_version": "scgsim.palace.epr-result.v1",
                "rows": [],
                "provenance": {},
            }
        )


def test_index_attributes_must_be_the_manifest_attributes(run):
    """The index entries name attributes and groups that the mesh manifest records for the same mesh."""
    entry = next(e for e in run.index["entries"] if e["section"].endswith("Dielectric"))
    broken = deepcopy(run.index)
    target = next(
        e
        for e in broken["entries"]
        if e["index"] == entry["index"] and e["section"] == entry["section"]
    )
    target["metadata"]["rows"][0]["group"] = "somewhere else"
    target["physical_names"] = sorted(
        {row["group"] for row in target["metadata"]["rows"]}
    )
    with pytest.raises(ValueError, match="manifest"):
        run.consistency(index_map=broken).validate()
    manifest = deepcopy(run.manifest)
    manifest["groups"] = [g for g in manifest["groups"] if g["dimension"] != 3]
    with pytest.raises(ValueError, match="manifest"):
        run.consistency(mesh_manifest=manifest).validate()


def test_a_margin_must_bind_its_whole_surface_and_bands(run):
    """A margin entry that loses its inset, its bands or its place in the margin sequence is rejected."""
    margin_position = next(
        i
        for i, e in enumerate(run.index["entries"])
        if e["section"].endswith("Dielectric") and "inset" in e
    )
    for change, message in (
        (lambda e: e["inset"].__setitem__("bands", [99]), "band rows"),
        (lambda e: e["inset"].__setitem__("margin_um", 9.0), "margin"),
        (lambda e: e.__setitem__("baseline_index", e["index"]), "whole-surface entry"),
    ):
        broken = deepcopy(run.index)
        change(broken["entries"][margin_position])
        with pytest.raises(ValueError, match=message):
            validate_config_index_correspondence(
                run.config, broken, count_index_entries(broken), run.problem
            )


def test_internal_boundary_cracking_follows_the_shared_face_rows(tmp_path):
    """Route A sheets share a face between their MA and MS rows, so the config must crack it; Route B must not."""
    sheets = Run("A", "Electrostatic", tmp_path / "a")
    assert sheets.config["Model"]["CrackInternalBoundaryElements"] is True
    del sheets.config["Model"]["CrackInternalBoundaryElements"]
    with pytest.raises(ValueError, match="CrackInternalBoundaryElements"):
        validate_config_index_correspondence(
            sheets.config,
            sheets.index,
            count_index_entries(sheets.index),
            "Electrostatic",
        )
    plain = Run("B", "Electrostatic", tmp_path / "b")
    assert "CrackInternalBoundaryElements" not in plain.config["Model"]


@pytest.fixture(params=["A", "B"])
def eigenmode_run(request, tmp_path):
    """A producer-made Eigenmode run of each route."""
    return Run(request.param, "Eigenmode", tmp_path)


def _resolved_eigenmode(run: Run) -> ResolvedRunData:
    """A resolved Eigenmode run whose native tables carry one mode, with a value for every index entry."""
    sections = {}
    for entry in run.index["entries"]:
        sections.setdefault(entry["section"], []).append(entry["index"])
    surfaces = sections["Boundaries.Postprocessing.Dielectric"]
    domains = sections["Domains.Postprocessing.Energy"]

    def table(name: str, row: dict) -> ParsedTable:
        return ParsedTable(name, Path(f"{name}.csv"), tuple(row), (row,))

    tables = {
        "eig": table("eig", {"m": 1, "Re{f} (GHz)": 5.0}),
        "domain-E": table(
            "domain-E",
            {
                "m": 1,
                "E_elec (J)": 2.0,
                "E_cap (J)": 2.0,
                **{f"p_elec[{k}]": 0.1 * k for k in domains},
            },
        ),
        "port-EPR": table("port-EPR", {"m": 1, "p[1]": -0.5}),
        "surface-Q": table(
            "surface-Q", {"m": 1, **{f"p_surf[{k}]": 0.01 * k for k in surfaces}}
        ),
    }
    receipt = SimpleNamespace(
        handoff_id="h" * 8,
        schema="receipt",
        timestamp_utc="2026-01-01T00:00:00Z",
        solver_identity={},
        output_files=(),
        input_hashes=(),
    )
    provenance = SimpleNamespace(
        index_map=run.index, config=run.config, palace_json={"Problem": {}}
    )
    return ResolvedRunData(
        run_dir=run.root,
        problem="Eigenmode",
        route=run.route,
        has_returned_outputs=True,
        status="completed",
        performance=None,
        cost=None,
        tables=tables,
        returned_receipt=receipt,
        provenance=provenance,
    )


def test_eigenmode_epr_reads_every_row_of_the_index_map(eigenmode_run):
    """Whole-surface and margin rows come out of the one surface-Q table, bound to rows, Entities and Nets."""
    result = epr_result(_resolved_eigenmode(eigenmode_run))
    (row,) = result.rows
    kinds = [item["evaluation_kind"] for item in row["surface"]]
    assert kinds.count("requested_margin") == len(
        read_surface_margin_bindings(eigenmode_run.index)
    )
    assert kinds.count("unmasked_baseline") == len(
        read_surface_bindings(eigenmode_run.index)
    )
    margin = next(
        i for i in row["surface"] if i["evaluation_kind"] == "requested_margin"
    )
    assert margin["margin_um"] in (0.5, 1.0) and margin["bands"]
    assert margin["group_id"] == f"palace_surface_{margin['baseline_index']}"
    assert all(item["entities"] and item["rows"] for item in row["surface"])
    (port,) = row["junction"]
    assert len(port["host_entity_ids"]) == 2
    again = PalaceEprResult.from_payload(json.loads(json.dumps(result.to_payload())))
    assert again.rows == result.rows
    assert reanalyze_epr(result, surface_defaults={"MA": {"loss_tangent": 0.01}}).rows


@pytest.mark.parametrize(
    ("file", "key", "message"),
    [
        ("metadata/mesh_manifest.json", "manifest", "mesh manifest schema_version 1"),
        (
            "metadata/palace_index_map.json",
            "index",
            "palace index map schema_version 2",
        ),
    ],
)
def test_the_geometry_preview_does_not_read_older_schemas(tmp_path, file, key, message):
    """The semantic preview binds the manifest and index map, so it refuses the old ones too."""
    pytest.importorskip("meshio")
    pytest.importorskip("pyvista")
    from scgsim.visualization.loaders.palace import (  # noqa: PLC0415 -- optional visualization dependency.
        inspect_palace_geometry,
    )

    run = Run("B", "Eigenmode", tmp_path)
    (run.root / "palace.msh").write_text("", encoding="utf-8")
    payload = run.manifest if key == "manifest" else run.index
    old = {**payload, "schema_version": 1 if key == "manifest" else 2}
    (run.root / file).write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        inspect_palace_geometry(run.root)


def test_margin_rows_are_plotted_against_their_whole_surface(eigenmode_run, tmp_path):
    """The margin figure puts each margin row of a whole surface on one line, with the whole surface at zero."""
    pytest.importorskip("plotly")
    run = eigenmode_run
    table = tmp_path / "surface-Q.csv"
    _surface_q(run, table)
    wholes = read_surface_bindings(run.index)
    margins = read_surface_margin_bindings(run.index)
    kwargs = {
        "problem": "Eigenmode",
        "pass_index": 0,
        "source": "final",
        "expected_rows": 1,
        "surface_count": len(wholes) + len(margins),
    }
    (series,) = SurfaceEprReader(
        table, frequencies_ghz=None, bindings=wholes, **kwargs
    ).read()
    (margin_series,) = SurfaceMarginEprReader(table, bindings=margins, **kwargs).read()
    figure = surface_margin_participation_figure(margin_series, series, "MA", "light")
    assert figure is not None
    lines = [t for t in figure.data if t.mode == "lines+markers"]
    assert lines and all(list(t.x) == sorted(t.x) for t in lines)
    assert any(t.mode == "markers" and list(t.x) == [0.0] for t in figure.data)
    assert surface_ranking_figure(series, None, "light") is not None


def test_the_preview_loader_pins_the_same_schema_numbers():
    """The loader cannot import Palace, so it repeats the schema numbers; they must not drift."""
    from scgsim.visualization.loaders import palace as loader  # noqa: PLC0415 -- optional visualization dependency, imported where it is pinned.

    assert loader.MANIFEST_SCHEMA_VERSION == MANIFEST_SCHEMA_VERSION
    assert loader.INDEX_MAP_SCHEMA_VERSION == INDEX_MAP_SCHEMA_VERSION
