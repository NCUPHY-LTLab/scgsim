"""Protect boundary collection order in GeometryPlan snapshots."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
)
from scgsim.geometry.source import plan as source_plan


def _make_plan(monkeypatch, semantics_by_path):
    """Build a plan double that reaches only its source-boundary collection."""
    root = SimpleNamespace(name="root-cell", kcl=SimpleNamespace(dbu=0.001))
    root.write_gds = lambda path: Path(path).write_bytes(b"source")
    child = SimpleNamespace(name="child-cell")
    plan = source_plan.GeometryPlan.__new__(source_plan.GeometryPlan)
    plan.root_id = "root"
    vars(plan).update(
        {
            "_component": root,
            "_grid_size_um": None,
            "_nets": {},
            "_vacuum_region": None,
            "_instances": {
                "root": (root, None, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
                "root/child": (child, None, (1.0, 0.0, 0.0, 1.0, 2.0, 3.0)),
            },
            "_boundary_curves": (),
            "_boundary_reconstruction": (),
            "_lumped_supports": (),
            "_rectangular_junctions": (),
        }
    )

    monkeypatch.setattr(
        source_plan.GeometryPlan, "_validate_registered_sources", lambda _self: None
    )
    monkeypatch.setattr(
        source_plan.GeometryPlan,
        "_stack",
        lambda _self: {
            "layers": [],
            "metadata": {
                "port_sheet_source_layers": [],
                "geometry_grid": {},
                "source_dbu_um": 1.0,
                "unquantized_source_sha256": "source-hash",
            },
        },
    )
    monkeypatch.setattr(
        source_plan,
        "_local_semantics",
        lambda _cell, path: semantics_by_path[path],
    )
    monkeypatch.setattr(
        source_plan,
        "_source_recipe_map",
        lambda _runner, jobs: list(jobs),
    )
    monkeypatch.setattr(
        source_plan,
        "build_gds_stack_geometry_input",
        lambda **_kwargs: GeometryBuildInput(
            (),
            (),
            metadata={
                "geometry_grid": {},
                "source_dbu_um": 1.0,
                "unquantized_source_sha256": "source-hash",
            },
        ),
    )
    monkeypatch.setattr(source_plan, "_stage", lambda _name: nullcontext())
    fake_gdstk = ModuleType("gdstk")
    vars(fake_gdstk)["read_gds"] = lambda _path: SimpleNamespace(cells=[])
    monkeypatch.setitem(sys.modules, "gdstk", fake_gdstk)
    return plan


def test_prepare_collects_both_boundary_record_types_once_per_instance(monkeypatch):
    """Preserve mixed instance order and consume each boundary iterator once."""
    consumed = {
        "root_curves": 0,
        "root_reconstruction": 0,
        "child_curves": 0,
        "child_reconstruction": 0,
    }

    def one_shot(key, records):
        def values():
            consumed[key] += 1
            yield from records

        return values()

    semantics = {
        "root": {
            "boundary_curves": one_shot(
                "root_curves",
                (BoundaryCurveChainSpec("root-edge", "metal", (0, 0), ()),),
            ),
            "boundary_reconstruction": one_shot(
                "root_reconstruction",
                (GdsBoundaryReconstructionSpec("root-arc", "metal", (0, 0), "circle"),),
            ),
        },
        "root/child": {
            "boundary_curves": one_shot(
                "child_curves",
                (BoundaryCurveChainSpec("child-edge", "metal", (0, 0), ()),),
            ),
            "boundary_reconstruction": one_shot(
                "child_reconstruction",
                (
                    GdsBoundaryReconstructionSpec(
                        "child-arc", "metal", (0, 0), "circle"
                    ),
                ),
            ),
        },
    }
    plan = _make_plan(monkeypatch, semantics)

    result = plan.prepare().geometry_input

    assert [record.boundary_id for record in result.boundary_curves] == [
        "root/root-edge",
        "root/child/child-edge",
    ]
    assert [record.boundary_id for record in result.boundary_reconstruction] == [
        "root/root-arc",
        "root/child/child-arc",
    ]
    assert all(count == 1 for count in consumed.values())


def test_prepare_preserves_per_instance_boundary_error_priority(monkeypatch):
    """Raise the first instance reconstruction error before later curve errors."""
    semantics = {
        "root": {
            "boundary_curves": (),
            "boundary_reconstruction": (
                {
                    "boundary_id": "bad-arc",
                    "entity_id": "metal",
                    "selector_point_um": (0, 0),
                    "kind": "circle",
                    "role": "unknown",
                },
            ),
        },
        "root/child": {
            "boundary_curves": (object(),),
            "boundary_reconstruction": (),
        },
    }
    plan = _make_plan(monkeypatch, semantics)

    with pytest.raises(ValueError, match="source boundary role"):
        plan.prepare()
