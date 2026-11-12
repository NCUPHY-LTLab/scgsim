"""Check exact curved construction's native lifecycle in isolated processes."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_geometry_context_propagation import conductor_input

from scgsim.geometry import ApproximationProfile, interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    SourceCurveSpec,
)

# The first arc of the "no_circle" boundary runs along y = 2 um, so no circle passes through it.
NO_CIRCLE = (
    "boundary (Entity 'metal', layer metal) curve 0: the three arc points are collinear, "
    "so they define no circle: its start (0.0, 2.0) um, middle point (2.0, 2.0) um and end "
    "(4.0, 2.0) um lie on one line. Fix the layout:"
)


@pytest.mark.parametrize("initialized", (False, True))
@pytest.mark.parametrize(
    "outcome", ("success", "no_circle", "occ_error", "binding_error")
)
def test_curved_construction_preserves_native_lifecycle(
    initialized: bool, outcome: str
):
    """Preserve caller models and terminal state on success and both failure stages."""
    pytest.importorskip("gmsh")
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        (source, environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, __file__, outcome, str(int(initialized))],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _verify_lifecycle(outcome: str, initialized: bool) -> None:
    # Native state is process-global, so every case owns a fresh subprocess.
    import gmsh  # noqa: PLC0415 - optional native geometry dependency

    effective, context = _construction_input(outcome)
    if initialized:
        gmsh.initialize(readConfigFiles=False)
        gmsh.model.add("caller-model")
        gmsh.model.occ.addPoint(17, 23, 31)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("General.Terminal", 0)
        models = gmsh.model.list()
        terminal = gmsh.option.getNumber("General.Terminal")
        points = gmsh.model.getEntities(0)
    try:
        if outcome == "no_circle":
            # The binding refuses the arc before the native model opens, in SGB's words.
            with pytest.raises(ValueError, match=re.escape(NO_CIRCLE)):
                build_route_construction_plan(
                    effective, route="B", representation_context=context
                )
        elif outcome == "occ_error":
            with pytest.raises(Exception, match="Number of BSpline weights"):
                build_route_construction_plan(
                    effective, route="B", representation_context=context
                )
        elif outcome == "binding_error":
            with pytest.raises(
                ValueError, match="requires explicit original segment indices"
            ):
                build_route_construction_plan(
                    effective, route="B", representation_context=context
                )
        else:
            plan = build_route_construction_plan(
                effective, route="B", representation_context=context
            )
            assert plan.surfaces and plan.volumes
        assert bool(gmsh.isInitialized()) == initialized
        if initialized:
            assert gmsh.model.getCurrent() == "caller-model"
            assert gmsh.model.list() == models
            assert gmsh.model.getEntities(0) == points
            assert gmsh.option.getNumber("General.Terminal") == terminal
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _construction_input(outcome: str):
    source, stack = conductor_input()
    if outcome == "binding_error":
        source = replace(
            source,
            boundary_reconstruction=(
                GdsBoundaryReconstructionSpec("round", "metal", (2, 2), "circular_arc"),
            ),
        )
    else:
        first_arc = (
            ((0, 2), (2, 2), (4, 2))
            if outcome == "no_circle"
            else ((0, 2), (2, 0), (4, 2))
        )
        first_curve = (
            SourceCurveSpec("bspline", first_arc, degree=2, weights=(1.0,))
            if outcome == "occ_error"
            else SourceCurveSpec("circular_arc", first_arc)
        )
        source = replace(
            source,
            boundary_curves=(
                BoundaryCurveChainSpec(
                    "boundary",
                    "metal",
                    (2, 2),
                    (
                        first_curve,
                        SourceCurveSpec("circular_arc", ((4, 2), (2, 4), (0, 2))),
                    ),
                ),
            ),
        )
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    return effective, physical.context


if __name__ == "__main__":
    _verify_lifecycle(sys.argv[1], bool(int(sys.argv[2])))
