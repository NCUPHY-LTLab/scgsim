"""Check that a layout can be modeled, by running the modeling stages of ``mesh()`` and stopping before meshing."""

from __future__ import annotations

from collections.abc import Mapping
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.geometry.source.plan import GeometryPlan

from .eigenmode import EigenmodeSim

_ROOT_ID = "chip"
# The geometry build never reads a port's inductance, but ``add_port`` requires a positive one.
_PORT_INDUCTANCE_H = 1.0e-9


def simulation_drc(
    component: Any,
    *,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Any] | None = None,
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Return ``None`` when SCGSim can model ``component``; otherwise raise the error the modeling stage raises.

    This builds the ``GeometryPlan`` and ``EigenmodeSim`` a user would build, runs the preparation and the
    SGB geometry build that ``EigenmodeSim.mesh()`` runs, and stops before meshing.  Nothing is added to
    the error: the type and message are the ones ``mesh()`` gives.  The geometry is written to a temporary
    folder that is removed.

    The settings are fixed: Route B, every authored port, no nets and zero coupon padding.  The plan's
    root is ``"chip"``, so ``instances`` keys read ``"chip/<name>"``.  With ``surface_epr_specs`` the bands
    are cut as ``set_surface_epr(representation="B", specs=...)`` would cut them.

    A pass means the layout can be modeled with these settings.  It does not promise that a simulation
    with nets, Electrostatic terminals, Route A, junctions, lumped supports or a vacuum region will model;
    run that simulation's own ``mesh()`` for that.

    Args:
        component: The layout Component under check.
        layer_stack: The PDK layer stack, as ``GeometryPlan`` takes it.
        material_records: The PDK material records, as ``GeometryPlan`` takes them.
        instances: ``{"chip/device": reference, ...}`` for an assembly, as ``GeometryPlan.register_instance``.
        surface_epr_specs: The specs ``set_surface_epr`` takes; their margins become bands.
    """
    plan = GeometryPlan(
        component,
        root_id=_ROOT_ID,
        layer_stack=layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in (instances or {}).items():
        plan.register_instance(path, reference)
    snapshot = plan.prepare()

    sim = EigenmodeSim()
    sim.set_plan(snapshot)
    for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
        sim.add_port(
            port["name"],
            layer=port["target_layer"],
            layout_sheet=True,
            inductance=_PORT_INDUCTANCE_H,
        )
    if surface_epr_specs is not None:
        sim.set_surface_epr(representation="B", specs=surface_epr_specs)
    with TemporaryDirectory(prefix="scgsim-simulation-drc-") as folder:
        # build_route_geometry requires an empty folder; the temporary one is, and goes away with the check.
        sim.build_geometry(folder)
