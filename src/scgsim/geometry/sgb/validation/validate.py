"""Structural validation of a built model: fail closed, never disabled (design brief section 11, stage S9).

Owns running every invariant over the labelled arrangements, the topology and the route's
group plan, and turning any failure into a :class:`ValidationError`.  A production build
always validates before anything is written; there is no switch to turn it off.  Tests that
need to look at a *failing* report call :func:`validate_model` themselves and decide.
"""

from __future__ import annotations

from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.derivation.labelled import LabelledArrangements
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.routes.groups import RoutedModel
from scgsim.geometry.sgb.validation import arrangements
from scgsim.geometry.sgb.validation import topology as topology_checks
from scgsim.geometry.sgb.validation.report import ValidationReport


def validate_model(
    labelled: LabelledArrangements,
    topology: Topology,
    routed: RoutedModel,
    rt: Runtime,
) -> ValidationReport:
    """Run every invariant and return the report (call :meth:`ValidationReport.raise_if_failed` to stop)."""
    report = ValidationReport()
    layers = [*labelled.slabs, *labelled.level_layers]
    catalogue = labelled.catalogue
    with rt.serial("validation_serial"):
        with report.timed("grid_and_lengths"):
            arrangements.check_grid_and_lengths(
                report, layers, labelled.levels, catalogue.grid
            )
        for name in labelled.structural_checks:
            report.add(
                name,
                True,
                "checked while labelling; a violation stops the build before this point",
            )
    with report.timed("planarity"):
        arrangements.check_planar(report, layers, rt)
    with rt.serial("validation_serial"):
        with report.timed("arcs"):
            arrangements.check_arcs(report, layers)
        with report.timed("bands"):
            arrangements.check_band_areas(report, labelled.level_layers)
        with report.timed("areas"):
            arrangements.check_areas(report, layers)
        with report.timed("prism_areas"):
            arrangements.check_prism_areas(
                report, catalogue, labelled.levels, labelled.slabs
            )
    with report.timed("shells"):
        topology_checks.check_shells(report, topology, rt)
    with rt.serial("validation_serial"):
        with report.timed("face_sides"):
            topology_checks.check_face_sides(report, topology)
        with report.timed("cylinders"):
            topology_checks.check_cylinders(report, topology)
        with report.timed("groups"):
            topology_checks.check_groups(report, topology, routed)
        with report.timed("edges_in_faces"):
            topology_checks.check_edges_in_faces(report, topology)
            topology_checks.check_vertical_edges(
                report, topology, catalogue.grid.size_um
            )
    return report
