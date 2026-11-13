"""Structural validation of a built model: fail closed, never disabled (design brief section 11, stage S9).

Owns running every invariant over the labelled arrangements, the topology and the route's
group plan, and turning any failure into a :class:`ValidationError`.  A production build
always validates before anything is written; there is no switch to turn it off.  Tests that
need to look at a *failing* report call :func:`validate_model` themselves and decide.
"""

from __future__ import annotations

from scgsim.geometry.sgb.arrangement.derivation.labelled import LabelledArrangements
from scgsim.geometry.sgb.checks.validation import arrangements, ids
from scgsim.geometry.sgb.checks.validation import topology as topology_checks
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch
from scgsim.geometry.sgb.topology.assembly.records import Topology
from scgsim.geometry.sgb.topology.routes.groups import RoutedModel


def validate_model(
    labelled: LabelledArrangements,
    topology: Topology,
    routed: RoutedModel,
    watch: Stopwatch,
) -> ValidationReport:
    """Run every invariant and return the report (call :meth:`ValidationReport.raise_if_failed` to stop).

    ``watch`` receives the build-stage time of the checks; the report keeps the time of each check.
    """
    report = ValidationReport()
    layers = [*labelled.slabs, *labelled.level_layers]
    catalogue = labelled.catalogue
    with watch.stage("validation_serial"):
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
    with report.timed("atomic_ids"):
        ids.check_atomic_ids_unique(report, layers, topology)
    with report.timed("planarity"):
        arrangements.check_planar(report, layers, watch)
    with watch.stage("validation_serial"):
        with report.timed("arcs"):
            arrangements.check_arcs(report, layers)
        with report.timed("bands"):
            arrangements.check_band_areas(
                report, labelled.level_layers, _support_areas(catalogue)
            )
        with report.timed("areas"):
            arrangements.check_areas(report, layers)
        with report.timed("prism_areas"):
            arrangements.check_prism_areas(
                report, catalogue, labelled.levels, labelled.slabs
            )
    with report.timed("shells"):
        topology_checks.check_shells(report, topology, watch)
    with watch.stage("validation_serial"):
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


def _support_areas(catalogue: PrismCatalogue) -> dict[str, tuple[float, float]]:
    """Support area and tolerance of every inset family: each boundary point moves by less than one grid unit per margin."""
    return {
        family.family_id: (
            family.support.area(),
            family.support.perimeter() * family.support.band_count,
        )
        for family in catalogue.families
        if family.support is not None
    }
