"""Exercise existing compiler validation messages and aggregation with real records."""

from dataclasses import replace

import pytest

from scgsim.geometry.compiler import (
    coverage_validation,
    topology_validation,
    validation,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    ConstructionPlanRecord,
    CutHostOperationRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.models.tags import BackendEntityTagRecord, TagPlanRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    PointPlanRecord,
    RouteABVolumePlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)

VALIDATOR_NAMES = (
    "validate_backend_tag_ledger",
    "validate_curve_plan_coverage",
    "validate_interface_surface_source_of_truth",
    "validate_no_surface_overlap",
    "validate_route_operation_coverage",
    "validate_route_volume_surface_refs",
    "validate_selected_route",
    "validate_surface_deduplication",
    "validate_surface_partition_coverage",
    "validate_surface_sheet_interface_coverage",
    "validate_surface_use_counts",
    "validate_tag_plan_coverage",
    "validate_volume_surface_closure",
)


def _tetrahedron():
    points = tuple(
        PointPlanRecord(f"P{i}", xyz)
        for i, xyz in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    )
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    curves = tuple(
        CurvePlanRecord(f"C{a}{b}", "line_segment", f"P{a}", f"P{b}") for a, b in edges
    )
    faces = (
        ((0, 1, 2), (("C01", 1), ("C12", 1), ("C20", 1))),
        ((0, 3, 1), (("C03", 1), ("C13", -1), ("C01", -1))),
        ((1, 3, 2), (("C13", 1), ("C23", -1), ("C12", -1))),
        ((2, 3, 0), (("C23", 1), ("C03", -1), ("C20", -1))),
    )
    loops = tuple(
        SurfaceLoopRecord(
            f"L{i}",
            tuple(CurveRefRecord(name, side, "boundary") for name, side in refs),
            "outer",
        )
        for i, (_, refs) in enumerate(faces)
    )
    surfaces = tuple(
        SurfacePlanRecord(
            f"S{i}",
            "V",
            "domain_boundary",
            {"planar_loops_3d": (tuple(points[p].coordinate for p in face),)},
            outer_loop_ref=f"L{i}",
        )
        for i, (face, _) in enumerate(faces)
    )
    volume = VolumePlanRecord(
        "V",
        "V",
        "dielectric",
        tuple(SurfaceRefRecord(s.surface_id, 1, "exterior") for s in surfaces),
    )
    tags = tuple(
        TagPlanRecord(f"surface_{s.surface_id}", 2, "surface", s.surface_id, "domain")
        for s in surfaces
    )
    tags += (TagPlanRecord("volume_V", 3, "volume", "V", "domain"),)
    return ConstructionPlanRecord(
        "C",
        points=points,
        curves=curves,
        surface_loops=loops,
        surfaces=surfaces,
        volumes=(volume,),
        tags=tags,
    )


def _valid_calls():
    plan = _tetrahedron()
    interface = InterfacePlanRecord(
        "MS__fixture", "MS", ("metal", "substrate"), "authored"
    )
    body = ConstructionBodyPlanRecord(
        "body", "metal", "V", "cutout_boundary_shell", {}, ("S0",)
    )
    return {
        "validate_selected_route": (
            (GeometryBuildInput((), ()), "B"),
            {"representation_context": RepresentationContext(entities={})},
        ),
        "validate_route_operation_coverage": (
            (),
            {
                "construction_bodies": (body,),
                "cut_operations": (
                    CutHostOperationRecord("cut", "V", ("body",), ("S0",)),
                ),
                "surfaces": plan.surfaces,
            },
        ),
        "validate_surface_sheet_interface_coverage": (
            (GeometryBuildInput((), ()),),
            {"route": "A", "surfaces": ()},
        ),
        "validate_route_volume_surface_refs": (
            (),
            {"route": "C", "volumes": plan.volumes},
        ),
        "validate_curve_plan_coverage": (
            (),
            {
                "points": plan.points,
                "curves": plan.curves,
                "surface_loops": plan.surface_loops,
                "surfaces": plan.surfaces,
            },
        ),
        "validate_interface_surface_source_of_truth": (
            (),
            {"interfaces": (), "surfaces": plan.surfaces},
        ),
        "validate_surface_deduplication": ((), {"surfaces": plan.surfaces}),
        "validate_no_surface_overlap": ((), {"surfaces": plan.surfaces}),
        "validate_volume_surface_closure": (
            (),
            {
                "volumes": plan.volumes,
                "surfaces": plan.surfaces,
                "surface_loops": plan.surface_loops,
            },
        ),
        "validate_surface_use_counts": (
            (),
            {"volumes": plan.volumes, "surfaces": plan.surfaces},
        ),
        "validate_surface_partition_coverage": (
            (),
            {
                "interfaces": (interface,),
                "surface_partitions": (
                    SurfacePartitionRecord("partition", "MS__fixture", "S0", "child"),
                ),
                "surfaces": plan.surfaces,
            },
        ),
        "validate_tag_plan_coverage": (
            (),
            {"surfaces": plan.surfaces, "volumes": plan.volumes, "tags": plan.tags},
        ),
        "validate_backend_tag_ledger": (
            (),
            {
                "backend_tags": tuple(
                    BackendEntityTagRecord(
                        tag.source_record_kind, tag.source_record_id, (tag.dimension, i)
                    )
                    for i, tag in enumerate(plan.tags, 1)
                )
            },
        ),
    }


@pytest.mark.parametrize("name", VALIDATOR_NAMES)
def test_valid_compiler_records_retain_their_validation_result(name):
    """Accept a closed tetrahedron or the corresponding valid coverage records."""
    args, kwargs = _valid_calls()[name]
    assert getattr(validation, name)(*args, **kwargs) is None


@pytest.mark.parametrize("name", VALIDATOR_NAMES)
def test_facade_reexports_the_owning_function_without_a_wrapper(name):
    """Keep established imports bound directly to their one implementation."""
    owner = (
        coverage_validation
        if hasattr(coverage_validation, name)
        else topology_validation
    )
    assert getattr(validation, name) is getattr(owner, name)


@pytest.mark.parametrize("alias", ["_validate_surface_partition_coverage"])
def test_legacy_partition_alias_keeps_function_identity(alias):
    """Retain the old dispatcher spelling as an alias of the public validator."""
    assert getattr(validation, alias) is validation.validate_surface_partition_coverage
    assert set(validation.__all__) == set(VALIDATOR_NAMES)


def _curve_call(plan):
    return {
        "points": plan.points,
        "curves": plan.curves,
        "surface_loops": plan.surface_loops,
        "surfaces": plan.surfaces,
    }


def test_missing_canonical_records_are_reported_before_surface_references():
    """Aggregate the initial live-surface failure before individual missing loops."""
    surface = replace(
        _tetrahedron().surfaces[0], outer_loop_ref=None, hole_loop_refs=("missing",)
    )
    with pytest.raises(ValueError) as error:
        validation.validate_curve_plan_coverage(
            points=(), curves=(), surface_loops=(), surfaces=(surface,)
        )
    assert str(error.value) == (
        "Invalid curve plan coverage: backend-live surfaces require canonical points, curves, and loops; "
        "S0 lacks outer_loop_ref; S0 references unknown loop missing"
    )


def test_point_and_curve_identity_errors_preserve_source_order():
    """Report duplicate points before duplicate curves and missing endpoints."""
    plan = _tetrahedron()
    duplicate = replace(plan.points[0], coordinate=plan.points[1].coordinate)
    missing = CurvePlanRecord("missing", "line_segment", "unknown", "P0")
    collapsed = CurvePlanRecord("collapsed", "line_segment", "P0", "P0")
    invalid = replace(
        plan,
        points=(*plan.points, duplicate),
        curves=(*plan.curves, plan.curves[0], missing, collapsed),
    )
    with pytest.raises(ValueError) as error:
        validation.validate_curve_plan_coverage(**_curve_call(invalid))
    assert str(error.value) == (
        "Invalid curve plan coverage: duplicate point_id: P0; duplicate point coordinate: P1 and P0; "
        "duplicate curve_id: C01; duplicate curve geometry: C01 and C01; "
        "missing references unknown point unknown; collapsed has identical start/end points"
    )


def test_loop_errors_preserve_reference_and_cycle_order():
    """Keep repeated-ref, orientation, closure and short-loop error precedence."""
    plan = _tetrahedron()
    repeated = replace(
        plan.surface_loops[0],
        curve_refs=(
            CurveRefRecord("C01", 1, "boundary"),
            CurveRefRecord("C01", 1, "boundary"),
            CurveRefRecord("C12", 0, "boundary"),
            CurveRefRecord("C20", 1, "boundary"),
        ),
    )
    short = SurfaceLoopRecord(
        "short", (CurveRefRecord("unknown", 1, "boundary"),), "outer"
    )
    with pytest.raises(ValueError) as error:
        validation.validate_curve_plan_coverage(
            **_curve_call(
                replace(plan, surface_loops=(repeated, short, *plan.surface_loops[1:]))
            )
        )
    assert str(error.value) == (
        "Invalid curve plan coverage: L0 repeats curve refs inside one surface loop: ['C01']; "
        "L0 curve C12 has invalid orientation 0; L0 curve refs do not form a closed loop; "
        "short requires at least three straight curves"
    )


@pytest.mark.parametrize("end", [(2, 0, 0), (2, 2, 2)], ids=["axis", "diagonal"])
def test_interior_point_detection_keeps_axis_and_general_index_paths(end):
    """Use the existing coordinate search to reject an un-noded line interior."""
    middle = tuple(value / 2 for value in end)
    points = (
        PointPlanRecord("A", (0, 0, 0)),
        PointPlanRecord("I", middle),
        PointPlanRecord("B", end),
    )
    with pytest.raises(ValueError) as error:
        validation.validate_curve_plan_coverage(
            points=points,
            curves=(CurvePlanRecord("line", "line_segment", "A", "B"),),
            surface_loops=(),
            surfaces=(),
        )
    assert (
        str(error.value)
        == "Invalid curve plan coverage: line contains planned points in its interior: ['I']"
    )


def _curved_circle():
    points = (PointPlanRecord("A", (1, 0, 0)), PointPlanRecord("B", (-1, 0, 0)))
    top = CurvePlanRecord(
        "top",
        "circular_arc",
        "A",
        "B",
        geometry={"points_um": ((1, 0), (0, 1), (-1, 0))},
    )
    bottom = CurvePlanRecord(
        "bottom",
        "circular_arc",
        "B",
        "A",
        geometry={"points_um": ((-1, 0), (0, -1), (1, 0))},
    )
    loop = SurfaceLoopRecord(
        "circle",
        (CurveRefRecord("top", 1, "boundary"), CurveRefRecord("bottom", 1, "boundary")),
        "outer",
    )
    return ConstructionPlanRecord(
        "C", points=points, curves=(top, bottom), surface_loops=(loop,)
    )


def test_curved_loop_and_duplicate_geometry_keep_their_signature_contract():
    """Allow two curved edges while comparing their original geometry signatures."""
    plan = _curved_circle()
    validation.validate_curve_plan_coverage(**_curve_call(plan))
    duplicate = replace(plan.curves[0], curve_id="top_copy")
    with pytest.raises(ValueError) as error:
        validation.validate_curve_plan_coverage(
            **_curve_call(replace(plan, curves=(*plan.curves, duplicate)))
        )
    assert (
        str(error.value)
        == "Invalid curve plan coverage: duplicate curve geometry: top and top_copy"
    )


def _flat(name, x0, x1, *, z=0, holes=()):
    return SurfacePlanRecord(
        name,
        name,
        "domain_boundary",
        {
            "outer_loop": ((x0, 0), (x1, 0), (x1, 2), (x0, 2)),
            "hole_loops": holes,
            "z_um": z,
        },
    )


def test_overlap_keeps_first_plane_and_source_order():
    """Report the first occupied plane rather than sorting plane identities."""
    surfaces = (
        _flat("A", 0, 2, z=5),
        _flat("B", 1, 3, z=5),
        _flat("C", 0, 2),
        _flat("D", 1, 3),
    )
    with pytest.raises(ValueError) as error:
        validation.validate_no_surface_overlap(surfaces=surfaces)
    assert str(error.value) == "Invalid surface overlap: A overlaps B on z=5.0"


def test_overlap_keeps_touching_boundaries_and_authored_holes():
    """Retain non-overlap for touching edges and an explicitly excluded hole."""
    validation.validate_no_surface_overlap(
        surfaces=(_flat("left", 0, 1), _flat("right", 1, 2))
    )
    hole = ((1, 0), (2, 0), (2, 2), (1, 2))
    validation.validate_no_surface_overlap(
        surfaces=(_flat("host", 0, 3, holes=(hole,)), _flat("inside_hole", 1, 2))
    )


def test_source_loop_error_precedes_collected_overlap_detection():
    """Validate every source region before searching for overlap pairs."""
    bad = replace(_flat("bad", 0, 1), geometry_ref={"outer_loop": ((0, 0), (1, 0))})
    with pytest.raises(ValueError) as error:
        validation.validate_no_surface_overlap(
            surfaces=(_flat("A", 0, 2), _flat("B", 1, 3), bad)
        )
    assert str(error.value) == "surface loop requires at least 3 points"


def test_shell_errors_keep_exterior_then_void_order():
    """Aggregate missing surfaces/loops and incidence before inner-void errors."""
    plan = _tetrahedron()
    surfaces = (
        *plan.surfaces,
        replace(plan.surfaces[0], surface_id="plain", outer_loop_ref=None),
        replace(
            plan.surfaces[1], surface_id="lost_loop", outer_loop_ref="missing_loop"
        ),
    )
    refs = tuple(
        SurfaceRefRecord(name, 1, "exterior")
        for name in ("missing_surface", "plain", "lost_loop", "S0")
    )
    void = InnerPecVoidShellRecord(
        "void", (SurfaceRefRecord("missing_void", 1, "inner"),), "metal", ()
    )
    volume = RouteABVolumePlanRecord(
        "V",
        "V",
        "dielectric",
        plan.volumes[0].surface_refs,
        exterior_surface_refs=refs,
        inner_pec_void_shells=(void,),
    )
    with pytest.raises(ValueError) as error:
        validation.validate_volume_surface_closure(
            volumes=(volume,), surfaces=surfaces, surface_loops=plan.surface_loops
        )
    assert str(error.value) == (
        "Invalid volume surface closure: V/exterior references unknown surface missing_surface; "
        "V/exterior uses non-canonical surface plain; V/exterior references unknown loop missing_loop; "
        "V/exterior shell is not closed; curve incidence {'C01': 1, 'C12': 1, 'C20': 1}; "
        "V/void references unknown surface missing_void"
    )


def test_missing_volume_boundaries_precede_solution_shell_ambiguity():
    """Retain the first failure when a solution volume violates both checks."""
    volume = VolumePlanRecord(
        "V", "V", "vacuum", (), metadata={"representation": "solution_volume"}
    )
    with pytest.raises(ValueError) as error:
        validation.validate_route_volume_surface_refs(route="B", volumes=(volume,))
    assert (
        str(error.value)
        == "Backend-live volumes require planned boundary surfaces: ['V']"
    )
    with pytest.raises(ValueError) as error:
        validation.validate_route_volume_surface_refs(
            route="B",
            volumes=(
                replace(volume, surface_refs=(SurfaceRefRecord("S", 1, "exterior"),)),
            ),
        )
    assert (
        str(error.value)
        == "solution volumes require explicit exterior shell refs: ['V']"
    )


def test_route_context_error_precedes_unsupported_route():
    """Keep the mandatory physical-context failure before route selection."""
    source = GeometryBuildInput((), ())
    with pytest.raises(ValueError) as error:
        validation.validate_selected_route(source, "unsupported")
    assert (
        str(error.value)
        == "current compilation requires interpret_physical_geometry() and plan.context"
    )
    with pytest.raises(ValueError) as error:
        validation.validate_selected_route(
            source,
            "unsupported",
            representation_context=RepresentationContext(entities={}),
        )
    assert str(error.value) == "Unsupported route: unsupported"


def _metal(name, **kwargs):
    return SemanticEntitySpec(
        name, "conductor", "metal", "conductor", 1, "polygon", **kwargs
    )


def test_route_errors_keep_port_then_entity_order():
    """Aggregate existing Route C restrictions in original input order."""
    first = _metal("first", geometry={"route_ab_fused_selector_mode": True})
    second = _metal("second")
    port = PortSheetRegionRecord("port", "port", "polygon", ((0, 0), (1, 0), (1, 1)))
    source = GeometryBuildInput((), (first, second), port_sheet_regions=(port,))
    context = RepresentationContext(
        entities={"second": EntityRepresentation(None, "sheet", (0, 1), (0, 0))}
    )
    with pytest.raises(ValueError) as error:
        validation.validate_selected_route(source, "C", representation_context=context)
    assert str(error.value) == (
        "Invalid selected route: Route C does not support Palace lumped-port sheets; "
        "first does not define Route C; first Route C cannot use route_ab_fused_selector_mode; "
        "second Route C representation 'surface_sheet' is not supported"
    )


def test_route_operation_errors_keep_authored_then_sorted_coverage_order():
    """Keep body-local source order before sorted missing-cut and surface lists."""
    body = ConstructionBodyPlanRecord(
        "B2", "metal", "host", "cutout_boundary_shell", {}, ("lost_b", "lost_a")
    )
    bodies = (body, replace(body, construction_body_id="B1", expected_surface_ids=()))
    operation = CutHostOperationRecord("cut", "host", (), ("lost_z", "lost_x"))
    with pytest.raises(ValueError) as error:
        validation.validate_route_operation_coverage(
            construction_bodies=bodies, cut_operations=(operation,), surfaces=()
        )
    assert str(error.value) == (
        "Invalid route operation plan: B2 expects missing shell surfaces ['lost_b', 'lost_a']; "
        "construction bodies without cut operations ['B1', 'B2']; "
        "cut operations expose missing surfaces ['lost_x', 'lost_z']"
    )


def test_interface_and_partition_errors_keep_source_order():
    """Preserve interface-kind failures and parent-before-child partition errors."""
    surface = replace(
        _tetrahedron().surfaces[0], metadata={"interface_kinds": ("MS", "MA", "SA")}
    )
    unknown = replace(
        surface, surface_id="unknown", interface_id="MS__missing", metadata={}
    )
    with pytest.raises(ValueError) as error:
        validation.validate_interface_surface_source_of_truth(
            interfaces=(), surfaces=(surface, unknown)
        )
    assert str(error.value) == (
        "Invalid interface source of truth: S0 has too many interface kinds ('MA', 'MS', 'SA'); "
        "S0 has interface kinds ('MA', 'MS', 'SA') without InterfacePlan source; "
        "unknown references unknown interface MS__missing"
    )
    partition = SurfacePartitionRecord(
        "partition", "missing_parent", "missing_child", "child"
    )
    with pytest.raises(ValueError) as error:
        validation.validate_surface_partition_coverage(
            interfaces=(), surface_partitions=(partition,), surfaces=()
        )
    assert str(error.value) == (
        "Invalid surface partition plan: partition references unknown parent interface missing_parent; "
        "partition references unknown child surface missing_child"
    )


def test_surface_sheet_owner_string_is_one_semantic_identity():
    """Retain string-owner normalization and sorted missing sheet identities."""
    source = GeometryBuildInput(
        (),
        (
            _metal("A", route_representations={"A": "surface_sheet"}),
            _metal("B", route_representations={"A": "surface_sheet"}),
        ),
    )
    surface = replace(
        _tetrahedron().surfaces[0],
        interface_id="MS__owner",
        metadata={"owner_semantic_ids": "A"},
    )
    with pytest.raises(ValueError) as error:
        validation.validate_surface_sheet_interface_coverage(
            source, route="A", surfaces=(surface,)
        )
    assert (
        str(error.value)
        == "surface_sheet entities require planned interface surface coverage: ['B']"
    )


def test_surface_use_errors_keep_port_and_interface_policies():
    """Preserve boundary-port counts, embedded-sheet skipping and exterior checks."""
    plan = _tetrahedron()
    surfaces = (
        replace(
            plan.surfaces[0],
            surface_role="lumped_port",
            metadata={"route_a_boundary_port": True},
        ),
        replace(plan.surfaces[1], metadata={"embedded_surface_sheet": True}),
        plan.surfaces[2],
        replace(plan.surfaces[3], interface_id="MS__fixture"),
    )
    volume = replace(
        plan.volumes[0], surface_refs=(SurfaceRefRecord("S3", 1, "exterior"),) * 3
    )
    with pytest.raises(ValueError) as error:
        validation.validate_surface_use_counts(volumes=(volume,), surfaces=surfaces)
    assert str(error.value) == (
        "Invalid surface use counts: S0 Route A port use-count must be 2, got 0; "
        "S2 exterior use-count must be 1, got 0; S3 interface use-count must be 1 or 2, got 3"
    )


def test_tag_and_backend_ledger_errors_keep_their_aggregate_order():
    """Keep missing-before-extra tags and source-before-backend ledger conflicts."""
    plan = _tetrahedron()
    extra = TagPlanRecord("extra", 2, "surface", "orphan", "domain")
    with pytest.raises(ValueError) as error:
        validation.validate_tag_plan_coverage(
            surfaces=plan.surfaces, volumes=plan.volumes, tags=(extra,)
        )
    assert str(error.value) == (
        "Invalid tag plan coverage: missing tags for [('surface', 'S0'), ('surface', 'S1'), "
        "('surface', 'S2'), ('surface', 'S3'), ('volume', 'V')]; "
        "tags reference non-live records [('surface', 'orphan')]"
    )
    tags = (
        BackendEntityTagRecord("surface", "A", (2, 1)),
        BackendEntityTagRecord("surface", "A", (2, 2)),
        BackendEntityTagRecord("surface", "B", (2, 2)),
    )
    with pytest.raises(ValueError) as error:
        validation.validate_backend_tag_ledger(backend_tags=tags)
    text = str(error.value)
    assert text.startswith(
        "Invalid backend tag ledger: ('surface', 'A') maps to multiple backend tags "
    )
    assert "; backend tag (2, 2) is shared by sources " in text
