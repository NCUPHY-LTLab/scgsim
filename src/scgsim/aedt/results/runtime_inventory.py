"""Version-bound source inventories; historical paths never follow relocations."""

from __future__ import annotations

from typing import Any

SOURCE_SCHEMA_V1 = "scgsim.aedt.runtime-source.v1"


SOURCE_SCHEMA_V2 = "scgsim.aedt.runtime-source.v2"


SOURCE_SCHEMA_V3 = "scgsim.aedt.runtime-source.v3"


SOURCE_SCHEMA_V4 = "scgsim.aedt.runtime-source.v4"


SOURCE_SCHEMA_V5 = "scgsim.aedt.runtime-source.v5"


SOURCE_SCHEMA_V6 = "scgsim.aedt.runtime-source.v6"


SOURCE_SCHEMA_V7 = "scgsim.aedt.runtime-source.v7"


SOURCE_SCHEMA_V8 = "scgsim.aedt.runtime-source.v8"


SOURCE_SCHEMA_V9 = "scgsim.aedt.runtime-source.v9"


SOURCE_SCHEMA_V10 = "scgsim.aedt.runtime-source.v10"


SOURCE_SCHEMA_V11 = "scgsim.aedt.runtime-source.v11"


SOURCE_SCHEMA_V12 = "scgsim.aedt.runtime-source.v12"


SOURCE_SCHEMA_V13 = "scgsim.aedt.runtime-source.v13"


SOURCE_SCHEMA_V15 = "scgsim.aedt.runtime-source.v15"


SOURCE_SCHEMA_V16 = "scgsim.aedt.runtime-source.v16"


SOURCE_SCHEMA_V17 = "scgsim.aedt.runtime-source.v17"


SOURCE_SCHEMA_V18 = "scgsim.aedt.runtime-source.v18"


SOURCE_SCHEMA_V19 = "scgsim.aedt.runtime-source.v19"


SOURCE_SCHEMA_V20 = "scgsim.aedt.runtime-source.v20"


SOURCE_SCHEMA_V21 = "scgsim.aedt.runtime-source.v21"


SOURCE_SCHEMA_V22 = "scgsim.aedt.runtime-source.v22"


SOURCE_SCHEMA_V23 = "scgsim.aedt.runtime-source.v23"


SOURCE_SCHEMA_V24 = "scgsim.aedt.runtime-source.v24"


SOURCE_SCHEMA_V25 = "scgsim.aedt.runtime-source.v25"


SOURCE_SCHEMA_V26 = "scgsim.aedt.runtime-source.v26"


SOURCE_SCHEMA = SOURCE_SCHEMA_V26


_RUNTIME_SOURCE_V1_MODULES = (
    "_hfss_convergence.py",
    "_hfss_runtime.py",
    "_matrix_export.py",
    "_native_common.py",
    "_q2d_convergence.py",
    "_q2d_runtime.py",
    "_q3d_runtime.py",
    "_runtime_provenance.py",
    "handoff.py",
    "resolve.py",
    "run.py",
    "spec.py",
    "util.py",
)


_RUNTIME_SOURCE_V1_PATHS = tuple(
    f"scgsim/aedt/{name}" for name in sorted(_RUNTIME_SOURCE_V1_MODULES)
)


_RUNTIME_SOURCE_V2_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V1_PATHS,
            "scgsim/aedt/_epr_eigenmode.py",
            "scgsim/aedt/_epr_fields.py",
            "scgsim/aedt/_epr_geometry.py",
            "scgsim/aedt/_epr_models.py",
            "scgsim/aedt/_epr_results.py",
            "scgsim/semantics/route_a.py",
            "scgsim/sgb/planning.py",
        )
    )
)


_RUNTIME_SOURCE_V3_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V2_PATHS, "scgsim/aedt/_benchmark.py"))
)


_RUNTIME_SOURCE_V4_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V3_PATHS, "scgsim/aedt/_presentation.py"))
)


_RUNTIME_SOURCE_V5_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V4_PATHS,
            "scgsim/sgb/geometry_plan.py",
            "scgsim/sgb/__init__.py",
        )
    )
)


_RUNTIME_SOURCE_V6_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V5_PATHS,
            "scgsim/sgb/summary.py",
        )
    )
)


_RUNTIME_SOURCE_V7_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V6_PATHS,
            "scgsim/_notebook_presentation.py",
        )
    )
)


_RUNTIME_SOURCE_V8_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V7_PATHS,
            "scgsim/aedt/_junction_partition.py",
        )
    )
)


_RUNTIME_SOURCE_V9_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V8_PATHS,
            "scgsim/aedt/__init__.py",
            "scgsim/aedt/_q3d_geometry.py",
            "scgsim/sgb/adapter.py",
            "scgsim/sgb/stack.py",
        )
    )
)


_RUNTIME_SOURCE_V10_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V9_PATHS, "scgsim/aedt/_handoff_cohort.py"))
)


_RUNTIME_SOURCE_V11_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V10_PATHS, "scgsim/aedt/_q3d_bodies.py"))
)


_RUNTIME_SOURCE_V12_PATHS = _RUNTIME_SOURCE_V11_PATHS


_RUNTIME_SOURCE_V13_PATHS = (
    "src/scgsim/aedt/__init__.py",
    "src/scgsim/aedt/_io.py",
    "src/scgsim/aedt/eigenmode.py",
    "src/scgsim/aedt/epr/analysis.py",
    "src/scgsim/aedt/epr/cache.py",
    "src/scgsim/aedt/epr/fields.py",
    "src/scgsim/aedt/epr/geometry.py",
    "src/scgsim/aedt/epr/junction_partition.py",
    "src/scgsim/aedt/epr/models.py",
    "src/scgsim/aedt/epr/native.py",
    "src/scgsim/aedt/epr/selection.py",
    "src/scgsim/aedt/epr/workflow.py",
    "src/scgsim/aedt/preparation/cohort.py",
    "src/scgsim/aedt/preparation/geometry.py",
    "src/scgsim/aedt/preparation/handoff.py",
    "src/scgsim/aedt/preparation/q3d_geometry.py",
    "src/scgsim/aedt/presentation/benchmark.py",
    "src/scgsim/aedt/presentation/epr.py",
    "src/scgsim/aedt/presentation/runs.py",
    "src/scgsim/aedt/results/benchmark.py",
    "src/scgsim/aedt/results/convergence/common.py",
    "src/scgsim/aedt/results/convergence/hfss.py",
    "src/scgsim/aedt/results/convergence/q2d.py",
    "src/scgsim/aedt/results/convergence/q3d.py",
    "src/scgsim/aedt/results/epr.py",
    "src/scgsim/aedt/results/matrices.py",
    "src/scgsim/aedt/results/provenance.py",
    "src/scgsim/aedt/results/resolve.py",
    "src/scgsim/aedt/run.py",
    "src/scgsim/aedt/runtime/benchmark.py",
    "src/scgsim/aedt/runtime/families/hfss.py",
    "src/scgsim/aedt/runtime/families/q2d.py",
    "src/scgsim/aedt/runtime/families/q3d.py",
    "src/scgsim/aedt/runtime/native/common.py",
    "src/scgsim/aedt/runtime/native/q3d_bodies.py",
    "src/scgsim/aedt/runtime/transaction.py",
    "src/scgsim/aedt/specs/common.py",
    "src/scgsim/aedt/specs/hfss.py",
    "src/scgsim/aedt/specs/parse.py",
    "src/scgsim/aedt/specs/q2d.py",
    "src/scgsim/aedt/specs/q3d.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/_primitives/entities.py",
    "src/scgsim/geometry/_primitives/loops.py",
    "src/scgsim/geometry/_primitives/spatial.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/route_a.py",
)


_RUNTIME_SOURCE_V15_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V13_PATHS,
            "src/scgsim/aedt/specs/modeling.py",
            "src/scgsim/geometry/_primitives/geometry_refs.py",
            "src/scgsim/geometry/_primitives/surface_records.py",
            "src/scgsim/geometry/planning/evidence.py",
            "src/scgsim/geometry/source/intents.py",
            "src/scgsim/geometry/source/_normalization.py",
        )
    )
)


_RUNTIME_SOURCE_V16_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V15_PATHS,
            "src/scgsim/aedt/runtime/native/hfss_eigenmode.py",
        )
    )
)


_RUNTIME_SOURCE_V17_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V16_PATHS,
            "src/scgsim/geometry/models/lumped.py",
            "src/scgsim/aedt/specs/__init__.py",
            "src/scgsim/aedt/runtime/native/hfss_lumped.py",
        )
    )
)


_RUNTIME_SOURCE_V18_PATHS = tuple(
    sorted(
        (
            set(_RUNTIME_SOURCE_V17_PATHS)
            - {"src/scgsim/aedt/epr/junction_partition.py"}
            | {
                "src/scgsim/aedt/_data.py",
                "src/scgsim/aedt/_rlc.py",
                "src/scgsim/aedt/_terminal_chain.py",
                "src/scgsim/aedt/specs/geometry.py",
                "src/scgsim/aedt/specs/epr.py",
                "src/scgsim/aedt/specs/_support.py",
                "src/scgsim/aedt/preparation/modeling.py",
                "src/scgsim/aedt/preparation/planar.py",
                "src/scgsim/aedt/preparation/junction_partition.py",
                "src/scgsim/aedt/runtime/native/hfss_geometry.py",
                "src/scgsim/geometry/models/representation.py",
                "src/scgsim/geometry/planning/representation.py",
                "src/scgsim/geometry/source/ground.py",
                "src/scgsim/geometry/source/_polygon_ops.py",
            }
        )
    )
)


_RUNTIME_SOURCE_V19_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V18_PATHS, "src/scgsim/aedt/results/prepared.py"))
)


_RUNTIME_SOURCE_V20_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V19_PATHS,
            "src/scgsim/geometry/models/physical.py",
            "src/scgsim/geometry/planning/physical.py",
            "src/scgsim/semantics/snapshot.py",
        )
    )
)


_RUNTIME_SOURCE_V21_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V20_PATHS, "src/scgsim/aedt/_benchmark.py"))
)


_RUNTIME_SOURCE_V22_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V21_PATHS,
            "src/scgsim/aedt/versions.py",
            "src/scgsim/aedt/runtime/native/compatibility.py",
        )
    )
)


_RUNTIME_SOURCE_V23_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V22_PATHS, "src/scgsim/aedt/batch.py"))
)


_RUNTIME_SOURCE_V24_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V23_PATHS,
            "src/scgsim/aedt/epr/evidence.py",
            "src/scgsim/aedt/runtime/batch_protocol.py",
            "src/scgsim/aedt/runtime/batch_runner.py",
        )
    )
)


_RUNTIME_SOURCE_V25_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V24_PATHS,
            "src/scgsim/aedt/runtime/control.py",
        )
    )
)


_RUNTIME_SOURCE_V26_PATHS = (
    "src/scgsim/aedt/__init__.py",
    "src/scgsim/aedt/batch.py",
    "src/scgsim/aedt/batch_stages.py",
    "src/scgsim/aedt/benchmark_data.py",
    "src/scgsim/aedt/data.py",
    "src/scgsim/aedt/eigenmode.py",
    "src/scgsim/aedt/epr/adaptive.py",
    "src/scgsim/aedt/epr/analysis.py",
    "src/scgsim/aedt/epr/cache.py",
    "src/scgsim/aedt/epr/cache_config.py",
    "src/scgsim/aedt/epr/evidence.py",
    "src/scgsim/aedt/epr/expression_authoring.py",
    "src/scgsim/aedt/epr/expression_codec.py",
    "src/scgsim/aedt/epr/expression_loading.py",
    "src/scgsim/aedt/epr/fields.py",
    "src/scgsim/aedt/epr/geometry.py",
    "src/scgsim/aedt/epr/integrals.py",
    "src/scgsim/aedt/epr/models.py",
    "src/scgsim/aedt/epr/native.py",
    "src/scgsim/aedt/epr/native_workflow.py",
    "src/scgsim/aedt/epr/reanalysis.py",
    "src/scgsim/aedt/epr/selection.py",
    "src/scgsim/aedt/epr/surface.py",
    "src/scgsim/aedt/epr/surface_references.py",
    "src/scgsim/aedt/epr/workflow.py",
    "src/scgsim/aedt/io.py",
    "src/scgsim/aedt/preparation/cohort.py",
    "src/scgsim/aedt/preparation/gds.py",
    "src/scgsim/aedt/preparation/geometry.py",
    "src/scgsim/aedt/preparation/handoff.py",
    "src/scgsim/aedt/preparation/handoff_files.py",
    "src/scgsim/aedt/preparation/junction_partition.py",
    "src/scgsim/aedt/preparation/modeling.py",
    "src/scgsim/aedt/preparation/planar.py",
    "src/scgsim/aedt/preparation/planar_junctions.py",
    "src/scgsim/aedt/preparation/planar_masks.py",
    "src/scgsim/aedt/preparation/planar_source.py",
    "src/scgsim/aedt/preparation/q3d_geometry.py",
    "src/scgsim/aedt/presentation/benchmark.py",
    "src/scgsim/aedt/presentation/epr.py",
    "src/scgsim/aedt/presentation/epr_projection.py",
    "src/scgsim/aedt/presentation/runs.py",
    "src/scgsim/aedt/results/benchmark.py",
    "src/scgsim/aedt/results/convergence/common.py",
    "src/scgsim/aedt/results/convergence/files.py",
    "src/scgsim/aedt/results/convergence/hfss.py",
    "src/scgsim/aedt/results/convergence/q2d.py",
    "src/scgsim/aedt/results/convergence/q3d.py",
    "src/scgsim/aedt/results/epr.py",
    "src/scgsim/aedt/results/hfss_readback.py",
    "src/scgsim/aedt/results/matrices.py",
    "src/scgsim/aedt/results/prepared.py",
    "src/scgsim/aedt/results/provenance.py",
    "src/scgsim/aedt/results/q2d_readback.py",
    "src/scgsim/aedt/results/q3d_readback.py",
    "src/scgsim/aedt/results/readback_common.py",
    "src/scgsim/aedt/results/readback_dispatch.py",
    "src/scgsim/aedt/results/receipt_validation.py",
    "src/scgsim/aedt/results/resolve.py",
    "src/scgsim/aedt/results/runtime_inventory.py",
    "src/scgsim/aedt/results/terminal_readback.py",
    "src/scgsim/aedt/rlc.py",
    "src/scgsim/aedt/run.py",
    "src/scgsim/aedt/runtime/batch_protocol.py",
    "src/scgsim/aedt/runtime/batch_runner.py",
    "src/scgsim/aedt/runtime/benchmark.py",
    "src/scgsim/aedt/runtime/control.py",
    "src/scgsim/aedt/runtime/desktop_lifecycle.py",
    "src/scgsim/aedt/runtime/families/hfss.py",
    "src/scgsim/aedt/runtime/families/hfss_body_eigenmode.py",
    "src/scgsim/aedt/runtime/families/hfss_exports.py",
    "src/scgsim/aedt/runtime/families/hfss_history.py",
    "src/scgsim/aedt/runtime/families/hfss_mesh.py",
    "src/scgsim/aedt/runtime/families/hfss_port_properties.py",
    "src/scgsim/aedt/runtime/families/hfss_ports.py",
    "src/scgsim/aedt/runtime/families/hfss_requests.py",
    "src/scgsim/aedt/runtime/families/hfss_setup.py",
    "src/scgsim/aedt/runtime/families/q2d.py",
    "src/scgsim/aedt/runtime/families/q3d.py",
    "src/scgsim/aedt/runtime/families/q3d_exports.py",
    "src/scgsim/aedt/runtime/families/q3d_prepared.py",
    "src/scgsim/aedt/runtime/families/q3d_region.py",
    "src/scgsim/aedt/runtime/families/q3d_setup.py",
    "src/scgsim/aedt/runtime/families/q3d_terminals.py",
    "src/scgsim/aedt/runtime/families/q3d_termination.py",
    "src/scgsim/aedt/runtime/handoff_validation.py",
    "src/scgsim/aedt/runtime/native/common.py",
    "src/scgsim/aedt/runtime/native/compatibility.py",
    "src/scgsim/aedt/runtime/native/gds_import.py",
    "src/scgsim/aedt/runtime/native/hfss_eigenmode.py",
    "src/scgsim/aedt/runtime/native/hfss_geometry.py",
    "src/scgsim/aedt/runtime/native/hfss_lumped.py",
    "src/scgsim/aedt/runtime/native/junction_extrusion.py",
    "src/scgsim/aedt/runtime/native/junction_readback.py",
    "src/scgsim/aedt/runtime/native/junction_topology.py",
    "src/scgsim/aedt/runtime/native/planar_bodies.py",
    "src/scgsim/aedt/runtime/native/planar_region.py",
    "src/scgsim/aedt/runtime/native/q3d_bodies.py",
    "src/scgsim/aedt/runtime/native/resources.py",
    "src/scgsim/aedt/runtime/native/saved_geometry.py",
    "src/scgsim/aedt/runtime/receipt.py",
    "src/scgsim/aedt/runtime/transaction.py",
    "src/scgsim/aedt/specs/__init__.py",
    "src/scgsim/aedt/specs/common.py",
    "src/scgsim/aedt/specs/epr.py",
    "src/scgsim/aedt/specs/geometry.py",
    "src/scgsim/aedt/specs/hfss.py",
    "src/scgsim/aedt/specs/hfss_common.py",
    "src/scgsim/aedt/specs/hfss_driven.py",
    "src/scgsim/aedt/specs/hfss_eigenmode.py",
    "src/scgsim/aedt/specs/hfss_epr.py",
    "src/scgsim/aedt/specs/modeling.py",
    "src/scgsim/aedt/specs/parse.py",
    "src/scgsim/aedt/specs/q2d.py",
    "src/scgsim/aedt/specs/q3d.py",
    "src/scgsim/aedt/specs/support.py",
    "src/scgsim/aedt/terminal_chain.py",
    "src/scgsim/aedt/versions.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/compiler/coverage_validation.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/stages.py",
    "src/scgsim/geometry/compiler/topology_validation.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/construction/curved.py",
    "src/scgsim/geometry/construction/curved_geometry/__init__.py",
    "src/scgsim/geometry/construction/curved_geometry/arrangement.py",
    "src/scgsim/geometry/construction/curved_geometry/boundaries.py",
    "src/scgsim/geometry/construction/curved_geometry/kernel.py",
    "src/scgsim/geometry/construction/curved_geometry/records.py",
    "src/scgsim/geometry/construction/curved_geometry/surfaces.py",
    "src/scgsim/geometry/models/lumped.py",
    "src/scgsim/geometry/models/physical.py",
    "src/scgsim/geometry/models/representation.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/physical.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/representation.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/primitives/entities.py",
    "src/scgsim/geometry/primitives/geometry_refs.py",
    "src/scgsim/geometry/primitives/grid_overlay.py",
    "src/scgsim/geometry/primitives/grid_regions.py",
    "src/scgsim/geometry/primitives/loops.py",
    "src/scgsim/geometry/primitives/planar_loops.py",
    "src/scgsim/geometry/primitives/spatial.py",
    "src/scgsim/geometry/primitives/surface_records.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/adapter_records.py",
    "src/scgsim/geometry/source/ground.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/normalization.py",
    "src/scgsim/geometry/source/occurrence_validation.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/polygon_ops.py",
    "src/scgsim/geometry/source/ports.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/route_a.py",
    "src/scgsim/semantics/snapshot.py",
)


def runtime_source_paths(schema: Any) -> tuple[str, ...] | None:
    """Resolve the declared inventory without rewriting historical module paths."""
    return {
        SOURCE_SCHEMA_V1: _RUNTIME_SOURCE_V1_PATHS,
        SOURCE_SCHEMA_V2: _RUNTIME_SOURCE_V2_PATHS,
        SOURCE_SCHEMA_V3: _RUNTIME_SOURCE_V3_PATHS,
        SOURCE_SCHEMA_V4: _RUNTIME_SOURCE_V4_PATHS,
        SOURCE_SCHEMA_V5: _RUNTIME_SOURCE_V5_PATHS,
        SOURCE_SCHEMA_V6: _RUNTIME_SOURCE_V6_PATHS,
        SOURCE_SCHEMA_V7: _RUNTIME_SOURCE_V7_PATHS,
        SOURCE_SCHEMA_V8: _RUNTIME_SOURCE_V8_PATHS,
        SOURCE_SCHEMA_V9: _RUNTIME_SOURCE_V9_PATHS,
        SOURCE_SCHEMA_V10: _RUNTIME_SOURCE_V10_PATHS,
        SOURCE_SCHEMA_V11: _RUNTIME_SOURCE_V11_PATHS,
        SOURCE_SCHEMA_V12: _RUNTIME_SOURCE_V12_PATHS,
        SOURCE_SCHEMA_V13: _RUNTIME_SOURCE_V13_PATHS,
        SOURCE_SCHEMA_V15: _RUNTIME_SOURCE_V15_PATHS,
        SOURCE_SCHEMA_V16: _RUNTIME_SOURCE_V16_PATHS,
        SOURCE_SCHEMA_V17: _RUNTIME_SOURCE_V17_PATHS,
        SOURCE_SCHEMA_V18: _RUNTIME_SOURCE_V18_PATHS,
        SOURCE_SCHEMA_V19: _RUNTIME_SOURCE_V19_PATHS,
        SOURCE_SCHEMA_V20: _RUNTIME_SOURCE_V20_PATHS,
        SOURCE_SCHEMA_V21: _RUNTIME_SOURCE_V21_PATHS,
        SOURCE_SCHEMA_V22: _RUNTIME_SOURCE_V22_PATHS,
        SOURCE_SCHEMA_V23: _RUNTIME_SOURCE_V23_PATHS,
        SOURCE_SCHEMA_V24: _RUNTIME_SOURCE_V24_PATHS,
        SOURCE_SCHEMA_V25: _RUNTIME_SOURCE_V25_PATHS,
        SOURCE_SCHEMA_V26: _RUNTIME_SOURCE_V26_PATHS,
    }.get(schema)


def current_runtime_source_paths() -> tuple[str, ...]:
    """Return the current producer inventory in canonical order."""
    return _RUNTIME_SOURCE_V26_PATHS
