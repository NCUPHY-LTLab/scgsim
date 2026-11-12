"""Version-bound source inventories; historical paths never follow relocations."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# long-file: Keep immutable ordered rosters beside their schema readers; splitting the catalog into imported modules would alter future source closures.

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


SOURCE_SCHEMA_V27 = "scgsim.aedt.runtime-source.v27"


SOURCE_SCHEMA_V28 = "scgsim.aedt.runtime-source.v28"


SOURCE_SCHEMA_V29 = "scgsim.aedt.runtime-source.v29"


SOURCE_SCHEMA_V30 = "scgsim.aedt.runtime-source.v30"


SOURCE_SCHEMA = SOURCE_SCHEMA_V30


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


def runtime_import_closure(
    source_root: Path, seed_paths: Iterable[str]
) -> tuple[str, ...]:
    """Return source paths reached by static internal Python imports."""
    return _RuntimeImportClosure(source_root).build(seed_paths)


class _RuntimeImportClosure:
    """Collect source modules reachable from one set of canonical source seeds."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.paths: set[str] = set()
        self.pending: list[Path] = []

    def build(self, seed_paths: Iterable[str]) -> tuple[str, ...]:
        """Add all seeds, walk their imports, and return sorted source paths."""
        for seed in seed_paths:
            self.add_seed(seed)
        while self.pending:
            self.expand(self.pending.pop())
        return tuple(sorted(self.paths))

    def add_seed(self, seed: str) -> None:
        """Validate and add one canonical source path as a closure root."""
        relative = Path(seed)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "src"
            or ".." in relative.parts
        ):
            raise ValueError("runtime source seed must be a contained src path")
        path = self.source_root.joinpath(*relative.parts[1:])
        if not path.is_file():
            raise FileNotFoundError(f"runtime source seed is missing: {seed}")
        self.add_path(path)
        self.add_module(_module_name(path, self.source_root))

    def add_path(self, path: Path) -> None:
        """Schedule one source file once for import parsing."""
        canonical = f"src/{path.relative_to(self.source_root).as_posix()}"
        if canonical not in self.paths:
            self.paths.add(canonical)
            self.pending.append(path)

    def add_module(self, module: str) -> None:
        """Add an internal module plus every source package initializer."""
        parts = module.split(".")
        if not parts or parts[0] != "scgsim" or any(not part for part in parts):
            return
        for depth in range(1, len(parts)):
            package_init = self.source_root.joinpath(*parts[:depth], "__init__.py")
            if package_init.is_file():
                self.add_path(package_init)
        package_init = self.source_root.joinpath(*parts, "__init__.py")
        if package_init.is_file():
            self.add_path(package_init)
            return
        module_path = self.source_root.joinpath(*parts[:-1], f"{parts[-1]}.py")
        if module_path.is_file():
            self.add_path(module_path)

    def expand(self, path: Path) -> None:
        """Walk every import node, including imports inside lazy code paths."""
        module = _module_name(path, self.source_root)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.add_import(node)
            elif isinstance(node, ast.ImportFrom):
                self.add_from_import(node, package)

    def add_import(self, node: ast.Import) -> None:
        """Resolve each absolute module in an import statement."""
        for alias in node.names:
            self.add_module(alias.name)

    def add_from_import(self, node: ast.ImportFrom, package: str) -> None:
        """Resolve a base module and any imported child modules."""
        base = _import_from_base(node, package)
        self.add_module(base)
        if base.startswith("scgsim"):
            for alias in node.names:
                if alias.name != "*":
                    self.add_module(f"{base}.{alias.name}")


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_from_base(node: ast.ImportFrom, package: str) -> str:
    """Resolve an absolute or relative ImportFrom base name."""
    base = node.module or ""
    if not node.level:
        return base
    package_parts = package.split(".") if package else []
    remove = node.level - 1
    if remove > len(package_parts):
        return ""
    prefix = ".".join(package_parts[: len(package_parts) - remove])
    return ".".join(part for part in (prefix, base) if part)


_RUNTIME_SOURCE_V27_PATHS = (
    "src/scgsim/__init__.py",
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
    "src/scgsim/diagnostics.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/compiler/__init__.py",
    "src/scgsim/geometry/compiler/builder.py",
    "src/scgsim/geometry/compiler/coverage_validation.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/stages.py",
    "src/scgsim/geometry/compiler/topology_validation.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/construction/__init__.py",
    "src/scgsim/geometry/construction/conformality.py",
    "src/scgsim/geometry/construction/curved.py",
    "src/scgsim/geometry/construction/curved_geometry/__init__.py",
    "src/scgsim/geometry/construction/curved_geometry/arrangement.py",
    "src/scgsim/geometry/construction/curved_geometry/boundaries.py",
    "src/scgsim/geometry/construction/curved_geometry/kernel.py",
    "src/scgsim/geometry/construction/curved_geometry/records.py",
    "src/scgsim/geometry/construction/curved_geometry/surfaces.py",
    "src/scgsim/geometry/construction/export.py",
    "src/scgsim/geometry/construction/gmsh_occ.py",
    "src/scgsim/geometry/construction/native.py",
    "src/scgsim/geometry/construction/native_geometry/__init__.py",
    "src/scgsim/geometry/construction/native_geometry/lowering.py",
    "src/scgsim/geometry/construction/native_geometry/port_mesh.py",
    "src/scgsim/geometry/construction/native_geometry/reports.py",
    "src/scgsim/geometry/construction/native_geometry/timing.py",
    "src/scgsim/geometry/construction/native_geometry/topology.py",
    "src/scgsim/geometry/construction/native_types.py",
    "src/scgsim/geometry/models/__init__.py",
    "src/scgsim/geometry/models/common.py",
    "src/scgsim/geometry/models/construction.py",
    "src/scgsim/geometry/models/grid.py",
    "src/scgsim/geometry/models/input.py",
    "src/scgsim/geometry/models/junctions.py",
    "src/scgsim/geometry/models/lumped.py",
    "src/scgsim/geometry/models/physical.py",
    "src/scgsim/geometry/models/regions.py",
    "src/scgsim/geometry/models/representation.py",
    "src/scgsim/geometry/models/tags.py",
    "src/scgsim/geometry/models/topology.py",
    "src/scgsim/geometry/planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/geometry.py",
    "src/scgsim/geometry/planning/contact_planning/intents.py",
    "src/scgsim/geometry/planning/contact_planning/membership.py",
    "src/scgsim/geometry/planning/contact_planning/overlap.py",
    "src/scgsim/geometry/planning/contact_planning/patches.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/domain_planning/__init__.py",
    "src/scgsim/geometry/planning/domain_planning/boundaries.py",
    "src/scgsim/geometry/planning/domain_planning/envelope.py",
    "src/scgsim/geometry/planning/domain_planning/faces.py",
    "src/scgsim/geometry/planning/domain_planning/hosts.py",
    "src/scgsim/geometry/planning/domain_planning/occupancy.py",
    "src/scgsim/geometry/planning/domain_planning/reconciliation.py",
    "src/scgsim/geometry/planning/domain_planning/sidewalls.py",
    "src/scgsim/geometry/planning/domain_planning/vacuum.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/junction_partition.py",
    "src/scgsim/geometry/planning/junction_planning/__init__.py",
    "src/scgsim/geometry/planning/junction_planning/boolean_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/contact.py",
    "src/scgsim/geometry/planning/junction_planning/core.py",
    "src/scgsim/geometry/planning/junction_planning/derivation.py",
    "src/scgsim/geometry/planning/junction_planning/emission.py",
    "src/scgsim/geometry/planning/junction_planning/exact_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/incidence.py",
    "src/scgsim/geometry/planning/junction_planning/local_frame.py",
    "src/scgsim/geometry/planning/junction_planning/model.py",
    "src/scgsim/geometry/planning/junction_planning/model_checks.py",
    "src/scgsim/geometry/planning/junction_planning/model_state.py",
    "src/scgsim/geometry/planning/junction_planning/partition.py",
    "src/scgsim/geometry/planning/junction_planning/selection.py",
    "src/scgsim/geometry/planning/junction_planning/terminal.py",
    "src/scgsim/geometry/planning/physical.py",
    "src/scgsim/geometry/planning/port_planning/__init__.py",
    "src/scgsim/geometry/planning/port_planning/geometry.py",
    "src/scgsim/geometry/planning/port_planning/lowering.py",
    "src/scgsim/geometry/planning/port_planning/route_a.py",
    "src/scgsim/geometry/planning/port_planning/route_b.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/representation.py",
    "src/scgsim/geometry/planning/surface_planning/__init__.py",
    "src/scgsim/geometry/planning/surface_planning/conductors.py",
    "src/scgsim/geometry/planning/surface_planning/contacts.py",
    "src/scgsim/geometry/planning/surface_planning/contracts.py",
    "src/scgsim/geometry/planning/surface_planning/contributions.py",
    "src/scgsim/geometry/planning/surface_planning/effective_sheets.py",
    "src/scgsim/geometry/planning/surface_planning/interface_surfaces.py",
    "src/scgsim/geometry/planning/surface_planning/partitions.py",
    "src/scgsim/geometry/planning/surface_planning/planner.py",
    "src/scgsim/geometry/planning/surface_planning/scope.py",
    "src/scgsim/geometry/planning/surface_planning/sheets.py",
    "src/scgsim/geometry/planning/surface_planning/sidewalls.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/topology_planning/__init__.py",
    "src/scgsim/geometry/planning/topology_planning/indexing.py",
    "src/scgsim/geometry/planning/topology_planning/ownership.py",
    "src/scgsim/geometry/planning/topology_planning/registry.py",
    "src/scgsim/geometry/planning/topology_planning/rings.py",
    "src/scgsim/geometry/planning/volume_planning/__init__.py",
    "src/scgsim/geometry/planning/volume_planning/bodies.py",
    "src/scgsim/geometry/planning/volume_planning/orientation.py",
    "src/scgsim/geometry/planning/volume_planning/planner.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/primitives/__init__.py",
    "src/scgsim/geometry/primitives/constants.py",
    "src/scgsim/geometry/primitives/entities.py",
    "src/scgsim/geometry/primitives/entity_validation.py",
    "src/scgsim/geometry/primitives/geometry_refs.py",
    "src/scgsim/geometry/primitives/grid.py",
    "src/scgsim/geometry/primitives/grid_cache.py",
    "src/scgsim/geometry/primitives/grid_overlay.py",
    "src/scgsim/geometry/primitives/grid_regions.py",
    "src/scgsim/geometry/primitives/loops.py",
    "src/scgsim/geometry/primitives/planar_loops.py",
    "src/scgsim/geometry/primitives/port_contract.py",
    "src/scgsim/geometry/primitives/spatial.py",
    "src/scgsim/geometry/primitives/surface_records.py",
    "src/scgsim/geometry/primitives/tagged_grid.py",
    "src/scgsim/geometry/primitives/tagged_grid_faces.py",
    "src/scgsim/geometry/primitives/tagged_grid_operations.py",
    "src/scgsim/geometry/source/__init__.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/adapter_records.py",
    "src/scgsim/geometry/source/curves.py",
    "src/scgsim/geometry/source/grid.py",
    "src/scgsim/geometry/source/ground.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/junctions.py",
    "src/scgsim/geometry/source/normalization.py",
    "src/scgsim/geometry/source/occurrence.py",
    "src/scgsim/geometry/source/occurrence_validation.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/plan_assembly.py",
    "src/scgsim/geometry/source/polygon_ops.py",
    "src/scgsim/geometry/source/ports.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/vacuum.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/preparation.py",
    "src/scgsim/presentation/__init__.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/__init__.py",
    "src/scgsim/semantics/epr.py",
    "src/scgsim/semantics/evidence.py",
    "src/scgsim/semantics/interfaces.py",
    "src/scgsim/semantics/materials.py",
    "src/scgsim/semantics/ownership.py",
    "src/scgsim/semantics/route_a.py",
    "src/scgsim/semantics/snapshot.py",
    "src/scgsim/visualization/__init__.py",
    "src/scgsim/visualization/loaders/__init__.py",
    "src/scgsim/visualization/loaders/aedt.py",
    "src/scgsim/visualization/loaders/aedt_receipt.py",
    "src/scgsim/visualization/loaders/common.py",
    "src/scgsim/visualization/loaders/palace.py",
    "src/scgsim/visualization/preview.py",
    "src/scgsim/visualization/rendering.py",
    "src/scgsim/visualization/scene.py",
)


def current_runtime_source_closure() -> tuple[str, ...]:
    """Recompute the current (V30) closure for roster generation and consistency checks.

    The shared seeds are V29's, less the modules issue #67 decision 8 removed.
    """
    source_root = Path(__file__).resolve().parents[3]
    aedt_root = source_root / "scgsim/aedt"
    aedt_modules = tuple(
        f"src/{path.relative_to(source_root).as_posix()}"
        for path in aedt_root.rglob("*.py")
    )
    retained_shared_seeds = tuple(
        path
        for path in _RUNTIME_SOURCE_V29_PATHS
        if not path.startswith("src/scgsim/aedt/")
        and path not in _REMOVED_BY_DECISION_8
    )
    visualization_seeds = (
        "src/scgsim/visualization/loaders/aedt.py",
        "src/scgsim/visualization/loaders/aedt_receipt.py",
    )
    return runtime_import_closure(
        source_root,
        (*aedt_modules, *visualization_seeds, *retained_shared_seeds),
    )


_RUNTIME_SOURCE_V28_PATHS = (
    "src/scgsim/__init__.py",
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
    "src/scgsim/diagnostics.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/compiler/__init__.py",
    "src/scgsim/geometry/compiler/builder.py",
    "src/scgsim/geometry/compiler/coverage_validation.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/stages.py",
    "src/scgsim/geometry/compiler/topology_validation.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/construction/__init__.py",
    "src/scgsim/geometry/construction/conformality.py",
    "src/scgsim/geometry/construction/curved.py",
    "src/scgsim/geometry/construction/curved_geometry/__init__.py",
    "src/scgsim/geometry/construction/curved_geometry/arrangement.py",
    "src/scgsim/geometry/construction/curved_geometry/boundaries.py",
    "src/scgsim/geometry/construction/curved_geometry/kernel.py",
    "src/scgsim/geometry/construction/curved_geometry/records.py",
    "src/scgsim/geometry/construction/curved_geometry/surfaces.py",
    "src/scgsim/geometry/construction/export.py",
    "src/scgsim/geometry/construction/gmsh_occ.py",
    "src/scgsim/geometry/construction/native.py",
    "src/scgsim/geometry/construction/native_geometry/__init__.py",
    "src/scgsim/geometry/construction/native_geometry/lowering.py",
    "src/scgsim/geometry/construction/native_geometry/port_mesh.py",
    "src/scgsim/geometry/construction/native_geometry/reports.py",
    "src/scgsim/geometry/construction/native_geometry/timing.py",
    "src/scgsim/geometry/construction/native_geometry/topology.py",
    "src/scgsim/geometry/construction/native_types.py",
    "src/scgsim/geometry/models/__init__.py",
    "src/scgsim/geometry/models/common.py",
    "src/scgsim/geometry/models/construction.py",
    "src/scgsim/geometry/models/grid.py",
    "src/scgsim/geometry/models/input.py",
    "src/scgsim/geometry/models/junctions.py",
    "src/scgsim/geometry/models/lumped.py",
    "src/scgsim/geometry/models/physical.py",
    "src/scgsim/geometry/models/regions.py",
    "src/scgsim/geometry/models/representation.py",
    "src/scgsim/geometry/models/tags.py",
    "src/scgsim/geometry/models/topology.py",
    "src/scgsim/geometry/planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/geometry.py",
    "src/scgsim/geometry/planning/contact_planning/intents.py",
    "src/scgsim/geometry/planning/contact_planning/membership.py",
    "src/scgsim/geometry/planning/contact_planning/overlap.py",
    "src/scgsim/geometry/planning/contact_planning/patches.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/domain_planning/__init__.py",
    "src/scgsim/geometry/planning/domain_planning/boundaries.py",
    "src/scgsim/geometry/planning/domain_planning/envelope.py",
    "src/scgsim/geometry/planning/domain_planning/faces.py",
    "src/scgsim/geometry/planning/domain_planning/hosts.py",
    "src/scgsim/geometry/planning/domain_planning/occupancy.py",
    "src/scgsim/geometry/planning/domain_planning/reconciliation.py",
    "src/scgsim/geometry/planning/domain_planning/sidewalls.py",
    "src/scgsim/geometry/planning/domain_planning/vacuum.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/junction_partition.py",
    "src/scgsim/geometry/planning/junction_planning/__init__.py",
    "src/scgsim/geometry/planning/junction_planning/boolean_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/contact.py",
    "src/scgsim/geometry/planning/junction_planning/core.py",
    "src/scgsim/geometry/planning/junction_planning/derivation.py",
    "src/scgsim/geometry/planning/junction_planning/emission.py",
    "src/scgsim/geometry/planning/junction_planning/exact_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/incidence.py",
    "src/scgsim/geometry/planning/junction_planning/local_frame.py",
    "src/scgsim/geometry/planning/junction_planning/model.py",
    "src/scgsim/geometry/planning/junction_planning/model_checks.py",
    "src/scgsim/geometry/planning/junction_planning/model_state.py",
    "src/scgsim/geometry/planning/junction_planning/partition.py",
    "src/scgsim/geometry/planning/junction_planning/selection.py",
    "src/scgsim/geometry/planning/junction_planning/terminal.py",
    "src/scgsim/geometry/planning/physical.py",
    "src/scgsim/geometry/planning/port_planning/__init__.py",
    "src/scgsim/geometry/planning/port_planning/geometry.py",
    "src/scgsim/geometry/planning/port_planning/lowering.py",
    "src/scgsim/geometry/planning/port_planning/route_a.py",
    "src/scgsim/geometry/planning/port_planning/route_b.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/representation.py",
    "src/scgsim/geometry/planning/surface_planning/__init__.py",
    "src/scgsim/geometry/planning/surface_planning/conductors.py",
    "src/scgsim/geometry/planning/surface_planning/contacts.py",
    "src/scgsim/geometry/planning/surface_planning/contracts.py",
    "src/scgsim/geometry/planning/surface_planning/contributions.py",
    "src/scgsim/geometry/planning/surface_planning/effective_sheets.py",
    "src/scgsim/geometry/planning/surface_planning/interface_surfaces.py",
    "src/scgsim/geometry/planning/surface_planning/partitions.py",
    "src/scgsim/geometry/planning/surface_planning/planner.py",
    "src/scgsim/geometry/planning/surface_planning/scope.py",
    "src/scgsim/geometry/planning/surface_planning/sheets.py",
    "src/scgsim/geometry/planning/surface_planning/sidewalls.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/topology_planning/__init__.py",
    "src/scgsim/geometry/planning/topology_planning/indexing.py",
    "src/scgsim/geometry/planning/topology_planning/ownership.py",
    "src/scgsim/geometry/planning/topology_planning/registry.py",
    "src/scgsim/geometry/planning/topology_planning/rings.py",
    "src/scgsim/geometry/planning/volume_planning/__init__.py",
    "src/scgsim/geometry/planning/volume_planning/bodies.py",
    "src/scgsim/geometry/planning/volume_planning/orientation.py",
    "src/scgsim/geometry/planning/volume_planning/planner.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/primitives/__init__.py",
    "src/scgsim/geometry/primitives/constants.py",
    "src/scgsim/geometry/primitives/entities.py",
    "src/scgsim/geometry/primitives/entity_validation.py",
    "src/scgsim/geometry/primitives/geometry_refs.py",
    "src/scgsim/geometry/primitives/grid.py",
    "src/scgsim/geometry/primitives/grid_cache.py",
    "src/scgsim/geometry/primitives/grid_overlay.py",
    "src/scgsim/geometry/primitives/grid_regions.py",
    "src/scgsim/geometry/primitives/loops.py",
    "src/scgsim/geometry/primitives/planar_loops.py",
    "src/scgsim/geometry/primitives/port_contract.py",
    "src/scgsim/geometry/primitives/spatial.py",
    "src/scgsim/geometry/primitives/surface_records.py",
    "src/scgsim/geometry/primitives/tagged_grid.py",
    "src/scgsim/geometry/primitives/tagged_grid_faces.py",
    "src/scgsim/geometry/primitives/tagged_grid_operations.py",
    "src/scgsim/geometry/primitives/topology_validation.py",
    "src/scgsim/geometry/source/__init__.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/adapter_records.py",
    "src/scgsim/geometry/source/curves.py",
    "src/scgsim/geometry/source/grid.py",
    "src/scgsim/geometry/source/ground.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/junctions.py",
    "src/scgsim/geometry/source/normalization.py",
    "src/scgsim/geometry/source/occurrence.py",
    "src/scgsim/geometry/source/occurrence_validation.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/plan_assembly.py",
    "src/scgsim/geometry/source/polygon_ops.py",
    "src/scgsim/geometry/source/ports.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/vacuum.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/preparation.py",
    "src/scgsim/presentation/__init__.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/__init__.py",
    "src/scgsim/semantics/epr.py",
    "src/scgsim/semantics/evidence.py",
    "src/scgsim/semantics/interfaces.py",
    "src/scgsim/semantics/materials.py",
    "src/scgsim/semantics/ownership.py",
    "src/scgsim/semantics/route_a.py",
    "src/scgsim/semantics/snapshot.py",
    "src/scgsim/visualization/__init__.py",
    "src/scgsim/visualization/loaders/__init__.py",
    "src/scgsim/visualization/loaders/aedt.py",
    "src/scgsim/visualization/loaders/aedt_receipt.py",
    "src/scgsim/visualization/loaders/common.py",
    "src/scgsim/visualization/loaders/palace.py",
    "src/scgsim/visualization/preview.py",
    "src/scgsim/visualization/rendering.py",
    "src/scgsim/visualization/scene.py",
)


_RUNTIME_SOURCE_V29_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V28_PATHS,
            "src/scgsim/geometry/planning/junction_planning/lattice.py",
        )
    )
)


_REMOVED_BY_DECISION_8 = ("src/scgsim/geometry/models/lumped.py",)

_RUNTIME_SOURCE_V30_PATHS = (
    "src/scgsim/__init__.py",
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
    "src/scgsim/aedt/epr/sgb_surfaces.py",
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
    "src/scgsim/aedt/preparation/sgb_model.py",
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
    "src/scgsim/aedt/runtime/families/sgb_boundaries.py",
    "src/scgsim/aedt/runtime/families/sgb_faces.py",
    "src/scgsim/aedt/runtime/families/sgb_import.py",
    "src/scgsim/aedt/runtime/handoff_validation.py",
    "src/scgsim/aedt/runtime/native/common.py",
    "src/scgsim/aedt/runtime/native/compatibility.py",
    "src/scgsim/aedt/runtime/native/gds_import.py",
    "src/scgsim/aedt/runtime/native/hfss_eigenmode.py",
    "src/scgsim/aedt/runtime/native/hfss_geometry.py",
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
    "src/scgsim/aedt/terminal_chain.py",
    "src/scgsim/aedt/versions.py",
    "src/scgsim/contracts/__init__.py",
    "src/scgsim/contracts/sgb_sidecar/__init__.py",
    "src/scgsim/contracts/sgb_sidecar/errors.py",
    "src/scgsim/contracts/sgb_sidecar/markers.py",
    "src/scgsim/contracts/sgb_sidecar/read.py",
    "src/scgsim/contracts/sgb_sidecar/records.py",
    "src/scgsim/contracts/sgb_sidecar/references.py",
    "src/scgsim/contracts/sgb_sidecar/schema.py",
    "src/scgsim/contracts/sgb_sidecar/structure.py",
    "src/scgsim/contracts/sgb_sidecar/version.py",
    "src/scgsim/diagnostics.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/compiler/__init__.py",
    "src/scgsim/geometry/compiler/builder.py",
    "src/scgsim/geometry/compiler/coverage_validation.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/stages.py",
    "src/scgsim/geometry/compiler/topology_validation.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/construction/__init__.py",
    "src/scgsim/geometry/construction/conformality.py",
    "src/scgsim/geometry/construction/curved.py",
    "src/scgsim/geometry/construction/curved_geometry/__init__.py",
    "src/scgsim/geometry/construction/curved_geometry/arrangement.py",
    "src/scgsim/geometry/construction/curved_geometry/boundaries.py",
    "src/scgsim/geometry/construction/curved_geometry/kernel.py",
    "src/scgsim/geometry/construction/curved_geometry/records.py",
    "src/scgsim/geometry/construction/curved_geometry/surfaces.py",
    "src/scgsim/geometry/construction/export.py",
    "src/scgsim/geometry/construction/gmsh_occ.py",
    "src/scgsim/geometry/construction/native.py",
    "src/scgsim/geometry/construction/native_geometry/__init__.py",
    "src/scgsim/geometry/construction/native_geometry/lowering.py",
    "src/scgsim/geometry/construction/native_geometry/port_mesh.py",
    "src/scgsim/geometry/construction/native_geometry/reports.py",
    "src/scgsim/geometry/construction/native_geometry/timing.py",
    "src/scgsim/geometry/construction/native_geometry/topology.py",
    "src/scgsim/geometry/construction/native_types.py",
    "src/scgsim/geometry/models/__init__.py",
    "src/scgsim/geometry/models/common.py",
    "src/scgsim/geometry/models/construction.py",
    "src/scgsim/geometry/models/grid.py",
    "src/scgsim/geometry/models/input.py",
    "src/scgsim/geometry/models/junctions.py",
    "src/scgsim/geometry/models/physical.py",
    "src/scgsim/geometry/models/regions.py",
    "src/scgsim/geometry/models/representation.py",
    "src/scgsim/geometry/models/tags.py",
    "src/scgsim/geometry/models/topology.py",
    "src/scgsim/geometry/planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/__init__.py",
    "src/scgsim/geometry/planning/contact_planning/geometry.py",
    "src/scgsim/geometry/planning/contact_planning/intents.py",
    "src/scgsim/geometry/planning/contact_planning/membership.py",
    "src/scgsim/geometry/planning/contact_planning/overlap.py",
    "src/scgsim/geometry/planning/contact_planning/patches.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/domain_planning/__init__.py",
    "src/scgsim/geometry/planning/domain_planning/boundaries.py",
    "src/scgsim/geometry/planning/domain_planning/envelope.py",
    "src/scgsim/geometry/planning/domain_planning/faces.py",
    "src/scgsim/geometry/planning/domain_planning/hosts.py",
    "src/scgsim/geometry/planning/domain_planning/occupancy.py",
    "src/scgsim/geometry/planning/domain_planning/reconciliation.py",
    "src/scgsim/geometry/planning/domain_planning/sidewalls.py",
    "src/scgsim/geometry/planning/domain_planning/vacuum.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/junction_partition.py",
    "src/scgsim/geometry/planning/junction_planning/__init__.py",
    "src/scgsim/geometry/planning/junction_planning/boolean_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/contact.py",
    "src/scgsim/geometry/planning/junction_planning/core.py",
    "src/scgsim/geometry/planning/junction_planning/derivation.py",
    "src/scgsim/geometry/planning/junction_planning/emission.py",
    "src/scgsim/geometry/planning/junction_planning/exact_geometry.py",
    "src/scgsim/geometry/planning/junction_planning/incidence.py",
    "src/scgsim/geometry/planning/junction_planning/lattice.py",
    "src/scgsim/geometry/planning/junction_planning/local_frame.py",
    "src/scgsim/geometry/planning/junction_planning/model.py",
    "src/scgsim/geometry/planning/junction_planning/model_checks.py",
    "src/scgsim/geometry/planning/junction_planning/model_state.py",
    "src/scgsim/geometry/planning/junction_planning/partition.py",
    "src/scgsim/geometry/planning/junction_planning/selection.py",
    "src/scgsim/geometry/planning/junction_planning/terminal.py",
    "src/scgsim/geometry/planning/physical.py",
    "src/scgsim/geometry/planning/port_planning/__init__.py",
    "src/scgsim/geometry/planning/port_planning/geometry.py",
    "src/scgsim/geometry/planning/port_planning/lowering.py",
    "src/scgsim/geometry/planning/port_planning/route_a.py",
    "src/scgsim/geometry/planning/port_planning/route_b.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/representation.py",
    "src/scgsim/geometry/planning/surface_planning/__init__.py",
    "src/scgsim/geometry/planning/surface_planning/conductors.py",
    "src/scgsim/geometry/planning/surface_planning/contacts.py",
    "src/scgsim/geometry/planning/surface_planning/contracts.py",
    "src/scgsim/geometry/planning/surface_planning/contributions.py",
    "src/scgsim/geometry/planning/surface_planning/effective_sheets.py",
    "src/scgsim/geometry/planning/surface_planning/interface_surfaces.py",
    "src/scgsim/geometry/planning/surface_planning/partitions.py",
    "src/scgsim/geometry/planning/surface_planning/planner.py",
    "src/scgsim/geometry/planning/surface_planning/scope.py",
    "src/scgsim/geometry/planning/surface_planning/sheets.py",
    "src/scgsim/geometry/planning/surface_planning/sidewalls.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/topology_planning/__init__.py",
    "src/scgsim/geometry/planning/topology_planning/indexing.py",
    "src/scgsim/geometry/planning/topology_planning/ownership.py",
    "src/scgsim/geometry/planning/topology_planning/registry.py",
    "src/scgsim/geometry/planning/topology_planning/rings.py",
    "src/scgsim/geometry/planning/volume_planning/__init__.py",
    "src/scgsim/geometry/planning/volume_planning/bodies.py",
    "src/scgsim/geometry/planning/volume_planning/orientation.py",
    "src/scgsim/geometry/planning/volume_planning/planner.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/primitives/__init__.py",
    "src/scgsim/geometry/primitives/constants.py",
    "src/scgsim/geometry/primitives/entities.py",
    "src/scgsim/geometry/primitives/entity_validation.py",
    "src/scgsim/geometry/primitives/geometry_refs.py",
    "src/scgsim/geometry/primitives/grid.py",
    "src/scgsim/geometry/primitives/grid_cache.py",
    "src/scgsim/geometry/primitives/grid_overlay.py",
    "src/scgsim/geometry/primitives/grid_regions.py",
    "src/scgsim/geometry/primitives/loops.py",
    "src/scgsim/geometry/primitives/planar_loops.py",
    "src/scgsim/geometry/primitives/port_contract.py",
    "src/scgsim/geometry/primitives/spatial.py",
    "src/scgsim/geometry/primitives/surface_records.py",
    "src/scgsim/geometry/primitives/tagged_grid.py",
    "src/scgsim/geometry/primitives/tagged_grid_faces.py",
    "src/scgsim/geometry/primitives/tagged_grid_operations.py",
    "src/scgsim/geometry/primitives/topology_validation.py",
    "src/scgsim/geometry/sgb/__init__.py",
    "src/scgsim/geometry/sgb/arrangement/__init__.py",
    "src/scgsim/geometry/sgb/arrangement/arcs.py",
    "src/scgsim/geometry/sgb/arrangement/dcel.py",
    "src/scgsim/geometry/sgb/assembly/__init__.py",
    "src/scgsim/geometry/sgb/assembly/builder.py",
    "src/scgsim/geometry/sgb/assembly/curved_walls.py",
    "src/scgsim/geometry/sgb/assembly/lineage.py",
    "src/scgsim/geometry/sgb/assembly/records.py",
    "src/scgsim/geometry/sgb/assembly/shells.py",
    "src/scgsim/geometry/sgb/assembly/walls.py",
    "src/scgsim/geometry/sgb/build.py",
    "src/scgsim/geometry/sgb/cutfamilies/__init__.py",
    "src/scgsim/geometry/sgb/cutfamilies/die_cuts.py",
    "src/scgsim/geometry/sgb/cutfamilies/gap_intervals.py",
    "src/scgsim/geometry/sgb/cutfamilies/generate.py",
    "src/scgsim/geometry/sgb/cutfamilies/inset.py",
    "src/scgsim/geometry/sgb/cutfamilies/inset_plan.py",
    "src/scgsim/geometry/sgb/cutfamilies/inset_regions.py",
    "src/scgsim/geometry/sgb/cutfamilies/requests.py",
    "src/scgsim/geometry/sgb/cutfamilies/supports.py",
    "src/scgsim/geometry/sgb/derivation/__init__.py",
    "src/scgsim/geometry/sgb/derivation/arc_edges.py",
    "src/scgsim/geometry/sgb/derivation/contacts.py",
    "src/scgsim/geometry/sgb/derivation/ids.py",
    "src/scgsim/geometry/sgb/derivation/labelled.py",
    "src/scgsim/geometry/sgb/derivation/layer.py",
    "src/scgsim/geometry/sgb/derivation/level_layers.py",
    "src/scgsim/geometry/sgb/derivation/slabs.py",
    "src/scgsim/geometry/sgb/emission/__init__.py",
    "src/scgsim/geometry/sgb/emission/measures.py",
    "src/scgsim/geometry/sgb/emission/numbers.py",
    "src/scgsim/geometry/sgb/emission/plan.py",
    "src/scgsim/geometry/sgb/emission/records.py",
    "src/scgsim/geometry/sgb/emission/surfaces.py",
    "src/scgsim/geometry/sgb/emission/walk.py",
    "src/scgsim/geometry/sgb/emission/writer.py",
    "src/scgsim/geometry/sgb/errors.py",
    "src/scgsim/geometry/sgb/exchange/__init__.py",
    "src/scgsim/geometry/sgb/exchange/guard.py",
    "src/scgsim/geometry/sgb/exchange/step.py",
    "src/scgsim/geometry/sgb/forking.py",
    "src/scgsim/geometry/sgb/identity.py",
    "src/scgsim/geometry/sgb/labelling/__init__.py",
    "src/scgsim/geometry/sgb/labelling/bands.py",
    "src/scgsim/geometry/sgb/labelling/dissolve.py",
    "src/scgsim/geometry/sgb/labelling/linking.py",
    "src/scgsim/geometry/sgb/labelling/membership.py",
    "src/scgsim/geometry/sgb/labelling/prism_set.py",
    "src/scgsim/geometry/sgb/model.py",
    "src/scgsim/geometry/sgb/noding/__init__.py",
    "src/scgsim/geometry/sgb/noding/snap_rounding.py",
    "src/scgsim/geometry/sgb/noding/spatial_hash.py",
    "src/scgsim/geometry/sgb/noding/tiling.py",
    "src/scgsim/geometry/sgb/parallel.py",
    "src/scgsim/geometry/sgb/plane.py",
    "src/scgsim/geometry/sgb/prisms/__init__.py",
    "src/scgsim/geometry/sgb/prisms/arcs.py",
    "src/scgsim/geometry/sgb/prisms/catalogue.py",
    "src/scgsim/geometry/sgb/prisms/curves.py",
    "src/scgsim/geometry/sgb/prisms/entities.py",
    "src/scgsim/geometry/sgb/prisms/footprints.py",
    "src/scgsim/geometry/sgb/prisms/levels.py",
    "src/scgsim/geometry/sgb/prisms/ports.py",
    "src/scgsim/geometry/sgb/records/__init__.py",
    "src/scgsim/geometry/sgb/records/band_regions.py",
    "src/scgsim/geometry/sgb/records/carriers.py",
    "src/scgsim/geometry/sgb/records/circle_math.py",
    "src/scgsim/geometry/sgb/records/families.py",
    "src/scgsim/geometry/sgb/records/grid.py",
    "src/scgsim/geometry/sgb/records/labels.py",
    "src/scgsim/geometry/sgb/records/prisms.py",
    "src/scgsim/geometry/sgb/records/support.py",
    "src/scgsim/geometry/sgb/routes/__init__.py",
    "src/scgsim/geometry/sgb/routes/face_names.py",
    "src/scgsim/geometry/sgb/routes/groups.py",
    "src/scgsim/geometry/sgb/routes/naming.py",
    "src/scgsim/geometry/sgb/routes/policy.py",
    "src/scgsim/geometry/sgb/sidecar/__init__.py",
    "src/scgsim/geometry/sgb/sidecar/aedt.py",
    "src/scgsim/geometry/sgb/sidecar/build.py",
    "src/scgsim/geometry/sgb/sidecar/expected.py",
    "src/scgsim/geometry/sgb/sidecar/features.py",
    "src/scgsim/geometry/sgb/sidecar/files.py",
    "src/scgsim/geometry/sgb/sidecar/groups.py",
    "src/scgsim/geometry/sgb/sidecar/lineage.py",
    "src/scgsim/geometry/sgb/slivers/__init__.py",
    "src/scgsim/geometry/sgb/slivers/feature_size.py",
    "src/scgsim/geometry/sgb/slivers/report.py",
    "src/scgsim/geometry/sgb/stopwatch.py",
    "src/scgsim/geometry/sgb/validation/__init__.py",
    "src/scgsim/geometry/sgb/validation/arrangements.py",
    "src/scgsim/geometry/sgb/validation/ids.py",
    "src/scgsim/geometry/sgb/validation/report.py",
    "src/scgsim/geometry/sgb/validation/topology.py",
    "src/scgsim/geometry/sgb/validation/validate.py",
    "src/scgsim/geometry/sgb/verification/__init__.py",
    "src/scgsim/geometry/sgb/verification/occt.py",
    "src/scgsim/geometry/sgb/verification/ocp.py",
    "src/scgsim/geometry/sgb/verification/tolerances.py",
    "src/scgsim/geometry/source/__init__.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/adapter_records.py",
    "src/scgsim/geometry/source/curves.py",
    "src/scgsim/geometry/source/grid.py",
    "src/scgsim/geometry/source/ground.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/junctions.py",
    "src/scgsim/geometry/source/normalization.py",
    "src/scgsim/geometry/source/occurrence.py",
    "src/scgsim/geometry/source/occurrence_validation.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/plan_assembly.py",
    "src/scgsim/geometry/source/polygon_ops.py",
    "src/scgsim/geometry/source/ports.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/vacuum.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/geometry/surface_epr/__init__.py",
    "src/scgsim/geometry/surface_epr/build.py",
    "src/scgsim/geometry/surface_epr/insets.py",
    "src/scgsim/geometry/surface_epr/specs.py",
    "src/scgsim/preparation.py",
    "src/scgsim/presentation/__init__.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/__init__.py",
    "src/scgsim/semantics/epr.py",
    "src/scgsim/semantics/evidence.py",
    "src/scgsim/semantics/interfaces.py",
    "src/scgsim/semantics/materials.py",
    "src/scgsim/semantics/ownership.py",
    "src/scgsim/semantics/route_a.py",
    "src/scgsim/semantics/snapshot.py",
    "src/scgsim/visualization/__init__.py",
    "src/scgsim/visualization/loaders/__init__.py",
    "src/scgsim/visualization/loaders/aedt.py",
    "src/scgsim/visualization/loaders/aedt_receipt.py",
    "src/scgsim/visualization/loaders/common.py",
    "src/scgsim/visualization/loaders/palace.py",
    "src/scgsim/visualization/preview.py",
    "src/scgsim/visualization/rendering.py",
    "src/scgsim/visualization/scene.py",
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
        SOURCE_SCHEMA_V27: _RUNTIME_SOURCE_V27_PATHS,
        SOURCE_SCHEMA_V28: _RUNTIME_SOURCE_V28_PATHS,
        SOURCE_SCHEMA_V29: _RUNTIME_SOURCE_V29_PATHS,
        SOURCE_SCHEMA_V30: _RUNTIME_SOURCE_V30_PATHS,
    }.get(schema)


def current_runtime_source_paths() -> tuple[str, ...]:
    """Return the current producer inventory in canonical order."""
    return _RUNTIME_SOURCE_V30_PATHS
