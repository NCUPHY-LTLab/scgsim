"""Version-bound source inventories; historical paths never follow relocations."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
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


SOURCE_SCHEMA_V27 = "scgsim.aedt.runtime-source.v27"


SOURCE_SCHEMA = SOURCE_SCHEMA_V27


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


def _current_runtime_source_v27_paths() -> tuple[str, ...]:
    """Build V27 from every AEDT module and the retained shared entry points."""
    source_root = Path(__file__).resolve().parents[3]
    aedt_root = source_root / "scgsim/aedt"
    aedt_modules = tuple(
        f"src/{path.relative_to(source_root).as_posix()}"
        for path in aedt_root.rglob("*.py")
    )
    retained_shared_seeds = tuple(
        path
        for path in _RUNTIME_SOURCE_V26_PATHS
        if not path.startswith("src/scgsim/aedt/")
    )
    visualization_seeds = (
        "src/scgsim/visualization/loaders/aedt.py",
        "src/scgsim/visualization/loaders/aedt_receipt.py",
    )
    return runtime_import_closure(
        source_root,
        (*aedt_modules, *visualization_seeds, *retained_shared_seeds),
    )


_RUNTIME_SOURCE_V27_PATHS = _current_runtime_source_v27_paths()


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
    }.get(schema)


def current_runtime_source_paths() -> tuple[str, ...]:
    """Return the current producer inventory in canonical order."""
    return _RUNTIME_SOURCE_V27_PATHS
