"""Submit live EPR expression caches and verify exact saved setup configuration."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.epr.cache_config import (
    cache_items,
    identity_sha256,
    native_expression_cache,
)
from scgsim.aedt.epr.evidence import SavedCacheEvidence
from scgsim.aedt.epr.expression_authoring import author_epr_expressions
from scgsim.aedt.runtime.native.common import detached_data
from scgsim.aedt.runtime.native.hfss_eigenmode import (
    submit_eigenmode_cache,
)
from scgsim.aedt.runtime.native.saved_geometry import read_saved_hfss_design
from scgsim.aedt.specs.hfss import HfssEprSpec
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl


def _submit_epr_cache(
    app: Any,
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    authoring: dict[str, Any],
) -> dict[str, Any]:
    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR cache submission requires an analysis request")
    selected_modes = (
        tuple(range(1, cast(EigenmodeRunControl, spec.run_control).num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    items = cache_items(
        expressions,
        cast(EigenmodeRunControl, spec.run_control).num_modes,
        selected_modes,
        spec.expression_convergence,
    )
    raw_cache = native_expression_cache(items)
    submit_eigenmode_cache(
        app,
        cast(EigenmodeRunControl, spec.run_control),
        expression_cache=raw_cache,
        use_cache_for=["Pass"],
    )
    return {
        "schema_version": "scgsim.aedt.epr-cache-request.v1",
        "setup_name": cast(EigenmodeRunControl, spec.run_control).setup_name,
        "use_cache_for": ["Pass"],
        "items": items,
        "source_assignment": authoring["source_assignment"],
        "postprocessing_variables": authoring["postprocessing_variables"],
        "native_surface_references": detached_data(
            authoring["native_surface_references"]
        ),
        "native_readback": {
            "status": "PENDING_SAVED_SETUP_READBACK",
            "reason": (
                "submitted cache settings are not saved-setup readback; "
                "saved properties are inspected after the project save"
            ),
        },
        "raw_cache_args_sha256": identity_sha256(raw_cache),
    }


def prepare_expressions_and_cache(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Author the prepared library before submitting its solve-time cache request."""
    if inset_plan is None:
        raise RuntimeError("EPR inset geometry must be precomputed before Desktop")
    expressions, authoring, phase_timings = author_epr_expressions(
        app,
        run_dir,
        spec,
        native_geometry,
        inset_plan,
    )
    started = time.perf_counter()
    cache = _submit_epr_cache(app, spec, expressions, authoring)
    phase_timings["cache_seconds"] = round(time.perf_counter() - started, 6)
    return expressions, cache, phase_timings


def read_cache(
    app: Any, spec: HfssEprSpec, submitted: list[dict[str, Any]]
) -> dict[str, Any]:
    """Read exact saved setup bytes and retain requested cache configuration evidence."""
    return _SavedCacheReadback(app, spec, submitted).read()


@dataclass
class _SavedCacheReadback:
    """Keep saved setup identity, persisted items, and comparisons together."""

    app: Any
    spec: HfssEprSpec
    submitted: list[dict[str, Any]]

    def read(self) -> dict[str, Any]:
        self._read_saved_setup()
        self._decode_items()
        self._compare_targets()
        return self._payload()

    def _read_saved_setup(self) -> None:
        if self.app.design_name != self.spec.design_name:
            raise RuntimeError(
                "EPR cache readback design differs from the bound request"
            )
        self.project_sha256, self.design = read_saved_hfss_design(
            self.app.project_file,
            self.spec.design_name,
            purpose="EPR expression-cache configuration",
        )
        analysis = self.design.get("AnalysisSetup")
        setups = analysis.get("SolveSetups") if isinstance(analysis, Mapping) else None
        setup_names = (
            [name for name, value in setups.items() if isinstance(value, Mapping)]
            if isinstance(setups, Mapping)
            else []
        )
        if setup_names != [cast(EigenmodeRunControl, self.spec.run_control).setup_name]:
            raise RuntimeError("saved EPR project does not contain one exact setup")
        self.raw = cast(Mapping[str, Any], setups)[
            cast(EigenmodeRunControl, self.spec.run_control).setup_name
        ]
        self.use_cache_for = self.raw.get("UseCacheFor")
        if isinstance(self.use_cache_for, str):
            self.use_cache_for = [self.use_cache_for]
        expression_cache = self.raw.get("ExpressionCache")
        if not isinstance(expression_cache, dict):
            raise RuntimeError("saved EPR setup lacks ExpressionCache")
        self.raw_items = expression_cache.get("CacheItem")
        if isinstance(self.raw_items, dict):
            self.raw_items = [self.raw_items]
        if not isinstance(self.raw_items, list):
            raise RuntimeError("saved EPR ExpressionCache items are invalid")

    def _decode_items(self) -> None:
        self.persisted = []
        self.convergence_field_names = (
            "IsConvergence",
            "UseRelativeConvergence",
            "MaxConvergenceDelta",
            "MaxConvergeValue",
        )
        self.convergence_fields = []
        for item in cast(list[object], self.raw_items):
            if not isinstance(item, dict):
                raise RuntimeError("saved EPR cache item is invalid")
            self.convergence_fields.append(
                {
                    "title": item.get("Title"),
                    "values": {
                        name: item[name]
                        for name in self.convergence_field_names
                        if name in item
                    },
                    "unavailable": [
                        name
                        for name in self.convergence_field_names
                        if name not in item
                    ],
                }
            )
            self.persisted.append(
                {
                    "title": item.get("Title"),
                    "expression": item.get("Expression"),
                    "intrinsics": str(item.get("Intrinsics", "")).replace("\\'", "'"),
                }
            )

    def _compare_targets(self) -> None:
        by_title = {item["title"]: item for item in self.convergence_fields}
        self.targets = []
        for item in self.submitted:
            if not item["is_convergence"]:
                continue
            native_request = native_expression_cache([item])[1]
            requested_fields = {
                name: native_request[native_request.index(f"{name}:=") + 1]
                for name in self.convergence_field_names
            }
            observation = by_title[item["title"]]
            self.targets.append(
                {
                    "control_id": item["control_id"],
                    "mode": item["mode"],
                    "title": item["title"],
                    "expression": item["expression"],
                    "expression_identity_sha256": item["expression_identity_sha256"],
                    "requested_control": item["requested_control"],
                    "requested_native_fields": requested_fields,
                    "observed_fields": observation,
                    "raw_representation_differences": [
                        name
                        for name, value in observation["values"].items()
                        if value != requested_fields[name]
                    ],
                    "comparison_qualification": (
                        "Raw native/request representations only; missing fields stay unavailable, "
                        "configuration equality is not convergence evidence"
                    ),
                }
            )

    def _payload(self) -> dict[str, Any]:
        frequency_fields = (
            "MaxDeltaFreq",
            "MaximumPasses",
            "MinimumPasses",
            "MinimumConvergedPasses",
            "PercentRefinement",
        )

        return SavedCacheEvidence.from_payload(
            self.submitted,
            {
                "convergence_targets": self.targets,
                "saved_configuration_source": {
                    "project_sha256": self.project_sha256,
                    "project_file_name": Path(self.app.project_file).name,
                    "design_name": self.spec.design_name,
                    "setup_name": cast(
                        EigenmodeRunControl, self.spec.run_control
                    ).setup_name,
                    "basis": "uncached pinned load_entire_aedt_file of exact saved project",
                },
                "frequency_control": {
                    "requested": cast(
                        EigenmodeRunControl, self.spec.run_control
                    ).to_payload(),
                    "observed_fields": {
                        name: self.raw[name]
                        for name in frequency_fields
                        if name in self.raw
                    },
                    "unavailable": [
                        name for name in frequency_fields if name not in self.raw
                    ],
                },
                "use_cache_for": self.use_cache_for,
                "items": self.persisted,
                "convergence_fields": self.convergence_fields,
            },
        ).to_payload()
