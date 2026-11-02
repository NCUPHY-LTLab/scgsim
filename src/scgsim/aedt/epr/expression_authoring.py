"""Compile the requested modal, surface, junction, and convergence field expressions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.epr.analysis import (
    electric_domain_energy_coefficient,
    epr_domain_context,
    junction_capacitive_energy_coefficient,
)
from scgsim.aedt.epr.cache_config import (
    expression_name,
    identity_sha256,
)
from scgsim.aedt.epr.expression_codec import compile_named_expression
from scgsim.aedt.epr.expression_loading import load_compiled_expressions
from scgsim.aedt.epr.fields import (
    field_integral_operations,
    install_variables,
    junction_voltage_operations,
)
from scgsim.aedt.epr.geometry import plan_inset_sheet_names
from scgsim.aedt.epr.native import (
    InsetSurfaceBinding,
)
from scgsim.aedt.epr.selection import surface_integral_groups
from scgsim.aedt.epr.surface import surface_energy_coefficients
from scgsim.aedt.epr.surface_references import bind_native_surface_references
from scgsim.aedt.runtime.native.common import detached_data
from scgsim.aedt.specs.epr import (
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    iter_expression_convergence,
    surface_evaluations,
)
from scgsim.aedt.specs.hfss import HfssEprAnalysisSpec, HfssEprSpec
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl


def _weighted_native_terms(terms: list[tuple[str, float]]) -> list[str]:
    operations: list[str] = []
    for name, coefficient in terms:
        if coefficient == 0.0:
            continue
        has_prior_term = bool(operations)
        operations.extend(
            (
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            )
        )
        if has_prior_term:
            operations.append("Operation('+')")
    if not operations:
        raise RuntimeError("normalized expression has no weighted terms")
    return operations


def _native_term_sum(terms: list[list[str]]) -> list[str]:
    if not terms:
        raise RuntimeError("normalized expression has no energy terms")
    operations: list[str] = []
    for term in terms:
        has_prior_term = bool(operations)
        operations.extend(term)
        if has_prior_term:
            operations.append("Operation('+')")
    return operations


def _normalized_surface_total_operations(
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    target: NormalizedSurfaceEprTotal,
) -> list[str]:
    if spec.epr_request is None:
        raise RuntimeError("normalized surface convergence requires an EPR request")
    groups = [
        group
        for group in surface_integral_groups(spec.geometry, spec.epr_request)
        if group["interface_kind"] == target.interface_kind
        and group.get("evaluation_kind", "requested_margin") == target.evaluation_kind
        and float(group["margin_um"]) == target.margin_um
    ]
    if not groups:
        raise RuntimeError("normalized surface convergence target has no owner groups")

    numerator_terms: list[tuple[str, float]] = []
    for group in groups:
        normal_coefficient, tangential_coefficient = surface_energy_coefficients(
            group["interface_kind"],
            float(group["film_thickness_m"]),
            float(group["film_relative_permittivity"]),
            float(group["substrate_relative_permittivity"]),
        )
        group_id = str(group["group_id"])
        if normal_coefficient != 0.0:
            numerator_terms.append(
                (
                    expression_name(
                        expressions,
                        purpose=f"electric_normal_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    normal_coefficient,
                )
            )
        if tangential_coefficient != 0.0:
            numerator_terms.append(
                (
                    expression_name(
                        expressions,
                        purpose=f"electric_tangential_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    tangential_coefficient,
                )
            )
    numerator_operations = _weighted_native_terms(numerator_terms)

    _, materials, domains = epr_domain_context(spec.geometry)
    denominator_terms: list[list[str]] = []
    for domain_id, domain in domains.items():
        material = materials[domain["material_id"]]
        coefficient = electric_domain_energy_coefficient(
            float(material["permittivity"])
        )
        name = expression_name(
            expressions,
            purpose=f"electric_volume_{domain_id}",
            selection_kind="volume",
            selection_id_key="semantic_id",
            selection_id=domain_id,
        )
        denominator_terms.append(
            [
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    for junction in spec.geometry.junctions:
        coefficient = junction_capacitive_energy_coefficient(
            junction.capacitance_f, junction.width_um
        )
        if coefficient == 0.0:
            continue
        real_name = expression_name(
            expressions,
            purpose=f"junction_voltage_real_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        imag_name = expression_name(
            expressions,
            purpose=f"junction_voltage_imag_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        denominator_terms.append(
            [
                f"NameOfExpression('{real_name}')",
                f"NameOfExpression('{real_name}')",
                "Operation('*')",
                f"NameOfExpression('{imag_name}')",
                f"NameOfExpression('{imag_name}')",
                "Operation('*')",
                "Operation('+')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    denominator_operations = _native_term_sum(denominator_terms)
    return [
        *numerator_operations,
        *denominator_operations,
        "Operation('/')",
    ]


def author_epr_expressions(  # noqa: PLR0913 -- preserve the saved-field and live authoring keyword contract.
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec | HfssEprAnalysisSpec,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    *,
    namespace: str = "prepared",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Author the complete requested field library and saved definition evidence."""
    return _ExpressionAuthor(
        app, run_dir, spec, native_geometry, inset_plan, namespace
    ).build()


@dataclass
class _ExpressionAuthor:
    """Keep modal source, compiled definitions, and binding evidence together."""

    app: Any
    run_dir: Path
    spec: HfssEprSpec | HfssEprAnalysisSpec
    native_geometry: dict[str, Any]
    inset_plan: dict[tuple[str, float], dict[str, Any]]
    namespace: str

    def build(self) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        self._initialize()
        self._install_sources()
        self._compile_volumes()
        self._compile_native_references()
        self._prepare_surface_groups()
        self._compile_constituents()
        self._compile_grouped_surfaces()
        self._compile_junctions()
        self._compile_convergence_targets()
        self._load_and_readback()
        return self._result()

    def _initialize(self) -> None:
        self.request = self.spec.epr_request
        self.surface_field_selectors = (
            self.spec.surface_field_selectors
            if isinstance(self.spec, HfssEprAnalysisSpec)
            else None
        )
        if self.request is None:
            raise RuntimeError(
                "EPR expression preparation requires an analysis request"
            )
        self.solution = f"{cast(EigenmodeRunControl, self.spec.run_control).setup_name} : LastAdaptive"
        self.evidence_dir = self.run_dir / "metadata/epr_expressions"
        self.expressions: list[dict[str, Any]] = []
        self.authoring_started = time.perf_counter()
        self.phase_seconds = {
            "pp_sources_seconds": 0.0,
            "sheet_binding_seconds": 0.0,
            "ordinary_compile_seconds": 0.0,
            "adjacent_compile_seconds": 0.0,
            "native_definition_readback_seconds": 0.0,
        }
        self.expression_counts = {"ordinary": 0, "adjacent": 0}
        self.expression_stage_seconds = {
            "volume": 0.0,
            "constituent": 0.0,
            "grouped": 0.0,
            "junction": 0.0,
        }
        self.compiled: list[
            tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]
        ] = []

    def _install_sources(self) -> None:
        started = time.perf_counter()
        # Keep a nonzero modal source selected when saving the project.
        pp_values = {
            f"scgsim_epr_pp_mode_{index}": "1" if index == 1 else "0"
            for index in range(
                1, cast(EigenmodeRunControl, self.spec.run_control).num_modes + 1
            )
        }
        self.pp_observed = install_variables(self.app, pp_values)
        self.source_assignment = {
            str(index): (f"scgsim_epr_pp_mode_{index}", "0deg")
            for index in range(
                1, cast(EigenmodeRunControl, self.spec.run_control).num_modes + 1
            )
        }
        if (
            self.app.edit_sources(self.source_assignment, eigenmode_stored_energy=False)
            is not True
        ):
            raise RuntimeError("native Eigenmode PP source-vector installation failed")
        self.phase_seconds["pp_sources_seconds"] = time.perf_counter() - started

    def _compile_volumes(self) -> None:
        self.objects = self.native_geometry["objects"]
        self.material_readback = self.native_geometry["material_readback"]
        for item in self.objects:
            if item["kind"] != "solution_domain":
                continue
            for quantity in ("effective_volume", "electric_volume", "magnetic_energy"):
                operations = field_integral_operations(
                    quantity=quantity, selection_name=item["object_name"]
                )
                self.expressions.append(
                    self._compile_timed(
                        "volume",
                        purpose=f"{quantity}_{item['semantic_id']}",
                        operations=operations,
                        solution=self.solution,
                        phase_degrees=0.0,
                        dependencies=self.pp_observed,
                        selection={
                            "kind": "volume",
                            "semantic_id": item["semantic_id"],
                            "object_name": item["object_name"],
                            "relative_permeability": self.material_readback[
                                item["material_id"]
                            ]["relative_permeability"],
                        },
                        namespace=self.namespace,
                    )
                )

    def _compile_native_references(self) -> None:
        self.selected_contributions = (
            {item.contribution_id for item in self.spec.geometry.contributions}
            if self.request.surface_contribution_ids is None
            else set(self.request.surface_contribution_ids)
        )
        requested_physical_surfaces = any(
            item.contribution_id in self.selected_contributions
            and item.interface_kind in {"MA", "MS", "SA"}
            for item in self.spec.geometry.contributions
        )
        self.native_references: list[dict[str, Any]] = []
        if self.spec.geometry.modeling == "solid" and requested_physical_surfaces:
            self.native_references = bind_native_surface_references(
                self.app, self.spec.geometry, self.native_geometry, self.request
            )
            if not self.native_references:
                raise RuntimeError(
                    "selected Solid EPR contributions have no native face references"
                )
            for reference in self.native_references:
                reference_id = reference["reference_id"]
                reference_selection = {
                    **detached_data(reference),
                    "kind": "native_surface_reference",
                }
                for integral_kind, quantity in (
                    ("native_surface_reference_normal", "electric_normal"),
                    ("native_surface_reference_tangential", "electric_tangential"),
                    ("native_surface_reference_area", "masked_area"),
                ):
                    operations = field_integral_operations(
                        quantity=quantity,
                        selection_name=reference["selection_name"],
                        adjacent_side=False,
                        normal_vector=(
                            reference["projection_normal"]
                            if quantity != "masked_area"
                            else None
                        ),
                    )
                    self.expressions.append(
                        self._compile_timed(
                            "native_surface_reference",
                            purpose=f"{integral_kind}_{reference_id}",
                            operations=operations,
                            solution=self.solution,
                            phase_degrees=0.0,
                            dependencies=self.pp_observed,
                            selection=reference_selection,
                            namespace=self.namespace,
                        )
                    )

    def _prepare_surface_groups(self) -> None:
        self.bindings = {
            str(item["binding_id"]): item
            for item in self.spec.geometry.surface_bindings
        }
        self.groups = surface_integral_groups(self.spec.geometry, self.request)
        self.native_by_binding = {
            item["binding_id"]: item
            for item in self.native_geometry["surface_selections"]
        }
        if len(self.native_by_binding) != len(
            self.native_geometry["surface_selections"]
        ):
            raise RuntimeError("native EPR surface binding IDs are not unique")
        self.group_by_member = {
            (member["binding_id"], group["evaluation_kind"], group["margin_um"]): group
            for group in self.groups
            for member in group["members"]
        }
        self.grouped_constituents: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.member_components: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
        # Reuse native sheet facts only within this authoring session; side stays per binding.
        self.sheet_facts: dict[str, dict[str, Any]] = {}
        self.sheet_names = plan_inset_sheet_names(
            self.spec.geometry.surface_bindings, self.inset_plan
        )
        self.selected_contributions = (
            {item.contribution_id for item in self.spec.geometry.contributions}
            if self.request.surface_contribution_ids is None
            else set(self.request.surface_contribution_ids)
        )

    def _compile_constituents(self) -> None:
        for selection in self.native_geometry["surface_selections"]:
            binding = self.bindings[selection["binding_id"]]
            if (
                selection["contribution_id"] not in self.selected_contributions
                or binding["contribution"]["classification"] == "MM"
            ):
                continue
            for evaluation_kind, margin in surface_evaluations(
                binding["margins_um"],
                policy=self.spec.geometry.source.get("surface_evaluation_policy"),
            ):
                margin_um = float(margin)
                member_key = (selection["binding_id"], evaluation_kind, margin_um)
                group = self.group_by_member[member_key]
                started = time.perf_counter()
                components = InsetSurfaceBinding(
                    self.app,
                    binding,
                    selection,
                    margin_um,
                    self.inset_plan,
                    self.sheet_facts,
                    self.sheet_names,
                    modeling=self.spec.geometry.modeling,
                    surface_field_selectors=self.surface_field_selectors,
                ).bind()
                self.phase_seconds["sheet_binding_seconds"] += (
                    time.perf_counter() - started
                )
                self.member_components[member_key] = components
                for component in components:
                    component_selection = {
                        **selection,
                        **component,
                        "margin_um": margin_um,
                        "evaluation_kind": evaluation_kind,
                        "integration_geometry": "klayout_complement_minkowski.v1",
                    }
                    for quantity in (
                        "electric_normal",
                        "electric_tangential",
                        "masked_area",
                    ):
                        operations = field_integral_operations(
                            quantity=quantity,
                            selection_name=component["selection_name"],
                            adjacent_side=component["adjacent_side"],
                            normal_vector=component["native_normal"],
                        )
                        authored = self._compile_timed(
                            "constituent",
                            purpose=(
                                f"{quantity}_{selection['binding_id']}_"
                                f"{evaluation_kind}_{margin_um:.17g}um_"
                                f"component_{component['component_index']}"
                            ),
                            operations=operations,
                            solution=self.solution,
                            phase_degrees=0.0,
                            dependencies=self.pp_observed,
                            selection={
                                **component_selection,
                                **(
                                    {"field_quantity": quantity}
                                    if self.surface_field_selectors
                                    and quantity != "masked_area"
                                    else {}
                                ),
                            },
                            namespace=self.namespace,
                            adjacent_selection_name=(
                                component["selection_name"]
                                if component["adjacent_side"]
                                and quantity != "masked_area"
                                else None
                            ),
                        )
                        self.grouped_constituents.setdefault(
                            (group["group_id"], quantity), []
                        ).append(authored)

    def _compile_grouped_surfaces(self) -> None:
        for group in self.groups:
            native_members = [
                {
                    "binding_id": member["binding_id"],
                    "source_selection_name": self.native_by_binding[
                        member["binding_id"]
                    ]["selection_name"],
                    "components": self.member_components[
                        (
                            member["binding_id"],
                            group["evaluation_kind"],
                            group["margin_um"],
                        )
                    ],
                    "geometry_empty": not self.member_components[
                        (
                            member["binding_id"],
                            group["evaluation_kind"],
                            group["margin_um"],
                        )
                    ],
                    "effective_domain_id": self.native_by_binding[member["binding_id"]][
                        "effective_domain_id"
                    ],
                }
                for member in group["members"]
            ]
            for quantity in ("electric_normal", "electric_tangential", "masked_area"):
                members = self.grouped_constituents.get(
                    (group["group_id"], quantity), []
                )
                if len(members) != sum(
                    len(item["components"]) for item in native_members
                ):
                    raise RuntimeError(
                        "grouped EPR expression lacks an inset component"
                    )
                members.sort(
                    key=lambda item: item["identity"]["selection"]["binding_id"]
                )
                operations: list[str] = []
                for member in members:
                    operations.append(f"NameOfExpression('{member['name']}')")
                    if len(operations) > 1:
                        operations.append("Operation('+')")
                if not operations:
                    # Every source member was geometrically empty at this margin.
                    operations = ["Scalar_Constant(0)"]
                dependencies = dict(self.pp_observed)
                for member in members:
                    dependencies.update(member["identity"]["dependencies"])
                self.expressions.append(
                    self._compile_timed(
                        "grouped",
                        purpose=f"{quantity}_{group['group_id']}",
                        operations=operations,
                        solution=self.solution,
                        phase_degrees=0.0,
                        dependencies=dependencies,
                        selection={
                            "kind": "surface_group",
                            **group,
                            "integration_geometry": "klayout_complement_minkowski.v1",
                            "polygon_approximation": {
                                "method": "klayout_complement_minkowski.v1",
                                "points_per_circle": 128,
                                "source_dbu_um": self.spec.geometry.source[
                                    "source_dbu_um"
                                ],
                                "klayout_version": self.bindings[
                                    group["members"][0]["binding_id"]
                                ]["mask_support"]["klayout_version"],
                            },
                            "native_member_selections": native_members,
                            "geometry_empty": not members,
                        },
                        namespace=self.namespace,
                    )
                )

    def _compile_junctions(self) -> None:
        junction_specs = {
            item.junction_id: item for item in self.spec.geometry.junctions
        }
        selected_junctions = (
            set(junction_specs)
            if self.request.junction_ids is None
            else set(self.request.junction_ids)
        )
        for selection in self.native_geometry["junctions"]:
            junction = junction_specs[selection["junction_id"]]
            # Every model junction contributes capacitive energy to the common
            # normalization.  junction_ids controls only reported junction rows.
            if junction.junction_id not in selected_junctions:
                report_selected = False
            else:
                report_selected = True
            for quantity in ("junction_voltage_real", "junction_voltage_imag"):
                operations = junction_voltage_operations(
                    quantity=quantity,
                    selection_name=selection["object_name"],
                    direction_xy=junction.direction_xy,
                )
                self.expressions.append(
                    self._compile_timed(
                        "junction",
                        purpose=f"{quantity}_{junction.junction_id}",
                        operations=operations,
                        solution=self.solution,
                        phase_degrees=0.0,
                        dependencies=self.pp_observed,
                        selection={
                            "kind": "junction_sheet",
                            "junction_id": junction.junction_id,
                            "object_name": selection["object_name"],
                            "direction_xy": list(junction.direction_xy),
                            "report_selected": report_selected,
                        },
                        namespace=self.namespace,
                    )
                )

    def _compile_convergence_targets(self) -> None:
        authored_targets: set[str] = set()
        convergence_control = (
            self.spec.expression_convergence
            if isinstance(self.spec, HfssEprSpec)
            else None
        )
        for convergence in iter_expression_convergence(convergence_control):
            target = convergence.target
            target_id = identity_sha256(target.to_payload())
            if target_id in authored_targets:
                continue
            authored_targets.add(target_id)
            if isinstance(target, NormalizedSurfaceEprTotal):
                purpose = "normalized_surface_epr_total"
                selection = {
                    "kind": "normalized_surface_total",
                    "interface_kind": target.interface_kind,
                    "evaluation_kind": target.evaluation_kind,
                    "margin_um": target.margin_um,
                }
                operations = _normalized_surface_total_operations(
                    cast(HfssEprSpec, self.spec), self.expressions, target
                )
            elif isinstance(target, NativeExpressionDefinition):
                purpose = "custom_convergence"
                selection = {
                    "kind": "custom_convergence",
                    "definition_name": target.name,
                }
                operations = list(target.operations)
            else:
                raise TypeError("unsupported expression-cache convergence target")
            authored = self._compile_timed(
                "convergence",
                purpose=purpose,
                operations=operations,
                solution=self.solution,
                phase_degrees=0.0,
                dependencies=self.pp_observed,
                selection=selection,
                namespace=self.namespace,
            )
            authored["cache_only"] = True
            authored["convergence_target"] = target.to_payload()
            self.expressions.append(authored)

    def _load_and_readback(self) -> None:
        self.batch = load_compiled_expressions(
            self.app, self.compiled, self.evidence_dir
        )
        for key in (
            "collision_check_seconds",
            "evidence_write_seconds",
            "batch_import_seconds",
            "postload_readback_seconds",
            "native_definition_readback_seconds",
        ):
            self.phase_seconds[key] = self.batch[key]
        self.selector_bindings = []
        if self.surface_field_selectors:
            for expression, _, _ in self.compiled:
                selection = expression["identity"]["selection"]
                if selection.get("field_quantity") not in {
                    "electric_normal",
                    "electric_tangential",
                }:
                    continue
                self.selector_bindings.append(
                    {
                        "binding_id": selection["binding_id"],
                        "contribution_id": selection["contribution_id"],
                        "component_index": selection["component_index"],
                        "margin_um": selection["margin_um"],
                        "evaluation_kind": selection["evaluation_kind"],
                        "selection_name": selection["selection_name"],
                        "field_quantity": selection["field_quantity"],
                        "field_selector_binding": selection["field_selector_binding"],
                        "expression_name": expression["name"],
                        "expression_identity_sha256": expression["identity"]["sha256"],
                        # Existing SaveNamedExpressions readback, not a reconstructed claim.
                        "native_readback": expression["native_readback"],
                    }
                )

    def _result(self) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        authoring_seconds = time.perf_counter() - self.authoring_started
        other_seconds = max(0.0, authoring_seconds - sum(self.phase_seconds.values()))
        return (
            self.expressions,
            {
                "source_assignment": self.source_assignment,
                "postprocessing_variables": self.pp_observed,
                "native_expression_readback": self.batch["native_expression_readback"],
                **(
                    {"surface_field_selector_bindings": self.selector_bindings}
                    if self.surface_field_selectors
                    else {}
                ),
                "native_surface_references": detached_data(self.native_references),
            },
            {
                **{key: round(value, 6) for key, value in self.phase_seconds.items()},
                "other_seconds": round(other_seconds, 6),
                "ordinary_expression_count": self.expression_counts["ordinary"],
                "adjacent_expression_count": self.expression_counts["adjacent"],
                "expression_stage_compile_seconds": {
                    key: round(value, 6)
                    for key, value in self.expression_stage_seconds.items()
                },
                "batch_library_sha256": self.batch["library_sha256"],
                "native_definition_count": self.batch["native_definition_count"],
                "unique_inset_sheet_count": len(self.sheet_facts),
            },
        )

    def _compile_timed(self, stage: str, **kwargs: Any) -> dict[str, Any]:
        adjacent = kwargs.get("adjacent_selection_name") is not None
        started = time.perf_counter()
        result = compile_named_expression(**kwargs)
        elapsed = time.perf_counter() - started
        kind = "adjacent" if adjacent else "ordinary"
        self.phase_seconds[f"{kind}_compile_seconds"] += elapsed
        self.expression_stage_seconds[stage] = (
            self.expression_stage_seconds.get(stage, 0.0) + elapsed
        )
        self.expression_counts[kind] += 1
        self.compiled.append(result)
        return result[0]
