"""Author the named field-integral expressions an EPR analysis needs.

Owns: deciding which field integrals an analysis needs (domain volumes, native
face references, inset surfaces, junctions and the optional convergence
expression) and compiling and loading them through `fields.py`.
Does not own: submitting them as the setup's expression cache (`cache.py`), the
normalized-total convergence stack (`normalized_total.py`), reading traces back
(`history.py`) or the energy arithmetic (`analysis.py`).

EPR is the energy participation ratio. Its surface terms use the MA
(metal-air), MS (metal-substrate) and SA (substrate-air) interfaces defined in
docs/eigenmode-epr.qmd; the formulas are in docs/specs/aedt-runtime.qmd.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scgsim.aedt.epr.fields import (
    ExpressionRecipe,
    compile_named_expression,
    field_integral_operations,
    install_variables,
    junction_voltage_operations,
    load_compiled_expressions,
)
from scgsim.aedt.epr.geometry import plan_inset_sheet_names
from scgsim.aedt.epr.native import (
    InsetSheetBook,
    bind_inset_surface_selections,
    bind_native_surface_references,
)
from scgsim.aedt.epr.normalized_total import _normalized_surface_total_operations
from scgsim.aedt.epr.selection import surface_integral_groups
from scgsim.aedt.runtime.native.common import detached_data
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    surface_evaluations,
)
from scgsim.aedt.specs.hfss_prepared import HfssEprSpec

# --- Entry point ------------------------------------------------------------


def _author_epr_expressions(  # noqa: PLR0913 -- one argument per authoring input.
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    *,
    namespace: str = "prepared",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compile and load every field integral the EPR analysis needs.

    Returns the authored expressions, the native evidence for the receipt and the
    per-phase timings. The steps run in a fixed order because AEDT objects, sheets
    and variables created by one step are read by the next.
    """

    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR expression preparation requires an analysis request")
    author = _ExpressionAuthor(app, spec, request, namespace)
    authoring_started = time.perf_counter()

    source_assignment = author.install_modal_sources()
    _author_volume_expressions(author, native_geometry)
    native_references = _author_native_reference_expressions(author, native_geometry)
    unique_sheet_count = _author_surface_expressions(
        author, native_geometry, inset_plan
    )
    _author_junction_expressions(author, native_geometry)
    _author_convergence_expression(author)

    batch = load_compiled_expressions(
        app, author.compiled, run_dir / "metadata/epr_expressions"
    )
    for key in (
        "collision_check_seconds",
        "evidence_write_seconds",
        "batch_import_seconds",
        "postload_readback_seconds",
        "native_definition_readback_seconds",
    ):
        author.phase_seconds[key] = batch[key]
    authoring_seconds = time.perf_counter() - authoring_started
    other_seconds = max(0.0, authoring_seconds - sum(author.phase_seconds.values()))
    return (
        author.expressions,
        {
            "source_assignment": source_assignment,
            "postprocessing_variables": author.pp_observed,
            "native_expression_readback": batch["native_expression_readback"],
            "native_surface_references": detached_data(native_references),
        },
        {
            **{key: round(value, 6) for key, value in author.phase_seconds.items()},
            "other_seconds": round(other_seconds, 6),
            "ordinary_expression_count": author.expression_counts["ordinary"],
            "adjacent_expression_count": author.expression_counts["adjacent"],
            "expression_stage_compile_seconds": {
                key: round(value, 6) for key, value in author.stage_seconds.items()
            },
            "batch_library_sha256": batch["library_sha256"],
            "native_definition_count": batch["native_definition_count"],
            "unique_inset_sheet_count": unique_sheet_count,
        },
    )


class _ExpressionAuthor:
    """Compiles expressions in call order and keeps the timing evidence.

    Compiling is local; nothing reaches AEDT until `load_compiled_expressions`.
    Every expression is evaluated at phase 0 deg with the one-hot mode variables
    (`scgsim_epr_pp_mode_<n>`) as dependencies, so the cache can later select one
    mode at a time.
    """

    def __init__(
        self,
        app: Any,
        spec: HfssEprSpec,
        request: EprAnalysisRequest,
        namespace: str,
    ) -> None:
        self.app = app
        self.spec = spec
        self.request = request
        self.namespace = namespace
        self.solution = f"{spec.run_control.setup_name} : LastAdaptive"
        self.expressions: list[dict[str, Any]] = []
        self.compiled: list[
            tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]
        ] = []
        self.pp_observed: dict[str, str] = {}
        self.phase_seconds = {
            "pp_sources_seconds": 0.0,
            "sheet_binding_seconds": 0.0,
            "ordinary_compile_seconds": 0.0,
            "adjacent_compile_seconds": 0.0,
            "native_definition_readback_seconds": 0.0,
        }
        self.expression_counts = {"ordinary": 0, "adjacent": 0}
        self.stage_seconds = {
            "volume": 0.0,
            "constituent": 0.0,
            "grouped": 0.0,
            "junction": 0.0,
        }

    def install_modal_sources(self) -> dict[str, tuple[str, str]]:
        """Create the one-hot mode variables and bind them as the source vector."""

        started = time.perf_counter()
        num_modes = self.spec.run_control.num_modes
        # Keep a nonzero modal source selected when saving the project.
        pp_values = {
            f"scgsim_epr_pp_mode_{index}": "1" if index == 1 else "0"
            for index in range(1, num_modes + 1)
        }
        self.pp_observed = install_variables(self.app, pp_values)
        source_assignment = {
            str(index): (f"scgsim_epr_pp_mode_{index}", "0deg")
            for index in range(1, num_modes + 1)
        }
        if (
            self.app.edit_sources(source_assignment, eigenmode_stored_energy=False)
            is not True
        ):
            raise RuntimeError("native Eigenmode PP source-vector installation failed")
        self.phase_seconds["pp_sources_seconds"] = time.perf_counter() - started
        return source_assignment

    def author(
        self,
        stage: str,
        purpose: str,
        operations: list[str],
        selection: dict[str, Any],
        *,
        adjacent_selection_name: str | None = None,
    ) -> dict[str, Any]:
        """Compile one expression, timed under `stage`; return its record."""

        return self._compile(
            stage,
            self._recipe(
                purpose,
                operations,
                selection,
                self.pp_observed,
                adjacent_selection_name,
            ),
        )

    def add(
        self,
        stage: str,
        purpose: str,
        operations: list[str],
        selection: dict[str, Any],
        *,
        dependencies: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compile an expression and keep it as a top-level analysis expression."""

        recipe = self._recipe(
            purpose,
            operations,
            selection,
            self.pp_observed if dependencies is None else dependencies,
            None,
        )
        record = self._compile(stage, recipe)
        self.expressions.append(record)
        return record

    def _recipe(
        self,
        purpose: str,
        operations: list[str],
        selection: dict[str, Any],
        dependencies: dict[str, str],
        adjacent_selection_name: str | None,
    ) -> ExpressionRecipe:
        return ExpressionRecipe(
            purpose=purpose,
            operations=operations,
            solution=self.solution,
            phase_degrees=0.0,
            dependencies=dependencies,
            selection=selection,
            namespace=self.namespace,
            adjacent_selection_name=adjacent_selection_name,
        )

    def _compile(self, stage: str, recipe: ExpressionRecipe) -> dict[str, Any]:
        started = time.perf_counter()
        result = compile_named_expression(recipe)
        elapsed = time.perf_counter() - started
        kind = "adjacent" if recipe.adjacent_selection_name is not None else "ordinary"
        self.phase_seconds[f"{kind}_compile_seconds"] += elapsed
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + elapsed
        self.expression_counts[kind] += 1
        self.compiled.append(result)
        return result[0]


def _selected_contribution_ids(
    spec: HfssEprSpec, request: EprAnalysisRequest
) -> set[str]:
    """Contribution ids the request asks for; all of them when it names none."""

    if request.surface_contribution_ids is None:
        return {item.contribution_id for item in spec.geometry.contributions}
    return set(request.surface_contribution_ids)


# --- Volumes ----------------------------------------------------------------


def _author_volume_expressions(
    author: _ExpressionAuthor, native_geometry: dict[str, Any]
) -> None:
    """Author volume, |E|^2 and |H|^2 integrals for every solution domain.

    The electric integral is the denominator of the participation ratio. The
    magnetic one is weighted later by the domain's relative permeability, which is
    stored with the selection.
    """

    material_readback = native_geometry["material_readback"]
    for item in native_geometry["objects"]:
        if item["kind"] != "solution_domain":
            continue
        for quantity in ("effective_volume", "electric_volume", "magnetic_energy"):
            operations = field_integral_operations(
                quantity=quantity, selection_name=item["object_name"]
            )
            author.add(
                "volume",
                purpose=f"{quantity}_{item['semantic_id']}",
                operations=operations,
                selection={
                    "kind": "volume",
                    "semantic_id": item["semantic_id"],
                    "object_name": item["object_name"],
                    "relative_permeability": material_readback[item["material_id"]][
                        "relative_permeability"
                    ],
                },
            )


# --- Native face references (Solid modeling) --------------------------------


def _author_native_reference_expressions(
    author: _ExpressionAuthor, native_geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    """For Solid models, integrate directly on AEDT's own interface faces.

    Only models built as solids have native faces to reference; planar models use
    the inset sheets of `_author_surface_expressions` instead.
    """

    spec = author.spec
    selected_contributions = _selected_contribution_ids(spec, author.request)
    requested_physical_surfaces = any(
        item.contribution_id in selected_contributions
        and item.interface_kind in {"MA", "MS", "SA"}
        for item in spec.geometry.contributions
    )
    if spec.geometry.modeling != "solid" or not requested_physical_surfaces:
        return []
    native_references = bind_native_surface_references(
        author.app, spec.geometry, native_geometry, author.request
    )
    if not native_references:
        raise RuntimeError(
            "selected Solid EPR contributions have no native face references"
        )
    for reference in native_references:
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
            author.add(
                "native_surface_reference",
                purpose=f"{integral_kind}_{reference['reference_id']}",
                operations=operations,
                selection=reference_selection,
            )
    return native_references


# --- Inset surfaces (planar modeling) ---------------------------------------


@dataclass(slots=True)
class _SurfacePlan:
    """Surface bindings, owner groups and native selections for one analysis."""

    bindings: dict[str, Any]
    groups: list[dict[str, Any]]
    native_by_binding: dict[str, Any]
    group_by_member: dict[tuple[str, str, float], dict[str, Any]]
    # Filled while authoring: per-group constituent expressions, per-member inset
    # components, and native sheet facts reused within this authoring session.
    grouped_constituents: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    member_components: dict[tuple[str, str, float], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    sheet_facts: dict[str, dict[str, Any]] = field(default_factory=dict)


def _surface_plan(
    author: _ExpressionAuthor, native_geometry: dict[str, Any]
) -> _SurfacePlan:
    """Pair every requested surface binding with the group that owns its member."""

    spec = author.spec
    bindings = {
        str(item["binding_id"]): item for item in spec.geometry.surface_bindings
    }
    groups = surface_integral_groups(spec.geometry, author.request)
    native_by_binding = {
        item["binding_id"]: item for item in native_geometry["surface_selections"]
    }
    if len(native_by_binding) != len(native_geometry["surface_selections"]):
        raise RuntimeError("native EPR surface binding IDs are not unique")
    group_by_member = {
        (member["binding_id"], group["evaluation_kind"], group["margin_um"]): group
        for group in groups
        for member in group["members"]
    }
    return _SurfacePlan(bindings, groups, native_by_binding, group_by_member)


def _author_surface_expressions(
    author: _ExpressionAuthor,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]],
) -> int:
    """Author the inset-surface integrals and the group sums built from them.

    Each owner group (an interface kind at one margin) is the sum of its
    members' per-component integrals. Returns the number of unique inset sheets.
    """

    plan = _surface_plan(author, native_geometry)
    _author_constituent_expressions(author, native_geometry, inset_plan, plan)
    for group in plan.groups:
        _author_group_expressions(author, group, plan)
    return len(plan.sheet_facts)


def _author_constituent_expressions(
    author: _ExpressionAuthor,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    plan: _SurfacePlan,
) -> None:
    """Bind each member's inset sheets and author one integral per component."""

    spec = author.spec
    sheet_names = plan_inset_sheet_names(spec.geometry.surface_bindings, inset_plan)
    sheets = InsetSheetBook(
        inset_plan, sheet_names, plan.sheet_facts, spec.geometry.modeling
    )
    selected_contributions = _selected_contribution_ids(spec, author.request)
    for selection in native_geometry["surface_selections"]:
        binding = plan.bindings[selection["binding_id"]]
        # MM (metal-metal) bindings carry no thin-film energy.
        if (
            selection["contribution_id"] not in selected_contributions
            or binding["contribution"]["classification"] == "MM"
        ):
            continue
        for evaluation_kind, margin in surface_evaluations(
            binding["margins_um"],
            policy=spec.geometry.source.get("surface_evaluation_policy"),
        ):
            margin_um = float(margin)
            member_key = (selection["binding_id"], evaluation_kind, margin_um)
            group = plan.group_by_member[member_key]
            started = time.perf_counter()
            components = bind_inset_surface_selections(
                author.app,
                binding,
                selection,
                margin_um,
                sheets,
            )
            author.phase_seconds["sheet_binding_seconds"] += (
                time.perf_counter() - started
            )
            plan.member_components[member_key] = components
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
                    authored = _author_component_integral(
                        author,
                        quantity,
                        component,
                        component_selection,
                        label=(
                            f"{selection['binding_id']}_{evaluation_kind}_"
                            f"{margin_um:.17g}um"
                        ),
                    )
                    plan.grouped_constituents.setdefault(
                        (group["group_id"], quantity), []
                    ).append(authored)


def _author_component_integral(
    author: _ExpressionAuthor,
    quantity: str,
    component: dict[str, Any],
    component_selection: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Author one surface integral on one inset component (not kept top-level)."""

    operations = field_integral_operations(
        quantity=quantity,
        selection_name=component["selection_name"],
        adjacent_side=component["adjacent_side"],
        normal_vector=component["native_normal"],
    )
    return author.author(
        "constituent",
        purpose=f"{quantity}_{label}_component_{component['component_index']}",
        operations=operations,
        selection=component_selection,
        adjacent_selection_name=(
            component["selection_name"]
            if component["adjacent_side"] and quantity != "masked_area"
            else None
        ),
    )


def _author_group_expressions(
    author: _ExpressionAuthor,
    group: dict[str, Any],
    plan: _SurfacePlan,
) -> None:
    """Author the three sums (normal, tangential, area) for one owner group."""

    spec = author.spec
    native_members = [
        {
            "binding_id": member["binding_id"],
            "source_selection_name": plan.native_by_binding[member["binding_id"]][
                "selection_name"
            ],
            "components": plan.member_components[
                (member["binding_id"], group["evaluation_kind"], group["margin_um"])
            ],
            "geometry_empty": not plan.member_components[
                (member["binding_id"], group["evaluation_kind"], group["margin_um"])
            ],
            "effective_domain_id": plan.native_by_binding[member["binding_id"]][
                "effective_domain_id"
            ],
        }
        for member in group["members"]
    ]
    for quantity in ("electric_normal", "electric_tangential", "masked_area"):
        members = plan.grouped_constituents.get((group["group_id"], quantity), [])
        if len(members) != sum(len(item["components"]) for item in native_members):
            raise RuntimeError("grouped EPR expression lacks an inset component")
        members.sort(key=lambda item: item["identity"]["selection"]["binding_id"])
        # Reverse-polish sum: push each member, then add once two are on the stack.
        operations: list[str] = []
        for member in members:
            operations.append(f"NameOfExpression('{member['name']}')")
            if len(operations) > 1:
                operations.append("Operation('+')")
        if not operations:
            # Every source member was geometrically empty at this margin.
            operations = ["Scalar_Constant(0)"]
        dependencies = dict(author.pp_observed)
        for member in members:
            dependencies.update(member["identity"]["dependencies"])
        author.add(
            "grouped",
            purpose=f"{quantity}_{group['group_id']}",
            operations=operations,
            dependencies=dependencies,
            selection={
                "kind": "surface_group",
                **group,
                "integration_geometry": "klayout_complement_minkowski.v1",
                "polygon_approximation": {
                    "method": "klayout_complement_minkowski.v1",
                    "points_per_circle": 128,
                    "source_dbu_um": spec.geometry.source["source_dbu_um"],
                    "klayout_version": plan.bindings[group["members"][0]["binding_id"]][
                        "mask_support"
                    ]["klayout_version"],
                },
                "native_member_selections": native_members,
                "geometry_empty": not members,
            },
        )


# --- Junctions and convergence ----------------------------------------------


def _author_junction_expressions(
    author: _ExpressionAuthor, native_geometry: dict[str, Any]
) -> None:
    """Author the real and imaginary junction-voltage integrals.

    Every model junction contributes its capacitive energy to the common
    normalization (docs/eigenmode-epr.qmd#quantities-and-normalization), so all
    are authored. `junction_ids` only decides which junctions get reported rows.
    """

    spec = author.spec
    request = author.request
    junction_specs = {item.junction_id: item for item in spec.geometry.junctions}
    selected_junctions = (
        set(junction_specs)
        if request.junction_ids is None
        else set(request.junction_ids)
    )
    for selection in native_geometry["junctions"]:
        junction = junction_specs[selection["junction_id"]]
        for quantity in ("junction_voltage_real", "junction_voltage_imag"):
            operations = junction_voltage_operations(
                quantity=quantity,
                selection_name=selection["object_name"],
                direction_xy=junction.direction_xy,
            )
            author.add(
                "junction",
                purpose=f"{quantity}_{junction.junction_id}",
                operations=operations,
                selection={
                    "kind": "junction_sheet",
                    "junction_id": junction.junction_id,
                    "object_name": selection["object_name"],
                    "direction_xy": list(junction.direction_xy),
                    "report_selected": junction.junction_id in selected_junctions,
                },
            )


def _author_convergence_expression(author: _ExpressionAuthor) -> None:
    """Author the optional expression AEDT watches for convergence.

    It is marked `cache_only`: it lives in the expression cache so the solver can
    test it each pass, but it is not an analysis integral.
    """

    spec = author.spec
    convergence = spec.expression_convergence if isinstance(spec, HfssEprSpec) else None
    if convergence is None:
        return
    target = convergence.target
    if isinstance(target, NormalizedSurfaceEprTotal):
        purpose = "normalized_surface_epr_total"
        selection = {
            "kind": "normalized_surface_total",
            "interface_kind": target.interface_kind,
            "evaluation_kind": target.evaluation_kind,
            "margin_um": target.margin_um,
        }
        operations = _normalized_surface_total_operations(
            spec, author.expressions, target
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
    authored = author.author(
        "convergence", purpose=purpose, operations=operations, selection=selection
    )
    authored["cache_only"] = True
    author.expressions.append(authored)
