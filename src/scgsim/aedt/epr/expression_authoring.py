"""Compile and load the EPR field expressions of an imported SGB model.

:func:`author_epr_expressions` runs, in order:

1. install the modal postprocessing sources, one variable per configured mode;
2. read what the model's EPR integrates (:func:`scgsim.aedt.epr.sgb_source.sgb_epr_source`) and the
   selection sheets the import created for it (``geometry["surface_selections"]`` and
   ``geometry["junction_selections"]``);
3. compile every expression :func:`scgsim.aedt.epr.sgb_expressions.sgb_expression_requests` lists, with each
   domain's relative permeability read from the AEDT material of its object;
4. compile the expression-cache convergence targets;
5. load the library once and read every definition back.

The field selector of a saved analysis (``surface`` or ``adjacent``, by Surface-EPR group) decides how
each row's field is read; the prepared model reads every row on its faces.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.epr.cache_config import identity_sha256
from scgsim.aedt.epr.expression_codec import compile_named_expression
from scgsim.aedt.epr.expression_loading import load_compiled_expressions
from scgsim.aedt.epr.fields import install_variables
from scgsim.aedt.epr.sgb_expressions import (
    normalized_surface_total_operations,
    sgb_expression_requests,
)
from scgsim.aedt.epr.sgb_source import SgbEprSource, sgb_epr_source
from scgsim.aedt.epr.sgb_surfaces import (
    FieldSelector,
    JunctionSelection,
    SurfaceSelection,
)
from scgsim.aedt.runtime.native.common import (
    native_object_property,
    read_native_material,
)
from scgsim.aedt.specs.epr import (
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    iter_expression_convergence,
)
from scgsim.aedt.specs.hfss import HfssEprAnalysisSpec, HfssEprSpec
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl

Compiled = tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]


def spec_epr_source(spec: HfssEprSpec | HfssEprAnalysisSpec) -> SgbEprSource:
    """What the request's EPR integrates: its model, with the Surface-EPR groups its request selects."""
    request = spec.epr_request
    return sgb_epr_source(
        spec.model,
        surface_epr_groups=None if request is None else request.surface_epr_groups,
    )


def author_epr_expressions(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec | HfssEprAnalysisSpec,
    geometry: Mapping[str, Any],
    *,
    namespace: str = "prepared",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Author the complete field library of the model and its saved-definition evidence (see the module)."""
    started = time.perf_counter()
    run_control = cast(EigenmodeRunControl, spec.run_control)
    solution = f"{run_control.setup_name} : LastAdaptive"
    pp_values, source_assignment = _install_sources(app, run_control.num_modes)
    source = spec_epr_source(spec)
    selectors: dict[str, FieldSelector] = (
        dict(spec.surface_field_selectors or {})
        if isinstance(spec, HfssEprAnalysisSpec)
        else {}
    )
    requests = sgb_expression_requests(
        source,
        surface_selections(geometry, source),
        junction_selections(geometry, source),
        relative_permeability=domain_permeabilities(app, source),
        field_selectors=selectors,
    )
    compiled: list[Compiled] = []
    compile_started = time.perf_counter()
    for request in requests:
        compiled.append(
            compile_named_expression(
                purpose=request.purpose,
                operations=request.operations,
                solution=solution,
                phase_degrees=0.0,
                dependencies=pp_values,
                selection=request.selection,
                namespace=namespace,
                adjacent_selection_name=request.adjacent_selection_name,
            )
        )
    names = {item[0]["identity"]["purpose"]: item[0]["name"] for item in compiled}
    expressions = [item[0] for item in compiled]
    if isinstance(spec, HfssEprSpec):
        targets = _convergence_targets(spec, source, names, solution, pp_values)
        compiled += targets
        expressions += [item[0] for item in targets]
    compile_seconds = time.perf_counter() - compile_started
    batch = load_compiled_expressions(
        app, compiled, run_dir / "metadata/epr_expressions"
    )
    authoring: dict[str, Any] = {
        "source_assignment": source_assignment,
        "postprocessing_variables": pp_values,
        "native_expression_readback": batch["native_expression_readback"],
    }
    if selectors:
        authoring["surface_field_selector_bindings"] = _selector_bindings(expressions)
    timings = {
        "compile_seconds": round(compile_seconds, 6),
        **{
            key: batch[key]
            for key in (
                "collision_check_seconds",
                "evidence_write_seconds",
                "batch_import_seconds",
                "postload_readback_seconds",
                "native_definition_readback_seconds",
            )
        },
        "expression_count": len(expressions),
        "batch_library_sha256": batch["library_sha256"],
        "native_definition_count": batch["native_definition_count"],
        "authoring_seconds": round(time.perf_counter() - started, 6),
    }
    return expressions, authoring, timings


def surface_selections(
    geometry: Mapping[str, Any], source: SgbEprSource
) -> dict[str, SurfaceSelection]:
    """The selection sheet of every row ``source`` integrates; a row the design has no sheet for is refused."""
    recorded = geometry.get("surface_selections", {})
    missing = sorted(set(source.rows) - set(recorded))
    if missing:
        raise RuntimeError(
            f"the AEDT design has no selection sheet for the Surface-EPR rows {missing}; "
            "prepare the model with their groups selected"
        )
    out = {}
    for row in source.rows:
        item = recorded[row]
        out[row] = SurfaceSelection(
            row=item["row"],
            group=item["group"],
            interface_type=item["interface_type"],
            entities=tuple(item["entities"]),
            band=item["band"],
            selection_name=item["selection_name"],
            faces=tuple((str(name), int(face)) for name, face in item["faces"]),
        )
    return out


def junction_selections(
    geometry: Mapping[str, Any], source: SgbEprSource
) -> dict[str, JunctionSelection]:
    """The voltage sheet of every junction of the model, as the import recorded it."""
    recorded = geometry.get("junction_selections", {})
    if set(recorded) != set(source.junctions):
        raise RuntimeError(
            f"the AEDT design's junction sheets {sorted(recorded)} are not the model's junctions "
            f"{sorted(source.junctions)}"
        )
    return {
        junction_id: JunctionSelection(
            junction_id=item["junction_id"],
            port_group=item["port_group"],
            selection_name=item["selection_name"],
            faces=tuple((str(name), int(face)) for name, face in item["faces"]),
        )
        for junction_id, item in recorded.items()
    }


def domain_permeabilities(app: Any, source: SgbEprSource) -> dict[str, float]:
    """The relative permeability of each domain's object, read from the AEDT material it carries."""
    out = {}
    for name in source.domains:
        material = native_object_property(app.modeler[name], "Material").strip('"')
        value = read_native_material(app, material)["relative_permeability"]
        if value is None:
            raise RuntimeError(
                f"AEDT reports no scalar permeability for {material!r} on {name}"
            )
        out[name] = float(value)
    return out


def _install_sources(
    app: Any, num_modes: int
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    # Keep a nonzero modal source selected when saving the project.
    pp_values = {
        f"scgsim_epr_pp_mode_{index}": "1" if index == 1 else "0"
        for index in range(1, num_modes + 1)
    }
    observed = install_variables(app, pp_values)
    assignment = {
        str(index): (f"scgsim_epr_pp_mode_{index}", "0deg")
        for index in range(1, num_modes + 1)
    }
    if app.edit_sources(assignment, eigenmode_stored_energy=False) is not True:
        raise RuntimeError("native Eigenmode PP source-vector installation failed")
    return observed, assignment


def _convergence_targets(
    spec: HfssEprSpec,
    source: SgbEprSource,
    names: Mapping[str, str],
    solution: str,
    pp_values: Mapping[str, str],
) -> list[Compiled]:
    out: list[Compiled] = []
    authored: set[str] = set()
    for convergence in iter_expression_convergence(spec.expression_convergence):
        target = convergence.target
        target_id = identity_sha256(target.to_payload())
        if target_id in authored:
            continue
        authored.add(target_id)
        if isinstance(target, NormalizedSurfaceEprTotal):
            purpose = "normalized_surface_epr_total"
            selection: dict[str, Any] = {
                "kind": "normalized_surface_total",
                "interface_kind": target.interface_kind,
                "evaluation_kind": target.evaluation_kind,
                "margin_um": target.margin_um,
            }
            operations = normalized_surface_total_operations(source, names, target)
        elif isinstance(target, NativeExpressionDefinition):
            purpose = "custom_convergence"
            selection = {"kind": "custom_convergence", "definition_name": target.name}
            operations = list(target.operations)
        else:
            raise TypeError("unsupported expression-cache convergence target")
        compiled = compile_named_expression(
            purpose=purpose,
            operations=operations,
            solution=solution,
            phase_degrees=0.0,
            dependencies=pp_values,
            selection=selection,
        )
        compiled[0]["cache_only"] = True
        compiled[0]["convergence_target"] = target.to_payload()
        out.append(compiled)
    return out


def _selector_bindings(expressions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Which expression reads each row's field and on which side, with its saved native definition."""
    out = []
    for expression in expressions:
        selection = expression["identity"]["selection"]
        purpose = expression["identity"]["purpose"]
        if selection.get("kind") != "surface_row" or purpose.startswith("masked_area"):
            continue
        out.append(
            {
                "row": selection["row"],
                "group": selection["group"],
                "selection_name": selection["selection_name"],
                "field_selector": selection["field_selector"],
                "purpose": purpose,
                "expression_name": expression["name"],
                "expression_identity_sha256": expression["identity"]["sha256"],
                "native_readback": expression["native_readback"],
            }
        )
    return out
