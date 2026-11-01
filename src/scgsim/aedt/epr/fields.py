"""Build field-integral operations and bind native postprocessing controls."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scgsim.aedt.epr.expression_codec import compile_named_expression
from scgsim.aedt.epr.expression_loading import load_compiled_expressions


def parse_native_scalar(value: Any) -> dict[str, Any]:
    """Parse one native scalar without treating false/missing as numeric zero."""

    if value is False or value is None or isinstance(value, bool):
        raise RuntimeError("HFSS field calculator returned no scalar value")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
        return {"value": number, "unit": "", "raw": value}
    text = str(value).strip()
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([^\s]*)",
        text,
    )
    if match is None:
        raise RuntimeError(f"HFSS field calculator scalar is invalid: {value!r}")
    number = float(match.group(1))
    if not math.isfinite(number):
        raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
    return {"value": number, "unit": match.group(2), "raw": text}


def _constant(value: float) -> list[str]:
    return [f"Scalar_Constant({value:.17g})"]


def _binary(left: Sequence[str], right: Sequence[str], operation: str) -> list[str]:
    return [*left, *right, f"Operation('{operation}')"]


def _square(value: Sequence[str]) -> list[str]:
    return [*value, "Scalar_Constant(2)", "Operation('Pow')"]


def _same_postprocessing_value(requested: str, actual: str) -> bool:
    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value  # noqa: PLC0415 -- optional native readback unit parser.

    requested_number, requested_unit = decompose_variable_value(requested)
    if requested == actual:
        return not (
            isinstance(requested_number, (int, float))
            and not math.isfinite(requested_number)
        ) and not re.fullmatch(
            r"[+-]?(?:nan|inf(?:inity)?)[A-Za-z]*", requested.strip(), re.I
        )
    actual_number, actual_unit = decompose_variable_value(actual)
    return (
        isinstance(requested_number, (int, float))
        and not isinstance(requested_number, bool)
        and isinstance(actual_number, (int, float))
        and not isinstance(actual_number, bool)
        and math.isfinite(requested_number)
        and math.isfinite(actual_number)
        and requested_unit == actual_unit
        and requested_number == actual_number
    )


def _validated_postprocessing_readback(
    values: Mapping[str, str], postprocessing: Mapping[str, Any]
) -> dict[str, str]:
    """Preserve exact expressions; accept rewritten finite literals only at exact parity."""

    observed: dict[str, str] = {}
    for name, requested in values.items():
        if name not in postprocessing:
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} is missing from readback"
            )
        actual = str(postprocessing[name])
        if not _same_postprocessing_value(requested, actual):
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} differs from requested value"
            )
        observed[name] = actual
    return observed


def install_variables(app: Any, values: Mapping[str, str]) -> dict[str, str]:
    """Install and read back complete postprocessing variable assignments."""
    manager = app.variable_manager
    existing_names = set(manager.design_variable_names)
    existing_postprocessing = manager.post_processing_variables
    new_props: list[Any] = ["NAME:NewProps"]
    for name, value in values.items():
        if not _same_postprocessing_value(value, value):
            raise RuntimeError(f"HFSS postprocessing variable {name!r} is nonfinite")
        if (
            name in existing_postprocessing
            and existing_postprocessing[name].sweep is False
            and _same_postprocessing_value(value, str(existing_postprocessing[name]))
        ):
            continue
        if name in existing_names or name.startswith("$"):
            if not manager.set_variable(
                name,
                expression=value,
                sweep=False,
                overwrite=True,
                is_post_processing=True,
            ):
                raise RuntimeError(f"HFSS postprocessing variable failed: {name!r}")
            continue
        new_props.append(
            [
                "NAME:" + name,
                "PropType:=",
                "PostProcessingVariableProp",
                "UserDef:=",
                True,
                "Value:=",
                value,
                "Description:=",
                "",
                "ReadOnly:=",
                False,
                "Hidden:=",
                False,
                "Sweep:=",
                False,
            ]
        )
    if len(new_props) > 1:
        changed = app.odesign.ChangeProperty(
            [
                "NAME:AllTabs",
                [
                    "NAME:LocalVariableTab",
                    ["NAME:PropServers", "LocalVariables"],
                    new_props,
                ],
            ]
        )
        if changed is False:
            raise RuntimeError("HFSS postprocessing variable batch creation failed")
    return _validated_postprocessing_readback(values, manager.post_processing_variables)


def field_integral_operations(
    *,
    quantity: str,
    selection_name: str,
    adjacent_side: bool = False,
    normal_vector: Sequence[float] | None = None,
) -> list[str]:
    """Return the documented HFSS stack for one raw integral quantity."""

    if quantity == "effective_volume":
        value = _constant(1.0)
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "electric_volume":
        value = [
            "Fundamental_Quantity('E')",
            "Operation('Conj')",
            "Fundamental_Quantity('E')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "magnetic_energy":
        value = [
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Conj')",
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"electric_normal", "electric_tangential", "masked_area"}:
        # Area is independent of which field side is sampled.
        enter_surface = (
            "EnterAdjacentSurf"
            if adjacent_side and quantity != "masked_area"
            else "EnterSurface"
        )
        if quantity == "masked_area":
            value = _constant(1.0)
            terminal = [
                f"{enter_surface}('{selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ]
            return [*value, *terminal]
        if (
            normal_vector is None
            or len(normal_vector) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in normal_vector
            )
        ):
            raise ValueError("surface field integral requires a finite native normal")
        length = math.sqrt(sum(float(item) ** 2 for item in normal_vector))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError(
                "surface field integral requires a finite nonzero native normal"
            )
        vector = (
            "Vector_Constant("
            + ", ".join(f"{float(item) / length:.17g}" for item in normal_vector)
            + ")"
        )
        normal_real = [
            "Fundamental_Quantity('E')",
            "Operation('Real')",
            vector,
            "Operation('Dot')",
        ]
        normal_imag = [
            "Fundamental_Quantity('E')",
            "Operation('Imag')",
            vector,
            "Operation('Dot')",
        ]
        normal_square = _binary(_square(normal_real), _square(normal_imag), "+")
        if quantity == "electric_normal":
            value = normal_square
        else:
            full_real = ["Fundamental_Quantity('E')", "Operation('Real')"]
            full_imag = ["Fundamental_Quantity('E')", "Operation('Imag')"]
            full_square = _binary(
                [*full_real, *full_real, "Operation('Dot')"],
                [*full_imag, *full_imag, "Operation('Dot')"],
                "+",
            )
            value = _binary(full_square, normal_square, "-")
        # This reporter-operation spelling records side intent in the identity;
        # the compiled CLC uses the native-saved EnterAdjacentSurface token.
        terminal = [
            f"{enter_surface}('{selection_name}')",
            "Operation('SurfaceValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError(
            "junction voltage requires an explicit directed vector; use "
            "junction_voltage_operations"
        )
    else:
        raise ValueError(f"unsupported HFSS EPR field quantity {quantity!r}")
    if adjacent_side and quantity not in {"electric_normal", "electric_tangential"}:
        raise ValueError("adjacent_side is valid only for surface field quantities")
    return [*value, *terminal]


def junction_voltage_operations(
    *, quantity: str, selection_name: str, direction_xy: Sequence[float]
) -> list[str]:
    """Integrate Re/Im(E) dot the declared in-plane terminal direction."""

    if quantity not in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError("junction voltage quantity must be real or imaginary")
    if len(direction_xy) != 2:
        raise ValueError("junction direction requires two values")
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("junction direction must be finite and nonzero")
    component = "Real" if quantity.endswith("real") else "Imag"
    return [
        "Fundamental_Quantity('E')",
        f"Operation('{component}')",
        f"Vector_Constant({dx / norm:.17g}, {dy / norm:.17g}, 0)",
        "Operation('Dot')",
        f"EnterSurface('{selection_name}')",
        "Operation('SurfaceValue')",
        "Operation('Integrate')",
    ]


def create_verified_face_list(
    app: Any, *, name: str, face_ids: Sequence[int], reuse_existing: bool = False
) -> dict[str, Any]:
    """Bind native list identity and check created/restored wrapper membership."""
    ids = [int(value) for value in face_ids]
    if not ids or len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise ValueError("face list requires distinct positive native face ids")
    existing = [item for item in app.modeler.user_lists if str(item.name) == name]
    if existing and not reuse_existing:
        raise RuntimeError(f"HFSS face-list name already exists: {name!r}")
    if len(existing) > 1:
        raise RuntimeError(f"HFSS face-list name is ambiguous: {name!r}")
    item = existing[0] if existing else app.modeler.create_face_list(ids, name=name)
    if (
        item is False
        or item is None
        or item.name != name
        or item.props.get("Type") != "Face"
    ):
        raise RuntimeError(f"HFSS face-list creation failed: {name!r}")
    native_id = app.modeler.get_entitylist_id(name)
    if (
        type(native_id) is not int
        or native_id <= 0
        or native_id != item.props.get("ID")
    ):
        raise RuntimeError(f"HFSS face-list readback failed: {name!r}")
    # PyAEDT exposes both created and restored EntityList membership as List.
    members = item.props.get("List")
    if isinstance(members, (int, str)):
        members = [members]
    if [int(value) for value in members or ()] != ids:
        raise RuntimeError(f"HFSS face-list membership differs: {name!r}")
    return {
        "name": name,
        "native_id": native_id,
        "face_ids": ids,
        "assignment_basis": (
            "PyAEDT created/restored List property; not independent native membership"
        ),
    }


def evaluate_named_expression(  # noqa: PLR0913 -- preserve the established field-expression keyword API.
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Author and evaluate one immutable named expression in a bound context."""

    authored = author_named_expression(
        app,
        purpose=purpose,
        operations=operations,
        solution=solution,
        phase_degrees=phase_degrees,
        dependencies=dependencies,
        selection=selection,
        evidence_dir=evidence_dir,
    )
    name = authored["name"]
    raw = app.post.fields_calculator.evaluate(
        name,
        setup=solution,
        intrinsics={"Phase": f"{phase_degrees:g}deg"},
    )
    return {
        **authored,
        "scalar": parse_native_scalar(raw),
    }


def author_named_expression(  # noqa: PLR0913 -- preserve the established field-expression keyword API.
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
    namespace: str = "prepared",
    adjacent_selection_name: str | None = None,
) -> dict[str, Any]:
    """Author one immutable expression through the same local batch compiler."""

    compiled = compile_named_expression(
        purpose=purpose,
        operations=operations,
        solution=solution,
        phase_degrees=phase_degrees,
        dependencies=dependencies,
        selection=selection,
        namespace=namespace,
        adjacent_selection_name=adjacent_selection_name,
    )
    load_compiled_expressions(app, [compiled], evidence_dir)
    # Loading attaches the native CLC readback to the authored expression.
    # ``compiled`` is (expression_dict, definitions, evidence), not a list of
    # per-expression records.
    return compiled[0]
