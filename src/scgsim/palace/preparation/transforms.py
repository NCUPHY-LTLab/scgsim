"""Input validators and the airbox transform shared by Palace problem classes.

Owns: the config ``SCHEMA_VERSION``, argument validators, Route-A thin-film
normalization and padding the stack with an airbox.  Does not own: the vacuum
region (``scgsim.geometry.source.vacuum``) or the order in which transforms run
(``workflow.py``).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from scgsim.geometry.models.input import GeometryBuildInput

SCHEMA_VERSION = "v0.16.0"

RouteAThinFilm = Literal["substrate_face", "metal_gap_equivalent"]


def validate_nonempty_string(value: Any, field: str) -> str:
    """Return a non-empty trimmed string."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


def validate_positive_number(value: Any, field: str) -> float:
    """Validate finite positive real values used by the mesh controls."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    if float(value) <= 0.0:
        raise ValueError(f"{field} must be > 0.")
    return float(value)


def validate_non_negative_int(value: Any, field: str) -> int:
    """Validate a non-negative integer control."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an int.")
    if value < 0:
        raise ValueError(f"{field} must be >= 0.")
    return int(value)


def normalize_route_a_thin_film(route: str, value: str | None) -> RouteAThinFilm | None:
    """Validate the explicit Route-A thin-film lowering choice."""
    if route == "B":
        if value is not None:
            raise ValueError("Route B does not accept route_a_thin_film.")
        return None
    if route != "A":
        raise ValueError("route must be either 'A' or 'B'.")
    normalized = value if value in {"substrate_face", "metal_gap_equivalent"} else None
    if normalized is not None:
        return normalized
    raise ValueError(
        "Route A requires route_a_thin_film='substrate_face' or 'metal_gap_equivalent'."
    )


def _normalized_component_input(
    component: Any, stack: Mapping[str, Any]
) -> GeometryBuildInput:
    """Acquire the existing public component/stack source before interpretation."""
    if component is None:
        raise ValueError(
            "Palace preparation requires component or normalized geometry input"
        )
    from scgsim.geometry.source.adapter import build_gds_stack_geometry_input

    from ..mesh.build import _component_gds_top_cell_name, _write_component_gds

    with TemporaryDirectory(prefix="scgsim-physical-source-") as directory:
        gds_path = Path(directory) / "design.gds"
        stack_path = Path(directory) / "design.stack.json"
        _write_component_gds(component, gds_path)
        stack_path.write_text(json.dumps(dict(stack)) + "\n", encoding="utf-8")
        return build_gds_stack_geometry_input(
            gds_file=gds_path,
            stack_file=stack_path,
            top_cell_name=_component_gds_top_cell_name(
                component=component, gds_path=gds_path
            ),
        )


def apply_airbox_to_stack(
    stack: Mapping[str, Any], airbox: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return a copy of stack with requested airbox envelope updates applied.

    The airbox is the solution region that pads the device with vacuum. ``airbox`` may
    set ``margin_x`` and ``margin_y`` (equal, added to every side in the plane) and
    ``z_below`` / ``z_above`` (added below the lowest and above the highest z).
    """
    if not airbox:
        return dict(stack)
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping.")

    solution_regions = stack.get("solution_regions")
    if not isinstance(solution_regions, Mapping):
        raise TypeError("stack must provide 'solution_regions' mapping.")

    targets = _airbox_targets(_marked_airbox_regions(solution_regions))
    margin = _plane_margin(airbox)

    work = copy.deepcopy(dict(stack))
    work_regions = dict(work["solution_regions"])
    shared_domain_bounds: dict[str, float] | None = None
    if len(targets) == 2 and margin is not None:
        shared_domain_bounds = _shared_two_sided_bounds(
            targets, solution_regions, margin, work_regions
        )
    z_below = (
        validate_positive_number(airbox["z_below"], "z_below")
        if "z_below" in airbox
        else None
    )
    z_above = (
        validate_positive_number(airbox["z_above"], "z_above")
        if "z_above" in airbox
        else None
    )
    for key, region, side in targets:
        geometry = region.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError(f"airbox solution region {key!r} must define geometry.")
        work_region = dict(region)
        work_geometry = dict(geometry)
        if margin is not None:
            _pad_in_plane(work_geometry, key, margin, shared_domain_bounds)
        if z_below is not None and side in {None, "below"}:
            work_geometry["z_min_um"] = (
                _number_field(work_geometry, "z_min_um", key) - z_below
            )
        if z_above is not None and side in {None, "above"}:
            work_geometry["z_max_um"] = (
                _number_field(work_geometry, "z_max_um", key) + z_above
            )
        work_region["geometry"] = work_geometry
        work_regions[key] = work_region
    work["solution_regions"] = work_regions
    return work


def _marked_airbox_regions(
    solution_regions: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    matched = []
    for key, region in solution_regions.items():
        if not isinstance(region, Mapping):
            raise TypeError("each solution region must be a mapping.")
        marker = region.get("is_airbox", False)
        if not isinstance(marker, bool):
            raise TypeError(f"solution region {key!r} is_airbox must be a bool.")
        if marker:
            if region.get("role") != "solution_region":
                raise ValueError(
                    f"airbox marker {key!r} must be on a solution_region record."
                )
            matched.append((str(key), region))
    return matched


def _airbox_targets(
    matched: list[tuple[str, Mapping[str, Any]]],
) -> tuple[tuple[str, Mapping[str, Any], Any], ...]:
    """The airbox regions with their side. One marker pads on both sides.

    Two markers must name opposite sides, so each side can have its own z extension.
    """
    if len(matched) == 1:
        return ((matched[0][0], matched[0][1], None),)
    if len(matched) != 2:
        raise TypeError(
            "Airbox requires either one legacy marker or explicit below/above markers."
        )
    targets = tuple((key, region, region.get("airbox_side")) for key, region in matched)
    sides = tuple(side for _, _, side in targets)
    if set(sides) != {"below", "above"} or len(set(sides)) != len(sides):
        raise ValueError(
            "Two airbox solution regions require one explicit airbox_side=below "
            "and one explicit airbox_side=above."
        )
    return targets


def _plane_margin(airbox: Mapping[str, Any]) -> float | None:
    """The in-plane padding to add, or None when the airbox sets no margin."""
    has_x = "margin_x" in airbox
    has_y = "margin_y" in airbox
    if not (has_x or has_y):
        return None
    if not (has_x and has_y):
        raise ValueError("margin_x and margin_y must both be provided together.")
    margin_x = validate_positive_number(airbox["margin_x"], "margin_x")
    margin_y = validate_positive_number(airbox["margin_y"], "margin_y")
    if margin_x != margin_y:
        raise ValueError(
            "margin_x and margin_y must be equal for scalar padding updates."
        )
    return margin_x


def _shared_two_sided_bounds(
    targets: tuple[tuple[str, Mapping[str, Any], Any], ...],
    solution_regions: Mapping[str, Any],
    margin: float,
    work_regions: dict[str, Any],
) -> dict[str, float]:
    """Pad the one footprint that every solution region shares, in ``work_regions``.

    With a below and an above airbox, all solution regions must keep the same
    footprint, so the margin is applied to that common footprint once.
    """
    marker_bounds = tuple(
        _explicit_domain_bounds(region, key) for key, region, _ in targets
    )
    if marker_bounds[0] != marker_bounds[1]:
        raise ValueError(
            "Two-sided airbox markers must have identical explicit domain_bounds_um."
        )
    common_bounds = marker_bounds[0]
    shared_regions = tuple(
        (str(key), region)
        for key, region in solution_regions.items()
        if region.get("role") == "solution_region"
    )
    for key, region in shared_regions:
        if _explicit_domain_bounds(region, key) != common_bounds:
            raise ValueError(
                "Two-sided airbox requires every solution_region to share its "
                "explicit domain_bounds_um footprint."
            )
    shared_domain_bounds = _padded_bounds(common_bounds, margin)
    for key, region in shared_regions:
        work_region = dict(region)
        work_geometry = dict(region["geometry"])
        work_geometry["domain_bounds_um"] = dict(shared_domain_bounds)
        work_region["geometry"] = work_geometry
        work_regions[key] = work_region
    return shared_domain_bounds


def _padded_bounds(bounds: Mapping[str, float], margin: float) -> dict[str, float]:
    return {
        "x_min_um": bounds["x_min_um"] - margin,
        "y_min_um": bounds["y_min_um"] - margin,
        "x_max_um": bounds["x_max_um"] + margin,
        "y_max_um": bounds["y_max_um"] + margin,
    }


def _pad_in_plane(
    geometry: dict[str, Any],
    key: str,
    margin: float,
    shared_domain_bounds: dict[str, float] | None,
) -> None:
    """Grow an airbox region in x and y: its explicit bounds, or else its padding."""
    if shared_domain_bounds is not None:
        geometry["domain_bounds_um"] = dict(shared_domain_bounds)
        return
    bounds = geometry.get("domain_bounds_um")
    if isinstance(bounds, Mapping):
        required_bounds = ("x_min_um", "y_min_um", "x_max_um", "y_max_um")
        if any(
            not isinstance(bounds.get(name), (int, float))
            or isinstance(bounds.get(name), bool)
            for name in required_bounds
        ):
            raise TypeError(
                f"airbox solution region {key!r} has invalid domain_bounds_um."
            )
        geometry["domain_bounds_um"] = _padded_bounds(
            {name: float(bounds[name]) for name in required_bounds}, margin
        )
        return
    padding = geometry.get("padding_um")
    if not isinstance(padding, (int, float)) or isinstance(padding, bool):
        raise TypeError(
            f"airbox solution region {key!r} must define padding_um or domain_bounds_um."
        )
    if float(padding) < 0.0:
        raise ValueError(f"airbox solution region {key!r} has invalid padding_um.")
    geometry["padding_um"] = float(padding) + margin


def _number_field(geometry: Mapping[str, Any], name: str, key: str) -> float:
    value = geometry.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"airbox solution region {key!r} must define {name}.")
    return float(value)


def _explicit_domain_bounds(region: Mapping[str, Any], key: str) -> dict[str, float]:
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError(f"solution region {key!r} must define geometry.")
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise TypeError(
            f"solution region {key!r} must define explicit domain_bounds_um."
        )
    required = ("x_min_um", "y_min_um", "x_max_um", "y_max_um")
    if any(
        not isinstance(bounds.get(name), (int, float))
        or isinstance(bounds.get(name), bool)
        or not math.isfinite(float(bounds[name]))
        for name in required
    ):
        raise TypeError(f"solution region {key!r} has invalid domain_bounds_um.")
    return {name: float(bounds[name]) for name in required}
