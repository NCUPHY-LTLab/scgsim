"""The values a Palace config is built from and the records it returns, with the scalar readers they share.

``ConfigInput`` is what both problems' configs are built from; ``EigenmodeSettings`` and
``ElectrostaticSettings`` are what each problem adds.  Each checks its values when it is built, so the config
builders start from values already refused or accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from scgsim.refusal import refusal

from ..mesh.physical_groups import Groups
from ..source.surface_epr import SurfaceEprSpecs
from ..source.transforms import validate_non_negative_int, validate_positive_number
from .index import IndexEntry
from .numerical import NumericalControls

MODEL_L0_M = 1e-6


class GroundResolution(TypedDict):
    """How the Ground boundary was resolved, as the index map records it: requested Nets, physical and exterior."""

    requested_net_ids: list[str]
    physical_net_ids: list[str]
    conductor_component_ids: list[str]
    physical_names: list[str]
    physical_attributes: list[int]
    exterior_physical_names: list[str]
    exterior_attributes: list[int]
    union_attributes: list[int]


@dataclass(frozen=True)
class TerminalBinding:
    """Exact net mapping record for a conductor terminal."""

    name: str
    net_id: str


@dataclass(frozen=True)
class ConfigInput:
    """What every Palace config is built from: the current mesh, the materials and the problem's shared settings.

    ``groups`` are the mesh's physical groups and ``mesh_path`` its ``palace.msh``; ``materials`` the material
    records of the prepared stack; ``surface_epr_specs`` is None when Surface EPR was not requested.  Building it
    refuses numerical controls the ``problem`` cannot be written from.
    """

    problem: Literal["Eigenmode", "Electrostatic"]
    groups: Groups
    mesh_path: Path
    materials: Mapping[str, Mapping[str, Any]]
    surface_epr_specs: SurfaceEprSpecs | None
    numerical: NumericalControls

    def __post_init__(self) -> None:
        """Refuse AMR settings left open (``NumericalControls.check_ready_for``)."""
        self.numerical.check_ready_for(self.problem)


@dataclass(frozen=True)
class EigenmodeSettings:
    """The mode controls of one Eigenmode solve: modes, target (Hz), solver tolerance and saved fields.

    Raises on build:
        ValueError: ``num_modes`` is not a positive integer, ``target_hz`` or ``tolerance`` is not finite and
            positive, or ``save_fields`` is negative.
        TypeError: ``save_fields`` is not an integer.
    """

    num_modes: int = 10
    target_hz: float | None = None
    tolerance: float = 1e-6
    save_fields: int = 0

    def __post_init__(self) -> None:
        """Refuse an invalid control, then hold the numbers as floats where Palace reads floats."""
        if (
            not isinstance(self.num_modes, int)
            or isinstance(self.num_modes, bool)
            or self.num_modes < 1
        ):
            raise ValueError(
                refusal(
                    "num_modes must be a positive integer: call set_eigenmode(num_modes=...)",
                    found=self.num_modes,
                )
            )
        # A frozen dataclass normalizes its own fields through object.__setattr__.
        if self.target_hz is not None:
            object.__setattr__(
                self,
                "target_hz",
                validate_positive_number(self.target_hz, "set_eigenmode(target=...)"),
            )
        object.__setattr__(
            self,
            "tolerance",
            validate_positive_number(self.tolerance, "set_eigenmode(tolerance=...)"),
        )
        validate_non_negative_int(self.save_fields, "set_eigenmode(save=...)")


@dataclass(frozen=True)
class ElectrostaticSettings:
    """The saved fields and the boundary policies of one Electrostatic solve.

    ``unassigned_conductor_policy`` decides what a conductor no terminal or ground names becomes (``"ground"``
    or an ``"error"``); ``exterior_boundary_policy="ground"`` also grounds the outer faces of the domain, and
    then every conductor must be assigned.
    """

    save_fields: int = 0
    unassigned_conductor_policy: Literal["ground", "error"] = "ground"
    exterior_boundary_policy: Literal["none", "ground"] = "none"

    def __post_init__(self) -> None:
        """Refuse an unknown policy or a pair of policies that contradict each other."""
        if self.unassigned_conductor_policy not in {"ground", "error"}:
            raise ValueError(
                refusal(
                    "unassigned_conductor_policy must be 'ground' or 'error': call "
                    "set_electrostatic(unassigned_conductor_policy=...)",
                    found=self.unassigned_conductor_policy,
                )
            )
        if self.exterior_boundary_policy not in {"none", "ground"}:
            raise ValueError(
                refusal(
                    "exterior_boundary_policy must be 'none' or 'ground': call "
                    "set_electrostatic(exterior_boundary_policy=...)",
                    found=self.exterior_boundary_policy,
                )
            )
        if (
            self.exterior_boundary_policy == "ground"
            and self.unassigned_conductor_policy != "error"
        ):
            raise ValueError(
                refusal(
                    "exterior ground requires unassigned_conductor_policy='error': call "
                    "set_electrostatic(unassigned_conductor_policy='error', exterior_boundary_policy='ground')",
                    found=self.unassigned_conductor_policy,
                    expected="error",
                )
            )
        validate_non_negative_int(self.save_fields, "save_fields")


@dataclass(frozen=True)
class ElectrostaticConfigBuildResult:
    """Electrostatic config, its index entries and the records derived from it."""

    config: dict[str, Any]
    index_entries: list[IndexEntry]
    material_resolution: list[dict[str, Any]]
    ground_boundary_resolution: GroundResolution


@dataclass(frozen=True)
class EigenmodeConfigBuildResult:
    """Eigenmode config, its index entries and the records derived from it."""

    config: dict[str, Any]
    port_information: list[dict[str, Any]]
    index_entries: list[IndexEntry]
    material_resolution: list[dict[str, Any]]


def optional_string(value: Any) -> str | None:
    """Normalize an optional textual identifier to non-empty text or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MODEL_L0_M",
    "ConfigInput",
    "EigenmodeConfigBuildResult",
    "EigenmodeSettings",
    "ElectrostaticConfigBuildResult",
    "ElectrostaticSettings",
    "TerminalBinding",
    "optional_string",
]
