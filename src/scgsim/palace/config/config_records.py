"""The values a Palace config is built from and the records it returns, with the scalar readers they share.

``ConfigInput`` is what every problem's config is built from; ``EigenmodeSettings``, ``ElectrostaticSettings``
and ``DrivenSweep`` are what each problem adds.  Each checks its values when it is built, so the config
builders start from values already refused or accepted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from scgsim.contracts.palace_run_folder import ProblemName
from scgsim.refusal import Declared, found_text

from ..mesh.physical_groups import Groups
from ..source.surface_epr import SurfaceEprSpecs
from ..source.transforms import validate_non_negative_int, validate_positive_number
from .index import IndexEntry
from .numerical import NumericalControls

_WHERE_EIGENMODE_SETTINGS = "the call to set_eigenmode"
_WHERE_ELECTROSTATIC_SETTINGS = "the call to set_electrostatic"
_WHERE_DRIVEN_SWEEP = "the call to set_driven"

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
    records of the prepared stack; ``surface_epr_specs`` is None when Surface EPR was not requested.
    ``convergence_declared`` says that ``set_convergence`` declared criteria, which then end the run in place
    of Palace's tolerance.  Building it refuses numerical controls the ``problem`` cannot be written from.
    """

    problem: ProblemName
    groups: Groups
    mesh_path: Path
    materials: Mapping[str, Mapping[str, Any]]
    surface_epr_specs: SurfaceEprSpecs | None
    numerical: NumericalControls
    convergence_declared: bool = False

    def __post_init__(self) -> None:
        """Refuse AMR settings left open (``NumericalControls.check_ready_for``)."""
        self.numerical.check_ready_for(
            self.problem, convergence_declared=self.convergence_declared
        )


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
                Declared(
                    "num_modes must be a positive integer: call set_eigenmode(num_modes=...)",
                    _WHERE_EIGENMODE_SETTINGS,
                ).refusal(found_text(self.num_modes))
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
                Declared(
                    "unassigned_conductor_policy must be 'ground' or 'error': call "
                    "set_electrostatic(unassigned_conductor_policy=...)",
                    _WHERE_ELECTROSTATIC_SETTINGS,
                ).refusal(found_text(self.unassigned_conductor_policy))
            )
        if self.exterior_boundary_policy not in {"none", "ground"}:
            raise ValueError(
                Declared(
                    "exterior_boundary_policy must be 'none' or 'ground': call "
                    "set_electrostatic(exterior_boundary_policy=...)",
                    _WHERE_ELECTROSTATIC_SETTINGS,
                ).refusal(found_text(self.exterior_boundary_policy))
            )
        if (
            self.exterior_boundary_policy == "ground"
            and self.unassigned_conductor_policy != "error"
        ):
            raise ValueError(
                Declared(
                    "exterior ground requires unassigned_conductor_policy='error': call "
                    "set_electrostatic(unassigned_conductor_policy='error', exterior_boundary_policy='ground')",
                    _WHERE_ELECTROSTATIC_SETTINGS,
                ).refusal(found_text(self.unassigned_conductor_policy, "error"))
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
class DrivenSweep:
    """The frequency sweep of one Driven solve: ``points`` frequencies from ``start_hz`` to ``stop_hz``, both ends included.

    ``adaptive_tolerance`` None solves the field at every frequency (a uniform sweep).  A tolerance turns on
    Palace's adaptive fast sweep: it solves a few frequencies, adds solves until a reduced model built from them
    agrees with them to that relative error, and evaluates the model at every frequency.
    ``adaptive_max_samples`` caps those solves per excitation; None leaves Palace's own cap.

    Raises on build:
        TypeError: a frequency or the tolerance is not a number, or ``points`` or ``adaptive_max_samples`` is not
            an integer.
        ValueError: ``start_hz`` is not finite and above zero, ``stop_hz`` is not finite and above ``start_hz``,
            ``points`` is below 2, ``adaptive_tolerance`` is not finite and above zero, or
            ``adaptive_max_samples`` is below 1 or is given without ``adaptive_tolerance``.
    """

    start_hz: float
    stop_hz: float
    points: int
    adaptive_tolerance: float | None = None
    adaptive_max_samples: int | None = None

    def __post_init__(self) -> None:
        """Refuse a sweep Palace cannot sample, then hold the numbers as floats where Palace reads floats."""
        where = Declared("the frequency sweep", _WHERE_DRIVEN_SWEEP)
        start = _sweep_number(where, "start", self.start_hz)
        stop = _sweep_number(where, "stop", self.stop_hz)
        # The frequency-domain problem is singular at zero frequency, so the sweep starts above it.
        if start <= 0.0:
            raise ValueError(where.expected("start", "above 0 Hz", self.start_hz))
        if stop <= start:
            raise ValueError(
                where.expected("stop", f"above start ({start!r} Hz)", self.stop_hz)
            )
        # Palace spaces NSample points by (stop - start) / (NSample - 1), both ends included.
        if _sweep_integer(where, "points", self.points) < 2:
            raise ValueError(where.expected("points", "at least 2", self.points))
        # A frozen dataclass normalizes its own fields through object.__setattr__.
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)
        if self.adaptive_tolerance is not None:
            tolerance = _sweep_number(
                where, "adaptive_tolerance", self.adaptive_tolerance
            )
            if tolerance <= 0.0:
                raise ValueError(
                    where.expected(
                        "adaptive_tolerance",
                        "above 0; leave it out for a uniform sweep",
                        self.adaptive_tolerance,
                    )
                )
            object.__setattr__(self, "adaptive_tolerance", tolerance)
        if self.adaptive_max_samples is not None:
            if self.adaptive_tolerance is None:
                raise ValueError(
                    where.refusal(
                        "adaptive_max_samples caps the solves of the adaptive sweep, and no "
                        "adaptive_tolerance turns that sweep on: give adaptive_tolerance too, or leave "
                        "adaptive_max_samples out"
                    )
                )
            if (
                _sweep_integer(where, "adaptive_max_samples", self.adaptive_max_samples)
                < 1
            ):
                raise ValueError(
                    where.expected(
                        "adaptive_max_samples", "at least 1", self.adaptive_max_samples
                    )
                )

    @property
    def adaptive(self) -> bool:
        """Whether the sweep is Palace's adaptive fast sweep rather than a solve at every frequency."""
        return self.adaptive_tolerance is not None

    def to_payload(self) -> dict[str, Any]:
        """The sweep as ``input_summary()`` shows it, frequencies in Hz."""
        return {
            "kind": "adaptive" if self.adaptive else "uniform",
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
            "adaptive_tolerance": self.adaptive_tolerance,
            "adaptive_max_samples": self.adaptive_max_samples,
        }


def _sweep_number(where: Declared, field: str, value: object) -> float:
    """``value`` as a finite float; a value that is not a finite number is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(where.expected(field, "a number", value))
    if not math.isfinite(float(value)):
        raise ValueError(where.expected(field, "finite", value))
    return float(value)


def _sweep_integer(where: Declared, field: str, value: object) -> int:
    """``value`` as an int; a value that is not an integer is refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(where.expected(field, "an integer", value))
    return value


@dataclass(frozen=True)
class PortConfigBuildResult:
    """A config with lumped ports (Eigenmode, Driven), its index entries and the records derived from it."""

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
    "DrivenSweep",
    "EigenmodeSettings",
    "ElectrostaticConfigBuildResult",
    "ElectrostaticSettings",
    "PortConfigBuildResult",
    "TerminalBinding",
    "optional_string",
]
