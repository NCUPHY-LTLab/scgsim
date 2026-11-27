"""The detached result of a Palace Driven run: its S-parameters with the Surface EPR and the adaptive sampling.

``PalaceDrivenResult`` is an ``SParameters`` (``scgsim.semantics.s_parameters``), the record every driven solver
returns, with two Palace facts beside it.  A driven participation is the share of the electric energy stored at
one frequency with one port driven; only near an isolated resonance does it approach a mode's, so no T1 or
device Q is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.palace.results.inspection_records import (
    SurfaceEprRecord,
    SurfaceMarginEprRecord,
)
from scgsim.semantics.s_parameters import SParameters


@dataclass(frozen=True)
class DrivenSurfaceEpr:
    """The Surface EPR Palace computed at one swept frequency with one port driven.

    ``surfaces`` are the whole-surface ``Dielectric`` rows and ``margins`` the margin rows, each bound to the
    sidecar rows it integrates as the Eigenmode readers bind them.  ``participation`` is Palace's ``p_surf``, the
    row's share of the electric energy (the field's and the lumped capacitors', as in Eigenmode);
    ``quality_factor`` is its ``Q_surf``, one over the participation times the row's loss tangent: a figure of
    the row, not a device Q.
    """

    excited_port: int
    frequency_hz: float
    surfaces: tuple[SurfaceEprRecord, ...]
    margins: tuple[SurfaceMarginEprRecord, ...]


@dataclass(frozen=True)
class AdaptiveSampling:
    """How Palace's adaptive fast sweep converged with one port excited, as its log records it.

    ``samples`` is the number of full solves the reduced model was built from, at ``sampled_frequencies_hz`` in
    the order Palace added them; ``error`` is the model's last estimated relative error, below the requested
    tolerance.
    """

    excited_port: int
    samples: int
    error: float
    sampled_frequencies_hz: tuple[float, ...]


@dataclass(frozen=True)
class PalaceDrivenResult(SParameters):
    """The S-parameters of a Palace Driven run, with its Surface EPR and its adaptive sampling.

    ``surface_epr`` holds one record per excited port and frequency, the ports in order and each port's
    frequencies in sweep order; it is empty when the plan sets no Surface EPR.  ``adaptive_sampling`` holds one
    record per excited port for an adaptive sweep, and is empty for a uniform one.
    """

    surface_epr: tuple[DrivenSurfaceEpr, ...]
    adaptive_sampling: tuple[AdaptiveSampling, ...]
