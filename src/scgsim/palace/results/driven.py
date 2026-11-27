"""Detach the result of a resolved Palace Driven run: its S-parameters, Surface EPR and adaptive sampling.

Columns are bound to ports by name, never by position.  ``|S[o][e]|`` is the port with ``Index`` o in the index
map, which names it as the plan does, at excitation e, which the config gives (``driven_excitations``: each
port is excited by its own ``Index``).  The S-matrix is over the plan's ports, in the index map's order, the
plan's: port i is row and column i.  The plan's other lumped elements are unexcited lumped ports; Palace writes
their columns too, which the table validators check and the record leaves out, since an element is a
termination of the device, not one of its ports.

Whether the adaptive fast sweep converged is written only to Palace's log, so a run whose log says an excitation
reached its maximum number of samples is refused, naming the port and ``adaptive_max_samples``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.contracts.palace_index_map import LUMPED_PORT_SECTION
from scgsim.contracts.palace_run_folder import (
    DRIVEN,
    PORT_S_MAGNITUDE_COLUMN,
    PORT_S_PHASE_COLUMN,
    confined_path,
)
from scgsim.palace.config.driven_sweep import DrivenSweep
from scgsim.palace.results.artifacts import checked_number, expect_mapping
from scgsim.palace.results.driven_records import (
    AdaptiveSampling,
    DrivenSurfaceEpr,
    PalaceDrivenResult,
)
from scgsim.palace.results.driven_tables import (
    FREQUENCY_COLUMN,
    driven_excitations,
    excitation_suffix,
    requested_sweep,
)
from scgsim.palace.results.inspection_records import (
    SurfaceEprRecord,
    SurfaceMarginEprRecord,
)
from scgsim.palace.results.resolved_records import ParsedTable, ResolvedRunData
from scgsim.palace.results.surface_readers import (
    read_surface_bindings,
    read_surface_margin_bindings,
)
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared, found_text, internal
from scgsim.semantics.s_parameters import SParameterPort

_WHERE_DRIVEN_RESULT_CALL = "the call to driven_result"
_WHERE_SWEEP = "the call to DrivenSim.set_driven"


def driven_result(run: ResolvedRunData) -> PalaceDrivenResult:
    """The S-parameters, Surface EPR and adaptive sampling of a resolved Driven run, detached from its folder.

    Raises:
        TypeError: ``run`` is not a resolved Palace Driven run.
        ValueError: the adaptive sweep reached its maximum number of samples for a port, or the log does not
            record the adaptive sampling of every port.
    """
    _require_driven(run)
    config = run.provenance.config
    # The ports are the excited lumped ports; each excitation is its port's Index.
    excitations = driven_excitations(config)
    ports = _ports(run.provenance.index_map, config, excitations)
    port_s = run.tables["port-S"]
    # The frequencies as Palace printed them, which the table validators checked are the requested ones.
    frequencies_hz = tuple(row[FREQUENCY_COLUMN] * 1e9 for row in port_s.rows)
    sweep = requested_sweep(config)
    return PalaceDrivenResult(
        ports=ports,
        frequencies_hz=frequencies_hz,
        magnitude_db=_s_matrices(port_s, excitations, PORT_S_MAGNITUDE_COLUMN),
        phase_deg=_s_matrices(port_s, excitations, PORT_S_PHASE_COLUMN),
        sweep=sweep.to_payload(),
        provenance=_provenance(run, sweep),
        surface_epr=_surface_epr(run, excitations, frequencies_hz),
        adaptive_sampling=_adaptive_sampling(run, excitations, ports, sweep),
    )


def _require_driven(run: object) -> None:
    """Refuse anything but a resolved Driven run, naming what to call for the other problems."""
    if not isinstance(run, ResolvedRunData) or run.problem != DRIVEN.name:
        raise TypeError(
            Declared(
                "driven_result requires a resolved Palace Driven run",
                _WHERE_DRIVEN_RESULT_CALL,
            ).refusal(
                found_text(getattr(run, "problem", type(run).__name__), DRIVEN.name)
                + "; an Eigenmode run's results are in epr_result()"
            )
        )


def _ports(
    index_map: Mapping[str, Any],
    config: Mapping[str, Any],
    excitations: tuple[int, ...],
) -> tuple[SParameterPort, ...]:
    """The S-matrix ports, the excited ones, in the index map's order, each with the impedance its config row has."""
    rows = {
        row["Index"]: row
        for row in expect_mapping(config.get("Boundaries"), "config.Boundaries")[
            "LumpedPort"
        ]
    }
    entries = sorted(
        (
            entry
            for entry in index_map["entries"]
            if entry["section"] == LUMPED_PORT_SECTION and entry["index"] in excitations
        ),
        key=lambda entry: entry["index"],
    )
    return tuple(
        SParameterPort(
            name=entry["port_name"],
            sheet_id=entry["metadata"]["port_sheet_id"],
            impedance_ohm=checked_number(
                rows[entry["index"]].get("R"),
                f"config LumpedPort {entry['index']} R",
                positive=True,
            ),
        )
        for entry in entries
    )


def _s_matrices(
    port_s: ParsedTable, ports: tuple[int, ...], column: str
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """One value of ``column`` (``{o}`` the output port, ``{e}`` the excitation) per frequency, output and port.

    ``ports`` are the Palace indices of the plan's ports, which are also their excitations.
    """
    return tuple(
        tuple(tuple(float(row[column.format(o=o, e=e)]) for e in ports) for o in ports)
        for row in port_s.rows
    )


def _provenance(run: ResolvedRunData, sweep: DrivenSweep) -> dict[str, Any]:
    """The run the values came from, and the returned files they were read from with their SHA-256."""
    read = {
        table.path.relative_to(run.run_dir).as_posix()
        for name, table in run.tables.items()
        if name in {"port-S", "surface-Q"}
    }
    if sweep.adaptive and run.returned_receipt.log is not None:
        read.add(run.returned_receipt.log["path"])
    return {
        "solver": "Palace",
        "problem": DRIVEN.name,
        "handoff_id": run.returned_receipt.handoff_id,
        "route": run.route,
        "solver_git_tag": run.cost.git_tag,
        "receipt_timestamp_utc": run.returned_receipt.timestamp_utc,
        "source_files": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in run.returned_receipt.output_files
            if item.get("path") in read
        ],
    }


def _surface_epr(
    run: ResolvedRunData,
    excitations: tuple[int, ...],
    frequencies_hz: tuple[float, ...],
) -> tuple[DrivenSurfaceEpr, ...]:
    """Each Surface-EPR row's participation and Q for every excited port at every frequency."""
    table = run.tables.get("surface-Q")
    if table is None:
        return ()
    surfaces = read_surface_bindings(run.provenance.index_map) or ()
    margins = read_surface_margin_bindings(run.provenance.index_map) or ()
    # The validated table has two columns per row and excitation; every one of its rows must be bound.
    if 2 * (len(surfaces) + len(margins)) * len(excitations) != len(table.headers) - 1:
        raise ValueError(
            internal(
                "the Surface-EPR bindings do not cover the surface-Q columns: "
                + found_text(
                    len(surfaces) + len(margins),
                    (len(table.headers) - 1) // (2 * len(excitations)),
                )
            )
        )
    suffix = {e: excitation_suffix(e, excitations) for e in excitations}
    return tuple(
        DrivenSurfaceEpr(
            excited_port=excitation,
            frequency_hz=frequency,
            surfaces=tuple(
                SurfaceEprRecord(
                    **binding.record_fields(),
                    **_row_values(row, binding.index, suffix[excitation]),
                )
                for binding in surfaces
            ),
            margins=tuple(
                SurfaceMarginEprRecord(
                    **binding.record_fields(),
                    **_row_values(row, binding.index, suffix[excitation]),
                )
                for binding in margins
            ),
        )
        for excitation in excitations
        for frequency, row in zip(frequencies_hz, table.rows, strict=True)
    )


def _row_values(row: Mapping[str, Any], surface: int, suffix: str) -> dict[str, Any]:
    """One Surface-EPR row's participation and Q at one frequency, ``suffix`` naming the excited port."""
    return {
        "participation": float(row[f"p_surf[{surface}]{suffix}"]),
        "quality_factor": float(row[f"Q_surf[{surface}]{suffix}"]),
    }


# ---- The adaptive sweep in Palace's log ------------------------------------------------------------------

# Palace 0.18.1 prints, for each excitation, "Adding excitation index {k} ({i}/{n}):", then once its
# reduced model is built "Adaptive sampling converged with {N} frequency samples:" or "Adaptive sampling
# reached maximum {N} frequency samples:", then " n = .., error = {e}, tol = .., memory = ../.." and
# " Sampled frequencies (GHz): {f}, {f}, ...".  A run with one excitation prints no "Adding excitation" line
# (issue 187, round 2 of slice 5).  An AMR pass repeats them; the last pass's are the result's.
_EXCITATION_LINE = re.compile(r"^Adding excitation index (\d+) \(\d+/\d+\):")
_STATUS_LINE = re.compile(
    r"^Adaptive sampling (converged with|reached maximum) (\d+) frequency samples:"
)
_ERROR_LINE = re.compile(r"^ n = \d+, error = (\S+), tol = ")
_SAMPLED_LINE = re.compile(r"^ Sampled frequencies \(GHz\): (.+)$")
# Palace wraps a long list of values: a line that ends in "," goes on in the next line, indented under the
# first value (five samples or more, issue 307).
_WRAPPED_LIST = re.compile(r",\n\s+(?=[-+0-9.])")


@dataclass
class _LoggedSampling:
    """What the log says about the adaptive sampling of one excitation, filled line by line."""

    reached_maximum: bool
    samples: int
    error: float | None = None
    sampled_ghz: tuple[float, ...] | None = None


def _adaptive_sampling(
    run: ResolvedRunData,
    excitations: tuple[int, ...],
    ports: tuple[SParameterPort, ...],
    sweep: DrivenSweep,
) -> tuple[AdaptiveSampling, ...]:
    """The adaptive sampling of every port, from the log; a uniform sweep has none.

    ``excitations`` are the ports' excitation indices, in the order of ``ports``.
    """
    if not sweep.adaptive:
        return ()
    log = run.returned_receipt.log
    if log is None:
        raise ValueError(internal("a resolved run has no returned log record"))
    logged = _logged_sampling(
        confined_path(run.run_dir, log["path"]).read_text(encoding="utf-8"), excitations
    )
    sampling = []
    for number, port in zip(excitations, ports, strict=True):
        status = logged.get(number)
        if status is not None and status.reached_maximum:
            raise ValueError(_not_converged(number, port, status, sweep))
        if (
            status is None
            or status.error is None
            or status.sampled_ghz is None
            or len(status.sampled_ghz) != status.samples
        ):
            raise ValueError(
                Declared(f"the solver log {log['path']}", RETURNED_RUN_FOLDER).refusal(
                    f"does not record how the adaptive sweep converged with port {number} ({port.name}) "
                    "excited, which only the log says"
                )
            )
        sampling.append(
            AdaptiveSampling(
                excited_port=number,
                samples=status.samples,
                error=status.error,
                sampled_frequencies_hz=tuple(f * 1e9 for f in status.sampled_ghz),
            )
        )
    return tuple(sampling)


def _logged_sampling(
    text: str, excitations: tuple[int, ...]
) -> dict[int, _LoggedSampling]:
    """The last adaptive-sampling record of each excitation in a Palace log."""
    logged: dict[int, _LoggedSampling] = {}
    # With one excitation, no line names it, so every record is that excitation's.
    excitation = excitations[0] if len(excitations) == 1 else None
    current: _LoggedSampling | None = None
    for line in _WRAPPED_LIST.sub(", ", text).splitlines():
        if match := _EXCITATION_LINE.match(line):
            excitation, current = int(match[1]), None
        elif (match := _STATUS_LINE.match(line)) and excitation is not None:
            current = _LoggedSampling(match[1] == "reached maximum", int(match[2]))
            logged[excitation] = current
        elif current is not None and (match := _ERROR_LINE.match(line)):
            current.error = float(match[1])
        elif current is not None and (match := _SAMPLED_LINE.match(line)):
            current.sampled_ghz = tuple(float(value) for value in match[1].split(","))
    return logged


def _not_converged(
    number: int, port: SParameterPort, status: _LoggedSampling, sweep: DrivenSweep
) -> str:
    """The refusal of a sweep whose reduced model, with port ``number`` excited, did not reach its tolerance."""
    cap = (
        f"adaptive_max_samples={sweep.adaptive_max_samples}"
        if sweep.adaptive_max_samples is not None
        else "Palace's default cap, adaptive_max_samples unset"
    )
    return Declared(
        f"the adaptive sweep with port {number} ({port.name}) excited", _WHERE_SWEEP
    ).refusal(
        f"reached its maximum of {status.samples} frequency samples ({cap}) before its reduced model "
        f"converged to adaptive_tolerance={sweep.adaptive_tolerance!r} (last error {status.error!r}), as "
        "Palace's log says, so the S-parameters it evaluated are not converged. Raise adaptive_max_samples, "
        "loosen adaptive_tolerance or sweep uniformly, and run again"
    )
