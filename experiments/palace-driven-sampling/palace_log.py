"""What a Palace 0.18.1 Driven or Eigenmode log says about its run: the adaptive sampling and the time spent.

Palace writes these only to its log: each excitation's adaptive sampling ("Adaptive sampling converged with N
frequency samples" or "reached maximum N"), the offline phase's elapsed time, the elapsed time at the start
of each frequency ("It k/N: ... (total elapsed time = T s"), to 3 significant digits, and the elapsed-time
report at the end.
"""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER = r"([-+0-9.eE]+)"
_OFFLINE = re.compile(rf"Total offline phase elapsed time: {_NUMBER} s")
_ITERATION = re.compile(
    rf"^It (\d+)/(\d+): .* = {_NUMBER} GHz \(total elapsed time = {_NUMBER} s[,)]"
)
_SAMPLING = re.compile(
    r"^Adaptive sampling (converged with|reached maximum) (\d+) frequency samples:"
)
_SAMPLING_ERROR = re.compile(rf"error = {_NUMBER},")
_LINEAR_SOLVE = re.compile(r"^GMRES solver converged in (\d+) iterations")
_TIMER_ROW = re.compile(
    rf"^(\s*[A-Za-z][A-Za-z ]*?)\s+{_NUMBER}\s+{_NUMBER}\s+{_NUMBER}$"
)


def log_timing(log: str) -> dict[str, Any]:
    """The adaptive sampling, the linear solves, the offline time, each excitation's time per frequency, and the
    elapsed-time report, from Palace's log.

    Palace prints the elapsed time at the start of each frequency to 3 significant digits.  An excitation's time
    runs from its first frequency to the next excitation's first, the last one's to the run's total.
    """
    offline = _OFFLINE.search(log)
    sweeps: list[dict[str, Any]] = []
    timers: dict[str, float] = {}
    sampling: list[dict[str, Any]] = []
    gmres_iterations: list[int] = []
    in_report = False
    parent = ""
    lines = log.splitlines()
    for k, line in enumerate(lines):
        if sampled := _SAMPLING.match(line):
            # The next line holds the model's last error: " n = 18, error = 3.190e-05, tol = ...".
            error = _SAMPLING_ERROR.search(lines[k + 1])
            sampling.append(
                {
                    "converged": sampled.group(1) == "converged with",
                    "samples": int(sampled.group(2)),
                    "error": float(error.group(1)) if error else None,
                }
            )
        if solve := _LINEAR_SOLVE.match(line):
            gmres_iterations.append(int(solve.group(1)))
        iteration = _ITERATION.match(line)
        if iteration and int(iteration.group(1)) == 1:
            sweeps.append(
                {
                    "frequencies": int(iteration.group(2)),
                    "start_s": float(iteration.group(4)),
                }
            )
        if line.startswith("Elapsed Time Report"):
            in_report = True
        elif in_report and (row := _TIMER_ROW.match(line)):
            # An indented row is part of the row above it.  Columns: minimum, maximum and average over the
            # ranks; the maximum is kept.
            name = row.group(1).strip()
            if row.group(1).startswith(" "):
                name = f"{parent} / {name}"
            else:
                parent = name
            timers[name] = float(row.group(3))
            in_report = name != "Total"
    # An Eigenmode run sweeps nothing: no ends either.
    ends = [sweep["start_s"] for sweep in sweeps[1:]] + [timers.get("Total", math.nan)]
    for sweep, end in zip(sweeps, ends[: len(sweeps)], strict=True):
        sweep["seconds"] = end - sweep["start_s"]
        sweep["seconds_per_frequency"] = sweep["seconds"] / sweep["frequencies"]
    return {
        "offline_seconds": float(offline.group(1)) if offline else None,
        "adaptive_sampling": sampling,
        "gmres_solves": len(gmres_iterations),
        "gmres_iterations": sum(gmres_iterations),
        "gmres_iterations_range": [min(gmres_iterations), max(gmres_iterations)]
        if gmres_iterations
        else None,
        "excitations": sweeps,
        "elapsed_time_report_max_s": timers,
    }
