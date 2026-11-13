"""Own Q3D matrix export and normalized result files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from scgsim.aedt.io import file_sha256, write_csv
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q3d import Q3dSpec


def export_q3d_matrices(
    app: Any, run_dir: Path, spec: Q3dSpec
) -> tuple[dict[str, str], dict[str, Any]]:
    """Export each requested matrix and retain independent native readback summaries."""
    control = _matrix_control(spec)
    output_dir = run_dir / "results" / "q3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    row_counts: dict[str, int] = {}
    exports = (("C", "c"), ("AC RL", "ac_rl")) if spec.solve_ac_rl else (("C", "c"),)
    normalized: list[dict[str, Any]] = []
    for problem, stem in exports:
        path = output_dir / f"{stem}_matrix.csv"
        export_matrix_file(app, path, spec, problem)
        titles = {
            "C": {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
            "AC RL": {"AC Inductance Matrix": "L", "AC Resistance Matrix": "R"},
        }[problem]
        matrix_options = matrix_parse_options(spec, problem)
        rows, summary = parse_matrix_export(
            path,
            "Q3D",
            problem,
            control.frequency_ghz,
            titles,
            **matrix_options,
        )
        if spec.solve_ac_rl:
            normalized.extend(rows)
        summaries[stem] = summary
        row_counts[stem] = len(rows)
        hashes[path.relative_to(run_dir).as_posix()] = file_sha256(path)
    if not spec.solve_ac_rl:
        path = output_dir / "c_matrix.csv"
        return hashes, {
            "matrices": {
                "path": path.relative_to(run_dir).as_posix(),
                "frequency_ghz": control.frequency_ghz,
                "native": summaries["c"],
                "primary_rows": row_counts["c"],
            }
        }
    normalized_path = output_dir / "matrices.csv"
    write_csv(
        normalized_path,
        list[Mapping[str, Any]](normalized),
        fieldnames=["problem_type", "quantity", "row", "column", "value", "unit"],
    )
    hashes[normalized_path.relative_to(run_dir).as_posix()] = file_sha256(
        normalized_path
    )
    return hashes, {
        "matrices": {
            "frequency_ghz": control.frequency_ghz,
            "native": summaries,
            "normalized_rows": len(normalized),
        }
    }


def export_matrix_file(app: Any, path: Path, spec: Q3dSpec, problem: str) -> None:
    """One native LastAdaptive/Original export authority for full and stopped runs."""
    control = _matrix_control(spec)
    frequency_hz = control.frequency_ghz * 1e9
    app.odesign.ExportMatrixData(
        str(path),
        problem,
        "",
        f"{control.setup_name} : LastAdaptive",
        "Original",
        "ohm",
        "nH",
        "pF",
        "mho",
        frequency_hz,
        "Maxwell",
        0,
        False,
        15,
        20,
        1,
    )


class MatrixParseOptions(TypedDict, total=False):
    """Only capacitance parsing checks the declared signal-net label set."""

    expected_label_set: set[str]


def matrix_parse_options(spec: Q3dSpec, problem: str) -> MatrixParseOptions:
    """Select the label check for a matrix: a capacitance matrix must name exactly the Signal nets."""
    if problem == "C":
        return {
            "expected_label_set": {
                net.name for net in spec.nets if net.net_type == "Signal"
            }
        }
    return {}


def _matrix_control(spec: Q3dSpec) -> MatrixRunControl:
    if spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    return spec.run_control
