"""Strict AEDT matrix export parsers and canonical readback."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q2d import Q2dSpec


def parse_matrix_export(  # noqa: PLR0913 -- preserve the matrix codec public validation keyword API.
    path: Path,
    solver: str,
    problem: str,
    frequency_ghz: float,
    titles: dict[str, str],
    *,
    expected_setup: str | None = None,
    expected_unit_line: str | None = None,
    expected_labels: list[str] | None = None,
    expected_label_set: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read exact native matrix headers, ordered labels and finite square rows."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{solver} {problem} matrix export is missing or empty")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    expectation = _MatrixExpectation(
        solver,
        problem,
        frequency_ghz,
        titles,
        expected_setup,
        expected_unit_line,
        expected_labels,
        expected_label_set,
    )
    units = _read_matrix_header(lines, expectation)
    result, labels_by_quantity = _read_matrix_blocks(lines, units, expectation)
    return result, {
        "problem_type": problem,
        "frequency_ghz": frequency_ghz,
        "quantities": list(titles.values()),
        "labels": labels_by_quantity,
        "rows": len(result),
        "bytes": path.stat().st_size,
    }


def read_q2d_rlgc_matrix(
    path: Path, spec: Q2dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify the combined Q2D distributed R/L/G/C native export."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Q2D combined CG/RL matrix export is missing or empty")
    control = cast(MatrixRunControl, spec.run_control)
    expected_unit_line = (
        "C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter"
    )
    if path.read_text(encoding="utf-8", errors="strict").splitlines()[:5] != [
        f"{control.setup_name}:LastAdaptive",
        "Problem Type:  CG, RL",
        expected_unit_line,
        "Reduce Matrix:  Original",
        f"Frequency:  {control.frequency_ghz:g}GHz",
    ]:
        raise RuntimeError("Q2D combined CG/RL matrix header is invalid")
    labels = [
        conductor.name
        for conductor in spec.conductors
        if conductor.conductor_type == "SignalLine"
    ]
    return parse_matrix_export(
        path,
        "Q2D",
        "CG, RL",
        control.frequency_ghz,
        {
            "Capacitance Matrix": "C",
            "Conductance Matrix": "G",
            "Inductance Matrix": "L",
            "Resistance Matrix": "R",
        },
        expected_setup=f"{control.setup_name}:LastAdaptive",
        expected_unit_line=expected_unit_line,
        expected_labels=labels,
    )


@dataclass(frozen=True)
class _MatrixExpectation:
    """The existing optional native header and matrix-label requirements."""

    solver: str
    problem: str
    frequency_ghz: float
    titles: dict[str, str]
    expected_setup: str | None
    expected_unit_line: str | None
    expected_labels: list[str] | None
    expected_label_set: set[str] | None

    @property
    def strict(self) -> bool:
        """Enable exact header/block counting when any optional check is authored."""
        return any(
            value is not None
            for value in (
                self.expected_setup,
                self.expected_unit_line,
                self.expected_labels,
                self.expected_label_set,
            )
        )


def _read_matrix_header(
    lines: list[str], expectation: _MatrixExpectation
) -> dict[str, str]:
    """Validate header identity before reading the declared unit line."""
    required_lines = {
        f"Problem Type:  {expectation.problem}",
        f"Frequency:  {expectation.frequency_ghz:g}GHz",
    }
    if any(line not in lines for line in required_lines):
        raise RuntimeError(
            f"{expectation.solver} {expectation.problem} matrix header is invalid"
        )
    if expectation.strict:
        required_lines.add("Reduce Matrix:  Original")
        if expectation.expected_setup is not None:
            required_lines.add(expectation.expected_setup)
        if any(lines.count(line) != 1 for line in required_lines):
            raise RuntimeError(
                f"{expectation.solver} {expectation.problem} matrix header is invalid"
            )
    unit_line = next((line for line in lines if " Units:" in line), None)
    if unit_line is None or (
        expectation.expected_unit_line is not None
        and unit_line != expectation.expected_unit_line
    ):
        raise RuntimeError(
            f"{expectation.solver} {expectation.problem} matrix units are invalid"
        )
    units = dict(re.findall(r"([CGRL]) Units:([^,]+)", unit_line))
    return units


def _read_matrix_blocks(
    lines: list[str], units: dict[str, str], expectation: _MatrixExpectation
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Read the requested quantity blocks in their declared order."""
    result: list[dict[str, Any]] = []
    labels_by_quantity: dict[str, list[str]] = {}
    for title, quantity in expectation.titles.items():
        if title not in lines or (expectation.strict and lines.count(title) != 1):
            raise RuntimeError(
                f"{expectation.solver} export must contain one exact {title!r}"
            )
        title_index = lines.index(title)
        header_index = title_index + 1
        while header_index < len(lines) and not lines[header_index].strip():
            header_index += 1
        header = next(csv.reader([lines[header_index]]))
        labels = [value.strip() for value in header[1:] if value.strip()]
        if (
            not labels
            or len(set(labels)) != len(labels)
            or (
                expectation.expected_labels is not None
                and labels != expectation.expected_labels
            )
            or (
                expectation.expected_label_set is not None
                and set(labels) != expectation.expected_label_set
            )
        ):
            raise RuntimeError(f"{expectation.solver} {title} labels are invalid")
        labels_by_quantity[quantity] = labels
        for row_index, row_name in enumerate(labels, start=header_index + 1):
            values = next(csv.reader([lines[row_index]]))
            if values[0].strip() != row_name or len(values[1:]) != len(labels):
                raise RuntimeError(f"{expectation.solver} {title} matrix is not square")
            for column, raw in zip(labels, values[1:], strict=True):
                value = float(raw)
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"{expectation.solver} {title} contains a non-finite value"
                    )
                result.append(
                    {
                        "problem_type": expectation.problem,
                        "quantity": quantity,
                        "row": row_name,
                        "column": column,
                        "value": value,
                        "unit": units[quantity],
                    }
                )
        block_end = header_index + 1 + len(labels)
        if expectation.strict and block_end < len(lines) and lines[block_end].strip():
            raise RuntimeError(
                f"{expectation.solver} {title} contains unexpected matrix rows"
            )
    return result, labels_by_quantity
