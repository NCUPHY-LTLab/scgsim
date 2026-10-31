"""Export of HFSS solutions to Touchstone and eigenmode CSV files.

Owns writing the exported files and verifying them against the solved setup.
It does not own the solve or the native readback records."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, write_csv
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.parse import HfssSpec


def _export(
    hfss: Any, run_dir: Path, spec: HfssSpec, ports: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, Any]]:
    output_dir = run_dir / "results" / spec.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(spec, HfssEigenmodeSpec):
        return _export_eigenmode(hfss, run_dir, output_dir, spec)
    setup_sweep = f"{spec.run_control.setup_name} : {spec.run_control.sweep_name}"
    terminal = spec.mode == "terminal"
    names = [
        record["terminal_excitation" if terminal else "modal_excitation"]
        for record in ports
    ]
    hashes: dict[str, str] = {}
    readback: dict[str, Any] = {}
    touchstone = output_dir / f"{spec.mode}.s2p"
    if (
        not hfss.export_touchstone(
            setup=spec.run_control.setup_name,
            sweep=spec.run_control.sweep_name,
            output_file=str(touchstone),
        )
        or not touchstone.is_file()
    ):
        raise RuntimeError(f"{spec.mode} Touchstone export failed")
    readback["touchstone"] = _verify_touchstone(touchstone, spec, names)
    prefix = "St" if terminal else "S"
    expressions = [f"{prefix}({left},{right})" for left in names for right in names]
    data = hfss.post.get_solution_data(
        expressions=expressions,
        setup_sweep_name=setup_sweep,
        report_category=(
            "Terminal Solution Data" if terminal else "Modal Solution Data"
        ),
    )
    csv_path = output_dir / f"{spec.mode}_{prefix.casefold()}.csv"
    readback[f"{spec.mode}_{prefix.casefold()}"] = _write_complex_csv(
        data, expressions, csv_path, suffix="", spec=spec
    )
    hashes[touchstone.relative_to(run_dir).as_posix()] = file_sha256(touchstone)
    hashes[csv_path.relative_to(run_dir).as_posix()] = file_sha256(csv_path)
    return hashes, readback


def _export_eigenmode(
    hfss: Any,
    run_dir: Path,
    output_dir: Path,
    spec: HfssEigenmodeSpec,
) -> tuple[dict[str, str], dict[str, Any]]:
    setup = f"{spec.run_control.setup_name} : LastAdaptive"
    raw = output_dir / "eigenmodes.eig"
    hfss.osolution.ExportEigenmodes(setup, "", str(raw))
    if not raw.is_file() or raw.stat().st_size == 0:
        raise RuntimeError("HFSS Eigenmode native export is missing or empty")
    rows = _parse_eigenmode_export(raw, spec)
    path = output_dir / "eigenmodes.csv"
    write_csv(path, rows, fieldnames=["mode", "frequency_ghz", "q_factor"])
    raw_relative = raw.relative_to(run_dir).as_posix()
    path_relative = path.relative_to(run_dir).as_posix()
    return {
        raw_relative: file_sha256(raw),
        path_relative: file_sha256(path),
    }, {
        "eigenmodes": {
            "modes": len(rows),
            "frequency_unit": "GHz",
            "mode_indices": [row["mode"] for row in rows],
            "frequencies_ghz": [row["frequency_ghz"] for row in rows],
            "q_factors": [row["q_factor"] for row in rows],
            "native_export": raw.name,
            "native_export_bytes": raw.stat().st_size,
        }
    }


def _parse_eigenmode_export(
    path: Path, spec: HfssEigenmodeSpec
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_header = next(
        (
            index
            for index, line in enumerate(lines)
            if "Mode" in line and "Frequency (GHz)" in line and "Q" in line
        ),
        None,
    )
    if (
        not lines
        or lines[0] != "# Ansys eigenmode data file.  Version 2.0"
        or f"# Design:     {spec.design_name}" not in lines
        or f"# Solution:   {spec.run_control.setup_name} : LastAdaptive" not in lines
        or table_header is None
    ):
        raise RuntimeError("HFSS Eigenmode native export header is invalid")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        lines[table_header + 1 :], start=table_header + 2
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = line.split()
        if (
            not tokens
            or re.fullmatch(r"[0-9]+", tokens[0]) is None
            or len(tokens) not in {3, 6}
        ):
            raise RuntimeError(
                f"HFSS Eigenmode native mode row at line {line_number} is malformed"
            )
        mode = int(tokens[0])
        if mode <= 0:
            raise RuntimeError(
                f"HFSS Eigenmode native mode row at line {line_number} is malformed"
            )
        try:
            frequency = float(tokens[1])
            if len(tokens) == 3:
                q_factor = float(tokens[2])
                imaginary_frequency = None
            else:
                if tokens[2] not in {"+", "-"} or tokens[4] != "j":
                    raise ValueError
                imaginary_magnitude = float(tokens[3])
                imaginary_frequency = (
                    imaginary_magnitude if tokens[2] == "+" else -imaginary_magnitude
                )
                q_factor = float(tokens[5])
        except ValueError:
            raise RuntimeError(
                f"HFSS Eigenmode native mode row at line {line_number} is malformed"
            ) from None
        if (
            not math.isfinite(frequency)
            or frequency <= 0
            or not math.isfinite(q_factor)
            or q_factor < 0
            or (
                imaginary_frequency is not None
                and not math.isfinite(imaginary_frequency)
            )
        ):
            raise RuntimeError(
                "HFSS Eigenmode native mode row at line "
                f"{line_number} has invalid numeric values"
            )
        rows.append({"mode": mode, "frequency_ghz": frequency, "q_factor": q_factor})
    if [row["mode"] for row in rows] != list(range(1, spec.run_control.num_modes + 1)):
        raise RuntimeError("HFSS Eigenmode native mode count/order is invalid")
    return rows


def _write_complex_csv(
    data: Any, expressions: list[str], path: Path, *, suffix: str, spec: HfssDrivenSpec
) -> dict[str, Any]:
    if not data:
        raise RuntimeError("solution-data extraction failed")
    point_count = int(spec.run_control.sweep.points)
    frequency, _ = data.get_expression_data(expressions[0], "real")
    if len(frequency) != point_count:
        raise RuntimeError(
            f"extracted point count must be {point_count}, got {len(frequency)}"
        )
    unit = data.units_sweeps.get(data.primary_sweep)
    values = [float(value) for value in frequency]
    if (
        data.primary_sweep != "Freq"
        or unit != "GHz"
        or not math.isclose(values[0], spec.run_control.sweep.start_ghz, abs_tol=1e-9)
        or not math.isclose(values[-1], spec.run_control.sweep.stop_ghz, abs_tol=1e-9)
        or any(right <= left for left, right in pairwise(values))
    ):
        raise RuntimeError(
            "solution frequency records have invalid units, endpoints, or ordering"
        )
    columns: dict[str, tuple[Any, Any]] = {
        expression: (
            data.get_expression_data(expression, "real")[1],
            data.get_expression_data(expression, "imag")[1],
        )
        for expression in expressions
    }
    if any(
        len(real) != point_count or len(imag) != point_count
        for real, imag in columns.values()
    ):
        raise RuntimeError("solution-data expression length mismatch")
    names = [
        "frequency_ghz",
        *[
            name
            for expression in expressions
            for name in (f"Re({expression}){suffix}", f"Im({expression}){suffix}")
        ],
    ]
    rows = []
    for index in range(point_count):
        row: dict[str, Any] = {"frequency_ghz": frequency[index]}
        for expression, (real, imag) in columns.items():
            row[f"Re({expression}){suffix}"] = real[index]
            row[f"Im({expression}){suffix}"] = imag[index]
        rows.append(row)
    write_csv(path, rows, fieldnames=names)
    return {
        "records": len(values),
        "frequency_unit": unit,
        "first_frequency_ghz": values[0],
        "last_frequency_ghz": values[-1],
        "strictly_increasing": True,
    }


def _verify_touchstone(
    path: Path, spec: HfssDrivenSpec, native_port_names: list[str]
) -> dict[str, Any]:
    """Check an exported two-port Touchstone file against the sweep and native port order."""
    if path.suffix.lower() != ".s2p" or path.stat().st_size == 0:
        raise RuntimeError("Touchstone must be a nonempty .s2p file")
    if len(native_port_names) != 2 or len(set(native_port_names)) != 2:
        raise RuntimeError("Touchstone requires two ordered native port names")
    point_count = int(spec.run_control.sweep.points)
    unit, frequencies, header_ports = _read_touchstone_records(path)
    if (
        unit != "GHz"
        or len(frequencies) != point_count
        or not math.isclose(
            frequencies[0], spec.run_control.sweep.start_ghz, abs_tol=1e-9
        )
        or not math.isclose(
            frequencies[-1], spec.run_control.sweep.stop_ghz, abs_tol=1e-9
        )
        or any(right <= left for left, right in pairwise(frequencies))
    ):
        raise RuntimeError(
            "Touchstone records have invalid count, units, endpoints, or ordering"
        )
    if [header_ports.get(1), header_ports.get(2)] != native_port_names:
        raise RuntimeError(
            "Touchstone indexed port-name order does not match native ports"
        )
    return {
        "path": path.name,
        "ports": 2,
        "records": len(frequencies),
        "frequency_unit": unit,
        "first_frequency_ghz": frequencies[0],
        "last_frequency_ghz": frequencies[-1],
        "strictly_increasing": True,
        "port_order": native_port_names,
        "bytes": path.stat().st_size,
    }


def _read_touchstone_records(
    path: Path,
) -> tuple[str | None, list[float], dict[int, str]]:
    """Read the frequency unit, data-record frequencies and ``! Port[n]`` names of a Touchstone file.

    Lines that are neither headers nor numeric data records are skipped.
    """
    unit: str | None = None
    frequencies: list[float] = []
    header_ports: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            match = re.fullmatch(r"!\s*Port\[(1|2)]\s*=\s*(.+?)\s*", stripped)
            if match:
                index, name = int(match.group(1)), match.group(2)
                if index in header_ports:
                    raise RuntimeError("Touchstone repeats an indexed port-name header")
                header_ports[index] = name
            continue
        if stripped.startswith("#"):
            tokens = stripped[1:].split()
            if not tokens or tokens[0].casefold() != "ghz":
                raise RuntimeError("Touchstone frequency unit must be GHz")
            unit = "GHz"
            continue
        if stripped.startswith("["):
            continue
        tokens = stripped.split()
        try:
            frequency = float(tokens[0])
        except (IndexError, ValueError):
            continue
        if len(tokens) != 9:
            raise RuntimeError("Touchstone is not a two-port data record")
        frequencies.append(frequency)
    return unit, frequencies, header_ports
