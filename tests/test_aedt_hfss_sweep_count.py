"""Accepted caller-owned sweep counts of an HFSS Driven Terminal request; transport data below is synthetic.

The request is a real one, on the SGB model of two lumped ports (``sgb_driven_inputs``).  No fixture
represents a completed native run.  Complete and partial synthetic readbacks exercise the real validator
without bypassing native identity checks.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("OCP")

from sgb_driven_inputs import driven_model  # noqa: E402

from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.results.readback_dispatch import validate_result_readback
from scgsim.aedt.runtime.families import hfss_exports as exports
from scgsim.aedt.runtime.families import hfss_setup as setup
from scgsim.aedt.specs.hfss import (
    FrequencySweepSpec,
    HfssDrivenGeometrySpec,
    HfssRunControl,
)
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard

pytestmark = pytest.mark.occt


def _spec(points=3):
    return HfssDrivenGeometrySpec(
        modeling="solid",
        aedt_version="2024.2",
        project_name="Synthetic",
        design_name="Design",
        model=driven_model(),
        run_control=HfssRunControl(
            "Setup", "Sweep", FrequencySweepSpec(1, 2, points=points)
        ),
    )


def _port_records(spec):
    """The lumped-port records an import makes, Terminals ``P1`` and ``P2``, each on one face."""
    return [
        {
            "index": port.index,
            "kind": "lumped",
            "requested": port.to_payload(),
            "boundary": port.name,
            "terminal_excitation": f"P{port.index}",
            "port_group": f"PORT::{port.port_sheet_id}",
            "faces": [{"object_name": "substrate", "face_id": 10 + port.index}],
            "host_entity_ids": ["chip/ground", f"chip/pad_{port.index}"],
            "signal_host": f"chip/pad_{port.index}",
            "terminal_edges": [20 + port.index],
        }
        for port in spec.ports
    ]


def _geometry(spec, ports):
    guard = read_step_guard(spec.model.path("guard"))
    return {
        "objects": {solid["solid"]: solid["aedt_name"] for solid in guard["solids"]},
        "model_identity_sha256": spec.model.identity_sha256,
        "lumped_ports": ports,
    }


class _SolutionData:
    """Double only for PyAEDT's returned solution-data transport."""

    primary_sweep = "Freq"
    units_sweeps = {"Freq": "GHz"}

    def __init__(self, count, short_expression=False):
        self.frequencies = [1 + i / (count - 1) for i in range(count)]
        self.count = count - int(short_expression)

    def get_expression_data(self, expression, component):
        return self.frequencies, [
            i + (0 if component == "real" else 0.5) for i in range(self.count)
        ]


class _NativeTransport:
    """Synthetic native-call boundary; no Desktop, solver or fake service."""

    setup_names = ()

    def __init__(self, count):
        self.count = count
        self.sweep_kwargs = None
        self.query = None
        self.export_ok = True
        self.readback_count = count
        self.sweep_error = None
        self.post = SimpleNamespace(get_solution_data=self.get_solution_data)

    def create_setup(self, name):
        return SimpleNamespace(
            props={},
            enable_adaptive_setup_broadband=lambda *a, **k: True,
            update=lambda: True,
        )

    def create_linear_count_sweep(self, *args, **kwargs):
        self.sweep_kwargs = kwargs
        if self.sweep_error:
            raise self.sweep_error
        return SimpleNamespace(
            props={
                "Type": "Fast",
                "RangeStart": "1.0GHz",
                "RangeEnd": "2.0GHz",
                "RangeCount": self.readback_count,
            }
        )

    def export_touchstone(self, **kwargs):
        if not self.export_ok:
            return False
        data = _SolutionData(self.count)
        Path(kwargs["output_file"]).write_text(
            "! Port[1] = P1\n! Port[2] = P2\n# GHz S RI R 50\n"
            + "".join(f"{f} 1 0 2 0 3 0 4 0\n" for f in data.frequencies)
        )
        return True

    def get_solution_data(self, **kwargs):
        self.query = kwargs
        return _SolutionData(self.count)


class HfssSweepCountTests(unittest.TestCase):
    """Trace authored sweep counts through setup, exports and strict readback."""

    def test_complete_synthetic_strict_readback_accepts_authored_count(self):
        """Accept the authored sweep count with complete synthetic strict readback."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _spec(5.0)
            transport = _NativeTransport(5)
            _, readback = exports.export_hfss_results(
                transport,
                root,
                spec,
                [{"terminal_excitation": "P1"}, {"terminal_excitation": "P2"}],
            )
            assigned = _port_records(spec)
            ports = [
                {
                    **record,
                    "native": {
                        "saved_boundary": {
                            "BoundType": "Lumped Port",
                            "LumpedPortType": "Terminal",
                            "Faces": [10 + record["index"]],
                        },
                        "saved_terminal": {
                            "BoundType": "Terminal",
                            "Edges": [20 + record["index"]],
                        },
                    },
                }
                for record in assigned
            ]
            results = root / "Synthetic.aedtresults"
            profiles = results / "Design.results"
            profiles.mkdir(parents=True)
            (results / "Design.asol").write_text(
                "SimSetupName='Setup'\nP( File='Adaptive.profile')\n"
            )
            (profiles / "Adaptive.profile").write_text(
                "Name='Adaptive Pass 1'\n\\'Max Mag. Delta S\\', 0.01,\n"
                "Adaptive Passes converged\n\\'Max solved tets\\', 100,\n"
            )
            # Isolated synthetic diagnostic text; never a protected workspace log.
            log = root / "batch.log"
            log.write_text(
                "Synthetic unittest transport evidence; no native execution.\n"
            )
            receipt = {
                "connected": {
                    "aedt_version": spec.aedt_version,
                    "pyaedt_version": spec.pyaedt_version,
                },
                "ports": ports,
                "geometry": _geometry(spec, assigned),
                "result_readback": readback,
                "setup": {
                    "name": "Setup",
                    "native": {
                        "solve_type": "Broadband",
                        "low_frequency": "1GHz",
                        "high_frequency": "2GHz",
                        "maximum_delta_s": 0.02,
                        "maximum_passes": 99,
                        "minimum_passes": 1,
                        "minimum_converged_passes": 1,
                        "percent_refinement": 30.0,
                    },
                },
                "convergence": read_hfss_convergence(root, spec),
                "diagnostics": {
                    "batch_log": "batch.log",
                    "present": True,
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    "physics_warnings": [],
                },
            }
            self.assertIsNone(validate_result_readback(root, receipt, spec))
            moved = deepcopy(receipt)
            moved["ports"][1]["native"]["saved_boundary"]["Faces"] = [99]
            with self.assertRaisesRegex(RuntimeError, "port 'p2' is not the requested"):
                validate_result_readback(root, moved, spec)
            for key in ("terminal_st", "touchstone"):
                with self.subTest(count_mismatch=key):
                    bad = deepcopy(receipt)
                    bad["result_readback"][key]["records"] = 20000
                    with self.assertRaisesRegex(
                        RuntimeError, "frequency readback|Touchstone readback"
                    ):
                        validate_result_readback(root, bad, spec)

    def test_default_and_historical_payload_identity(self):
        """Retain default and historical sweep payload identities."""
        self.assertEqual(FrequencySweepSpec(1, 2).points, 20000)
        # These canonical hashes freeze historical authored numeric representation.
        for value, digest in (
            (20000, "4fbbe08ff714f18363f8d8fa77951ea9e9d99d5a750aa25ba88a271855c1c086"),
            (
                20000.0,
                "a580178fbd641852eaa5f89c504aa1c4ad1c16b4168a32dd4afa79781057e901",
            ),
        ):
            with self.subTest(points=value, type=type(value)):
                payload = FrequencySweepSpec(1, 2, points=value).to_payload()
                raw = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                self.assertIs(type(FrequencySweepSpec(**payload).points), type(value))

    def test_whole_numeric_count_is_lossless_and_fractional_is_not_truncated(self):
        """Accept whole-valued counts and reject fractional counts without truncation."""
        for value in (7, 7.0):
            self.assertEqual(FrequencySweepSpec(1, 2, points=value).points, value)
        for value in (7.5, True, "7"):
            with (
                self.subTest(points=value),
                self.assertRaisesRegex(ValueError, "whole-number"),
            ):
                FrequencySweepSpec(1, 2, points=value)

    def test_authored_count_flows_through_setup_export_and_strict_readback(self):
        """Carry authored counts through native setup, exported files and strict readback."""
        mode = "terminal"
        for points in (3, 5.0):
            with (
                self.subTest(points=points),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                spec = _spec(points)
                transport = _NativeTransport(int(points))
                setup.setup_hfss(transport, spec)
                self.assertEqual(
                    transport.sweep_kwargs["num_of_freq_points"], int(points)
                )
                self.assertIs(type(transport.sweep_kwargs["num_of_freq_points"]), int)
                ports = [
                    {
                        "index": i,
                        "terminal_excitation": f"P{i}",
                        "unresolved_native": True,
                    }
                    for i in (1, 2)
                ]
                hashes, readback = exports.export_hfss_results(
                    transport, root, spec, ports
                )
                prefix = "St"
                self.assertEqual(
                    transport.query["expressions"],
                    [f"{prefix}({a},{b})" for a in ("P1", "P2") for b in ("P1", "P2")],
                )
                csv_path = root / "results" / mode / f"{mode}_{prefix.lower()}.csv"
                with csv_path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), int(points))
                self.assertEqual(
                    (float(rows[0]["frequency_ghz"]), float(rows[-1]["frequency_ghz"])),
                    (1, 2),
                )
                for relative, digest in hashes.items():
                    self.assertEqual(
                        hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                        digest,
                    )
                self.assertEqual(readback["touchstone"]["records"], int(points))
                # Partial transport evidence is never marked completed: reaching the
                # untouched port check proves count acceptance without bypassing it.
                receipt = {
                    "connected": {
                        "aedt_version": spec.aedt_version,
                        "pyaedt_version": spec.pyaedt_version,
                    },
                    "ports": ports,
                    "result_readback": readback,
                }
                error = "not the request's SGB model"
                with self.assertRaisesRegex(RuntimeError, error):
                    validate_result_readback(root, receipt, spec)
                result_key = f"{mode}_{prefix.lower()}"
                for label, mutate, message in (
                    (
                        "csv count",
                        lambda r, result_key=result_key: r["result_readback"][
                            result_key
                        ].update(records=9),
                        "frequency readback",
                    ),
                    (
                        "touchstone count",
                        lambda r: r["result_readback"]["touchstone"].update(records=9),
                        "Touchstone readback",
                    ),
                    (
                        "endpoint",
                        lambda r, result_key=result_key: r["result_readback"][
                            result_key
                        ].update(first_frequency_ghz=1.1),
                        "frequency readback",
                    ),
                    (
                        "port order",
                        lambda r: r["result_readback"]["touchstone"].update(
                            port_order=["P2", "P1"]
                        ),
                        "Touchstone readback",
                    ),
                    (
                        "identity",
                        lambda r: r["connected"].update(pyaedt_version="other"),
                        "version identity",
                    ),
                ):
                    with self.subTest(mismatch=label):
                        bad = deepcopy(receipt)
                        mutate(bad)
                        with self.assertRaisesRegex(RuntimeError, message):
                            validate_result_readback(root, bad, spec)

    def test_native_sweep_mismatch_and_exception_are_not_hidden(self):
        """Propagate native sweep-count mismatches and native setup exceptions."""
        transport = _NativeTransport(3)
        transport.readback_count = 4
        with self.assertRaisesRegex(RuntimeError, "sweep readback mismatch"):
            setup.setup_hfss(transport, _spec())
        failure = RuntimeError("synthetic native transport failure")
        transport.sweep_error = failure
        with self.assertRaises(RuntimeError) as caught:
            setup.setup_hfss(transport, _spec())
        self.assertIs(caught.exception, failure)

    def test_export_failure_and_expression_mismatch_remain_failures(self):
        """Reject failed exports and mismatched solution-expression data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = _NativeTransport(3)
            transport.export_ok = False
            with self.assertRaisesRegex(RuntimeError, "Touchstone export failed"):
                exports.export_hfss_results(
                    transport,
                    root,
                    _spec(),
                    [{"terminal_excitation": "P1"}, {"terminal_excitation": "P2"}],
                )
            with self.assertRaisesRegex(RuntimeError, "expression length mismatch"):
                exports.write_complex_csv(
                    _SolutionData(3, True),
                    ["St(P1,P1)"],
                    root / "short.csv",
                    suffix="",
                    spec=_spec(),
                )


if __name__ == "__main__":
    unittest.main()
