"""Five real Palace Driven runs of issue 187, kept as test fixtures, and the result each resolves to.

``tests/fixtures/palace_driven_runs`` keeps, for each run, the files of its returned run folder that the readback
reads, at their run-folder paths: the sealed ``config.json`` and ``metadata/palace_index_map.json``; the six
tables and the ``palace.json`` stock Palace 0.18.1 wrote into ``results/palace``; and its log
``logs/palace-manual.log``, whose launch line is scrubbed of the local install paths.  Nothing else is changed.
The layout is the roster's launcher-to-launcher composition (``scripts/roster_cases.py``; MA, MS and SA Surface EPR without
bands, 7 ``Dielectric`` rows; Route B; the left sheet a 50 ohm port, port 1), prepared with ``DrivenSim`` and
solved on 8 MPI ranks, the same mesh in each.  E1's two runs have the right sheet as port 2, 50 ohm:

- ``r1_order1_uniform5``: FEM order 1, a uniform sweep of 5 points from 2 to 10 GHz.
- ``r3_order2_adaptive81``: FEM order 2, the adaptive sweep (tolerance 1e-3) of 81 points from 2 to 10 GHz.

The runs of round 2 of slice 5 have the right sheet as an unexcited ``LumpedElement``, at FEM order 1, so they
have one excitation:

- ``ra_element_r50_uniform5``: ``resistance_ohm=50.0``, r1's sweep.  Port 1 sees the same 50 ohm termination as
  in r1, so its S11 is r1's.
- ``rb_element_l10nh_uniform5``: ``inductance_h=10e-9``, r1's sweep.
- ``rc_element_r50_adaptive11``: ra's element, the adaptive sweep (tolerance 1e-3) of 11 points.

The 12 MB mesh is not kept, so these files alone cannot pass the run folder's identity checks.
``fixture_result`` builds the ``ResolvedPalaceResult`` those checks lead to, from the real tables read by the
real table validators and the cost and timers of the real ``palace.json``, standing in only for the identity
checks; ``test_palace_driven_launchers`` (PDK) runs the real ``resolve_palace_result`` on a run folder of the
same config whose solver returns these files.  The handoff record is not kept either: ``write_handoff`` writes
what a report reads of it, the identity each run's handoff recorded (problem, route, the ``direct-local``
profile) and the config it sealed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scgsim.contracts.palace_convergence import ConvergenceCheck
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    HANDOFF_METADATA,
    INDEX_MAP,
    NATIVE_METADATA,
    output_table,
    sha256,
    solver_log,
)
from scgsim.palace import ResolvedPalaceResult
from scgsim.palace.results.convergence import SolverEnd
from scgsim.palace.results.resolved_records import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
)
from scgsim.palace.results.tables import (
    count_index_entries,
    output_families,
    read_result_table,
)

FIXTURES = Path(__file__).parent / "fixtures" / "palace_driven_runs"
UNIFORM_RUN = "r1_order1_uniform5"
ADAPTIVE_RUN = "r3_order2_adaptive81"
RESISTOR_RUN = "ra_element_r50_uniform5"
INDUCTOR_RUN = "rb_element_l10nh_uniform5"
RESISTOR_ADAPTIVE_RUN = "rc_element_r50_adaptive11"
LOG = solver_log("manual")


def copied_run(name: str, directory: Path) -> Path:
    """A copy of fixture run ``name`` in ``directory``, for a test to read or to change."""
    root = directory / name
    shutil.copytree(FIXTURES / name, root)
    return root


def read_json(root: Path, relative: str) -> dict:
    """One JSON record of the run folder ``root``."""
    return json.loads((root / relative).read_text(encoding="utf-8"))


def handoff_id(root: Path) -> str:
    """The handoff id the fixture folder ``root`` stands in with."""
    return f"fixture-{root.name}"


def file_record(root: Path, relative: str) -> dict[str, object]:
    """The record a handoff or a receipt keeps of one file of the run folder ``root``, or that it is absent."""
    path = root / relative
    if not path.is_file():
        return {"path": relative, "bytes": None, "sha256": None, "present": False}
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "present": True,
    }


def write_handoff(root: Path) -> list[dict[str, object]]:
    """Write the handoff record of the fixture folder ``root``: its identity and the config it sealed.

    Returns the input hashes it records, which a receipt of the run repeats.
    """
    hashes = [file_record(root, CONFIG)]
    record = {
        "problem": "Driven",
        "route": "B",
        "profile": "direct-local",
        "handoff_id": handoff_id(root),
        "status": "prepared",
        "hashes": hashes,
    }
    (root / HANDOFF_METADATA).write_text(json.dumps(record), encoding="utf-8")
    return hashes


def fixture_result(root: Path) -> ResolvedPalaceResult:
    """The resolved Driven run of the fixture folder ``root``: its tables read and checked as resolve reads them."""
    config = read_json(root, CONFIG)
    index_map = read_json(root, INDEX_MAP)
    palace_json = read_json(root, NATIVE_METADATA)
    counts = count_index_entries(index_map)
    tables = {
        family: read_result_table(root / output_table(family), "Driven", counts, config)
        for family in output_families("Driven", config, counts)
    }
    returned = [
        *(output_table(family) for family in tables),
        NATIVE_METADATA,
        LOG,
    ]
    output_files = tuple(
        {"path": path, "sha256": sha256(root / path), "present": True}
        for path in returned
    )
    problem = palace_json["Problem"]
    elapsed = palace_json["ElapsedTime"]
    return ResolvedPalaceResult(
        run_dir=root,
        problem="Driven",
        route="B",
        has_returned_outputs=True,
        status="completed",
        performance=PalacePerformance(
            counts=elapsed["Counts"], durations=elapsed["Durations"]
        ),
        cost=PalaceCost(
            problem_degrees_of_freedom=problem["DegreesOfFreedom"],
            mesh_elements=problem["MeshElements"],
            mpi_size=problem["MPISize"],
            openmp_threads=problem["OpenMPThreads"],
            peak_memory_megabytes=palace_json["PeakMemoryMegabytes"],
            peak_node_memory_megabytes=palace_json["PeakNodeMemoryMegabytes"],
            linear_solver=palace_json["LinearSolver"],
            git_tag=palace_json["GitTag"],
        ),
        tables=tables,
        returned_receipt=PalaceReturnedReceipt(
            schema="palace-returned-run-receipt.v2",
            schema_version=2,
            handoff_id=handoff_id(root),
            status="completed",
            ending="solver_ended",
            exit_code=0,
            solver_exit_code=0,
            tee_exit_code=0,
            route="B",
            problem="Driven",
            identity_verified=True,
            timestamp_utc="2026-11-25T00:00:00Z",
            job_identity={"job_id": "manual", "job_name": "manual"},
            input_hashes=(),
            output_files=output_files,
            log={"path": LOG, "present": True},
            convergence_decision=None,
            solver_identity={"GitTag": palace_json["GitTag"]},
        ),
        provenance=PalaceProvenance(
            handoff_metadata={},
            run_metadata={},
            resource_record={},
            handoff_archive_manifest={},
            index_map=index_map,
            mesh_manifest={},
            config=config,
            palace_json=palace_json,
            returned_receipt={},
        ),
        convergence=ConvergenceCheck.not_declared(),
        solver_end=SolverEnd(
            ending="solver_ended", stopped_after=None, passes_allowed=1
        ),
    )


def read_touchstone(path: Path) -> tuple[list[str], list[float], list[list[float]]]:
    """A Touchstone 1.1 file as text: its option line's words, its frequencies and each frequency's numbers.

    A test-side reader, independent of the writer: comments are skipped, and a line that starts a new
    frequency is told from a continuation line by the number of values a frequency's block holds.
    """
    option: list[str] = []
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("!"):
            continue
        if line.startswith("#"):
            option = line.split()
            continue
        values.extend(float(value) for value in line.split())
    ports = int(path.suffix[2:-1])
    width = 1 + 2 * ports * ports
    blocks = [values[start : start + width] for start in range(0, len(values), width)]
    return option, [block[0] for block in blocks], [block[1:] for block in blocks]
