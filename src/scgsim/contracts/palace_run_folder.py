"""The Palace run folder: where each file lives, the schema of each record, and what each problem writes.

Every Palace stage that writes or reads a run folder takes a file's path, a record's schema and a problem's
outputs from here.  A path is relative to the run folder, in POSIX form::

    config.json                         CONFIG           (config)
    palace.msh                          MESH             (mesh)
    geometry/design.gds, .stack.json    SOURCE_GDS, SOURCE_STACK (mesh)
    mesh/                               MESH_DIR: the meshing module's records (mesh)
    metadata/mesh_manifest.json         MESH_MANIFEST    (mesh)
    metadata/route_<a|b>_provenance.json  route_provenance(route) (mesh)
    metadata/palace_index_map.json      INDEX_MAP        (config), with the problem's other metadata files
    metadata/palace_convergence_plan.json  CONVERGENCE_PLAN (config), only when convergence is declared
    run_palace.sh | run_palace.sbatch   script_name(profile) (run)
    metadata/palace_handoff_*.json ...  HANDOFF_METADATA, RUN_METADATA, RESOURCE_RECORD, ARCHIVE_MANIFEST (run)
    metadata/record_palace_return.py    RECORDER: the receipt recorder (run)
    metadata/run_palace_until_converged.py  CONVERGENCE_RUNNER (run), only when convergence is declared
    logs/palace-<job>.log               solver_log(job): the solver log the run script writes
    results/palace/                     OUTPUT_DIR: what Palace writes, see ``PalaceProblem``
    metadata/palace_convergence_decision.json  CONVERGENCE_DECISION, written by the runner during the solve
    metadata/palace_returned_run_receipt.json  RETURNED_RECEIPT, written by the recorder after the solve

The receipt recorder (``scgsim.palace.run.returned_receipt``) and the convergence runner
(``scgsim.palace.run.convergence_runner``) run in the run folder with only a Python interpreter.  When
``prepare_handoff`` copies them there, the text of this module takes the place of their import of it
(``scgsim.palace.run.standalone_copy``), so each copy carries the same layout, problems and helpers, and
nothing in the run folder imports SCGSim.  For that reason this module imports only the
standard library (``.importlinter``: ``contracts-import-nothing-else``) and has no ``__future__`` import,
and it words its refusals with its own ``refused``, the sentence ``scgsim.refusal`` words for every other
stage.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

# ---- Paths, relative to the run folder ----------------------------------------------------------------

CONFIG = "config.json"
MESH = "palace.msh"

GEOMETRY_DIR = "geometry"
SOURCE_GDS = "geometry/design.gds"
SOURCE_STACK = "geometry/design.stack.json"
MESH_DIR = "mesh"
LOGS_DIR = "logs"
METADATA_DIR = "metadata"

MESH_MANIFEST = "metadata/mesh_manifest.json"
INDEX_MAP = "metadata/palace_index_map.json"
MATERIAL_RESOLUTION = "metadata/palace_material_resolution.json"
PORT_INFORMATION = "metadata/port_information.json"
NUMERICAL_CONTROLS = "metadata/palace_numerical_controls.json"
CONVERGENCE_PLAN = "metadata/palace_convergence_plan.json"
"""The convergence plan (``scgsim.contracts.palace_convergence``), written only when convergence is declared."""

HANDOFF_METADATA = "metadata/palace_handoff_metadata.json"
RUN_METADATA = "metadata/palace_run_metadata.json"
RESOURCE_RECORD = "metadata/palace_resource_record.json"
ARCHIVE_MANIFEST = "metadata/palace_handoff_archive_manifest.json"
RECORDER = "metadata/record_palace_return.py"
CONVERGENCE_RUNNER = "metadata/run_palace_until_converged.py"
"""The convergence runner the run script starts the solver through; copied only when convergence is declared."""
CONVERGENCE_DECISION = "metadata/palace_convergence_decision.json"
"""The runner's record of the passes it read, its verdict and how the solver ended."""
RETURNED_RECEIPT = "metadata/palace_returned_run_receipt.json"
RUNTIME_COMPATIBILITY = "metadata/palace_runtime_compatibility.json"

DIRECT_LOCAL_SCRIPT = "run_palace.sh"
SLURM_SCRIPT = "run_palace.sbatch"

OUTPUT_DIR = "results/palace"
NATIVE_METADATA_NAME = "palace.json"
"""The record Palace writes beside its tables, in ``OUTPUT_DIR`` and in each ``iteration<N>`` folder."""
NATIVE_METADATA = f"{OUTPUT_DIR}/{NATIVE_METADATA_NAME}"
ITERATION_PREFIX = "iteration"
"""Palace names the output folder of adaptive pass ``N`` ``iteration<N>`` (``iteration_number`` reads it)."""

SEALED_INPUTS = (MESH, CONFIG, MESH_MANIFEST, INDEX_MAP)
"""The inputs the handoff hashes, in the order it records them; the receipt and ``resolve`` check each."""
SEALED_WHEN_PRESENT = (CONVERGENCE_PLAN,)
"""The inputs the handoff hashes after ``SEALED_INPUTS`` when the run folder has them; a reader refuses one
that is present but not hashed."""


def route_provenance(route: str) -> str:
    """The provenance record of the mesh built on Surface-EPR representation ``route`` (``"A"`` or ``"B"``)."""
    return f"metadata/route_{route.lower()}_provenance.json"


def solver_log(job: str) -> str:
    """The solver log the run script writes for Slurm job ``job`` (``manual`` outside Slurm).

    The run script passes a shell or Slurm pattern as ``job`` (``*``, ``%j``, ``${SLURM_JOB_ID:-manual}``).
    """
    return f"{LOGS_DIR}/palace-{job}.log"


def script_name(profile: str) -> str:
    """The run script of handoff ``profile``: a shell script for ``direct-local``, a batch script for Slurm."""
    return DIRECT_LOCAL_SCRIPT if profile == "direct-local" else SLURM_SCRIPT


def table_name(family: str) -> str:
    """The file name of the CSV table Palace writes for output ``family``."""
    return f"{family}.csv"


def output_table(family: str, directory: str = OUTPUT_DIR) -> str:
    """The CSV table Palace writes for output ``family`` in ``directory``."""
    return f"{directory}/{table_name(family)}"


def iteration_number(directory_name: str) -> int | None:
    """The pass number of an ``iteration<N>`` output folder name, None for any other name."""
    number = directory_name.removeprefix(ITERATION_PREFIX)
    if directory_name.startswith(ITERATION_PREFIX) and number.isdigit():
        return int(number)
    return None


# ---- Schemas ------------------------------------------------------------------------------------------

CONFIG_SCHEMA = "1-7-0"
"""The Palace config schema ``config.json`` is written for, recorded as ``palace_identity.config_schema``."""
MANIFEST_SCHEMA_VERSION = 2
INDEX_MAP_SCHEMA_VERSION = 3
RUN_RECORD_SCHEMA_VERSION = 1
"""The ``schema_version`` of the handoff metadata, run metadata, resource record and archive manifest."""
EPR_SELECTION_SCHEMA = "scgsim.palace.epr-selection.v1"
SURFACE_FILM_SCHEMA = "scgsim.palace.surface-film.v2"
RETURNED_RECEIPT_SCHEMA_VERSION = 2
RETURNED_RECEIPT_SCHEMA = (
    f"palace-returned-run-receipt.v{RETURNED_RECEIPT_SCHEMA_VERSION}"
)
"""The receipt the recorder writes; a receipt of an earlier version is refused, never read."""
RUNTIME_COMPATIBILITY_SCHEMA = "palace-runtime-compatibility.v1"


def collects_adapted_meshes(palace_identity: Any) -> bool:
    """Whether a run's outputs include the adapted meshes Palace saves, read from its handoff's identity.

    ``palace_identity`` is the ``palace_identity`` the handoff recorded; its ``config_schema`` names the
    config schema the run was written for.  The recorder lists adapted meshes and ``resolve`` requires
    them only for a run written for ``CONFIG_SCHEMA``, the schema whose ``SaveAdaptMesh`` they read, so
    raising the schema here moves both readers with the writer.
    """
    return (
        isinstance(palace_identity, dict)
        and palace_identity.get("config_schema") == CONFIG_SCHEMA
    )


# ---- Refusals -----------------------------------------------------------------------------------------

WHERE_RUN_FOLDER = (
    "the run folder: write it again with mesh(), write_config() and prepare_handoff(), "
    "or fetch the returned run again"
)
"""Where a refused file of a run folder is set right: every file in it is written by one of those steps."""


def refused(subject: str, problem: str, where: str) -> str:
    """The refusal of ``problem`` on ``subject``, which is changed in ``where``.

    The sentence of ``scgsim.refusal.refused``, written again here because a run folder has no SCGSim.
    """
    return f"{subject}: {problem}. Change it in {where}."


# ---- The records the recorder writes -----------------------------------------------------------------

ReceiptStatus = Literal["completed", "failed", "diagnostic_unverified"]
"""``completed`` when the sealed inputs match, every required output is present and the solver ended in one
of the two ways a run completes (``COMPLETED_ENDINGS``); ``diagnostic_unverified`` for a receipt written after
the fact, which ``resolve`` never accepts."""

RunEnding = Literal[
    "not_started",
    "solver_ended",
    "stopped_at_convergence",
    "stop_escalated",
    "runner_terminated",
    "not_recorded",
]
"""How the solver of a run ended, as the receipt states it.

``not_started``: the run stopped before the solve (the config's dry run).  ``solver_ended``: the solver ended
without a stop from SCGSim, by itself or ended by something else; its exit code says which, and only exit
code 0 completes the run.  ``stopped_at_convergence``: the convergence runner stopped it with SIGTERM after a
pass at which the declared criteria had converged, and it ended within the runner's wait; this completes the
run whatever exit code the stop gave.  ``stop_escalated``: it did not end on SIGTERM and the runner killed it.
``runner_terminated``: the runner itself was told to end (by the script, the scheduler or a person) and ended
the solver first.  ``not_recorded``: a declared run whose runner left no readable decision record.
"""
COMPLETED_ENDINGS: tuple[RunEnding, ...] = ("solver_ended", "stopped_at_convergence")
"""The endings a completed receipt can state."""


class FileRecord(TypedDict):
    """One run-folder file in a record: its path, and its size and SHA-256 when it is present."""

    path: str
    bytes: int | None
    sha256: str | None
    present: bool


class JobIdentity(TypedDict):
    """The Slurm job the run ran as; the run script passes ``manual`` outside Slurm."""

    job_id: str | None
    job_name: str | None


class _ReceiptFields(TypedDict):
    """The fields every returned-run receipt has, in the order the receipt writes them."""

    schema: str
    schema_version: int
    handoff_id: str
    route: str
    problem: str
    status: ReceiptStatus
    exit_code: int
    execution_stage: str
    solver_invoked: bool
    preflight_exit_code: int
    solver_exit_code: int | None
    tee_exit_code: int | None
    ending: RunEnding
    identity_verified: bool
    timestamp_utc: str
    job_identity: JobIdentity
    input_hashes: list[dict[str, Any]]
    output_files: list[FileRecord]
    log: FileRecord
    convergence_decision: FileRecord | None
    solver_identity: dict[str, Any]


class ReturnedReceipt(_ReceiptFields, total=False):
    """The returned-run receipt; the two fields at its end are written only when the handoff has them."""

    runtime_compatibility: dict[str, Any]
    route_a_thin_film: Any


class ObservedIdentity(TypedDict):
    """The Palace the dry run met, as ``verify-config`` records it: observed, never required."""

    native_git_tag: str | None
    supported_upstream_release: str | None
    native_schema_version: str | None
    version_probe_exit_code: int | None
    runtime_binary_identity: str | None
    runtime_binary_sha256: str | None


# ---- Problems -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PalaceProblem:
    """What differs between the Palace problems SCGSim writes: one record per problem, read by every stage.

    ``name`` is the problem's ``Problem.Type`` and the key of its ``Solver`` block in ``config.json``.
    ``output_families`` are the tables Palace can write for it, in the order the recorder lists them;
    ``port_families`` of them are written only when the config has lumped ports, and ``surface-Q`` only when
    it has Surface-EPR rows.  ``excitation`` is the ``Boundaries`` block whose rows drive the problem and
    ``excitation_section`` its index-map section.  ``series`` says what one row of a table is: a ``mode``
    (rows ``m`` = 1..N of the requested modes) or an ``excitation`` (rows ``i``, one per terminal).
    ``primary_table`` holds the problem's physics, ``snapshot_tables`` the files that make one result
    snapshot complete, and ``fem_space`` the finite-element space of its unknowns.
    """

    name: str
    output_families: tuple[str, ...]
    port_families: tuple[str, ...]
    excitation: str
    excitation_section: str
    series: Literal["mode", "excitation"]
    series_index: Literal["m", "i"]
    primary_table: str
    snapshot_tables: tuple[str, ...]
    fem_space: Literal["ND", "H1"]

    @property
    def solves_modes(self) -> bool:
        """Whether each table row is a mode (Eigenmode) rather than an excitation (Electrostatic)."""
        return self.series == "mode"

    def written_families(
        self, *, lumped_ports: bool, surfaces: bool
    ) -> tuple[str, ...]:
        """The output families Palace writes for a config with or without lumped ports and Surface-EPR rows."""
        return tuple(
            family
            for family in self.output_families
            if (lumped_ports or family not in self.port_families)
            and (surfaces or family != "surface-Q")
        )


ELECTROSTATIC = PalaceProblem(
    name="Electrostatic",
    output_families=(
        "terminal-C",
        "terminal-Cm",
        "terminal-Cinv",
        "terminal-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
    port_families=(),
    excitation="Terminal",
    excitation_section="Boundaries.Terminal",
    series="excitation",
    series_index="i",
    primary_table="terminal-C",
    snapshot_tables=("terminal-C", "surface-Q", "error-indicators"),
    fem_space="H1",
)
EIGENMODE = PalaceProblem(
    name="Eigenmode",
    output_families=(
        "eig",
        "port-EPR",
        "port-I",
        "port-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
    port_families=("port-EPR", "port-I", "port-V"),
    excitation="LumpedPort",
    excitation_section="Boundaries.LumpedPort",
    series="mode",
    series_index="m",
    primary_table="eig",
    snapshot_tables=("eig", "port-EPR", "surface-Q", "error-indicators"),
    fem_space="ND",
)
PROBLEMS = {problem.name: problem for problem in (ELECTROSTATIC, EIGENMODE)}


def palace_problem(name: Any) -> PalaceProblem:
    """The problem record of ``name``; a name SCGSim does not write is refused with the names it does."""
    problem = PROBLEMS.get(name) if isinstance(name, str) else None
    if problem is None:
        raise ValueError(
            refused(
                f"Palace problem {name!r}",
                f"is not one SCGSim writes; expected one of {tuple(PROBLEMS)}",
                "the problem passed to prepare_handoff, or the run folder's handoff metadata",
            )
        )
    return problem


# ---- The handoff id -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoffIdentity:
    """The identities a handoff id is the hash of; the handoff writes the id and ``resolve`` checks it.

    ``hashes`` are the sealed inputs' path, size and SHA-256 records; ``route_a_thin_film`` is None on
    Route B and then left out of the hashed record.
    """

    profile: str
    route: str
    problem: str
    source_revisions: Any
    palace_identity: Any
    hashes: list[Any]
    execution_identity: Any
    route_a_thin_film: Any = None

    def digest(self) -> str:
        """The handoff id: the canonical SHA-256 of these identities."""
        return canonical_sha256(
            {
                "route": self.route,
                "problem": self.problem,
                "profile": self.profile,
                "source_revisions": self.source_revisions,
                "palace_identity": self.palace_identity,
                "hashes": self.hashes,
                "execution_identity": self.execution_identity,
                **(
                    {"route_a_thin_film": self.route_a_thin_film}
                    if self.route_a_thin_film is not None
                    else {}
                ),
            }
        )


def canonical_sha256(payload: Any) -> str:
    """The SHA-256 of ``payload`` as compact JSON with sorted keys."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- Reading and writing run-folder files ------------------------------------------------------------


def sha256(path: Path) -> str:
    """The SHA-256 of the bytes of the file at ``path``, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_for_path(root: Path, relative_path: str) -> FileRecord:
    """The record of run-folder file ``relative_path``: its size and SHA-256, or that it is not present."""
    path = confined_path(root, relative_path)
    if not path.is_file():
        return {"path": relative_path, "bytes": None, "sha256": None, "present": False}
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "present": True,
    }


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in the file at ``path``; a file holding any other JSON value is refused."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(
            refused(str(path), "must contain a JSON object", WHERE_RUN_FOLDER)
        )
    return parsed


def json_text(payload: Any) -> str:
    """The text of a run-folder JSON record: two-space indentation and a final newline."""
    return json.dumps(payload, indent=2) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as a run-folder JSON record at ``path``, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a run-folder JSON record through a temporary file beside it, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(json_text(payload))
        temporary = Path(handle.name)
    os.replace(temporary, path)


def utc_timestamp() -> str:
    """The time now in UTC as the run folder's records write it, ``2026-11-14T08:00:00.123456Z``."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def confined_path(root: Path, relative: str) -> Path:
    """Resolve ``relative`` beneath run folder ``root``; an absolute path or one that escapes it is refused."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(
            refused(
                f"run-folder path {relative!r}",
                f"must be a non-empty path relative to the run folder {root}",
                WHERE_RUN_FOLDER,
            )
        )
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(
            refused(
                f"run-folder path {relative!r}",
                f"escapes the run folder {root}",
                WHERE_RUN_FOLDER,
            )
        )
    return candidate
