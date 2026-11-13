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
    run_palace.sh | run_palace.sbatch   script_name(profile) (run)
    metadata/palace_handoff_*.json ...  HANDOFF_METADATA, RUN_METADATA, RESOURCE_RECORD, ARCHIVE_MANIFEST (run)
    metadata/record_palace_return.py    RECORDER: the receipt recorder (run)
    logs/palace-<job>.log               the solver log the run script writes
    results/palace/                     OUTPUT_DIR: what Palace writes, see ``PalaceProblem``
    metadata/palace_returned_run_receipt.json  RETURNED_RECEIPT, written by the recorder after the solve

The receipt recorder (``scgsim.palace.run.returned_receipt``) runs in the run folder with only a Python
interpreter.  When ``prepare_handoff`` copies it there, the text of this module takes the place of the
recorder's import of it (``scgsim.palace.run.handoff``), so the copy carries the same layout, problems and
helpers, and nothing in the run folder imports SCGSim.  For that reason this module imports only the
standard library (``.importlinter``: ``contracts-import-nothing-else``) and has no ``__future__`` import.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

HANDOFF_METADATA = "metadata/palace_handoff_metadata.json"
RUN_METADATA = "metadata/palace_run_metadata.json"
RESOURCE_RECORD = "metadata/palace_resource_record.json"
ARCHIVE_MANIFEST = "metadata/palace_handoff_archive_manifest.json"
RECORDER = "metadata/record_palace_return.py"
RETURNED_RECEIPT = "metadata/palace_returned_run_receipt.json"
RUNTIME_COMPATIBILITY = "metadata/palace_runtime_compatibility.json"

DIRECT_LOCAL_SCRIPT = "run_palace.sh"
SLURM_SCRIPT = "run_palace.sbatch"
SOLVER_LOG_PREFIX = "logs/palace-"
"""The run script writes the solver log to ``logs/palace-<job id>.log`` (``manual`` outside Slurm)."""

OUTPUT_DIR = "results/palace"
NATIVE_METADATA_NAME = "palace.json"
"""The record Palace writes beside its tables, in ``OUTPUT_DIR`` and in each ``iteration<N>`` folder."""
NATIVE_METADATA = f"{OUTPUT_DIR}/{NATIVE_METADATA_NAME}"
ITERATION_PREFIX = "iteration"

SEALED_INPUTS = (MESH, CONFIG, MESH_MANIFEST, INDEX_MAP)
"""The inputs the handoff hashes, in the order it records them; the receipt and ``resolve`` check each."""


def route_provenance(route: str) -> str:
    """The provenance record of the mesh built on Surface-EPR representation ``route`` (``"A"`` or ``"B"``)."""
    return f"metadata/route_{route.lower()}_provenance.json"


def script_name(profile: str) -> str:
    """The run script of handoff ``profile``: a shell script for ``direct-local``, a batch script for Slurm."""
    return DIRECT_LOCAL_SCRIPT if profile == "direct-local" else SLURM_SCRIPT


def table_name(family: str) -> str:
    """The file name of the CSV table Palace writes for output ``family``."""
    return f"{family}.csv"


def output_table(family: str, directory: str = OUTPUT_DIR) -> str:
    """The CSV table Palace writes for output ``family`` in ``directory``."""
    return f"{directory}/{table_name(family)}"


def iteration_dir(directory_name: str) -> int | None:
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
RETURNED_RECEIPT_SCHEMA = "palace-returned-run-receipt.v1"
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
            f"unsupported problem {name!r}; expected one of {tuple(PROBLEMS)}."
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


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in the file at ``path``; a file holding any other JSON value is refused."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return parsed


def json_text(payload: Any) -> str:
    """The text of a run-folder JSON record: two-space indentation and a final newline."""
    return json.dumps(payload, indent=2) + "\n"


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as a run-folder JSON record at ``path``, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


def confined_path(root: Path, relative: str) -> Path:
    """Resolve ``relative`` beneath run folder ``root``; an absolute path or one that escapes it is refused."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact paths must be non-empty relative paths.")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes the run directory.")
    return candidate
