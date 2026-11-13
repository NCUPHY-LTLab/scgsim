"""Render the run script of a handoff, direct-local or Slurm, from checked settings."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    HANDOFF_METADATA,
    INDEX_MAP,
    LOGS_DIR,
    MESH,
    MESH_MANIFEST,
    METADATA_DIR,
    OUTPUT_DIR,
    RECORDER,
    RETURNED_RECEIPT,
    RUN_METADATA,
    RUNTIME_COMPATIBILITY,
    solver_log,
)
from scgsim.refusal import Declared, found_text

_WHERE_HANDOFF_CALL = "the call to prepare_handoff"


@dataclass(frozen=True)
class ExecutionSettings:
    """How the run script starts Palace: the caller's profile, executable, resources and commands, checked.

    ``prepare_handoff`` builds it after checking each value; the script and the handoff's execution identity
    are rendered from it.
    """

    profile: str
    executable: str
    resources: Mapping[str, Any]
    setup_commands: tuple[str, ...]
    petsc_options: tuple[str, ...]

    @property
    def command_style(self) -> str:
        """``binary`` (Palace's own executable) or ``wrapper`` (the ``palace`` launcher script)."""
        return str(self.resources.get("command_style", "binary"))

    def threads(self) -> int:
        """The OpenMP threads per process: ``threads``, else Slurm's ``cpus_per_task``, else 1."""
        resources = self.resources
        return positive_int(
            resources.get("threads", resources.get("cpus_per_task", 1)), "threads"
        )

    def processes(self) -> int:
        """The MPI processes: ``processes``, else Slurm's ``ntasks``, else 1."""
        resources = self.resources
        return positive_int(
            resources.get("processes", resources.get("ntasks", 1)), "processes"
        )


def render_script(execution: ExecutionSettings, *, config_schema: str) -> str:
    """Render a dependency-free script for a manual Palace handoff whose config has ``config_schema``."""
    # The thread and process counts are refused before any line is rendered.
    execution.threads()
    execution.processes()
    directives = _slurm_directives(execution)
    command = _solver_command(execution)
    config_schema = single_line(str(config_schema), "palace_identity.config_schema")
    lines = _script_start_lines(
        execution, directives, _run_folder_lines(execution.profile)
    )
    lines.extend(_petsc_lines(execution.petsc_options))
    lines.extend(_setup_lines(execution.setup_commands))
    lines.extend(
        _compatibility_lines(
            profile=execution.profile,
            probe=_compatibility_probe(config_schema),
            receipt_command=_receipt_command(
                "--execution-stage config_compatibility "
                '--preflight-exit-code "$COMPATIBILITY_VERIFY_EXIT" '
            ),
        )
    )
    lines.extend(
        _solver_lines(
            profile=execution.profile,
            command=command,
            receipt_command=_receipt_command(
                "--execution-stage solver --preflight-exit-code 0 "
                '--solver-exit-code "$PALACE_EXIT_CODE" '
            ),
        )
    )
    return "\n".join(lines)


_NOT_SBATCH_DIRECTIVES = {
    "launcher",
    "command_style",
    "threads",
    "processes",
    "output",
    "error",
    "open_mode",
}
"""Resource keys the script uses itself, or binds to the solver log, rather than passing to ``sbatch``."""


def _slurm_directives(execution: ExecutionSettings) -> list[str]:
    """The ``#SBATCH`` lines of a Slurm profile, none for ``direct-local``."""
    if execution.profile == "direct-local":
        return []
    directives: list[str] = []
    for key, value in execution.resources.items():
        if key in _NOT_SBATCH_DIRECTIVES:
            continue
        single_line(str(value), f"resource {key}")
        directives.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    # Slurm defaults to slurm-%j.out in the submit directory. Palace stdout
    # already belongs in logs/palace-%j.log, so bind both streams there and
    # do not tee (tee would duplicate every line into the same file).
    directives.extend(
        [
            f"#SBATCH --output={solver_log('%j')}",
            f"#SBATCH --error={solver_log('%j')}",
            "#SBATCH --open-mode=append",
        ]
    )
    return directives


def _solver_command(execution: ExecutionSettings) -> str:
    """The command line that starts Palace: the Slurm launcher, then the executable and the config."""
    launcher = (
        []
        if execution.profile == "direct-local"
        else launcher_tokens(execution.resources.get("launcher", ("srun",)))
    )
    if execution.profile != "direct-local" and not launcher:
        raise ValueError(
            Declared("Slurm profile launcher", _WHERE_HANDOFF_CALL).refusal(
                f"must be non-empty: profile {execution.profile!r} got resources.launcher={execution.resources.get('launcher')!r}; pass a launcher such as ('srun',) or leave it out"
            )
        )
    arguments = ['"$PALACE_EXECUTABLE"']
    if execution.command_style == "wrapper":
        arguments.extend(["-np", '"$PALACE_PROCESSES"', "-nt", '"$PALACE_THREADS"'])
    arguments.append('"$PALACE_CONFIG"')
    return " ".join([*(shlex.quote(item) for item in launcher), *arguments])


def _receipt_command(stage_arguments: str) -> str:
    """The recorder call that writes the returned receipt after one execution stage, keeping its exit code."""
    return (
        f'python3 {RECORDER} "$(pwd)" '
        '"$PIPELINE_EXIT_CODE" --log-path "$PALACE_LOG" '
        '--tee-exit-code "$TEE_EXIT_CODE" '
        '--job-id "${SLURM_JOB_ID:-manual}" '
        '--job-name "${SLURM_JOB_NAME:-manual}" '
        + stage_arguments
        + " || RECEIPT_EXIT=$?"
    )


def _run_folder_lines(profile: str) -> list[str]:
    """Set ``RUN_DIR`` to the run folder: the script's own folder locally, the submit folder under Slurm."""
    if profile == "direct-local":
        return [
            'SCRIPT_SOURCE="${BASH_SOURCE[0]}"',
            'RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd -P)" || exit 2',
        ]
    return [
        '[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || { echo "missing SLURM_SUBMIT_DIR" >&2; exit 2; }',
        'RUN_DIR="$(CDPATH= cd -- "$SLURM_SUBMIT_DIR" && pwd -P)" || exit 2',
    ]


def _petsc_lines(petsc_options: Sequence[str]) -> list[str]:
    """Append the caller's PETSc options to any the environment already sets."""
    if not petsc_options:
        return []
    petsc = " ".join(petsc_options)
    return [
        f"PALACE_PETSC_OPTIONS={shlex.quote(petsc)}",
        'export PETSC_OPTIONS="${PETSC_OPTIONS:-} ${PALACE_PETSC_OPTIONS}"',
    ]


def _compatibility_probe(config_schema: str) -> str:
    """The recorder call that dry-runs Palace on the sealed config before the solve."""
    return (
        f"python3 {RECORDER} verify-config "
        '"$PALACE_EXECUTABLE" "$PALACE_COMMAND_STYLE" '
        '"$PALACE_RUNTIME_COMPATIBILITY" "$PALACE_CONFIG" '
        f"--config-schema {shlex.quote(config_schema)}"
    )


def _script_start_lines(
    execution: ExecutionSettings,
    directives: Sequence[str],
    run_directory_lines: Sequence[str],
) -> list[str]:
    """Set run-directory, cleanup, and solver variables in the generated script."""
    profile = execution.profile
    return [
        "#!/usr/bin/env bash",
        *directives,
        "",
        "set -u",
        *run_directory_lines,
        *(
            _require_file_line(f"$RUN_DIR/{name}", f"$RUN_DIR/{name}")
            for name in (CONFIG, MESH)
        ),
        _require_file_line(
            f"$RUN_DIR/{HANDOFF_METADATA}", "canonical handoff metadata"
        ),
        _require_file_line(f"$RUN_DIR/{RUN_METADATA}", "canonical run metadata"),
        _require_file_line(f"$RUN_DIR/{MESH_MANIFEST}", "canonical mesh manifest"),
        _require_file_line(f"$RUN_DIR/{INDEX_MAP}", "canonical index map"),
        'cd "$RUN_DIR" || exit 2',
        f"mkdir -p {LOGS_DIR} {OUTPUT_DIR} {METADATA_DIR}",
        f"rm -f {RETURNED_RECEIPT}",
        f"rm -f {RUNTIME_COMPATIBILITY}",
        *([f"rm -f {solver_log('*')}"] if profile == "direct-local" else []),
        f"rm -rf {OUTPUT_DIR}",
        f"mkdir -p {OUTPUT_DIR}",
        f'PALACE_CONFIG="{CONFIG}"',
        f'PALACE_MESH="{MESH}"',
        f"PALACE_EXECUTABLE={shlex.quote(execution.executable)}",
        f"PALACE_PROCESSES={execution.processes()}",
        f"PALACE_THREADS={execution.threads()}",
        f"PALACE_COMMAND_STYLE={shlex.quote(execution.command_style)}",
        'export OMP_NUM_THREADS="$PALACE_THREADS"',
        f'PALACE_LOG="{solver_log("${SLURM_JOB_ID:-manual}")}"',
        f'PALACE_RUNTIME_COMPATIBILITY="{RUNTIME_COMPATIBILITY}"',
        '[[ -f "$PALACE_CONFIG" ]] || { echo "missing $PALACE_CONFIG" >&2; exit 2; }',
        '[[ -f "$PALACE_MESH" ]] || { echo "missing $PALACE_MESH" >&2; exit 2; }',
    ]


def _require_file_line(path: str, missing: str) -> str:
    """The script line that exits with status 2 unless ``path`` is a file, printing ``missing <missing>``."""
    return f'[[ -f "{path}" ]] || {{ echo "missing {missing}" >&2; exit 2; }}'


def _setup_lines(setup: Sequence[str]) -> list[str]:
    """Render environment initialization and caller-provided setup commands."""
    lines: list[str] = []
    if any(command.lstrip().startswith("module ") for command in setup):
        lines.extend(
            [
                "if ! type module >/dev/null 2>&1; then",
                '  [[ -r /etc/profile.d/modules.sh ]] || { echo "missing Environment Modules initialization" >&2; exit 2; }',
                "  set +u",
                "  source /etc/profile.d/modules.sh || { echo 'Environment Modules initialization failed' >&2; exit 2; }",
                "  set -u",
                "fi",
            ]
        )
    for setup_command in setup:
        lines.extend(
            [
                "set +u",
                f"{{ {setup_command}; }} || {{ echo 'setup command failed' >&2; exit 2; }}",
                "set -u",
            ]
        )
    return lines


def _compatibility_lines(
    *, profile: str, probe: str, receipt_command: str
) -> list[str]:
    """Require the native dry-run before the generated solver command."""
    lines: list[str] = []
    if profile == "direct-local":
        lines.extend(
            [
                "set -o pipefail",
                f'{probe} 2>&1 | tee -a "$PALACE_LOG"',
                'COMPATIBILITY_PIPE_STATUS=("${PIPESTATUS[@]}")',
                "COMPATIBILITY_VERIFY_EXIT=${COMPATIBILITY_PIPE_STATUS[0]:-1}",
                "COMPATIBILITY_TEE_EXIT=${COMPATIBILITY_PIPE_STATUS[1]:-1}",
            ]
        )
    else:
        lines.extend(
            [
                probe,
                "COMPATIBILITY_VERIFY_EXIT=$?",
                "COMPATIBILITY_TEE_EXIT=0",
            ]
        )
    lines.extend(
        [
            'if [ "$COMPATIBILITY_VERIFY_EXIT" -ne 0 ] || [ "$COMPATIBILITY_TEE_EXIT" -ne 0 ]; then',
            "  TEE_EXIT_CODE=$COMPATIBILITY_TEE_EXIT",
            "  PIPELINE_EXIT_CODE=$COMPATIBILITY_VERIFY_EXIT",
            '  if [ "$PIPELINE_EXIT_CODE" -eq 0 ]; then PIPELINE_EXIT_CODE=$TEE_EXIT_CODE; fi',
            f"  {receipt_command}",
            "  RECEIPT_EXIT=${RECEIPT_EXIT:-0}",
            '  if [ "$PIPELINE_EXIT_CODE" -ne 0 ]; then exit "$PIPELINE_EXIT_CODE"; fi',
            '  exit "$RECEIPT_EXIT"',
            "fi",
        ]
    )
    return lines


def _solver_lines(*, profile: str, command: str, receipt_command: str) -> list[str]:
    """Run Palace, preserve its status, and record the completed attempt."""
    execution_lines = (
        [
            "set -o pipefail",
            f'{command} 2>&1 | tee -a "$PALACE_LOG"',
            'PALACE_PIPE_STATUS=("${PIPESTATUS[@]}")',
            "PALACE_EXIT_CODE=${PALACE_PIPE_STATUS[0]:-1}",
            "TEE_EXIT_CODE=${PALACE_PIPE_STATUS[1]:-1}",
            "PIPELINE_EXIT_CODE=$PALACE_EXIT_CODE",
            'if [ "$PIPELINE_EXIT_CODE" -eq 0 ] && [ "$TEE_EXIT_CODE" -ne 0 ]; then PIPELINE_EXIT_CODE=$TEE_EXIT_CODE; fi',
        ]
        if profile == "direct-local"
        else [
            command,
            "PALACE_EXIT_CODE=$?",
            "TEE_EXIT_CODE=0",
            "PIPELINE_EXIT_CODE=$PALACE_EXIT_CODE",
        ]
    )
    lines: list[str] = []
    lines.extend(
        [
            *execution_lines,
            receipt_command,
            "RECEIPT_EXIT=${RECEIPT_EXIT:-0}",
            'if [ "$PIPELINE_EXIT_CODE" -ne 0 ]; then exit "$PIPELINE_EXIT_CODE"; fi',
            'if [ "$RECEIPT_EXIT" -ne 0 ]; then exit "$RECEIPT_EXIT"; fi',
            'exit "$PIPELINE_EXIT_CODE"',
        ]
    )
    return lines


def launcher_tokens(value: Any) -> list[str]:
    """Treat a launcher string as one executable token, never as characters."""
    if isinstance(value, str):
        return [single_line(value, "launcher")]
    if not isinstance(value, Sequence):
        raise TypeError(
            Declared(
                "resources.launcher must be a single string or a sequence of strings",
                _WHERE_HANDOFF_CALL,
            ).refusal(found_text(type(value).__name__))
        )
    tokens = [single_line(item, "launcher token") for item in value]
    if not tokens:
        raise ValueError(
            Declared(
                "resources.launcher must be non-empty", _WHERE_HANDOFF_CALL
            ).refusal(found_text(list(value)))
        )
    return tokens


def positive_int(value: Any, field: str) -> int:
    """Require a positive integer value for a named resource field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            Declared(f"{field}", _WHERE_HANDOFF_CALL).refusal(
                "must be a positive integer"
            )
        )
    return value


def single_line(value: str, field: str) -> str:
    """Require non-empty single-line text for a shell or resource field."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(
            Declared(f"{field}", _WHERE_HANDOFF_CALL).refusal(
                "must be non-empty single-line text"
            )
        )
    return value
