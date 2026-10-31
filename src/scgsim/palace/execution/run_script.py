"""Render the bash script that runs one manual Palace handoff.

The script is a list of bash lines. Its run order is: check the inputs, set up the
environment, ask Palace to ``--dry-run`` the config (record a receipt and stop if it
refuses), solve, record a receipt. Both receipts go through the recorder that
``handoff.py`` copies into the run directory. ``direct-local`` runs in the current
shell and tees Palace output into the log; Slurm profiles bind the job's output
to the log with ``#SBATCH`` directives instead.

The rendered bytes are hashed into the handoff ID, so any change to a line here is a
change of identity.

Not here: writing the script to disk and sealing the run (``handoff.py``).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from ._checks import positive_int, single_line

# Resource keys the script handles itself instead of turning into #SBATCH lines.
_NOT_SBATCH_DIRECTIVES = {
    "launcher",
    "command_style",
    "threads",
    "processes",
    "output",
    "error",
    "open_mode",
}


def render_script(
    profile: str,
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    *,
    returned_receipt_recorder_name: str,
    palace_identity: Mapping[str, Any],
) -> str:
    """Return the run script text, without the trailing newline.

    ``resources`` may leave ``command_style`` out, which means ``binary``: Palace is
    called directly. ``wrapper`` adds ``-np`` and ``-nt`` and is local only.
    """
    direct = profile == "direct-local"
    style = str(resources.get("command_style", "binary"))
    threads = positive_int(
        resources.get("threads", resources.get("cpus_per_task", 1)), "threads"
    )
    processes = positive_int(
        resources.get("processes", resources.get("ntasks", 1)), "processes"
    )
    directives = [] if direct else _sbatch_directives(resources)
    launcher = [] if direct else _launcher_tokens(resources.get("launcher", ("srun",)))
    if not direct and not launcher:
        raise ValueError("Slurm profile launcher must be non-empty.")

    arguments = ['"$PALACE_EXECUTABLE"']
    if style == "wrapper":
        arguments.extend(["-np", '"$PALACE_PROCESSES"', "-nt", '"$PALACE_THREADS"'])
    arguments.append('"$PALACE_CONFIG"')
    command = " ".join([*(shlex.quote(item) for item in launcher), *arguments])
    config_schema = single_line(
        str(palace_identity["config_schema"]), "palace_identity.config_schema"
    )

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "",
            "set -u",
            *_run_directory_lines(direct),
            *_input_check_lines(),
            *_variable_lines(executable, processes, threads, style, direct),
            *_petsc_lines(petsc),
            *_environment_setup_lines(setup),
            *_preflight_lines(direct, returned_receipt_recorder_name, config_schema),
            *_solver_lines(direct, command, returned_receipt_recorder_name),
        ]
    )


def _sbatch_directives(resources: Mapping[str, Any]) -> list[str]:
    directives: list[str] = []
    for key, value in resources.items():
        if key in _NOT_SBATCH_DIRECTIVES:
            continue
        single_line(str(value), f"resource {key}")
        directives.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    # Slurm defaults to slurm-%j.out in the submit directory. Palace stdout
    # already belongs in logs/palace-%j.log, so bind both streams there and
    # do not tee (tee would duplicate every line into the same file).
    directives.extend(
        [
            "#SBATCH --output=logs/palace-%j.log",
            "#SBATCH --error=logs/palace-%j.log",
            "#SBATCH --open-mode=append",
        ]
    )
    return directives


def _run_directory_lines(direct: bool) -> list[str]:
    """Find the run directory: next to the script locally, the submit dir on Slurm."""
    if direct:
        return [
            'SCRIPT_SOURCE="${BASH_SOURCE[0]}"',
            'RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd -P)" || exit 2',
        ]
    return [
        '[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || { echo "missing SLURM_SUBMIT_DIR" >&2; exit 2; }',
        'RUN_DIR="$(CDPATH= cd -- "$SLURM_SUBMIT_DIR" && pwd -P)" || exit 2',
    ]


def _input_check_lines() -> list[str]:
    return [
        '[[ -f "$RUN_DIR/config.json" ]] || { echo "missing $RUN_DIR/config.json" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/palace.msh" ]] || { echo "missing $RUN_DIR/palace.msh" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_handoff_metadata.json" ]] || { echo "missing canonical handoff metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_run_metadata.json" ]] || { echo "missing canonical run metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/mesh_manifest.json" ]] || { echo "missing canonical mesh manifest" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_index_map.json" ]] || { echo "missing canonical index map" >&2; exit 2; }',
    ]


def _variable_lines(
    executable: str, processes: int, threads: int, style: str, direct: bool
) -> list[str]:
    """Enter the run directory, clear old outputs, and define the PALACE_* variables."""
    return [
        'cd "$RUN_DIR" || exit 2',
        "mkdir -p logs results/palace metadata",
        "rm -f metadata/palace_returned_run_receipt.json",
        "rm -f metadata/palace_runtime_compatibility.json",
        *(["rm -f logs/palace-*.log"] if direct else []),
        "rm -rf results/palace",
        "mkdir -p results/palace",
        'PALACE_CONFIG="config.json"',
        'PALACE_MESH="palace.msh"',
        f"PALACE_EXECUTABLE={shlex.quote(executable)}",
        f"PALACE_PROCESSES={processes}",
        f"PALACE_THREADS={threads}",
        f"PALACE_COMMAND_STYLE={shlex.quote(style)}",
        'export OMP_NUM_THREADS="$PALACE_THREADS"',
        'PALACE_LOG="logs/palace-${SLURM_JOB_ID:-manual}.log"',
        'PALACE_RUNTIME_COMPATIBILITY="metadata/palace_runtime_compatibility.json"',
        '[[ -f "$PALACE_CONFIG" ]] || { echo "missing $PALACE_CONFIG" >&2; exit 2; }',
        '[[ -f "$PALACE_MESH" ]] || { echo "missing $PALACE_MESH" >&2; exit 2; }',
    ]


def _petsc_lines(petsc: Sequence[str]) -> list[str]:
    if not petsc:
        return []
    return [
        f"PALACE_PETSC_OPTIONS={shlex.quote(' '.join(petsc))}",
        'export PETSC_OPTIONS="${PETSC_OPTIONS:-} ${PALACE_PETSC_OPTIONS}"',
    ]


def _environment_setup_lines(setup: Sequence[str]) -> list[str]:
    """Run the caller's setup commands with ``set +u`` (module scripts use unset names)."""
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


def _receipt_command(recorder_name: str) -> str:
    """The recorder call both receipts share; each adds its own stage arguments."""
    return (
        f'python3 metadata/{recorder_name} "$(pwd)" '
        '"$PIPELINE_EXIT_CODE" --log-path "$PALACE_LOG" '
        '--tee-exit-code "$TEE_EXIT_CODE" '
        '--job-id "${SLURM_JOB_ID:-manual}" '
        '--job-name "${SLURM_JOB_NAME:-manual}" '
    )


def _preflight_lines(direct: bool, recorder_name: str, config_schema: str) -> list[str]:
    """Ask Palace to --dry-run the config. If it refuses, record a receipt and exit."""
    probe = (
        f"python3 metadata/{recorder_name} verify-config "
        '"$PALACE_EXECUTABLE" "$PALACE_COMMAND_STYLE" '
        '"$PALACE_RUNTIME_COMPATIBILITY" "$PALACE_CONFIG" '
        f"--config-schema {shlex.quote(config_schema)}"
    )
    preflight_receipt = (
        _receipt_command(recorder_name) + "--execution-stage config_compatibility "
        '--preflight-exit-code "$COMPATIBILITY_VERIFY_EXIT" '
        " || RECEIPT_EXIT=$?"
    )
    if direct:
        run_probe = [
            "set -o pipefail",
            f'{probe} 2>&1 | tee -a "$PALACE_LOG"',
            'COMPATIBILITY_PIPE_STATUS=("${PIPESTATUS[@]}")',
            "COMPATIBILITY_VERIFY_EXIT=${COMPATIBILITY_PIPE_STATUS[0]:-1}",
            "COMPATIBILITY_TEE_EXIT=${COMPATIBILITY_PIPE_STATUS[1]:-1}",
        ]
    else:
        run_probe = [
            probe,
            "COMPATIBILITY_VERIFY_EXIT=$?",
            "COMPATIBILITY_TEE_EXIT=0",
        ]
    return [
        *run_probe,
        'if [ "$COMPATIBILITY_VERIFY_EXIT" -ne 0 ] || [ "$COMPATIBILITY_TEE_EXIT" -ne 0 ]; then',
        "  TEE_EXIT_CODE=$COMPATIBILITY_TEE_EXIT",
        "  PIPELINE_EXIT_CODE=$COMPATIBILITY_VERIFY_EXIT",
        '  if [ "$PIPELINE_EXIT_CODE" -eq 0 ]; then PIPELINE_EXIT_CODE=$TEE_EXIT_CODE; fi',
        f"  {preflight_receipt}",
        "  RECEIPT_EXIT=${RECEIPT_EXIT:-0}",
        '  if [ "$PIPELINE_EXIT_CODE" -ne 0 ]; then exit "$PIPELINE_EXIT_CODE"; fi',
        '  exit "$RECEIPT_EXIT"',
        "fi",
    ]


def _solver_lines(direct: bool, command: str, recorder_name: str) -> list[str]:
    """Run Palace, record the solver receipt, and exit with the first failure."""
    solver_receipt = (
        _receipt_command(recorder_name) + "--execution-stage solver "
        "--preflight-exit-code 0 "
        '--solver-exit-code "$PALACE_EXIT_CODE" '
        " || RECEIPT_EXIT=$?"
    )
    if direct:
        run_palace = [
            "set -o pipefail",
            f'{command} 2>&1 | tee -a "$PALACE_LOG"',
            'PALACE_PIPE_STATUS=("${PIPESTATUS[@]}")',
            "PALACE_EXIT_CODE=${PALACE_PIPE_STATUS[0]:-1}",
            "TEE_EXIT_CODE=${PALACE_PIPE_STATUS[1]:-1}",
            "PIPELINE_EXIT_CODE=$PALACE_EXIT_CODE",
            'if [ "$PIPELINE_EXIT_CODE" -eq 0 ] && [ "$TEE_EXIT_CODE" -ne 0 ]; then PIPELINE_EXIT_CODE=$TEE_EXIT_CODE; fi',
        ]
    else:
        run_palace = [
            command,
            "PALACE_EXIT_CODE=$?",
            "TEE_EXIT_CODE=0",
            "PIPELINE_EXIT_CODE=$PALACE_EXIT_CODE",
        ]
    return [
        *run_palace,
        solver_receipt,
        "RECEIPT_EXIT=${RECEIPT_EXIT:-0}",
        'if [ "$PIPELINE_EXIT_CODE" -ne 0 ]; then exit "$PIPELINE_EXIT_CODE"; fi',
        'if [ "$RECEIPT_EXIT" -ne 0 ]; then exit "$RECEIPT_EXIT"; fi',
        'exit "$PIPELINE_EXIT_CODE"',
    ]


def _launcher_tokens(value: Any) -> list[str]:
    """Treat a launcher string as one executable token, never as characters."""
    if isinstance(value, str):
        return [single_line(value, "launcher")]
    if not isinstance(value, Sequence):
        raise TypeError(
            "resources.launcher must be a single string or a sequence of strings."
        )
    tokens = [single_line(item, "launcher token") for item in value]
    if not tokens:
        raise ValueError("resources.launcher must be non-empty.")
    return tokens
