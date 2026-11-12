"""Render portable Palace execution scripts and validate their shell inputs."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Any


def render_script(  # noqa: PLR0913 -- separate handoff settings feed script text
    profile: str,
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    *,
    returned_receipt_recorder_name: str,
    palace_identity: Mapping[str, Any],
) -> str:
    """Render a dependency-free script for a manual Palace handoff."""
    style = str(resources.get("command_style", "binary"))
    threads = positive_int(
        resources.get("threads", resources.get("cpus_per_task", 1)), "threads"
    )
    processes = positive_int(
        resources.get("processes", resources.get("ntasks", 1)), "processes"
    )
    directives: list[str] = []
    if profile != "direct-local":
        for key, value in resources.items():
            if key in {
                "launcher",
                "command_style",
                "threads",
                "processes",
                "output",
                "error",
                "open_mode",
            }:
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
    launcher = (
        []
        if profile == "direct-local"
        else launcher_tokens(resources.get("launcher", ("srun",)))
    )
    if profile != "direct-local" and not launcher:
        raise ValueError("Slurm profile launcher must be non-empty.")

    arguments = ['"$PALACE_EXECUTABLE"']
    if style == "wrapper":
        arguments.extend(["-np", '"$PALACE_PROCESSES"', "-nt", '"$PALACE_THREADS"'])
    arguments.append('"$PALACE_CONFIG"')
    command = " ".join([*(shlex.quote(item) for item in launcher), *arguments])
    config_schema = single_line(
        str(palace_identity["config_schema"]), "palace_identity.config_schema"
    )
    receipt_command = (
        f'python3 metadata/{returned_receipt_recorder_name} "$(pwd)" '
        '"$PIPELINE_EXIT_CODE" --log-path "$PALACE_LOG" '
        '--tee-exit-code "$TEE_EXIT_CODE" '
        '--job-id "${SLURM_JOB_ID:-manual}" '
        '--job-name "${SLURM_JOB_NAME:-manual}" '
    )
    preflight_receipt_command = (
        receipt_command + "--execution-stage config_compatibility "
        '--preflight-exit-code "$COMPATIBILITY_VERIFY_EXIT" '
        " || RECEIPT_EXIT=$?"
    )
    solver_receipt_command = (
        receipt_command + "--execution-stage solver --preflight-exit-code 0 "
        '--solver-exit-code "$PALACE_EXIT_CODE" '
        " || RECEIPT_EXIT=$?"
    )

    run_directory_lines = (
        [
            'SCRIPT_SOURCE="${BASH_SOURCE[0]}"',
            'RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd -P)" || exit 2',
        ]
        if profile == "direct-local"
        else [
            '[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || { echo "missing SLURM_SUBMIT_DIR" >&2; exit 2; }',
            'RUN_DIR="$(CDPATH= cd -- "$SLURM_SUBMIT_DIR" && pwd -P)" || exit 2',
        ]
    )

    lines = _script_start_lines(
        profile=profile,
        directives=directives,
        run_directory_lines=run_directory_lines,
        executable=executable,
        processes=processes,
        threads=threads,
        style=style,
    )
    if petsc:
        lines.extend(
            [
                f"PALACE_PETSC_OPTIONS={shlex.quote(' '.join(petsc))}",
                'export PETSC_OPTIONS="${PETSC_OPTIONS:-} ${PALACE_PETSC_OPTIONS}"',
            ]
        )
    lines.extend(_setup_lines(setup))

    compatibility_probe = (
        f"python3 metadata/{returned_receipt_recorder_name} verify-config "
        '"$PALACE_EXECUTABLE" "$PALACE_COMMAND_STYLE" '
        '"$PALACE_RUNTIME_COMPATIBILITY" "$PALACE_CONFIG" '
        f"--config-schema {shlex.quote(config_schema)}"
    )
    lines.extend(
        _compatibility_lines(
            profile=profile,
            probe=compatibility_probe,
            receipt_command=preflight_receipt_command,
        )
    )
    lines.extend(
        _solver_lines(
            profile=profile,
            command=command,
            receipt_command=solver_receipt_command,
        )
    )
    return "\n".join(lines)


def _script_start_lines(  # noqa: PLR0913 -- renders distinct script variables
    *,
    profile: str,
    directives: Sequence[str],
    run_directory_lines: Sequence[str],
    executable: str,
    processes: int,
    threads: int,
    style: str,
) -> list[str]:
    """Set run-directory, cleanup, and solver variables in the generated script."""
    return [
        "#!/usr/bin/env bash",
        *directives,
        "",
        "set -u",
        *run_directory_lines,
        '[[ -f "$RUN_DIR/config.json" ]] || { echo "missing $RUN_DIR/config.json" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/palace.msh" ]] || { echo "missing $RUN_DIR/palace.msh" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_handoff_metadata.json" ]] || { echo "missing canonical handoff metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_run_metadata.json" ]] || { echo "missing canonical run metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/mesh_manifest.json" ]] || { echo "missing canonical mesh manifest" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_index_map.json" ]] || { echo "missing canonical index map" >&2; exit 2; }',
        'cd "$RUN_DIR" || exit 2',
        "mkdir -p logs results/palace metadata",
        "rm -f metadata/palace_returned_run_receipt.json",
        "rm -f metadata/palace_runtime_compatibility.json",
        *(["rm -f logs/palace-*.log"] if profile == "direct-local" else []),
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
            "resources.launcher must be a single string or a sequence of strings."
        )
    tokens = [single_line(item, "launcher token") for item in value]
    if not tokens:
        raise ValueError("resources.launcher must be non-empty.")
    return tokens


def positive_int(value: Any, field: str) -> int:
    """Require a positive integer value for a named resource field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def single_line(value: str, field: str) -> str:
    """Require non-empty single-line text for a shell or resource field."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{field} must be non-empty single-line text.")
    return value
