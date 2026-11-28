"""Select and restore run-local AEDT solver resources around one blocking solve."""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.specs.common import AedtResources


@dataclass(frozen=True)
class _DsoConfiguration:
    """The selected resource file and prior Desktop registry identity."""

    desktop: Any
    design_type: str
    key: str
    previous: str
    config_name: str
    acf_path: Path


def analyze_with_resources(  # noqa: PLR0913 -- existing family solve callback passes the optional controlled analysis operation.
    app: Any,
    setup_name: str,
    run_dir: Path,
    resources: AedtResources | None,
    evidence: dict[str, Any],
    *,
    analysis_call: Callable[[Callable[[], Any]], Any] | None = None,
) -> bool:
    """Select one run-local DSO configuration only for the owned blocking solve."""

    def analyze() -> Any:
        def call() -> Any:
            return app.analyze_setup(name=setup_name, blocking=True)

        return analysis_call(call) if analysis_call is not None else call()

    evidence["requested"] = resources.to_payload() if resources else None
    if resources is None:
        evidence["status"] = "aedt_active_configuration_unchanged"
        solve_started = time.perf_counter()
        try:
            return bool(analyze())
        finally:
            evidence["solve_seconds"] = round(time.perf_counter() - solve_started, 6)

    configuration = _prepare_dso_configuration(app, run_dir, resources, evidence)
    started = time.perf_counter()
    load_attempted = False
    try:
        load_attempted = True
        if app.set_hpc_from_file(acf_file=str(configuration.acf_path)) is not True:
            raise RuntimeError("AEDT rejected the run-local ACF")
        observed = configuration.desktop.GetRegistryString(configuration.key)
        if observed != configuration.config_name:
            raise RuntimeError(f"AEDT active DSO configuration mismatch: {observed!r}")
        evidence["status"] = "configuration_selected"
        evidence["observed_active_configuration"] = observed
        evidence["selected_settings"] = {
            **resources.to_payload(),
            "tasks": 1,
            "gpus": 0,
            "use_auto_settings": False,
        }
        evidence["settings_evidence"] = "generated_acf_and_active_configuration_name"
        evidence["activation_seconds"] = round(time.perf_counter() - started, 6)
        solve_started = time.perf_counter()
        try:
            solved = analyze()
        finally:
            evidence["solve_seconds"] = round(time.perf_counter() - solve_started, 6)
        evidence["analyze_returned"] = solved is True
        if solved is not True:
            raise RuntimeError(f"AEDT setup {setup_name!r} analysis returned false")
        return True
    finally:
        _restore_dso_configuration(configuration, evidence, load_attempted)


def _prepare_dso_configuration(
    app: Any, run_dir: Path, resources: AedtResources, evidence: dict[str, Any]
) -> _DsoConfiguration:
    """Write the requested configuration after reading the current registry value."""
    desktop = app.desktop_class.odesktop
    design_type = app.design_type
    if design_type not in {"HFSS", "Q3D Extractor", "2D Extractor"}:
        raise RuntimeError(
            f"unsupported local AEDT resource design type: {design_type!r}"
        )
    key = f"Desktop/ActiveDSOConfigurations/{design_type}"
    previous = desktop.GetRegistryString(key)
    if not isinstance(previous, str):
        raise RuntimeError("previous AEDT active DSO configuration is unavailable")
    acf_started = time.perf_counter()
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16]
    config_name = f"scgsim_{digest}"
    template = Path(app.pyaedt_dir) / "misc/pyaedt_local_config.acf"
    content = template.read_text(encoding="utf-8")
    replacements = {
        "ConfigName": f"'{config_name}'",
        "DesignType": f"'{design_type}'",
        "NumCores": str(resources.cores),
        "NumEngines": "1",
        "NumGPUs": "0",
        "RAMPercent": str(resources.ram_limit_percent),
        "UseAutoSettings": "false",
    }
    for name, value in replacements.items():
        pattern = rf"(?m)^([ \t]*{name}=)[^\r\n]*$"
        content, count = re.subn(
            pattern, lambda match, value=value: match.group(1) + value, content
        )
        if count != 1:
            raise RuntimeError(f"AEDT ACF template has {count} {name} entries")
    acf_path = run_dir / "metadata" / f"{config_name}.acf"
    acf_path.write_text(content, encoding="utf-8")
    evidence.update(
        {
            "status": "acf_written",
            "acf": acf_path.relative_to(run_dir).as_posix(),
            "acf_sha256": file_sha256(acf_path),
            "design_type": design_type,
            "previous_active_configuration": previous,
            "acf_generation_seconds": round(time.perf_counter() - acf_started, 6),
        }
    )
    return _DsoConfiguration(desktop, design_type, key, previous, config_name, acf_path)


def _restore_dso_configuration(
    configuration: _DsoConfiguration, evidence: dict[str, Any], load_attempted: bool
) -> None:
    """Restore the prior registry selection while preserving a primary failure."""
    # Preserve a solve/activation error while still recording restoration failure.
    original_failure = sys.exc_info()[0] is not None
    restore_started = time.perf_counter()
    if load_attempted:
        try:
            configuration.desktop.SetRegistryString(
                configuration.key, configuration.previous
            )
            if (
                configuration.desktop.GetRegistryString(configuration.key)
                != configuration.previous
            ):
                raise RuntimeError("AEDT active DSO configuration was not restored")
            evidence["restored_previous_configuration"] = True
        except Exception as exc:
            evidence["restored_previous_configuration"] = False
            evidence["restoration_error"] = f"{type(exc).__name__}: {exc}"
            evidence["restoration_seconds"] = round(
                time.perf_counter() - restore_started, 6
            )
            if not original_failure:
                raise RuntimeError(
                    "AEDT active DSO configuration restoration failed"
                ) from exc
    evidence["restoration_seconds"] = round(time.perf_counter() - restore_started, 6)
