"""Read one saved HFSS design from its project file, without a cached parser or project mutation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from scgsim.aedt.io import file_sha256


def read_saved_hfss_design(project_path, design_name, *, purpose):
    """Read one saved HFSS design without a cached parser or project mutation."""
    from ansys.aedt.core.internal.load_aedt_file import load_entire_aedt_file  # noqa: PLC0415 -- optional PyAEDT parser is needed only when reading a saved native project, not during preparation.

    path = Path(project_path).resolve(strict=True)
    digest = file_sha256(path)
    native = load_entire_aedt_file(path)
    if file_sha256(path) != digest:
        raise RuntimeError(f"saved {purpose} native project changed during readback")
    project = native.get("AnsoftProject")
    designs = project.get("HFSSModel") if isinstance(project, Mapping) else None
    if isinstance(designs, Mapping):
        designs = [designs]
    if not isinstance(designs, list):
        raise TypeError(f"saved {purpose} design records are unavailable")
    matching = [
        d for d in designs if isinstance(d, Mapping) and d.get("Name") == design_name
    ]
    if len(matching) != 1:
        raise RuntimeError(f"saved {purpose} design is missing or ambiguous")
    return digest, matching[0]
