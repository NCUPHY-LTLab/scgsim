"""Write portable AEDT launchers and identify their handoff members."""

from __future__ import annotations

import time
from pathlib import Path

from scgsim.aedt.io import file_sha256


def write_script(path: Path, *, analyze_epr: bool = False) -> None:
    """Write the portable launcher with its preserved CLI arguments.

    ``analyze_epr`` adds the saved-field analysis flag.
    """
    flag = " --analyze-epr" if analyze_epr else ""
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'run_dir="$(cd "$(dirname "$0")" && pwd)"\n'
        f'python -m scgsim.aedt.run --handoff "$run_dir/metadata/aedt_handoff_metadata.json" --execute{flag} "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def handoff_member(path: Path, root: Path) -> dict[str, str | int]:
    """Record the exact path, byte count, and SHA-256 of one handoff file."""
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def utc_now() -> str:
    """Return the UTC timestamp representation used by prepared receipts."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
