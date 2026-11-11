"""Write the JSON Schema of the SGB sidecar from its definition in ``scgsim.contracts.sgb_sidecar``.

Run it after changing ``scgsim/contracts/sgb_sidecar/records.py`` (and raising the version beside the new
digest in ``version.py``).  With ``--check`` it writes nothing and fails when the committed file differs from
the definition; ``tests/test_sidecar_contract.py`` makes the same comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scgsim.contracts.sgb_sidecar.schema import schema_text, sidecar_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path("src/scgsim/geometry/sgb/sidecar/sidecar.schema.json")


def main(argv: list[str] | None = None) -> int:
    """Write the schema file, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="compare only; write nothing"
    )
    arguments = parser.parse_args(argv)
    text = schema_text(sidecar_schema())
    path = ROOT / SCHEMA_PATH
    if arguments.check:
        if path.read_text(encoding="utf-8") != text:
            print(
                f"{SCHEMA_PATH} differs from the sidecar definition; run scripts/generate_sidecar_schema.py"
            )
            return 1
        print(f"{SCHEMA_PATH} is current")
        return 0
    # Bytes, not text: no newline translation on any platform.
    path.write_bytes(text.encode("utf-8"))
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
