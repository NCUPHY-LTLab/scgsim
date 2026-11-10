"""Check that a rerun into a records folder never shows the stages of an earlier run.

    python check_rerun.py

The folder first holds a record with passing stages.  Then a case of the same file name runs through
``run_baseline.run_one`` and stops before it writes its first record: the name is no layout, so the case
fails at the PDK import or at the layout lookup, before its first stage.  The check passes when the record
left behind has no stage and names the exit code.  It needs no PDK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_baseline import record_file, run_one  # noqa: E402

CASE = "not_a_layout"
EARLIER = {
    "case": CASE,
    "bands": True,
    "stages": {"plan": {"result": "pass"}, "mesh": {"result": "pass"}},
}


def main() -> None:
    """Run the crashed rerun and refuse a record that still shows the earlier run's stages."""
    with TemporaryDirectory(prefix="baseline-rerun-") as folder:
        records = Path(folder)
        out = records / record_file(CASE, True, False)
        out.write_text(json.dumps(EARLIER))
        line = run_one(
            CASE, True, False, False, argparse.Namespace(records=records, timeout=300)
        )
        record = json.loads(out.read_text())
    print(line)
    if record["exit_code"] in (0, "timeout") or record["stages"]:
        raise SystemExit(f"a crashed rerun shows stages it did not run: {record}")
    print(
        "pass: the crashed rerun's record has no stage and exit code",
        record["exit_code"],
    )


if __name__ == "__main__":
    main()
