"""Check that the baseline's records say only what the run that wrote them did.

    python check_records.py

Three checks, each a way a record could show a run it does not describe:

* **A crashed rerun.**  The folder holds an earlier record with passing stages.  A case of the same file name
  then runs through ``run_baseline.run_one`` and stops before it writes its first record: the name is no
  layout, so the case fails at the PDK import or at the layout lookup, before its first stage.  The record
  left behind must have no stage and name the exit code.
* **A record read mid-write.**  A thread reads and parses the record without pause while ``run_case.write_record``
  rewrites it 200 times.  Every read must find a complete record.  A kill at any moment, as the time limit
  kills a case, leaves the file as such a read would find it at that moment; with a plain ``write_text`` almost
  every read finds a truncated file.
* **An unreadable record.**  ``run_baseline.stopped_record`` on a truncated file must return the empty
  record, not raise.

It needs no PDK.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_baseline import record_file, run_one, stopped_record  # noqa: E402
from run_case import write_record  # noqa: E402

CASE = "not_a_layout"
EARLIER = {
    "case": CASE,
    "bands": True,
    "stages": {"plan": {"result": "pass"}, "mesh": {"result": "pass"}},
}
REWRITES = 200
LARGE = {"stages": {"mesh": {"result": "pass", "history": ["x" * 80] * 20_000}}}


def main() -> None:
    """Run the three checks and stop at the first record that shows what its run did not do."""
    with TemporaryDirectory(prefix="baseline-records-") as folder:
        records = Path(folder)
        print(_crashed_rerun(records))
        print(_read_mid_write(records / "rewritten.json"))
        print(_unreadable(records / "truncated.json"))


def _crashed_rerun(records: Path) -> str:
    out = records / record_file(CASE, True, False)
    out.write_text(json.dumps(EARLIER))
    line = run_one(
        CASE, True, False, False, argparse.Namespace(records=records, timeout=300)
    )
    record = json.loads(out.read_text())
    if record["exit_code"] in (0, "timeout") or record["stages"]:
        raise SystemExit(f"a crashed rerun shows stages it did not run: {record}")
    return f"pass: {line}; the record has no stage and exit code {record['exit_code']}"


def _read_mid_write(out: Path) -> str:
    write_record(out, LARGE)
    done = threading.Event()
    reads, broken = [0], [0]

    def read_without_pause() -> None:
        while not done.is_set():
            reads[0] += 1
            try:
                json.loads(out.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                broken[0] += 1

    reader = threading.Thread(target=read_without_pause)
    reader.start()
    for _ in range(REWRITES):
        write_record(out, LARGE)
    done.set()
    reader.join()
    if broken[0]:
        raise SystemExit(f"{broken[0]} of {reads[0]} reads found a broken record")
    return f"pass: {reads[0]} reads during {REWRITES} rewrites, every one a complete record"


def _unreadable(out: Path) -> str:
    out.write_text(json.dumps(EARLIER)[:40])
    empty = {"case": CASE, "stages": {}}
    record = stopped_record(out, empty)
    if record != empty:
        raise SystemExit(
            f"an unreadable record was not replaced by the empty one: {record}"
        )
    return "pass: a truncated record reads as the empty record"


if __name__ == "__main__":
    main()
