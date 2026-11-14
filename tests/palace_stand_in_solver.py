"""A stand-in for Palace's adaptive loop, for the tests that run the real run script, runner and recorder.

It does what Palace 0.18.1 was recorded doing in the stop experiments of issue 248 (E1), for both problems:

* each solve writes its tables at the top of ``results/palace`` as regular files and its ``palace.json``
  recording ``Problem.Iteration`` k; an Eigenmode solve after the first starts ``eig.csv`` and
  ``port-EPR.csv`` with their header only;
* after solve k, when another pass follows, it prints ``Adaptive mesh refinement (AMR) iteration k:``,
  makes ``iteration<k>/``, moves the tables into it, writes ``iteration<k>/palace.json``, leaves a link to
  each table at the top level and rewrites the top-level ``palace.json`` for the next mesh;
* then it pauses for "refinement" and prints ``Proceeding with solve/estimate iteration k+1...``;
* the last solve stays at the top level, and it prints ``Completed N iterations of adaptive mesh refinement``.

Like a launcher (``prterun``) it starts one "rank" in its process group and answers SIGTERM by ending the
rank and exiting 1.  ``behind_wrapper`` puts it behind a shell script that starts it as a child and passes
no signal on, as Palace's own ``palace`` wrapper does with ``prterun`` (issue 248, the stop check with real
Palace): a signal to the solver's first process then reaches the wrapper only.  A schedule (JSON beside the
script) sets the pass tables, the pauses, a failure after a pass, and whether it ignores SIGTERM.  It writes
the ids of itself, the runner, its rank and its wrapper to ``pids`` in the schedule, so a test can check that
none of them is left.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

_SOURCE = r"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEDULE = json.loads(Path(__file__).with_suffix(".json").read_text())
RANK = "import signal, time\n" + (
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if SCHEDULE["ignore_sigterm"] else ""
) + "while True:\n    time.sleep(0.05)\n"


def main(arguments):
    arguments = [a for a in arguments if a != "-serial"]
    if arguments[:1] == ["--version"]:
        print("Palace version: 0.18.1")
        print("Schema version: 1-7-0")
        return 0
    if arguments[:1] == ["--dry-run"]:
        return 0
    rank = subprocess.Popen([sys.executable, "-I", "-c", RANK])
    pids = {"solver": os.getpid(), "runner": os.getppid(), "rank": rank.pid}
    if SCHEDULE["behind_wrapper"]:
        wrapper = os.getppid()
        # /proc/<pid>/stat: "pid (name) state ppid ..."; the runner started the wrapper.
        runner = int(Path(f"/proc/{wrapper}/stat").read_text().rsplit(") ", 1)[1].split()[1])
        pids.update(wrapper=wrapper, runner=runner)
    Path(SCHEDULE["pids"]).write_text(json.dumps(pids))
    if SCHEDULE["ignore_sigterm"]:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def stop(_signum, _frame):
            rank.terminate()
            rank.wait()
            os._exit(1)
        signal.signal(signal.SIGTERM, stop)
    results = Path("results/palace")
    passes = SCHEDULE["passes"]
    for number, entry in enumerate(passes, start=1):
        if number > 1:
            print(f"Proceeding with solve/estimate iteration {number}...", flush=True)
            if SCHEDULE["problem"] == "Eigenmode":
                for name in ("eig.csv", "port-EPR.csv"):
                    header = (results / name).read_text().splitlines()[0]
                    (results / name).unlink()
                    (results / name).write_text(header + "\n")
        time.sleep(entry["solve_s"])
        source = Path(entry["tables"])
        for table in sorted(source.glob("*.csv")):
            target = results / table.name
            if target.is_symlink() or target.exists():
                target.unlink()
            shutil.copyfile(table, target)
        shutil.copyfile(source / "palace.json", results / "palace.json")
        print(f" Indicator norm = {entry['norm']:.3e}", flush=True)
        if SCHEDULE["fail_after"] == number:
            print("Error: the stand-in solver failed", file=sys.stderr, flush=True)
            rank.terminate()
            rank.wait()
            return 2
        if number == len(passes):
            print(f"Completed {number - 1} iterations of adaptive mesh refinement (AMR):", flush=True)
            break
        print(f"Adaptive mesh refinement (AMR) iteration {number}:", flush=True)
        folder = results / f"iteration{number}"
        folder.mkdir()
        tables = sorted(results.glob("*.csv"))
        for table in tables:
            os.replace(table, folder / table.name)
        shutil.copyfile(results / "palace.json", folder / "palace.json")
        for table in tables:
            os.symlink(f"iteration{number}/{table.name}", table)
        metadata = json.loads((results / "palace.json").read_text())
        metadata["Problem"]["DegreesOfFreedom"] *= 4
        (results / "palace.json").write_text(json.dumps(metadata))
        time.sleep(entry["refine_s"])
    rank.terminate()
    rank.wait()
    return 0


raise SystemExit(main(sys.argv[1:]))
"""


def write_stand_in(  # noqa: PLR0913 -- one schedule, each switch independent
    path: Path,
    *,
    problem: str,
    passes: list[dict[str, Any]],
    pids: Path,
    fail_after: int | None = None,
    ignore_sigterm: bool = False,
    behind_wrapper: bool = False,
) -> Path:
    """Write the stand-in solver at ``path`` (its schedule beside it) and return ``path``.

    Each pass is ``{"tables": folder, "norm": float, "solve_s": seconds, "refine_s": seconds}``: the folder
    holds the pass's tables and ``palace.json`` in Palace's formats.  With ``behind_wrapper``, ``path`` is a
    shell script that runs the stand-in as its child.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    launcher = path.with_name(f"{path.name}-launcher") if behind_wrapper else path
    launcher.write_text(f"#!{sys.executable} -I\n{_SOURCE}", encoding="utf-8")
    launcher.chmod(0o755)
    launcher.with_suffix(".json").write_text(
        json.dumps(
            {
                "problem": problem,
                "passes": passes,
                "pids": str(pids),
                "fail_after": fail_after,
                "ignore_sigterm": ignore_sigterm,
                "behind_wrapper": behind_wrapper,
            }
        ),
        encoding="utf-8",
    )
    if behind_wrapper:
        path.write_text(f'#!/bin/sh\n"{launcher}" "$@"\n', encoding="utf-8")
        path.chmod(0o755)
    return path


def read_pids(path: Path, timeout_s: float = 20.0) -> dict[str, int]:
    """The ids the stand-in wrote once it started, waiting for it up to ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text():
            return json.loads(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"the stand-in solver did not start within {timeout_s} s")


def alive(pid: int) -> bool:
    """Whether a process with id ``pid`` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    return not (status.is_file() and status.read_text().split(") ")[-1][:1] == "Z")


def wait_until_gone(pids: dict[str, int], timeout_s: float = 10.0) -> list[str]:
    """The names of the processes in ``pids`` still alive after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        left = [name for name, pid in pids.items() if alive(pid)]
        if not left:
            return []
        time.sleep(0.05)
    return [name for name, pid in pids.items() if alive(pid)]


def runner_processes(run_dir: Path) -> list[int]:
    """The ids of the processes started from the runner copy of ``run_dir``: the runner and its guard."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if str(run_dir).encode() in command and any(
            part.endswith(b"run_palace_until_converged.py") for part in command
        ):
            found.append(int(entry.name))
    return [pid for pid in found if alive(pid)]


def end_left_over(pids: dict[str, int]) -> None:
    """End any process of a test's own run that is still there, so a failed test leaves nothing behind."""
    for pid in pids.values():
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
