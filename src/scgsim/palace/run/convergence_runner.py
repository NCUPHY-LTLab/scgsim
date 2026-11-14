"""The convergence runner: it starts the solver, checks each pass as it completes, and stops the solver at convergence.

With a declared convergence plan, the run script starts the solver through the copy of this module in the run
folder (``metadata/run_palace_until_converged.py``)::

    python3 metadata/run_palace_until_converged.py <run folder> -- <the solver command the script would run>

The runner

1. starts the solver command in a session of its own, with the runner's output streams, so the solver's output
   reaches the log exactly as before and the solver can be signalled without signalling the script;
2. looks at ``results/palace`` every ``LOOK_INTERVAL_S``; a pass counts once its ``iteration<k>/`` folder is
   unchanged between two looks, and after each new pass it runs the check of the plan on the passes read;
3. when the verdict is converged, writes the decision record, then sends SIGTERM to the solver's session and
   waits ``STOP_WAIT_S`` for all of it to end, sending SIGKILL to the session only after that wait;
4. when the solver has ended, by itself or after the stop, ends in the same way what its first process left
   behind (a launcher whose wrapper script ended first), writes the decision record again with the passes the
   results folder then holds and the solver's exit code, and exits with that exit code.

The solver has ended when its first process has ended and its session holds no process any more.  The first
process alone does not say it: Palace's ``palace`` is a wrapper script that SIGTERM ends at once, while the
launcher it started (``prterun``) needs 0.3 to 0.6 s more to bring its ranks down and ends after the last of
them.  The ranks are each in a process group of their own, so only their launcher ends them; a launcher ended
by SIGKILL before it has done that leaves them to end by themselves about a second later (measured with
Palace 0.18.1 and Open MPI 5.0.10).  Which processes a session holds is read from ``/proc``.  On a system
without it the runner waits for the solver's first process only, and the SIGKILL to its group at the end is
all that ends a launcher behind a wrapper script.

It signals only the process group of the session it started, and only while that session's first process has
not been reaped, so the signal cannot reach a process that took over its number.  Told to end itself (SIGTERM,
SIGINT or SIGHUP: the script, the scheduler or a person ending the job), it ends the solver the same way first
and records why; told so while the solver is already ending after the stop at convergence, the stop stands,
since that is what ended the solver.  Ended without that chance (SIGKILL, to it or to the script's whole process group), it leaves
the solver to its guard: a process in a session of its own, started before the solver, that sends SIGTERM to
the solver's session as soon as the runner is gone.

The stop comes one to two seconds after a pass is complete (``LOOK_INTERVAL_S``, two looks), so a pass shorter
than that still runs and is counted with the pass before it; a run that ends before the first look is
complete as the solver ended it.

Like the receipt recorder, the copy carries the contracts' text in place of their imports
(``standalone_copy``), so it runs with only a Python interpreter.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergenceDecision,
    ConvergencePlan,
    SolverExit,
    SolverStop,
    StopReason,
    check_convergence,
    recorded_check,
    returned_passes,
)
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    CONVERGENCE_PLAN,
    OUTPUT_DIR,
    read_json,
    sha256,
    utc_timestamp,
    write_json_atomic,
)

LOOK_INTERVAL_S = 1.0
"""How often the runner looks at ``results/palace``: a look reads a listing and a few small tables, so it costs
milliseconds, and the stop then comes one to two seconds after a pass, short against any solve worth stopping."""
STOP_WAIT_S = 60.0
"""How long the runner waits for the solver to end after SIGTERM before it sends SIGKILL: Palace 0.18.1 under
``mpirun`` ended in 0.4 s; the margin is for a launcher such as ``srun`` that forwards the signal to its tasks."""
KILL_WAIT_S = 10.0
"""How long the runner waits for the solver to end after SIGKILL before it reports the solver as it is."""
_STOOD_DOWN = b"done\n"
"""What the runner writes to its guard once it has ended the solver's session itself."""
_POLL_S = 0.1
"""How often the runner checks whether the solver has ended or the runner was told to end."""


def _script_main() -> None:
    """Run the solver command after ``--`` for the run folder given before it."""
    raise SystemExit(run(_request(sys.argv[1:])))


@dataclass(frozen=True)
class _Request:
    """The runner call the run script makes: the run folder and the solver command."""

    root: Path
    command: list[str]


def _request(argv: list[str]) -> _Request:
    parser = argparse.ArgumentParser(
        description="Run the Palace solver until the declared convergence criteria hold."
    )
    parser.add_argument("run_dir", type=Path, help="The run folder of the handoff.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("the solver command after -- is missing")
    return _Request(root=args.run_dir.expanduser().resolve(), command=command)


def run(request: _Request) -> int:
    """Run the solver until it ends or the plan's criteria converge; return the solver's exit code."""
    root = request.root
    plan_path = root / CONVERGENCE_PLAN
    plan = ConvergencePlan.from_payload(read_json(plan_path))
    looker = _Looker(plan, root, sha256(plan_path), utc_timestamp())
    endings = _EndRequests()
    solver = _start(request.command)
    try:
        stop = _supervise(solver, looker, endings)
    finally:
        # Whatever happened above, the runner does not leave its solver running unwatched.
        if not _ended(solver):
            _stop(solver, looker.stop_now("runner_terminated"))
    if stop is None:
        _end_what_is_left(solver)
    exit_code = _exit_code(_reap(solver))
    final = looker.decision(
        looker.final_check(),
        stop,
        SolverExit(exit_code, utc_timestamp()),
    )
    _write_decision(root, final)
    _say(_ended_words(final))
    return exit_code


@dataclass(frozen=True)
class _Running:
    """The solver process the runner started, and the pipe end that stands its guard down.

    The process's id is also its session's and its process group's.
    """

    process: subprocess.Popen[bytes]
    guard: int

    @property
    def pid(self) -> int:
        return self.process.pid

    def wait(self) -> int:
        return self.process.wait()


def _start(command: list[str]) -> _Running:
    """Start the guard, then the solver in a session of its own, with the runner's output streams.

    The guard comes first, so that no solver runs without one: when the guard cannot be started, no solver
    is.  The solver's first process tells the guard its session itself, before it becomes the solver command,
    so the guard knows the solver from the moment it exists, whatever happens to the runner meanwhile.
    """
    guard = _start_guard()

    def name_the_session() -> None:
        # Runs in the solver's process, once it leads its session and before the command replaces it.
        os.write(guard, f"{os.getpid()}\n".encode())

    process = subprocess.Popen(
        command, start_new_session=True, preexec_fn=name_the_session
    )
    return _Running(process, guard)


def _start_guard() -> int:
    """Start the guard; return the pipe end through which the runner tells it the session and stands it down.

    The runner can be ended in a way it cannot answer, and the solver, in a session of its own, would then run
    on unwatched to its pass limit: a launcher's wrapper script does not pass a signal on to the launcher it
    started.  The guard is a copy of the runner in a session of its own, so a signal to the script's process
    group does not reach it.  It waits on a pipe only the runner holds, which closes when the runner ends,
    however it ends.  The solver's first process writes its session there before it becomes the solver
    command, and the runner writes ``done`` once it has ended the session itself.  If the pipe closes with a
    session and without ``done``, the guard sends
    SIGTERM to the solver's process group, once: the runner is gone at that moment and the solver still
    runs, so the group is the solver's.  It sends nothing later, when the group's id could have passed to
    another process.
    """
    watched, held = os.pipe()
    try:
        guard = os.fork()
    except OSError:
        os.close(watched)
        os.close(held)
        raise
    if guard != 0:
        os.close(watched)
        return held
    try:
        os.close(held)
        os.setsid()
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        # The log's pipe must close when the script's processes end, not when the guard does.
        null = os.open(os.devnull, os.O_RDWR)
        for stream in (0, 1, 2):
            os.dup2(null, stream)
        told = b""
        while chunk := os.read(watched, 64):
            told += chunk
        session, _, after = told.partition(b"\n")
        if session and after != _STOOD_DOWN:
            _signal_group(int(session), signal.SIGTERM)
    finally:
        os._exit(0)


def _stand_down(solver: _Running) -> None:
    """Tell the guard that the runner has ended the solver's session itself."""
    os.write(solver.guard, _STOOD_DOWN)
    os.close(solver.guard)


def _signal_group(group: int, signum: int) -> None:
    """Signal a process group, if it has a process left."""
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass


def _session_members(session: int) -> list[int]:
    """The processes of a session that have not ended, as ``/proc`` lists them; none where there is no /proc.

    A process that has ended and waits to be reaped (the solver's first process, which the runner keeps so)
    is not one of them.
    """
    members = []
    try:
        entries = [entry for entry in os.scandir("/proc") if entry.name.isdigit()]
    except OSError:
        return members
    for entry in entries:
        try:
            # /proc/<pid>/stat: "pid (name) state ppid pgrp session ..."; the name may hold spaces.
            state, _parent, _group, recorded = (
                Path(entry.path, "stat").read_text().rsplit(") ", 1)[1].split()[:4]
            )
        except (OSError, IndexError, ValueError):
            continue  # the process ended while it was read
        if state != "Z" and int(recorded) == session:
            members.append(int(entry.name))
    return members


def _supervise(
    solver: _Running, looker: _Looker, endings: _EndRequests
) -> SolverStop | None:
    """Look at the passes until the solver ends; stop it at convergence, or when the runner is told to end."""
    next_look = time.monotonic() + LOOK_INTERVAL_S
    while not _ended(solver):
        if endings.received is not None:
            return _stop(solver, looker.stop_now("runner_terminated"))
        if time.monotonic() >= next_look:
            next_look += LOOK_INTERVAL_S
            check = looker.look()
            if check is not None and check.verdict.state == "converged":
                stop = looker.stop_now("converged")
                # The record comes first: whatever happens to the runner after the signal, it says why.
                _write_decision(looker.root, looker.decision(check, stop, None))
                _say(
                    f"converged at pass {check.verdict.pass_number}; stopping the solver after pass "
                    f"{stop.after_pass} (SIGTERM to its session)"
                )
                return _stop(solver, stop)
        time.sleep(_POLL_S)
    return None


def _stop(solver: _Running, stop: SolverStop) -> SolverStop:
    """End the solver's session for ``stop``; the stop says so when SIGKILL was needed."""
    escalated = _end(solver)
    return stop if escalated is None else replace(stop, escalated_utc=escalated)


def _end(solver: _Running) -> str | None:
    """Send SIGTERM to the solver's process group and wait for its session to end; SIGKILL after ``STOP_WAIT_S``.

    SIGTERM goes to the group: the command and the launcher it started, which ends its ranks.  SIGKILL goes
    to every process the session then still holds.  Returns when SIGKILL was sent, None when SIGTERM was
    enough.
    """
    _signal_session(solver, signal.SIGTERM)
    if _all_ended_within(solver, STOP_WAIT_S):
        return None
    escalated = utc_timestamp()
    _say(
        f"the solver did not end {STOP_WAIT_S:g} s after SIGTERM; sending SIGKILL to its session"
    )
    _signal_session(solver, signal.SIGKILL)
    for member in _session_members(solver.pid):
        try:
            os.kill(member, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _all_ended_within(solver, KILL_WAIT_S)
    return escalated


def _all_ended_within(solver: _Running, seconds: float) -> bool:
    """Whether, within ``seconds``, the solver's first process has ended and its session holds no process."""
    deadline = time.monotonic() + seconds
    while not (_ended(solver) and not _session_members(solver.pid)):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)
    return True


def _end_what_is_left(solver: _Running) -> None:
    """After the solver's first process ended by itself: end what it left running, as a stop would.

    A launcher outlives a wrapper script that was killed, and goes on solving.  Nothing is left when the
    launcher itself ended, which is how a run that completes ends.
    """
    if not _session_members(solver.pid):
        return
    _say("the solver's first process ended and left processes running; ending them")
    _end(solver)


def _reap(solver: _Running) -> int:
    """Collect the solver's exit status, after ending with SIGKILL any process its group still holds.

    Where the session's members can be read, none is left by now.  The signal goes out, and the guard is
    stood down, before the first process is reaped: until then the group's id is still the solver's, so
    neither the runner nor the guard can signal a process that took it over.
    """
    _signal_session(solver, signal.SIGKILL)
    _stand_down(solver)
    return solver.wait()


def _signal_session(solver: _Running, signum: int) -> None:
    """Signal the solver's process group while its first process is not reaped.

    Until it is reaped, even after it has ended, its id cannot be given to another process, so the group
    holds only the solver and the processes it started.
    """
    if solver.process.returncode is not None:
        return
    try:
        os.killpg(solver.pid, signum)
    except ProcessLookupError:
        pass


def _ended(solver: _Running) -> bool:
    """Whether the solver's first process has ended, without reaping it."""
    try:
        state = os.waitid(os.P_PID, solver.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return state is not None


def _exit_code(returncode: int) -> int:
    """The exit code as a shell reports it: 128 + N for a process ended by signal N."""
    return returncode if returncode >= 0 else 128 - returncode


class _EndRequests:
    """The runner's own SIGTERM, SIGINT and SIGHUP: each is remembered, and the runner ends the solver first."""

    def __init__(self) -> None:
        self.received: int | None = None
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, self._remember)

    def _remember(self, signum: int, _frame: FrameType | None) -> None:
        self.received = signum


class _Looker:
    """What the runner has read of ``results/palace``: each iteration folder's listing and the passes checked."""

    def __init__(
        self, plan: ConvergencePlan, root: Path, plan_sha256: str, started: str
    ) -> None:
        self.plan = plan
        self.root = root
        self.results = root / OUTPUT_DIR
        self.plan_sha256 = plan_sha256
        self.started = started
        self.listings: dict[Path, tuple[tuple[str, int, int], ...]] = {}
        self.checked: tuple[int, ...] = ()
        self.seen: dict[str, str] = {}
        self.unreadable = False

    def look(self) -> ConvergenceCheck | None:
        """The check of the passes read, when a pass was added since the last look; None otherwise.

        Palace moves, links and rewrites files while it runs, so a look can meet a file that is gone a moment
        later; such a look reads nothing and the next one tries again.
        """
        try:
            return self._look()
        except (OSError, ValueError) as error:
            if not self.unreadable:
                _say(f"a look at {OUTPUT_DIR} read nothing ({error}); looking again")
            self.unreadable = True
            return None

    def _look(self) -> ConvergenceCheck | None:
        iterations = [
            folder
            for folder in returned_passes(self.results).passes
            if folder.path != self.results
        ]
        listed = {folder.path: _listing(folder.path) for folder in iterations}
        # A pass counts once its folder did not change between two looks: Palace's order inside the move
        # of a pass into its folder is not known, and the folder never changes after the move.
        settled = [
            folder
            for folder in iterations
            if self.listings.get(folder.path) == listed[folder.path]
        ]
        self.listings = listed
        numbers = tuple(folder.number for folder in settled)
        if numbers == self.checked:
            return None
        self.checked = numbers
        for number in numbers:
            self.seen.setdefault(str(number), utc_timestamp())
        return check_convergence(self.plan, settled)

    def stop_now(self, reason: StopReason) -> SolverStop:
        """A stop for ``reason`` now, after the last pass read."""
        return SolverStop(
            reason=reason,
            after_pass=self.checked[-1] if self.checked else None,
            requested_utc=utc_timestamp(),
        )

    def final_check(self) -> ConvergenceCheck:
        """The check of every pass the results folder holds once the solver has ended, the top level included."""
        returned = returned_passes(self.results)
        for folder in returned.passes:
            self.seen.setdefault(str(folder.number), utc_timestamp())
        return check_convergence(self.plan, returned.passes, returned.set_aside)

    def decision(
        self,
        check: ConvergenceCheck,
        stop: SolverStop | None,
        solver: SolverExit | None,
    ) -> ConvergenceDecision:
        return ConvergenceDecision(
            plan_sha256=self.plan_sha256,
            started_utc=self.started,
            look_interval_s=LOOK_INTERVAL_S,
            stop_wait_s=STOP_WAIT_S,
            checked=recorded_check(check, self.root),
            seen_utc=dict(self.seen),
            stop=stop,
            solver=solver,
        )


def _listing(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """What a pass folder holds: each entry's name, size and modification time, so a change between looks shows."""
    entries = []
    for entry in os.scandir(folder):
        status = entry.stat(follow_symlinks=False)
        entries.append((entry.name, status.st_size, status.st_mtime_ns))
    return tuple(sorted(entries))


def _write_decision(root: Path, decision: ConvergenceDecision) -> None:
    write_json_atomic(root / CONVERGENCE_DECISION, decision.to_payload())


def _ended_words(decision: ConvergenceDecision) -> str:
    verdict = decision.checked["verdict"]
    solver = decision.solver
    code = solver.exit_code if solver is not None else None
    return (
        f"the solver ended ({decision.ending}, exit code {code}); verdict {verdict['state']}"
        + (
            f" at pass {verdict['pass_number']}"
            if verdict["state"] == "converged"
            else ""
        )
        + f"; recorded in {CONVERGENCE_DECISION}"
    )


def _say(message: str) -> None:
    """One line on the runner's error stream, which the run script sends to the solver log."""
    try:
        print(f"SCGSim convergence runner: {message}", file=sys.stderr, flush=True)
    except OSError:
        pass  # The log's reader is gone; the decision record still says it.


if __name__ == "__main__":
    _script_main()
