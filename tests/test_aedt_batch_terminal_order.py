"""Terminal batch evidence order and first-failure behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scgsim.aedt.runtime import batch_protocol as protocol
from scgsim.aedt.runtime import batch_runner


class TerminalAttemptOrderTests(unittest.TestCase):
    """Keep terminal evidence, time, inventory, and write in baseline order."""

    @staticmethod
    def _execution() -> batch_runner.AttemptExecution:
        attempt = protocol.BatchAttempt(
            attempt_id="attempt",
            package_sha256="package-hash",
            package_id="package",
            status="running",
            requested_aedt_version="2024.2",
            actual_aedt_version=None,
            version_observation="not queried",
            execution_location={},
            resources_requested=protocol.Resources(2, 8),
            resource_semantics="native limit",
            started_at_unix=1,
        )
        return batch_runner.AttemptExecution(attempt)

    def test_terminal_evidence_precedes_time_inventory_and_write(self) -> None:
        """Hash stdout then stderr before sampling time and inventorying work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            work = root / "work"
            output.mkdir()
            work.mkdir()
            (output / "solver.stdout").write_bytes(b"stdout")
            (output / "solver.stderr").write_bytes(b"stderr")
            execution = self._execution()
            events: list[tuple[str, str]] = []
            original_stat = Path.stat
            original_write_json = protocol.write_json
            written: list[dict[str, object]] = []

            def record_stat(path: Path, *args: object, **kwargs: object):
                if path.name in {"solver.stdout", "solver.stderr"}:
                    events.append(("stat", path.name))
                return original_stat(path, *args, **kwargs)

            def record_hash(path: Path) -> str:
                events.append(("hash", path.name))
                return f"digest:{path.name}"

            def record_time() -> float:
                events.append(("time", "finished_at_unix"))
                return 23.5

            def record_inventory(path: Path) -> list[protocol.Member]:
                events.append(("inventory", path.name))
                return [protocol.Member("model.aedt", 7, "model-hash")]

            def record_write(path: Path, value: dict[str, object]) -> None:
                events.append(("write", path.name))
                written.append(value)
                original_write_json(path, value)

            with (
                patch.object(Path, "stat", new=record_stat),
                patch.object(batch_runner.protocol, "sha256", new=record_hash),
                patch.object(batch_runner.time, "time", new=record_time),
                patch.object(batch_runner.protocol, "inventory", new=record_inventory),
                patch.object(batch_runner.protocol, "write_json", new=record_write),
            ):
                batch_runner.record_terminal_attempt(execution, output, work, 0)

            self.assertEqual(
                events,
                [
                    ("stat", "solver.stdout"),
                    ("hash", "solver.stdout"),
                    ("stat", "solver.stderr"),
                    ("hash", "solver.stderr"),
                    ("time", "finished_at_unix"),
                    ("inventory", "work"),
                    ("write", "batch-attempt.json"),
                ],
            )
            self.assertEqual(
                [item.path for item in execution.attempt.raw_evidence or ()],
                ["solver.stdout", "solver.stderr"],
            )
            self.assertEqual(
                [item.bytes for item in execution.attempt.raw_evidence or ()],
                [6, 6],
            )
            self.assertEqual(execution.attempt.finished_at_unix, 23.5)
            self.assertEqual(
                [item.path for item in execution.attempt.work_members or ()],
                ["model.aedt"],
            )
            self.assertEqual(written[0]["status"], "external_process_terminal")
            self.assertEqual(written[0]["returncode"], 0)
            self.assertEqual(written[0]["finished_at_unix"], 23.5)
            self.assertEqual(
                written[0]["raw_evidence"],
                [
                    {
                        "path": "solver.stdout",
                        "bytes": 6,
                        "sha256": "digest:solver.stdout",
                    },
                    {
                        "path": "solver.stderr",
                        "bytes": 6,
                        "sha256": "digest:solver.stderr",
                    },
                ],
            )

    def test_stream_failure_precedes_work_inventory_failure(self) -> None:
        """A raw stream error wins when work inventory would also fail."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            work = root / "work"
            output.mkdir()
            work.mkdir()
            (output / "solver.stdout").write_bytes(b"stdout")
            (output / "solver.stderr").write_bytes(b"stderr")
            events: list[tuple[str, str]] = []
            original_stat = Path.stat

            def record_stat(path: Path, *args: object, **kwargs: object):
                if path.name in {"solver.stdout", "solver.stderr"}:
                    events.append(("stat", path.name))
                return original_stat(path, *args, **kwargs)

            def fail_hash(path: Path) -> str:
                events.append(("hash", path.name))
                raise OSError("stdout evidence hash failed")

            def fail_inventory(path: Path) -> list[protocol.Member]:
                events.append(("inventory", path.name))
                raise OSError("work inventory failed")

            with (
                patch.object(Path, "stat", new=record_stat),
                patch.object(batch_runner.protocol, "sha256", new=fail_hash),
                patch.object(batch_runner.protocol, "inventory", new=fail_inventory),
                patch.object(
                    batch_runner.time,
                    "time",
                    side_effect=AssertionError("time must follow raw evidence"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "stdout evidence hash failed"):
                    batch_runner.record_terminal_attempt(
                        self._execution(), output, work, 0
                    )

            self.assertEqual(
                events,
                [("stat", "solver.stdout"), ("hash", "solver.stdout")],
            )


if __name__ == "__main__":
    unittest.main()
