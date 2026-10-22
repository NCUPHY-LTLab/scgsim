"""Artifact-only negative boundaries; no fabricated native-complete golden."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

from scgsim.aedt import AedtResources, batch
from scgsim.aedt.runtime import batch_protocol as protocol
from test_aedt_public_preparation import produce_request


class BatchFileBoundaryTests(unittest.TestCase):
    def test_paths_reject_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ("", "../escape", "/absolute"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    batch._inside(root, path)
            (root / "target").write_text("artifact")
            (root / "alias").symlink_to(root / "target")
            with self.assertRaises(ValueError):
                batch._inside(root, "alias")

    def test_inventory_hash_size_duplicates_and_nonhardlinked_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            (root / "member").write_bytes(b"generic file artifact")
            members = batch._inventory(root)
            copied = root.parent / "copy"
            batch._copy_members(root, copied, members)
            self.assertNotEqual((root / "member").stat().st_ino, (copied / "member").stat().st_ino)
            with self.assertRaises(ValueError):
                batch._verify(root, [*members, *members])
            (copied / "member").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                batch._verify(copied, members)

    def test_fresh_output_rejects_overlap_or_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                batch._fresh(root / "child", root)
            with self.assertRaises(FileExistsError):
                batch._fresh(root)

    def test_resource_recipe_rejects_unrepresentable_inputs(self):
        for value in ({"cores": 0, "ram_limit_percent": 8},
                      {"cores": True, "ram_limit_percent": 8},
                      {"cores": 2, "ram_limit_percent": 100},
                      {"cores": 2, "ram_limit_percent": 8, "unexpected": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                batch._resources(value)

    def test_package_and_collection_reject_unknown_or_missing_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native-batch-package.json").write_text(json.dumps({"schema_version": "unknown"}))
            with self.assertRaises(ValueError):
                batch._package(root)
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                batch._collection_source(root, {"package_sha256": "0" * 64, "attempt_sha256": "0" * 64})

    def test_pending_attempt_never_becomes_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for state in ("not_started", "starting", "running", "unsettled", "failed"):
                (root / "batch-attempt.json").write_text(json.dumps({
                    "schema_version": "scgsim.aedt.native-batch-attempt.v1",
                    "attempt_id": "pending", "package_sha256": "0" * 64, "package_id": "package",
                    "requested_aedt_version": "2024.2", "actual_aedt_version": None,
                    "version_observation": "unknown", "execution_location": {},
                    "resources_requested": {"cores": 2, "ram_limit_percent": 8},
                    "resource_semantics": "native limit", "started_at_unix": 0,
                    "status": state, "returncode": None,
                }))
                with self.subTest(state=state), self.assertRaises((TypeError, ValueError, RuntimeError)):
                    batch._attempt(root, root, {})

    def test_public_packager_rejects_real_file_only_not_run_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = produce_request(root / "request")
            before = plan.receipt_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "native prepared-model"):
                batch.prepare_native_batch(
                    plan.run_dir, output_dir=root / "package", solver_executable="unused",
                    resources=AedtResources(2, 8),
                )
            self.assertFalse((root / "package").exists())
            self.assertEqual(plan.receipt_path.read_bytes(), before)


class BatchTypedCodecTests(unittest.TestCase):
    """Wire DTOs are synthetic metadata, not native-prepared success receipts."""

    @staticmethod
    def package_payload():
        return {
            "schema_version": protocol.PACKAGE_SCHEMA, "status": "prepared_native",
            "package_id": "codec-only", "created_at_unix": 0, "source_location": "recorded-source",
            "requested_aedt_version": "2023.2", "prepared_connected": {"native": ["recorded"]},
            "spec_sha256": "spec", "model_sha256": "model", "source_sha256": "source",
            "project": "Example.aedt", "results": "Example.aedtresults", "design": "Design", "setup": "Setup",
            "resources": {"cores": 2, "ram_limit_percent": 8},
            "recipe": {"executable": "ansysedt", "mode": "prepared_project_batchsolve_ppe",
                       "allocation": "one_host_native_auto", "target": "Design:Nominal:Setup"},
            "declared_dependencies": [],
            # Historical single-file runner layout remains a valid v1 wire DTO.
            "members": [{"path": "run_batch.py", "bytes": 1, "sha256": "recorded"}],
            "retained_extension": {"nested": ["original"]},
        }

    def test_package_v1_single_file_roundtrip_and_deep_immutability(self):
        raw = self.package_payload()
        package = protocol.PackageManifest.from_payload(raw)
        self.assertEqual(package.to_payload(), raw)
        raw["prepared_connected"]["native"].append("mutation")
        raw["retained_extension"]["nested"].append("mutation")
        self.assertEqual(package.prepared_connected["native"], ("recorded",))
        self.assertEqual(package.extra["retained_extension"]["nested"], ("original",))
        with self.assertRaises(FrozenInstanceError):
            package.package_id = "changed"
        with self.assertRaises(TypeError):
            package.prepared_connected["native"] = ()
        self.assertEqual(len(package.members), 1)
        self.assertEqual(package.members[0].path, "run_batch.py")

    def test_attempt_preserves_unknown_version_null_and_pending_state(self):
        raw = {
            "schema_version": protocol.ATTEMPT_SCHEMA, "attempt_id": "codec-only",
            "package_sha256": "recorded", "package_id": "package", "status": "unsettled",
            "requested_aedt_version": "2024.2", "actual_aedt_version": None,
            "version_observation": "unknown", "execution_location": {"data": ["source"]},
            "resources_requested": {"cores": 2, "ram_limit_percent": 8},
            "resource_semantics": "native setting", "started_at_unix": 0,
            "returncode": None, "retained_extension": ["retained"],
        }
        attempt = protocol.BatchAttempt.from_payload(raw)
        self.assertEqual(attempt.to_payload(), raw)
        self.assertIsNone(attempt.actual_aedt_version)
        self.assertEqual(attempt.status, "unsettled")
        raw["execution_location"]["data"].append("mutation")
        self.assertEqual(attempt.execution_location["data"], ("source",))
        context = protocol.CollectionContext("package-sha", "attempt-sha")
        self.assertEqual(protocol.CollectionContext.from_payload(context.to_payload()), context)
        with self.assertRaises(ValueError):
            protocol.CollectionContext.from_payload({**context.to_payload(), "extra": 1})

    def test_tampered_helper_rejected_before_import_without_site_packages(self):
        from scgsim.aedt.runtime import batch_runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "run_batch.py"
            runner.write_bytes(Path(batch_runner.__file__).read_bytes())
            helper = root / "batch_protocol.py"
            helper.write_bytes(b"invalid Python must never be imported")
            (root / protocol.MANIFEST).write_text(json.dumps({
                "members": [{"path": "batch_protocol.py", "bytes": helper.stat().st_size,
                             "sha256": "not-the-helper-hash"}]}))
            result = subprocess.run([sys.executable, "-B", "-S", str(runner), "--help"],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("standalone protocol member identity differs", result.stderr)
            self.assertNotIn("SyntaxError", result.stderr)

    def test_new_runner_help_is_stdlib_only_and_inventory_cohort_preserves_v23(self):
        from scgsim.aedt.runtime import batch_runner
        from scgsim.aedt.results import provenance
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "run_batch.py"
            runner.write_bytes(Path(batch_runner.__file__).read_bytes())
            helper = root / "batch_protocol.py"
            helper.write_bytes(Path(protocol.__file__).read_bytes())
            record = protocol.Member("batch_protocol.py", helper.stat().st_size, protocol.sha256(helper))
            (root / protocol.MANIFEST).write_text(json.dumps({"members": [record.to_payload()]}))
            result = subprocess.run([sys.executable, "-B", "-S", str(runner), "--help"],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("solver-only", result.stdout)
            self.assertFalse((root / "batch-attempt.json").exists())
        self.assertEqual(set(provenance._RUNTIME_SOURCE_V24_PATHS) - set(provenance._RUNTIME_SOURCE_V23_PATHS), {
            "src/scgsim/aedt/epr/evidence.py",
            "src/scgsim/aedt/runtime/batch_protocol.py", "src/scgsim/aedt/runtime/batch_runner.py"})
        self.assertTrue(set(provenance._RUNTIME_SOURCE_V23_PATHS).issubset(provenance._RUNTIME_SOURCE_V24_PATHS))


    def test_plain_runner_import_preserves_exact_inventory_before_negative_native_boundary(self):
        """No native preparation receipt is invented: stop after real input verification."""
        from scgsim.aedt.runtime import batch_runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = produce_request(root / "request")
            package = root / "package"
            (package / "inputs").mkdir(parents=True)
            (package / "inputs/aedt_spec.json").write_bytes((request.run_dir / "aedt_spec.json").read_bytes())
            (package / "run_batch.py").write_bytes(Path(batch_runner.__file__).read_bytes())
            (package / "batch_protocol.py").write_bytes(Path(protocol.__file__).read_bytes())
            metadata = self.package_payload()
            metadata["spec_sha256"] = protocol.sha256(package / "inputs/aedt_spec.json")
            metadata["members"] = [member.to_payload() for member in protocol.inventory(package)]
            (package / protocol.MANIFEST).write_text(json.dumps(metadata))
            before = protocol.inventory(package)
            environment = dict(os.environ)
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            result = subprocess.run([
                sys.executable, "-S", str(package / "run_batch.py"), "solve", "--package", str(package),
                "--output", str(root / "attempt"),
            ], env=environment, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertIn("native-preparation-receipt.json", result.stderr)
            self.assertNotIn("undeclared or changed files", result.stderr)
            self.assertEqual(protocol.inventory(package), before)
            self.assertFalse((package / "__pycache__").exists())
            self.assertFalse((root / "attempt").exists())
