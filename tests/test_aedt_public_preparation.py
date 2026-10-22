"""Public file-only request production; no native project or success evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scgsim.aedt import (
    AedtResources, EigenmodeRunControl, PreparedPlanarGeometry,
    prepare_hfss_eigenmode_from_geometry,
)
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.geometry import ApproximationProfile, interpret_physical_layers


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_only_geometry():
    # Structural codec input: no claim of native body/field adequacy.
    physical = interpret_physical_layers((), ApproximationProfile("solid", "v1", {}))
    source = {"materials": {"vacuum": {"kind": "vacuum", "permittivity": 1.0}},
              "modeling": physical.to_payload(), "conductors": []}
    model = {"modeling": "solid", "source": source, "junctions": []}
    complete = {**model, "contribution_catalog": [], "contributions": [], "surface_bindings": []}
    return PreparedPlanarGeometry(
        modeling="solid", source=source, junctions=(), contribution_catalog=(),
        contributions=(), surface_bindings=(), model_sha256=_digest(model),
        source_sha256=_digest(complete),
    )


def produce_request(root: Path, version="2024.2"):
    return prepare_hfss_eigenmode_from_geometry(
        aedt_version=version, modeling="solid", geometry=file_only_geometry(),
        project_name="ProtocolSource", design_name="Design", output_dir=root,
        run_control=EigenmodeRunControl("Setup", 1.0, 1, 3, 1.0),
        resources=AedtResources(2, 8),
    )


class PublicPreparationTests(unittest.TestCase):
    def test_both_versions_produce_real_not_run_handoff_and_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            for version in ("2023.2", "2024.2"):
                with self.subTest(version=version):
                    plan = produce_request(Path(temporary) / version, version)
                    payload = json.loads(plan.spec_path.read_text())
                    parsed = parse_aedt_spec(payload, base_dir=plan.run_dir)
                    self.assertEqual(parsed.aedt_version, version)
                    self.assertEqual(parsed.modeling, "solid")
                    self.assertEqual(parsed.to_payload(), payload)
                    receipt = json.loads(plan.receipt_path.read_text())
                    self.assertEqual(receipt["status"], "not_run")
                    self.assertEqual(receipt["requested"]["aedt_version"], version)
                    self.assertFalse((plan.run_dir / "ProtocolSource.aedt").exists())
                    manifest = json.loads((plan.run_dir / "metadata/aedt_handoff_manifest.json").read_text())
                    for item in manifest["members"]:
                        body = (plan.run_dir / item["path"]).read_bytes()
                        self.assertEqual(len(body), item["bytes"])
                        self.assertEqual(hashlib.sha256(body).hexdigest(), item["sha256"])
                    with tarfile.open(plan.archive_path) as archive:
                        self.assertEqual(set(archive.getnames()), set(manifest["allowed_paths"]))

    def test_public_preparation_preserves_fresh_output_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            produce_request(root)
            before = (root / "aedt_spec.json").read_bytes()
            with self.assertRaises(FileExistsError):
                produce_request(root)
            self.assertEqual((root / "aedt_spec.json").read_bytes(), before)
