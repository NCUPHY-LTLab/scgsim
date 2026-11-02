"""SCGSim development, release-candidate, and stable version policy."""

from __future__ import annotations

import unittest

from scripts.check_version import validate_line, version_kind


class VersionPolicyTests(unittest.TestCase):
    """Check release metadata classes and repository-line restrictions."""

    def test_supported_version_classes(self) -> None:
        """Distinguish development, release-candidate and stable versions."""
        self.assertEqual(version_kind("1.0.0.dev1"), "development")
        self.assertEqual(version_kind("1.0.0rc1"), "release-candidate")
        self.assertEqual(version_kind("1.0.0"), "stable")

    def test_develop_rejects_stable_version(self) -> None:
        """Require prerelease metadata on the development line."""
        with self.assertRaisesRegex(ValueError, "develop"):
            validate_line("1.0.0", "develop")

    def test_main_rejects_prerelease_version(self) -> None:
        """Require stable metadata on the protected release line."""
        with self.assertRaisesRegex(ValueError, "main"):
            validate_line("1.0.0.dev1", "main")

    def test_invalid_version_is_rejected(self) -> None:
        """Reject version text outside the supported release grammar."""
        with self.assertRaisesRegex(ValueError, "unsupported"):
            version_kind("1.0")


if __name__ == "__main__":
    unittest.main()
