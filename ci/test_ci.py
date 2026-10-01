"""Tests for CI assertions and archive safety; no model downloads required."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package
import smoke


class SmokeTests(unittest.TestCase):
    def test_health_requires_correct_engine(self):
        smoke.validate_health(200, {"status": "ok", "engine": "comni"})
        for status, body in [(500, {"status": "ok", "engine": "comni"}),
                             (200, {"status": "ok", "engine": "llama"}), (200, {})]:
            with self.assertRaises(RuntimeError):
                smoke.validate_health(status, body)

    def test_missing_model_must_be_rejected_by_omni(self):
        body = {"error": {"message": "omni_init missing required model file (LLM): missing.gguf"}}
        smoke.validate_missing_model(500, body)
        for status, data in [(200, body), (500, {"error": {"message": "unrelated error"}}), (404, {})]:
            with self.assertRaises(RuntimeError):
                smoke.validate_missing_model(status, data)

    def test_help_checks_exit_status_and_banner(self):
        with patch("smoke.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "VoxCPM2")):
            smoke.run_checked(["voxcpm2-cli", "--help"], 1, "VoxCPM2")
            with self.assertRaises(RuntimeError):
                smoke.run_checked(["voxcpm2-cli", "--help"], 0, "VoxCPM2")
            with self.assertRaises(RuntimeError):
                smoke.run_checked(["voxcpm2-cli", "--help"], 1, "wrong banner")

    def test_binary_missing_is_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                smoke.binary(Path(root), "llama-omni-server")


class PackageTests(unittest.TestCase):
    def test_invalid_archive_name(self):
        for name in ("../escape", "a/b", "$(command)", ""):
            with self.assertRaises(ValueError):
                package.package(Path("unused"), name)

    def test_package_preserves_binaries_and_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "bin"
            binaries.mkdir()
            for name in package.TOOLS:
                tool = binaries / name
                tool.write_text("test executable")
                tool.chmod(0o755)
            (binaries / "libomni.so").write_text("runtime library")
            (binaries / "test-unrelated").write_text("not a release tool")
            extracted = root / "extracted"
            archive = package.package(binaries, "omni-test", extracted, root / "output")
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("libomni.so", zipped.namelist())
                self.assertNotIn("test-unrelated", zipped.namelist())
                self.assertIn("LICENSE", zipped.namelist())
            if os.name != "nt":
                self.assertTrue(os.access(extracted / "llama-omni-cli", os.X_OK))
            with self.assertRaises(RuntimeError):
                package.package(binaries, "omni-test", extracted, root / "output")


if __name__ == "__main__":
    unittest.main()
