"""Regression checks for the CI gate and model-factory naming policy."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


class WorkflowTests(unittest.TestCase):
    def test_python_checks_are_release_dependencies(self):
        build = workflow("build.yml")
        required = build["jobs"]["required"]
        self.assertIn("always()", required["if"])
        for job in ("python-lint", "python-types", "python-requirements", "platforms", "workflows", "ui",
                    "ui-checks", "naming", "formatting", "vendor", "cmake-package", "ops-docs"):
            self.assertIn(job, required["needs"])
        self.assertEqual(workflow("release.yml")["jobs"]["verify"]["uses"],
                         "./.github/workflows/build.yml")

    def test_reusable_python_checks_have_distinct_concurrency_groups(self):
        groups = []
        for name in ("python-lint.yml", "python-type-check.yml"):
            data = workflow(name)
            # PyYAML's YAML 1.1 loader interprets the unquoted 'on' key as True.
            triggers = data.get("on", data.get(True))
            self.assertIn("workflow_call", triggers)
            self.assertNotIn("pull_request", triggers)
            self.assertNotIn("push", triggers)
            groups.append(data["concurrency"]["group"])
        self.assertNotEqual(*groups)

    def test_labeler_uses_trusted_local_configuration(self):
        checkout = workflow("labeler.yml")["jobs"]["labeler"]["steps"][0]
        self.assertNotIn("repository", checkout["with"])
        self.assertEqual(checkout["with"]["ref"], "${{ github.event.pull_request.base.sha }}")
        self.assertFalse(checkout["with"]["persist-credentials"])

    def test_active_workflows_do_not_swallow_failures(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            data = workflow(path.name)
            for job in data.get("jobs", {}).values():
                self.assertNotIn("continue-on-error", job, path.name)
                for step in job.get("steps", []):
                    self.assertNotIn("continue-on-error", step, path.name)

    def test_model_factory_aliases_and_empty_input(self):
        command = workflow("code-style.yml")["jobs"]["model-naming"]["steps"][1]["run"]
        script = command.split("<< 'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/models").mkdir(parents=True)
            (root / "src/models/minicpm.cpp").write_text("", encoding="utf-8")
            factory = root / "src/llama-model.cpp"
            cases = [
                ("case LLM_ARCH_MINICPM:\ncase LLM_ARCH_MINICPM4:\n"
                 "return new llama_model_minicpm(params);", 0),
                ("case LLM_ARCH_WRONG:\nreturn new llama_model_minicpm(params);", 1),
                ("case LLM_ARCH_ABSENT:\nreturn new llama_model_absent(params);", 1),
                ("", 1),
            ]
            for source, expected in cases:
                with self.subTest(source=source):
                    factory.write_text(source, encoding="utf-8")
                    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
