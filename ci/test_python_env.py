"""Regression checks for the Python 3.12 roadmap typing corrections."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("roadmap_render", ROOT / "scripts/roadmap/render.py")
assert SPEC is not None and SPEC.loader is not None
ROADMAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROADMAP)


class LintCompatibilityTests(unittest.TestCase):
    def test_no_print_rule_and_existing_noqa_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text('print("forbidden")\n', encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "flake8", "--isolated", "--select=NP100", str(path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertIn("NP100", result.stdout)
            path.write_text('print("CLI output")  # noqa: NP100\n', encoding="utf-8")
            subprocess.run([sys.executable, "-m", "flake8", "--isolated", "--select=NP100", str(path)],
                           check=True, capture_output=True, text=True, timeout=30)


class TypeGateTests(unittest.TestCase):
    def test_warning_policy_does_not_hide_type_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.py"
            path.write_text('value: int = "wrong"\n', encoding="utf-8")
            result = subprocess.run(["ty", "check", "--exit-zero-on-warning", str(path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertIn("invalid-assignment", result.stdout)


class RoadmapTests(unittest.TestCase):
    def test_missing_and_invalid_toml_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.toml"
            with self.assertRaises(ROADMAP.ValidationError):
                ROADMAP.load_toml(path)
            path.write_text("not valid = [", encoding="utf-8")
            with self.assertRaises(ROADMAP.ValidationError):
                ROADMAP.load_toml(path)
            path.write_text('title = "test"\n', encoding="utf-8")
            self.assertEqual(ROADMAP.load_toml(path), {"title": "test"})

    def test_bilingual_render_keeps_structured_strings(self):
        tasks = [{"id": "T-001", "title": "Example", "title_zh": "示例",
                  "difficulty": "easy", "domain": "python"}]
        for language, title in (("en", "Example"), ("zh", "示例")):
            body = ROADMAP.render(tasks, "example/project", {}, lang=language)
            self.assertIn(f"**T-001** {title}", body)
            self.assertIn("`difficulty: easy`", body)
            self.assertIn("scripts/roadmap/render.py --check", body)

    def test_repository_tasks_validate_without_github(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/roadmap/render.py"), "--check"],
                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
