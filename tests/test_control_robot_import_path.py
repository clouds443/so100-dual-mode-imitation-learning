import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ControlRobotImportPathTest(unittest.TestCase):
    def test_control_robot_prefers_this_checkout_over_pythonpath_lerobot(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_package = Path(tmpdir) / "lerobot"
            fake_package.mkdir()
            (fake_package / "__init__.py").write_text("__version__ = 'foreign'\n", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(tmpdir)) + os.pathsep + env.get("PYTHONPATH", "")

            result = subprocess.run(
                [sys.executable, "lerobot/scripts/control_robot.py", "record", "--help"],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--robot-path", result.stdout)
        self.assertIn("--record-control-mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
