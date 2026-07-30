from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pull_component.sh"


class PullComponentCliTests(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(SCRIPT), *arguments], text=True, capture_output=True, check=False)

    def test_runtime_is_required(self) -> None:
        result = self.run_script("--work-dir", "/unused/work", "/unused/candidate", "/unused/output")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--container-runtime is required", result.stderr)

    def test_work_directory_is_required(self) -> None:
        result = self.run_script("--container-runtime", "docker", "/unused/candidate", "/unused/output")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--work-dir is required", result.stderr)

    def test_relative_work_directory_fails_before_candidate_read(self) -> None:
        result = self.run_script("--container-runtime", "docker", "--work-dir", "relative", "/unused/candidate", "/unused/output")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("work directory must be absolute", result.stderr)

    def test_existing_work_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            work.mkdir()
            marker = work / "marker"
            marker.write_text("preserved\n", encoding="utf-8")
            result = self.run_script("--container-runtime", "docker", "--work-dir", str(work), "/unused/candidate", "/unused/output")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("work directory must not exist", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserved\n")

    def test_podman_runtime_gate_keeps_calling_user_mapping(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('runtime_args=(run --rm --user "$(id -u):$(id -g)")', source)
        self.assertIn('runtime_args+=(--userns=keep-id --security-opt label=disable)', source)


if __name__ == "__main__":
    unittest.main()
