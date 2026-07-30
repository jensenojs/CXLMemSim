#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class ComponentScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "manifests").mkdir()
        for name in ("build_component.sh", "package_component.sh"):
            shutil.copy2(SOURCE_ROOT / "scripts" / name, self.repo / "scripts" / name)
        shutil.copy2(SOURCE_ROOT / "manifests/build-profile.json", self.repo / "manifests/build-profile.json")
        shutil.copy2(SOURCE_ROOT / "manifests/artifact-contract.json", self.repo / "manifests/artifact-contract.json")
        (self.repo / "scripts/component_artifact.py").write_text(
            """#!/usr/bin/env python3
import json, pathlib, sys
args=sys.argv[1:]
if args[0] == 'create-manifest':
    output=pathlib.Path(args[args.index('--output')+1])
    commit=args[args.index('--source-commit')+1]
    output.write_text(json.dumps({'schema_version':1,'source':{'commit':commit},'build':{'profile_sha256':'b'*64}},sort_keys=True)+'\\n')
elif args[0] not in {'verify-archive','verify-payload','verify-profile'}:
    raise SystemExit(2)
""",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        self.head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_build_rejects_dirty_before_creating_outputs(self) -> None:
        (self.repo / "dirty").write_text("x", encoding="utf-8")
        cache = self.root / "cache"
        cache.mkdir()
        work = self.root / "build-work"
        payload = self.root / "payload"
        result = subprocess.run(["bash", "scripts/build_component.sh", "--work-dir", str(work), "--cache-dir", str(cache), "--payload-dir", str(payload)], cwd=self.repo, text=True, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source checkout must be clean", result.stderr)
        self.assertFalse(work.exists())
        self.assertFalse(payload.exists())

    def test_package_is_deterministic_and_outputs_are_explicit(self) -> None:
        payload = self.root / "payload"
        payload.mkdir()
        (payload / "file").write_text("same bytes\n", encoding="utf-8")
        outputs = []
        for index in (1, 2):
            work = self.root / f"package-{index}"
            manifest = self.root / f"manifest-{index}.json"
            archive = self.root / f"archive-{index}.tar.zst"
            subprocess.run(["bash", "scripts/package_component.sh", "--payload-dir", str(payload), "--source-commit", self.head, "--work-dir", str(work), "--manifest-out", str(manifest), "--archive-out", str(archive)], cwd=self.repo, check=True)
            outputs.append((manifest.read_bytes(), archive.read_bytes()))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
