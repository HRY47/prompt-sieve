"""Check real Git ignore semantics while preserving intentional public evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_private_artifacts_ignored_public_fixtures_visible(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to verify ignore rules")
    policy = Path(__file__).resolve().parents[1] / ".gitignore"
    shutil.copyfile(policy, tmp_path / ".gitignore")
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    private = [
        ".aws/credentials", ".ssh/id_rsa", ".kube/config", ".git-credentials",
        ".netrc", ".pypirc", ".npmrc.local", ".env", ".env.production",
        "credentials.json", "config/credentials.local.json", "secrets.env",
        "config/secrets.local.json", "id_ed25519", "signing.pfx", "signing.p12",
        "private-data/records.csv", "data/private/records.parquet",
        "private-datasets/prompts.jsonl", "datasets/private/records.csv",
        "models/pytorch_model.bin", "model-cache/weights.bin", ".cache/huggingface/model.bin",
        "weights.safetensors", "model.gguf", "model.ckpt", ".venv/bin/python",
        "build/generated.py", "dist/package.whl", "src/pkg/__pycache__/x.pyc",
        ".review/local.log", ".tox/run/result", ".ruff_cache/result", "junit-results.xml",
    ]
    public = [
        ".env.example", "README.md", "SECURITY.md", "pyproject.toml", ".npmrc",
        "examples/cases.jsonl", "examples/pytest_run.log", "examples/prompt.md",
        "review/phase3-cases.jsonl", "review/phase3-benchmark.json",
        "review/phase2-pytest.xml", "review/phase3-pytest.xml",
        "tests/fixtures/public.csv", "scripts/phase3_benchmark.py",
    ]
    for expected, paths in ((True, private), (False, public)):
        for path in paths:
            result = subprocess.run([git, "-C", str(tmp_path), "check-ignore", "--quiet", "--no-index", path], capture_output=True)
            assert result.returncode in (0, 1)
            assert (result.returncode == 0) == expected, f"ignore policy mismatch: {path}"
