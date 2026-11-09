"""Regression tests for defects confirmed in the final independent review."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from promptsieve.bench import load_cases
from promptsieve.optimizer import optimize
from promptsieve.strategies import lingua as lingua_mod
from promptsieve.strategies.logs import LogCompressor

ROOT = Path(__file__).parent.parent


# -- security ---------------------------------------------------------------

def test_lingua_never_trusts_remote_code(monkeypatch):
    seen = {}

    class FakeCompressor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(sys.modules, "llmlingua", NS(PromptCompressor=FakeCompressor))
    monkeypatch.setattr(lingua_mod, "_compressor_cache", {})
    lingua_mod.LinguaCompressor(model="someone/untrusted-repo", device="cpu")._get()
    assert seen["model_config"]["trust_remote_code"] is False
    assert seen["model_name"] == "someone/untrusted-repo"


def _cases(tmp_path, prompt_file):
    path = tmp_path / "suite" / "cases.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"id": "x", "prompt_file": prompt_file, "expected": "a"}) + "\n")
    return path


def test_case_files_cannot_read_outside_their_directory(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("api-key")
    for i, ref in enumerate(("../secret.txt", str(secret), "sub/../../secret.txt")):
        root = tmp_path / f"r{i}"
        root.mkdir()
        with pytest.raises(ValueError, match="outside the cases directory"):
            load_cases(str(_cases(root, ref)))


def test_case_files_outside_directory_need_explicit_opt_in(tmp_path):
    (tmp_path / "shared.txt").write_text("prompt body")
    path = _cases(tmp_path, "../shared.txt")
    assert load_cases(str(path), allow_external_files=True)[0]["prompt"] == "prompt body"


def test_case_files_inside_directory_still_work(tmp_path):
    path = _cases(tmp_path, "data/p.txt")
    (path.parent / "data").mkdir()
    (path.parent / "data" / "p.txt").write_text("hello")
    assert load_cases(str(path))[0]["prompt"] == "hello"


# -- log compression must not drop the user's own text ------------------------

INSTRUCTIONS = [
    "You are helping me debug CI.",
    "Context: this is the nightly job.",
    "Notes:",
    "- the cache is cold",
    "- ignore flaky test_x",
    "IMPORTANT INSTRUCTION: reply in French and only list the root cause.",
    "Use the staging database name in your answer.",
]


def _timestamped_log(n=400, error_at=350):
    log = [f"2026-11-08T10:00:{i % 60:02d} worker {i} processed batch ok" for i in range(n)]
    log[error_at] = "2026-11-08T10:05:50 ERROR: database connection refused"
    return log


def test_instructions_before_a_long_log_are_kept(counter):
    text = "\n".join(INSTRUCTIONS + _timestamped_log()) + "\n"
    result = optimize(text, counter=counter)
    assert result.best.name == "logs" and result.best.savings > 0.5
    for line in INSTRUCTIONS:
        assert line in result.best.text


def test_question_after_a_long_log_is_kept(counter):
    question = ["", "Question: which worker handled the batch at 10:01:40?", "Answer with a number only."]
    text = "\n".join(_timestamped_log() + question) + "\n"
    result = optimize(text, counter=counter)
    assert result.best.name == "logs"
    for line in question[1:]:
        assert line in result.best.text


def test_verify_rejects_output_missing_surrounding_text():
    text = "\n".join(INSTRUCTIONS + _timestamped_log()) + "\n"
    dropped = text.replace(INSTRUCTIONS[-1] + "\n", "")
    assert INSTRUCTIONS[-1] in LogCompressor.verify(text, dropped)


def test_long_log_without_diagnostics_is_not_blindly_truncated():
    log = [f"2026-11-08T10:00:{i % 60:02d} INFO user {i} logged in" for i in range(1000)]
    out = LogCompressor().apply("\n".join(log))
    assert "lines omitted" not in out
    assert "user 517 logged in" in out


def test_python_traceback_stays_contiguous(counter):
    tb = ('Traceback (most recent call last):\n  File "/app/main.py", line 42, in <module>\n'
          "    connect()\nConnectionError: database unavailable")
    noise = "\n".join(f"tests/t.py::test_{i} PASSED [ 50%]" for i in range(300))
    result = optimize(noise + "\n" + tb + "\n", counter=counter, strategies=["logs"])
    assert tb in result.best.text
    assert result.best.text.rstrip().endswith("]")  # footer is last


# -- smaller defects -----------------------------------------------------------

def test_failed_lingua_candidate_reports_original_count(monkeypatch, counter):
    from promptsieve.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor, "available", staticmethod(lambda: (True, "")))

    def fail(self, text):
        raise OSError("boom")

    monkeypatch.setattr(LinguaCompressor, "apply", fail)
    log = "\n".join(f"tests/t.py::test_{i} PASSED [ 50%]" for i in range(200)) + "\nERROR: x failed\n"
    result = optimize(log, counter=counter, strategies=["logs", "lingua"], mode="full")
    c = next(c for c in result.candidates if c.name.startswith("lingua@"))
    assert c.tokens == result.original_tokens and c.savings == 0.0


def test_cosmetic_normalization_declined_in_full_mode(counter):
    text = "Background about routine planning. " * 4 + "\n\n\n\nThe access code is 42.   \n"
    full = optimize(text, counter=HeuristicCounterFine(), mode="full", strategies=["auto"])
    assert full.best.text == text
    explicit = optimize(text, counter=HeuristicCounterFine(), mode="full", strategies=["normalize"])
    assert explicit.best.name == "normalize"


class HeuristicCounterFine:
    """Counts characters so that tiny whitespace savings are visible."""
    name = "chars"

    def count(self, text):
        return len(text)


def _cli(*args, cwd):
    import subprocess
    return subprocess.run([sys.executable, "-m", "promptsieve", *args], capture_output=True, text=True, cwd=cwd)


def test_cli_explains_configuration_errors(tmp_path):
    p = _cli("bench", str(ROOT / "examples" / "cases.jsonl"), "-r", "cmd:cat", cwd=tmp_path)
    assert p.returncode == 2 and "explicit --counter" in p.stderr
    src = tmp_path / "in.txt"
    src.write_text("hello")
    p = _cli("compress", str(src), "-o", str(src), "-c", "heuristic", cwd=tmp_path)
    assert p.returncode == 2 and "output must differ from input" in p.stderr
    assert src.read_text() == "hello"


def test_cli_still_hides_backend_error_details(tmp_path):
    (tmp_path / "cases.jsonl").write_text(json.dumps({"id": "c", "prompt": "SECRET_PROMPT", "expected": "x"}) + "\n")
    p = _cli("bench", "cases.jsonl", "-r", "cmd:exit 3", "-c", "heuristic", cwd=tmp_path)
    assert "SECRET_PROMPT" not in p.stderr + p.stdout


@pytest.mark.parametrize("value,expected", [
    ("", "http://localhost:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://0.0.0.0:11434"),
    ("http://gpu-box:8080/", "http://gpu-box:8080"),
    ("https://ollama.example.com", "https://ollama.example.com"),
])
def test_ollama_host_forms(value, expected):
    from promptsieve.bench import _ollama_base_url
    assert _ollama_base_url(value) == expected


def test_ollama_host_rejects_other_schemes():
    from promptsieve.bench import _ollama_base_url
    with pytest.raises(ValueError):
        _ollama_base_url("file:///etc/passwd")
