"""Independent regressions. These tests do not establish ML model quality."""
import json
import subprocess
import sys

import pytest

from promptsieve.bench import CaseResult, CommandRunner, load_cases, run_bench, score, summarize
from promptsieve.check import preservation
from promptsieve.optimizer import optimize
from promptsieve.protect import mask, split_segments, unmask
from promptsieve.strategies.logs import LogCompressor
from promptsieve.strategies.normalize import normalize


@pytest.mark.parametrize("line", [
    "test_x PASSED WARNING: credentials exposed",
    "Collecting package ERROR: checksum mismatch",
    "PASS tests.js assertion failed",
    "Downloading dependency: permission denied",
])
def test_noise_cannot_override_diagnostics(line):
    out = LogCompressor().apply(line)
    assert line in out
    assert LogCompressor.verify(line, "") == [line]


def test_carriage_return_does_not_hide_error():
    assert "ERROR: denied" in normalize("ERROR: denied\rDone")


def test_normalize_keeps_fenced_code_whitespace():
    text = "```python\nx = '''a  \n\n\nb'''\n```"
    assert normalize(text) == text


def test_normalize_keeps_markdown_hard_break():
    assert normalize("one  \ntwo") == "one  \ntwo"


@pytest.mark.parametrize("original,candidate", [
    ("balance is -5", "balance is 5"),
    ("rate 1e-5", "rate 1e-4"),
    ("use -3.5%", "use 3.5%"),
])
def test_signed_and_scientific_numbers(original, candidate):
    assert not preservation(original, candidate).ok


@pytest.mark.parametrize("span", [r"C:\work\app.py", "/etc/config.yaml", "config.yaml", '"exact value"', "'literal'", "⟦0⟧"])
def test_mask_protects_fragile_literals(span):
    masked, saved = mask("Keep " + span)
    assert span in saved
    assert unmask(masked, saved) == ("Keep " + span, [])


@pytest.mark.parametrize("text", [
    "   ```py\nx = 1\n   ```",
    "````markdown\n```python\nx = 1\n```\n````",
    "```python\nx = 1",  # unfinished fence
    "    x = 1\n    y = 2",  # indented code
    "# Heading\n- first\n- second\n| a | b |",
])
def test_structures_are_protected(text):
    segs = split_segments(text)
    assert "".join(s.text for s in segs) == text
    assert all(s.protected for s in segs if s.text.strip())


def test_roundtrip_failure_is_candidate_rejection(counter):
    class BrokenBack:
        def translate(self, text, src, tgt):
            if tgt == "en":
                raise RuntimeError("missing reverse pack")
            return "字"
    res = optimize("Long ordinary prose. " * 30, counter=counter, langs=["zh"], translator=BrokenBack(), per_segment=True)
    assert res.best.name in {"original", "normalize"}
    assert not next(c for c in res.candidates if c.name == "translate:zh").passed


def test_roundtrip_negation_loss_rejected(counter):
    original = "The service must not delete the production database under any circumstances. " * 10
    class Inverting:
        def translate(self, text, src, tgt):
            return original.replace("not ", "") if tgt == "en" else "字"
    res = optimize(original, counter=counter, langs=["zh"], translator=Inverting())
    assert not next(c for c in res.candidates if c.name == "translate:zh").passed


@pytest.mark.parametrize("kwargs", [{"min_savings": -1}, {"min_savings": float("nan")}, {"min_chrf": 101}, {"lingua_rate": 0}, {"min_preservation": 2}, {"strategies": ["typo"]}])
def test_invalid_configuration_is_rejected(counter, kwargs):
    with pytest.raises(ValueError):
        optimize("text", counter=counter, **kwargs)


def test_translation_strategy_name_is_not_silently_ignored(counter, translator, prose):
    res = optimize(prose, counter=counter, strategies=["translate:xx"], translator=translator)
    assert any(c.name == "translate:xx" for c in res.candidates)


def test_agreement_is_not_accuracy():
    r = CaseResult("x", "original", 10, 10, 0, None, None, "wrong", "wrong")
    summary = summarize([r])
    assert summary["optimized_accuracy"] is None
    assert summary["agreement_rate"] == 1


def test_changed_wrong_answers_count_as_changed():
    r = CaseResult("x", "original", 10, 10, 0, False, False, "wrong A", "wrong B")
    assert summarize([r])["answers_changed_vs_original"] == 1


@pytest.mark.parametrize("mode", ["contains", "exact", "regex"])
def test_missing_expected_is_not_success(mode):
    with pytest.raises(ValueError):
        score("anything", None, mode)


def test_question_tokens_are_measured(counter):
    class Echo:
        def run(self, prompt):
            return "ok"
    prompt, question = "short", "large question " * 100
    r, = run_bench([{"id": "q", "prompt": prompt, "question": question, "expected": "ok", "match": "exact"}], Echo(), {"counter": counter})
    assert r.original_tokens == counter.count(prompt + "\n\n" + question)


def test_command_stdin_never_becomes_shell_code(tmp_path):
    script = tmp_path / "echo.py"
    script.write_text("import sys; sys.stdout.write(sys.stdin.read())", encoding="utf-8")
    prompt = '$(echo injected) & echo unsafe; `id` " 日本語'
    runner = CommandRunner(f'"{sys.executable}" "{script}"')
    assert runner.run(prompt) == prompt


def test_command_failure_does_not_leak_stderr():
    runner = CommandRunner(f'"{sys.executable}" -c "import sys;sys.stderr.write(\'PRIVATE_SENTINEL\');sys.exit(1)"')
    with pytest.raises(RuntimeError) as exc:
        runner.run("input")
    assert "PRIVATE_SENTINEL" not in str(exc.value)
