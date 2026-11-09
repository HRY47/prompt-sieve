import json
import subprocess
import sys
from pathlib import Path

from tokopt.bench import CommandRunner, load_cases, run_bench, score, summarize

ROOT = Path(__file__).parent.parent
PY = sys.executable


def test_score_modes():
    assert score("The failing test is test_checkout_total.", "test_checkout_total", "contains")
    assert score("26.25", "26.25", "exact")
    assert score("total=26.25", r"26\.2\d", "regex")
    assert score("Answer: B", None, "same_as_original", reference_answer="answer: b")
    assert not score("Answer: C", None, "same_as_original", reference_answer="Answer: B, because of the tax rule")


def test_bench_with_command_runner(tmp_path, counter):
    # a fake "model": answers with the first FAILED test name it sees in the prompt
    model = tmp_path / "model.py"
    model.write_text("import sys,re\nm=re.search(r'::(\\w+) FAILED', sys.stdin.read())\nprint(m.group(1) if m else 'none')\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "log1", "prompt_file": str(ROOT / "examples" / "pytest_run.log"),
                                 "question": "Which test failed?", "expected": "test_checkout_total"}) + "\n")
    results = run_bench(load_cases(str(cases)), CommandRunner(f"{PY} {model}"), {"counter": counter})
    s = summarize(results)
    assert s["original_accuracy"] == 1.0 and s["optimized_accuracy"] == 1.0
    assert s["token_savings"] > 0.9
    assert s["answers_changed_vs_original"] == 0


def run_cli(*args, stdin=None):
    return subprocess.run([PY, "-m", "tokopt", *args], input=stdin, capture_output=True, text=True, cwd=ROOT)


def test_cli_compress_stdin():
    log = (ROOT / "examples" / "pytest_run.log").read_text()
    p = run_cli("compress", "-", "-c", "heuristic", stdin=log)
    assert p.returncode == 0, p.stderr
    assert "test_checkout_total FAILED" in p.stdout and "PASSED" not in p.stdout
    assert "-> logs" in p.stderr


def test_cli_count():
    p = run_cli("count", "examples/prompt.md", "-c", "heuristic")
    assert p.returncode == 0 and "heuristic" in p.stdout


def test_cli_experiment_json():
    p = run_cli("experiment", "examples/prompt.md", "-c", "heuristic", "--json")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["best"] in ("original", "normalize")
