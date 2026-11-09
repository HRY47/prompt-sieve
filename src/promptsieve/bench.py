"""Same-answer benchmark: does the optimized prompt still get the right result?

Token savings mean nothing if accuracy drops. For each case this sends the
ORIGINAL prompt and the OPTIMIZED prompt to a real model and scores both.

Cases file (JSONL, one object per line):

    {"id": "fail-1", "prompt_file": "logs/run1.txt",
     "question": "Which test failed and why? Answer with the test name.",
     "expected": "test_checkout_total", "match": "contains"}

Fields:
  prompt | prompt_file   the content to optimize (required, one of the two)
  question              appended AFTER optimization, never compressed (optional)
  expected              reference answer (optional)
  match                 contains | exact | regex | same_as_original
                        (default: contains if expected is given, else same_as_original)

Runners (--runner):
  anthropic:MODEL   Anthropic API (ANTHROPIC_API_KEY)
  openai:MODEL      OpenAI API (OPENAI_API_KEY)
  ollama:MODEL      local Ollama at http://localhost:11434 (OLLAMA_HOST to override)
  cmd:SHELL CMD     any command; prompt on stdin, answer on stdout
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Protocol

from .check import chrf
from .errors import UsageError
from .counters import get_counter
from .optimizer import optimize


class Runner(Protocol):
    name: str

    def run(self, prompt: str) -> str: ...


class AnthropicRunner:
    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic

        self._c, self.model, self.max_tokens = anthropic.Anthropic(timeout=60, max_retries=0), model, max_tokens
        self.name = f"anthropic:{model}"

    def run(self, prompt: str) -> str:
        r = self._c.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=0,
                                    messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAIRunner:
    def __init__(self, model: str):
        from openai import OpenAI

        self._c, self.model = OpenAI(timeout=60, max_retries=0), model
        self.name = f"openai:{model}"

    def run(self, prompt: str) -> str:
        r = self._c.chat.completions.create(model=self.model, temperature=0, max_completion_tokens=1024,
                                            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


class OllamaRunner:
    def __init__(self, model: str):
        self.model = model
        self.host = _ollama_base_url(os.environ.get("OLLAMA_HOST", ""))
        self.name = f"ollama:{model}"

    def run(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        if not isinstance(result, dict) or not isinstance(result.get("response"), str) or result.get("error"):
            raise ValueError("malformed Ollama response")
        return result["response"]


def _ollama_base_url(value: str) -> str:
    """Accept Ollama's own OLLAMA_HOST forms (``host``, ``host:port``, URL)."""
    from urllib.parse import urlsplit

    value = value.strip().rstrip("/") or "http://localhost:11434"
    if "://" not in value:
        value = "http://" + value
        if urlsplit(value).port is None:
            value += ":11434"  # Ollama's default port applies to bare hosts
    if urlsplit(value).scheme.lower() not in ("http", "https"):
        raise UsageError("OLLAMA_HOST must be an http(s) address")
    return value


class CommandRunner:
    def __init__(self, cmd: str, timeout: float = 60):
        if not cmd.strip() or timeout <= 0:
            raise ValueError("command and positive timeout required")
        self.cmd = cmd
        self.timeout = timeout
        self.name = "cmd"  # commands may contain credentials

    def run(self, prompt: str) -> str:
        # The command itself is trusted configuration; document text only goes
        # to stdin. Keep subprocesses in a group so timeout kills descendants.
        with subprocess.Popen(self.cmd, shell=True, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                              start_new_session=os.name != "nt",
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0) as p:
            try:
                stdout, _stderr = p.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
                    try:
                        subprocess.run([taskkill, "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                else:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if p.poll() is None:
                    p.kill()
                p.communicate(timeout=5)
                raise subprocess.TimeoutExpired("runner command (withheld)", self.timeout) from None
        if p.returncode != 0:
            raise RuntimeError(f"runner command exited with status {p.returncode} (stderr withheld)")
        return stdout


def get_runner(spec: str) -> Runner:
    kind, _, arg = spec.partition(":")
    if kind == "anthropic":
        return AnthropicRunner(arg or "claude-haiku-5-5")
    if kind == "openai":
        return OpenAIRunner(arg or "gpt-4o-mini")
    if kind == "ollama":
        return OllamaRunner(arg or "llama3.1")
    if kind == "cmd":
        return CommandRunner(arg)
    raise UsageError(f"unknown runner type {kind!r} (use anthropic, openai, ollama or cmd)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".").lower()


def score(answer: str, expected: Optional[str], match: str, reference_answer: str = "") -> bool:
    if match != "same_as_original" and (not isinstance(expected, str) or not expected.strip()):
        raise UsageError(f"{match} requires a nonempty expected answer")
    if match == "exact":
        return _norm(answer) == _norm(expected or "")
    if match == "contains":
        return _norm(expected or "") in _norm(answer)
    if match == "regex":
        return re.search(expected or "", answer, re.I | re.S) is not None
    if match == "same_as_original":
        a, b = _norm(answer), _norm(reference_answer)
        return a == b or chrf(a, b) >= 70
    raise UsageError(f"unknown match mode {match!r}")


@dataclass
class CaseResult:
    id: str
    strategy: str
    original_tokens: int
    optimized_tokens: int
    savings: float
    original_correct: Optional[bool]
    optimized_correct: Optional[bool]
    original_answer: str = field(repr=False, default="")
    optimized_answer: str = field(repr=False, default="")
    error: str = ""
    counter: str = ""
    agreement: Optional[bool] = None
    optimization_seconds: float = 0.0
    original_seconds: float = 0.0
    optimized_seconds: float = 0.0
    original_output_tokens: Optional[int] = None
    optimized_output_tokens: Optional[int] = None


def load_cases(path: str, allow_external_files: bool = False) -> List[dict]:
    """Load JSONL cases. ``prompt_file`` must resolve inside the cases file's
    directory unless ``allow_external_files`` is set: a downloaded case file
    must not be able to read e.g. ``~/.ssh/id_rsa`` and send it to an API."""
    base = Path(path).resolve().parent
    cases = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        c = json.loads(line)
        if not isinstance(c, dict):
            raise UsageError(f"line {i + 1}: expected an object")
        c.setdefault("id", f"case-{i + 1}")
        if "prompt_file" in c:
            if not isinstance(c["prompt_file"], str):
                raise UsageError(f"{c['id']}: prompt_file must be a string")
            target = (base / c["prompt_file"]).resolve()
            if not allow_external_files and not target.is_relative_to(base):
                raise UsageError(f"{c['id']}: prompt_file resolves outside the cases directory; "
                                 "pass --allow-external-prompt-files if you trust this case file")
            c["prompt"] = target.read_text(encoding="utf-8")
        if "prompt" not in c:
            raise UsageError(f"{c['id']}: needs prompt or prompt_file")
        if not isinstance(c["prompt"], str) or not isinstance(c.get("question", ""), str):
            raise UsageError(f"{c['id']}: prompt and question must be strings")
        c.setdefault("match", "contains" if "expected" in c else "same_as_original")
        # Validate all scoring rules before sending any prompts to a backend.
        score("", c.get("expected"), c["match"])
        cases.append(c)
    return cases


def run_bench(cases: List[dict], runner: Runner, optimize_kwargs: dict,
              progress: Callable[[str], None] = lambda s: None) -> List[CaseResult]:
    results = []
    for case in cases:
        score("", case.get("expected"), case["match"])
    counter = optimize_kwargs.get("counter") or get_counter()
    optimize_kwargs = dict(optimize_kwargs, counter=counter)
    for c in cases:
        start = time.perf_counter()
        res = optimize(c["prompt"], **optimize_kwargs)
        optimization_seconds = time.perf_counter() - start
        q = ("\n\n" + c["question"]) if c.get("question") else ""
        original_prompt = c["prompt"] + q
        original_n = counter.count(original_prompt)
        # Tokenization is not additive across string boundaries. Include the
        # question and delimiters, then enforce the acceptance gate again.
        candidate = res.best
        optimized_prompt = candidate.text + q
        optimized_n = counter.count(optimized_prompt)
        floor = original_n * (1 - optimize_kwargs.get("min_savings", 0.1)) if candidate.lossy else original_n
        if optimized_n > floor:
            optimized_prompt, optimized_n = original_prompt, original_n
            strategy = "original"
        else:
            strategy = candidate.name
        limit = optimize_kwargs.get("max_tokens")
        if limit is not None and optimized_n > limit:
            raise UsageError("final benchmark prompt including question exceeds max_tokens")
        answers, times, errors = [], [], []
        for arm, prompt in (("original", original_prompt), ("optimized", optimized_prompt)):
            start = time.perf_counter()
            try:
                answer = runner.run(prompt)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty or malformed runner answer")
                answers.append(answer)
            except Exception as e:
                answers.append(None)
                errors.append(f"{arm} request failed ({type(e).__name__})")
            times.append(time.perf_counter() - start)
        a_orig, a_opt = answers
        agreement_only = c["match"] == "same_as_original"
        orig_ok = None if agreement_only else (False if a_orig is None else score(a_orig, c.get("expected"), c["match"]))
        opt_ok = None if agreement_only else (False if a_opt is None else score(a_opt, c.get("expected"), c["match"]))
        agreement = None if None in answers else score(a_opt, None, "same_as_original", a_orig)
        r = CaseResult(c["id"], strategy, original_n, optimized_n,
                       1 - optimized_n / original_n if original_n else 0,
                       orig_ok, opt_ok, a_orig or "", a_opt or "", "; ".join(errors),
                       counter.name, agreement, optimization_seconds, times[0], times[1],
                       counter.count(a_orig) if a_orig is not None else None,
                       counter.count(a_opt) if a_opt is not None else None)
        err = r.error
        progress(f"{r.id}: {r.strategy} {r.savings:+.0%} tokens, "
                 f"original={_yn(r.original_correct)} optimized={_yn(r.optimized_correct)}"
                 + (f" ERROR {err}" if err else ""))
        results.append(r)
    return results


def _yn(v: Optional[bool]) -> str:
    return "ref" if v is None else ("ok" if v else "WRONG")


def summarize(results: List[CaseResult]) -> dict:
    tot_o = sum(r.original_tokens for r in results)
    tot_p = sum(r.optimized_tokens for r in results)
    graded = [r for r in results if r.original_correct is not None]
    return {
        "cases": len(results),
        "original_tokens": tot_o,
        "optimized_tokens": tot_p,
        "token_savings": 0.0 if not tot_o else 1 - tot_p / tot_o,
        "original_accuracy": (sum(bool(r.original_correct) for r in graded) / len(graded)) if graded else None,
        "optimized_accuracy": (sum(bool(r.optimized_correct) for r in graded) / len(graded)) if graded else None,
        "graded_cases": len(graded),
        "paired_regressions": [r.id for r in graded if r.original_correct and not r.optimized_correct],
        "paired_improvements": [r.id for r in graded if not r.original_correct and r.optimized_correct],
        "paired_both_correct": sum(bool(r.original_correct and r.optimized_correct) for r in graded),
        "paired_both_wrong": sum(not r.original_correct and not r.optimized_correct for r in graded),
        "agreement_rate": (sum(r.agreement if r.agreement is not None else score(r.optimized_answer, None, "same_as_original", r.original_answer)
                               for r in results if not r.error) / sum(not r.error for r in results)) if any(not r.error for r in results) else None,
        "answers_changed_vs_original": sum(
            1 for r in results if not r.error and _norm(r.original_answer) != _norm(r.optimized_answer)),
        "errors": sum(1 for r in results if r.error),
    }


def results_jsonl(results: List[CaseResult], include_answers: bool = False) -> str:
    rows = []
    for r in results:
        row = asdict(r)
        if not include_answers:
            row.pop("original_answer")
            row.pop("optimized_answer")
        rows.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(rows) + "\n"
