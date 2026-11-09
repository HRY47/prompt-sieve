"""Isolated adapter tests. No API requests, model downloads or model inference."""
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from tokopt.bench import CommandRunner, OllamaRunner, OpenAIRunner, AnthropicRunner, results_jsonl, CaseResult, run_bench, summarize
from tokopt.counters import AnthropicCounter, CounterUnavailable, HFCounter
from tokopt.optimizer import optimize
from tokopt.strategies.translate import NLLBTranslator


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_api_runner_payload_comparable_and_bounded(monkeypatch, backend):
    calls = []
    def create(**kw):
        calls.append(kw)
        return NS(choices=[NS(message=NS(content="42"))], content=[NS(type="text", text="42")])
    client = NS(chat=NS(completions=NS(create=create)), messages=NS(create=create))
    monkeypatch.setitem(sys.modules, backend, NS(**{("OpenAI" if backend == "openai" else "Anthropic"): lambda **kw: client}))
    runner = (OpenAIRunner if backend == "openai" else AnthropicRunner)("test-model")
    assert runner.run("original") == runner.run("optimized") == "42"
    assert {k: v for k, v in calls[0].items() if k != "messages"} == {k: v for k, v in calls[1].items() if k != "messages"}
    assert calls[0]["temperature"] == 0


@pytest.mark.parametrize("payload", [{"response": "42"}, {"response": None}, {"error": "bad"}, []])
def test_ollama_response_contract(monkeypatch, payload):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    if payload == {"response": "42"}:
        assert OllamaRunner("test").run("prompt") == "42"
    else:
        with pytest.raises((ValueError, AttributeError)):
            OllamaRunner("test").run("prompt")


def test_anthropic_counter_keeps_actual_request_overhead(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-placeholder")
    client = NS(messages=NS(count_tokens=lambda **kw: NS(input_tokens=17)))
    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=lambda: client))
    assert AnthropicCounter("test").count("hello") == 17


def test_hf_loading_error_is_counter_unavailable(monkeypatch):
    def fail(*a, **kw): raise OSError("offline")
    monkeypatch.setitem(sys.modules, "transformers", NS(AutoTokenizer=NS(from_pretrained=fail)))
    with pytest.raises(CounterUnavailable):
        HFCounter("test")


def test_nllb_reuses_model_between_languages(monkeypatch):
    loads, calls = [], []
    def pipeline(*args, **kw):
        loads.append(kw)
        def run(text, **options):
            calls.append(options)
            return [{"translation_text": "translated"}]
        return run
    monkeypatch.setitem(sys.modules, "transformers", NS(pipeline=pipeline))
    tr = NLLBTranslator()
    tr.translate("one", "en", "fa")
    tr.translate("two", "fa", "en")
    assert len(loads) == 1
    assert calls[0]["tgt_lang"] == "pes_Arab" and calls[1]["tgt_lang"] == "eng_Latn"


def test_command_timeout_is_enforced():
    runner = CommandRunner(f'"{sys.executable}" -c "import time;time.sleep(0.2)"', timeout=0.01)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run("input")


def test_raw_answers_are_opt_in():
    row = CaseResult("x", "original", 1, 1, 0, True, True, "PRIVATE_SENTINEL", "PRIVATE_SENTINEL")
    assert "PRIVATE_SENTINEL" not in results_jsonl([row])
    assert "PRIVATE_SENTINEL" in results_jsonl([row], include_answers=True)


def test_failed_optimized_run_keeps_original_grade(counter):
    class Partial:
        calls = 0
        def run(self, prompt):
            self.calls += 1
            if self.calls == 2: raise RuntimeError("PRIVATE_SENTINEL")
            return "42"
    r, = run_bench([{"id": "x", "prompt": "text", "expected": "42", "match": "exact"}], Partial(), {"counter": counter})
    assert r.original_correct and not r.optimized_correct
    assert r.original_answer == "42" and "PRIVATE_SENTINEL" not in r.error


def test_configured_token_limit_cannot_silently_overflow(counter):
    with pytest.raises(ValueError, match="max_tokens"):
        optimize("long prose " * 100, counter=counter, max_tokens=1)


def test_critical_identifier_is_not_substring():
    from tokopt.check import preservation
    assert not preservation("use user_id", "use other_user_id").ok


def test_wrapper_counted_before_translation_acceptance(counter, translator):
    result = optimize("ordinary explanatory prose " * 10, counter=counter, langs=["xx"], translator=translator)
    candidate = next(c for c in result.candidates if c.name == "translate:xx")
    assert candidate.tokens == counter.count(candidate.text)
    assert candidate.text.endswith("(Answer in English.)")


def test_lingua_backend_failure_is_rejected(monkeypatch, counter):
    from tokopt.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor, "available", staticmethod(lambda: (True, "")))
    def fail(self, text): raise OSError("PRIVATE_SENTINEL")
    monkeypatch.setattr(LinguaCompressor, "apply", fail)
    result = optimize("prose " * 100, counter=counter, strategies=["lingua"])
    c = next(c for c in result.candidates if c.name.startswith("lingua@"))
    assert not c.passed and "PRIVATE_SENTINEL" not in result.to_json()


def test_lingua_protects_code_and_detects_changed_instructions(monkeypatch, counter):
    from tokopt.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor, "available", staticmethod(lambda: (True, "")))
    monkeypatch.setattr(LinguaCompressor, "compress_one", lambda self, text: text.replace("not ", ""))
    text = "The deployment must not delete data. " * 15 + "\n\n```sh\necho safe\n```"
    result = optimize(text, counter=counter, strategies=["lingua"], min_savings=0)
    c = next(c for c in result.candidates if c.name.startswith("lingua@"))
    assert "```sh\necho safe\n```" in c.text
    assert not c.passed


def test_invalid_utf8_cli_is_rejected(tmp_path):
    p = tmp_path / "invalid.txt"
    p.write_bytes(b"private\xffcontent")
    result = subprocess.run([sys.executable, "-m", "tokopt", "compress", str(p), "-c", "heuristic"], capture_output=True)
    assert result.returncode != 0
    assert b"private" not in result.stderr and not result.stdout


def test_empty_and_malformed_runner_answers_are_failures(counter):
    class Broken:
        def run(self, prompt): return {"answer": "42"}
    row, = run_bench([{"id": "x", "prompt": "text", "expected": "42", "match": "exact"}], Broken(), {"counter": counter})
    assert not row.original_correct and not row.optimized_correct
    assert summarize([row])["errors"] == 1


def test_roundtrip_is_necessary_but_not_proof_of_meaning(counter):
    # A correlated mistranslation can back-translate perfectly. This test
    # documents a known limitation rather than asserting semantic equivalence.
    original = "Alex lends Sam the book. " * 30
    class Correlated:
        def translate(self, text, src, tgt):
            return original if tgt == "en" else "反転"
    result = optimize(original, counter=counter, langs=["ja"], translator=Correlated())
    assert next(c for c in result.candidates if c.name == "translate:ja").passed
