"""Synthetic-only checks for CLI diagnostic disclosure; no network or API calls."""
import builtins
import io
import json
from types import SimpleNamespace
import warnings

import pytest

from promptsieve import cli
from promptsieve.counters import CounterUnavailable, HeuristicCounter


class ProviderError(Exception):
    pass


@pytest.fixture(params=[
    "sk-" + "proj-" + "SYNTHETIC_ONLY_" * 3,
    "gh" + "p_" + "SYNTHETIC_ONLY_" * 3,
    "Authorization: Bearer synthetic-authorization-value",
    "https://synthetic-user:synthetic-password@example.invalid/private",
    "/private/synthetic-person/credentials.env",
    "sensitive-input-sentinel\nsynthetic traceback details",
])
def sensitive(request):
    return request.param


def _failure(capsys, callback):
    with pytest.raises(SystemExit) as caught:
        callback()
    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    if isinstance(caught.value.code, str):
        diagnostic += caught.value.code
    assert "Traceback (most recent call last)" not in diagnostic
    return diagnostic


def _quiet_imports(monkeypatch):
    original = builtins.__import__
    def fake(name, *args, **kwargs):
        if name in ("anthropic", "llmlingua", "argostranslate.translate", "transformers"):
            return SimpleNamespace()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_runner_initialization_withholds_credentials(monkeypatch, capsys, caplog, sensitive):
    monkeypatch.setattr("promptsieve.bench.load_cases", lambda *a, **kw: [])
    def fail(spec):
        raise ProviderError(sensitive)
    monkeypatch.setattr("promptsieve.bench.get_runner", fail)
    diagnostic = _failure(capsys, lambda: cli.main([
        "bench", "/private/synthetic-cases.jsonl", "-r", sensitive, "-c", "heuristic",
    ]))
    assert sensitive not in diagnostic
    assert "runner unavailable" in diagnostic and "ProviderError" in diagnostic
    assert not caplog.records


def test_counter_initialization_withholds_spec_and_message(monkeypatch, capsys, sensitive):
    def fail(spec):
        raise CounterUnavailable(sensitive)
    monkeypatch.setattr(cli, "get_counter", fail)
    monkeypatch.setattr("sys.stdin", io.StringIO("ordinary input"))
    diagnostic = _failure(capsys, lambda: cli.main(["count", "-", "-c", "hf:" + sensitive]))
    assert sensitive not in diagnostic
    assert "counter unavailable" in diagnostic and "CounterUnavailable" in diagnostic
    assert "optional dependencies" in diagnostic


def test_counter_fallback_warning_withholds_details(monkeypatch, capsys, sensitive):
    def fallback(spec):
        warnings.warn(sensitive)
        return HeuristicCounter()
    monkeypatch.setattr(cli, "get_counter", fallback)
    assert isinstance(cli._counter(None), HeuristicCounter)
    captured = capsys.readouterr()
    assert sensitive not in captured.out + captured.err
    assert "rough heuristic counter" in captured.err


def test_doctor_withholds_errors_and_warnings(monkeypatch, capsys, caplog, sensitive):
    _quiet_imports(monkeypatch)
    def fail(spec):
        warnings.warn(sensitive)
        raise ProviderError(sensitive)
    monkeypatch.setattr(cli, "get_counter", fail)
    cli.main(["doctor"])
    captured = capsys.readouterr()
    assert sensitive not in captured.out + captured.err
    assert "missing tiktoken (ProviderError)" in captured.out
    assert "provider environment" in captured.out
    assert not caplog.records


def test_doctor_withholds_warning_on_success(monkeypatch, capsys, sensitive):
    _quiet_imports(monkeypatch)
    def warn(spec):
        warnings.warn(sensitive)
        return object()
    monkeypatch.setattr(cli, "get_counter", warn)
    cli.main(["doctor"])
    captured = capsys.readouterr()
    assert sensitive not in captured.out + captured.err
    assert "available tiktoken" in captured.out
    assert "warnings withheld" in captured.out


def test_unexpected_sdk_failure_has_no_traceback(monkeypatch, capsys, caplog, sensitive):
    def fail(text):
        raise ProviderError(sensitive)
    monkeypatch.setattr(cli, "_counter", lambda spec: SimpleNamespace(count=fail, name="test counter"))
    monkeypatch.setattr("sys.stdin", io.StringIO(sensitive))
    diagnostic = _failure(capsys, lambda: cli.main(["count", "-", "-c", "heuristic"]))
    assert sensitive not in diagnostic
    assert "command failed (ProviderError)" in diagnostic
    assert not caplog.records


@pytest.mark.parametrize("arguments", [
    lambda value: [value],
    lambda value: ["compress", "-", "--timeout", value],
    lambda value: ["compress", "-", "--strategy", value],
    lambda value: ["compress", value, "--unexpected", value],
])
def test_parser_does_not_echo_invalid_values(capsys, sensitive, arguments):
    diagnostic = _failure(capsys, lambda: cli.main(arguments(sensitive)))
    assert sensitive not in diagnostic
    assert "invalid command-line arguments" in diagnostic
    assert "--help" in diagnostic


@pytest.mark.parametrize("invalid_fields", [
    {"prompt_file": 123},
    {"prompt_file": "../private-file.txt"},
    {},
    {"prompt": []},
    {"prompt": "synthetic prompt", "question": []},
])
def test_case_validation_does_not_echo_case_id(tmp_path, capsys, sensitive, invalid_fields):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": sensitive, **invalid_fields}) + "\n", encoding="utf-8")
    diagnostic = _failure(capsys, lambda: cli.main([
        "bench", str(cases), "-r", "cmd:unused", "-c", "heuristic",
    ]))
    assert sensitive not in diagnostic
    assert "line 1:" in diagnostic


def test_invalid_match_mode_is_not_echoed(tmp_path, capsys, sensitive):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"prompt": "synthetic prompt", "match": sensitive}) + "\n", encoding="utf-8")
    diagnostic = _failure(capsys, lambda: cli.main([
        "bench", str(cases), "-r", "cmd:unused", "-c", "heuristic",
    ]))
    assert sensitive not in diagnostic
    assert "unknown match mode" in diagnostic


def test_error_message_is_never_stringified(monkeypatch, capsys):
    class UnprintableError(Exception):
        def __str__(self):
            raise AssertionError("raw exception formatting is forbidden")
    def fail(spec):
        raise UnprintableError()
    _quiet_imports(monkeypatch)
    monkeypatch.setattr(cli, "get_counter", fail)
    cli.main(["doctor"])
    assert "UnprintableError" in capsys.readouterr().out


def test_command_warnings_never_escape_on_failure_or_success(monkeypatch, capsys, sensitive):
    monkeypatch.setattr("sys.stdin", io.StringIO("ordinary input"))
    def count(text):
        warnings.warn(sensitive)
        return 1
    monkeypatch.setattr(cli, "_counter", lambda spec: SimpleNamespace(count=count, name="test counter"))
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        cli.main(["count", "-", "-c", "heuristic"])
    assert not escaped
    captured = capsys.readouterr()
    assert sensitive not in captured.out + captured.err
    assert "command emitted warnings" in captured.err

    def fail(text):
        warnings.warn(sensitive)
        raise ProviderError(sensitive)
    monkeypatch.setattr("sys.stdin", io.StringIO("ordinary input"))
    monkeypatch.setattr(cli, "_counter", lambda spec: SimpleNamespace(count=fail, name="test counter"))
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        diagnostic = _failure(capsys, lambda: cli.main(["count", "-", "-c", "heuristic"]))
    assert not escaped
    assert sensitive not in diagnostic
