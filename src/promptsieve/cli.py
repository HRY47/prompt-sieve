"""PromptSieve command-line interface.

  prompt-sieve count FILE [-c SPEC ...]          token counts under one or more tokenizers
  prompt-sieve compress FILE [-s logs|lingua|auto] [-o OUT]
                                           write the smallest passing version
  prompt-sieve experiment FILE -l fa,zh,ja       multilingual token experiment (table)
  prompt-sieve bench CASES.jsonl -r RUNNER       same-answer accuracy benchmark
  prompt-sieve setup-languages fa zh ja          install Argos language packs
  prompt-sieve doctor                            show which backends are available
  prompt-sieve run -- COMMAND ...                run a command, shorten noisy output, keep exit status
  prompt-sieve install codex|claude-code         install the agent skill (--scope user|project)
  prompt-sieve hook claude-code                  Claude Code PostToolUse hook (reads JSON on stdin)

FILE may be "-" for stdin, e.g.  pytest 2>&1 | prompt-sieve compress -
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import __version__
from .counters import CounterUnavailable, HeuristicCounter, get_counter
from .errors import UsageError, exception_summary


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse embeds invalid values, paths and unknown arguments in message.
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command-line arguments; "
                     f"run '{self.prog} --help' for valid options and values\n")


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _counter(spec):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = get_counter(spec)
        if w:
            if spec is None and isinstance(c, HeuristicCounter):
                print("warning: default tokenizer unavailable; falling back to the rough "
                      "heuristic counter; run 'prompt-sieve doctor' for dependency checks", file=sys.stderr)
            else:
                print("warning: counter initialization emitted warnings; details withheld; "
                      "run 'prompt-sieve doctor' for dependency checks", file=sys.stderr)
        return c
    except CounterUnavailable as e:
        sys.exit(exception_summary("error: counter unavailable", e) +
                 "; check the counter specification, optional dependencies and provider environment")


def _opt_kwargs(a) -> dict:
    kw = dict(
        counter=_counter(a.counter),
        mode=a.mode, timeout_seconds=a.timeout, cache=not a.no_cache,
        ml_min_tokens=a.ml_min_tokens, ml_min_saved_tokens=a.ml_min_saved_tokens,
        allow_mixed_lingua=a.allow_mixed_lingua,
        allow_sensitive_lingua=a.allow_sensitive_lingua,
        min_savings=a.min_savings,
        min_preservation=a.min_preservation,
        max_tokens=a.max_tokens,
        lingua_rate=a.rate,
        lingua_model=a.lingua_model,
        log_options={"strip_timestamps": a.strip_timestamps, "context": a.context},
    )
    if getattr(a, "langs", None):
        kw["translation_backend"] = a.backend
        kw.update(langs=[l.strip() for l in a.langs.split(",")], src_lang=a.src,
                  min_chrf=a.min_chrf, per_segment=a.per_segment,
                  roundtrip=not a.no_roundtrip, reply_hint=not a.no_reply_hint)
    return kw


def _add_common(p, strategies=True):
    p.add_argument("-c", "--counter", default=None,
                   help="tokenizer spec: tiktoken:o200k_base | openai:gpt-4o | anthropic:MODEL | "
                        "hf:MODEL_ID | heuristic (default: tiktoken o200k_base)")
    if strategies:
        p.add_argument("-s", "--strategy", action="append", choices=["auto", "logs", "lingua", "normalize"],
                       help="repeatable; default auto (= logs when the input looks like a log)")
    p.add_argument("--mode", choices=["fast", "balanced", "full"], default="fast",
                   help="fast: rules/cost gates; balanced: cost-gated ML for auto; full: explicitly requested experiments")
    p.add_argument("--timeout", type=float, help="hard optimization deadline in seconds, including worker startup")
    p.add_argument("--allow-sensitive-lingua", action="store_true", help="research override for unit/math contexts; may change answers")
    p.add_argument("--allow-mixed-lingua", action="store_true", help="research override for mixed-language pruning; may change answers")
    p.add_argument("--no-cache", action="store_true", help="disable bounded process-local rule-result cache")
    p.add_argument("--ml-min-tokens", type=int, default=1024, help="minimum source tokens for cost-gated ML")
    p.add_argument("--ml-min-saved-tokens", type=int, default=256, help="minimum optimistic token reduction for cost-gated ML")
    p.add_argument("--min-savings", type=float, default=0.10,
                   help="lossy candidates must save at least this fraction (default 0.10)")
    p.add_argument("--max-tokens", type=int, help="fail if no accepted final text fits this token limit")
    p.add_argument("--min-preservation", type=float, default=1.0,
                   help="fraction of critical items (numbers, paths, code...) that must survive")
    p.add_argument("--rate", type=float, default=0.5, help="LLMLingua keep-rate (default 0.5)")
    p.add_argument("--lingua-model", default="small", help="small | large | HF model id")
    p.add_argument("--strip-timestamps", action="store_true", help="logs: drop leading timestamps")
    p.add_argument("--context", type=int, default=3, help="logs: lines kept around each error")


def _add_translation(p, required=False):
    p.add_argument("-l", "--langs", required=required, help="comma-separated target languages, e.g. fa,zh,ja")
    p.add_argument("--src", default="en", help="source language (default en)")
    p.add_argument("--backend", default="argos", choices=["argos", "nllb"])
    p.add_argument("--min-chrf", type=float, default=55.0, help="round-trip chrF needed to accept a translation")
    p.add_argument("--per-segment", action="store_true", help="also try choosing a language per paragraph")
    p.add_argument("--no-roundtrip", action="store_true", help="skip the round-trip check (translations then never pass)")
    p.add_argument("--no-reply-hint", action="store_true", help="don't append '(Answer in English.)'")


def cmd_count(a):
    text = _read(a.file)
    specs = a.counter or [None]
    for s in specs:
        c = _counter(s)
        print(f"{c.count(text):>10,}  {c.name}")


def cmd_compress(a):
    from .optimizer import optimize

    if a.output and a.file != "-":
        from pathlib import Path
        source, dest = Path(a.file), Path(a.output)
        if source.resolve() == dest.resolve() or (dest.exists() and source.samefile(dest)):
            raise UsageError("output must differ from input; keep the original prompt")
    text = _read(a.file)
    kw = _opt_kwargs(a)
    res = optimize(text, strategies=a.strategy or ["auto"], **kw)
    out = res.best.text
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    if a.json:
        print(res.to_json(), file=sys.stderr)
    elif not a.quiet:
        print(f"\n{res.table()}\n-> {res.best.name}: {res.original_tokens} -> {res.best.tokens} tokens "
              f"({res.best.savings:+.1%})", file=sys.stderr)

    if not res.best.passed:
        raise UsageError("original fallback is not verified within --max-tokens; do not send it without checking")


def cmd_experiment(a):
    from .optimizer import optimize

    text = _read(a.file)
    kw = _opt_kwargs(a)
    res = optimize(text, strategies=a.strategy or ["auto"], measure_all=True, **kw)
    if a.json:
        print(res.to_json(include_text=a.show_text))
        return
    print(res.table())
    print(f"\nwinner: {res.best.name} ({res.best.tokens} tokens, {res.best.savings:+.1%})")
    extra = [_counter(s) for s in (a.also_count or [])]
    if extra:
        print("\nsame candidates under other tokenizers:")
        for c in extra:
            row = ", ".join(f"{x.name}={c.count(x.text):,}" for x in res.candidates)
            print(f"  {c.name}: {row}")
    if a.show_text:
        for c in res.candidates:
            print(f"\n===== {c.name} ({c.tokens:,} tokens) =====\n{c.text}")


def cmd_bench(a):
    from .bench import get_runner, load_cases, results_jsonl, run_bench, summarize

    cases = load_cases(a.cases, allow_external_files=a.allow_external_prompt_files)
    if a.counter is None:
        kind, _, model = a.runner.partition(":")
        if kind == "openai":
            a.counter = f"openai:{model or 'gpt-4o-mini'}"
        elif kind == "anthropic":
            a.counter = f"anthropic:{model or 'claude-haiku-5-5'}"
        else:
            raise UsageError("bench requires an explicit --counter for command and Ollama runners")
    try:
        runner = get_runner(a.runner)
    except UsageError:
        raise
    except Exception as e:
        sys.exit(exception_summary("error: runner unavailable", e) +
                 "; check the runner type, optional dependencies and provider environment")
    kw = _opt_kwargs(a)
    kw["strategies"] = a.strategy or ["auto"]
    results = run_bench(cases, runner, kw, progress=lambda s: print(s, file=sys.stderr))
    summary = summarize(results)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(results_jsonl(results, include_answers=a.include_answers))
    print(json.dumps(summary, indent=2))


def cmd_setup_languages(a):
    from .strategies.translate import install_argos_languages
    try:
        for line in install_argos_languages(a.langs, pivot=a.src):
            print(line)
    except ImportError:
        sys.exit("error: pip install 'prompt-sieve[argos]' first")


def cmd_doctor(a):
    print("Dependency/encoding checks only; model inference, language packs and APIs are not verified.")
    checks = {
        "tiktoken": lambda: get_counter("tiktoken:o200k_base"),
        "anthropic SDK (credentials/API not checked)": lambda: __import__("anthropic"),
        "llmlingua": lambda: __import__("llmlingua"),
        "Argos translation module (packs/inference not checked)": lambda: __import__("argostranslate.translate"),
        "transformers (hf counter / nllb)": lambda: __import__("transformers"),
    }
    for name, fn in checks.items():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fn()
            print(f"  available {name}")
            if caught:
                print("    initialization warnings withheld; check optional dependency versions")
        except Exception as e:
            print("  " + exception_summary(f"missing {name}", e) +
                  "; check optional dependencies, cached data and provider environment")


def cmd_run(a):
    from .integrations.run import run_command
    if a.min_chars < 0:
        raise UsageError("--min-chars must be nonnegative")
    sys.exit(run_command(a.command, min_chars=a.min_chars, save=not a.no_save))


def cmd_install(a):
    from .integrations.install import install
    for line in install(a.agent, scope=a.scope, uninstall=a.uninstall, force=a.force, dry_run=a.dry_run):
        print(line)


def cmd_hook(a):
    from .integrations import claude_hook
    sys.exit(claude_hook.main())


def main(argv=None):
    p = _ArgumentParser(prog="prompt-sieve", description="PromptSieve: token-aware prompt optimizer")
    p.add_argument("--version", action="version", version=f"PromptSieve {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("count", help="count tokens")
    s.add_argument("file")
    s.add_argument("-c", "--counter", action="append", help="repeatable")
    s.set_defaults(fn=cmd_count)

    s = sub.add_parser("compress", help="output the smallest version that passes all checks")
    s.add_argument("file")
    s.add_argument("-o", "--output")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--json", action="store_true", help="report as JSON on stderr")
    _add_common(s)
    _add_translation(s)
    s.set_defaults(fn=cmd_compress)

    s = sub.add_parser("experiment", help="compare languages/strategies side by side")
    s.add_argument("file")
    s.add_argument("--also-count", action="append", help="extra tokenizer specs to report")
    s.add_argument("--show-text", action="store_true")
    s.add_argument("--json", action="store_true")
    _add_common(s)
    _add_translation(s, required=False)
    s.set_defaults(fn=cmd_experiment, mode="full")

    s = sub.add_parser("bench", help="check optimized prompts still give the right answers")
    s.add_argument("cases", help="JSONL cases file")
    s.add_argument("-r", "--runner", required=True,
                   help="anthropic:MODEL | openai:MODEL | ollama:MODEL | 'cmd:SHELL COMMAND'")
    s.add_argument("-o", "--output", help="write per-case results (JSONL)")
    s.add_argument("--include-answers", action="store_true", help="include raw answers in report (may contain sensitive input)")
    s.add_argument("--allow-external-prompt-files", action="store_true",
                   help="let prompt_file entries point outside the cases directory (only for trusted case files)")
    _add_common(s)
    _add_translation(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("setup-languages", help="install Argos translation packs")
    s.add_argument("langs", nargs="+")
    s.add_argument("--src", default="en")
    s.set_defaults(fn=cmd_setup_languages)

    s = sub.add_parser("doctor", help="show which optional backends work")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("run", help="run a command and shorten noisy output (exit status preserved)")
    s.add_argument("--min-chars", type=int, default=6000,
                   help="only consider output at least this long (default 6000)")
    s.add_argument("--no-save", action="store_true", help="do not keep a private copy of the full output")
    s.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("install", help="install the PromptSieve skill for Codex or Claude Code")
    s.add_argument("agent", choices=["codex", "claude-code"])
    s.add_argument("--scope", choices=["user", "project"], default="user")
    s.add_argument("--uninstall", action="store_true")
    s.add_argument("--force", action="store_true", help="replace a skill folder this tool did not create")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_install)

    s = sub.add_parser("hook", help="agent hook entry point (used by the Claude Code plugin)")
    s.add_argument("agent", choices=["claude-code"])
    s.set_defaults(fn=cmd_hook)

    a = p.parse_args(argv)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a.fn(a)
        if caught:
            print("warning: command emitted warnings; details withheld; "
                  "check optional dependency versions", file=sys.stderr)
    except UsageError as e:
        p.exit(2, f"error: {e}\n")
    except Exception as e:
        # SDK and import errors also reach this boundary. Never format their
        # messages, attributes, chained exceptions or tracebacks into output.
        p.exit(2, exception_summary("error: command failed", e) +
               "; check input files, configuration and backend availability\n")


if __name__ == "__main__":
    main()
