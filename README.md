# tokopt: experimental token-aware prompt optimizer

`tokopt` builds smaller versions of text, counts them with a selected counter,
and selects the smallest candidate that passes its checks. Core strategies work
without dependencies. Translation and model-based pruning are experimental.

**Preservation checks are filters, not proof of equivalent meaning.** Use raw
source for editing, configuration, audit trails, and decisions requiring every
line. See [the independent review](review/REVIEW.md) for measured results and
integration limits. Phase 3 adds a conservative fast default, process-isolated
optimization deadlines, a bounded private result cache, and a 120-case local
benchmark. The Persian regression was triggered even by removing one space;
that normalization change is fixed. Experimental ML still requires task-specific
validation. General answer preservation and target-model memory savings remain
unproven.

## Install

Python 3.9+ is declared; the independent local review ran on Windows/Python
3.13.7; real optional-model testing now also runs on WSL/Python 3.12.3.
The CI matrix covers Python 3.9, 3.11, 3.13 and Windows, but has not yet
run on GitHub. Use Python 3.11 for optional ML backends until their complete
dependency stacks have been verified on your platform.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[openai,dev]'
pytest -v
tokopt doctor
```

Extras:

| Extra | Provides |
|---|---|
| `openai` | tiktoken text counter and OpenAI benchmark SDK |
| `anthropic` | Anthropic request counter and benchmark SDK |
| `hf` | Hugging Face tokenizer adapters |
| `lingua` | LLMLingua-2 and Transformers 4.x |
| `argos` | Argos translation; language packs installed separately |
| `nllb` | NLLB translation using Transformers 4.x, torch, SentencePiece |
| `dev` | pytest |
| `all` | all optional integrations above |

```bash
python -m pip install -e '.[openai,argos,dev]'
tokopt setup-languages fa zh ja  # installs both directions; requires downloads
```

First-use tokenizer/model downloads require network access. Dependencies have
version ranges, not a reproducible ML lockfile. A successful `doctor` import
check does not establish model inference, API access, or installed language packs.

## Fast defaults, deadlines, and recoverable originals

The CLI defaults to `--mode fast`: normalization and detected log compression,
with no implicit ML inference. Small cosmetic savings are ignored unless you
explicitly request normalization. `--mode balanced` can additionally try
LLMLingua for large `auto` inputs. Cost-gated ML requires at least 1,024 source
tokens and an optimistic 256-token reduction, configurable with
`--ml-min-tokens` and `--ml-min-saved-tokens`. These thresholds are a policy,
not a prediction of latency payback. Accepted rule savings short-circuit ML.
Translation backends load only after the cost gate.

```bash
tokopt compress examples/pytest_run.log --mode fast --timeout 5 -o compact.log
# Explicit experiments bypass cost gates, with preservation checks still active:
tokopt experiment examples/prompt.md -l fa,zh,ja --mode full
tokopt compress prompt.md -s lingua --mode full --timeout 10
```

The Python API selects fast mode for ordinary `auto`/rule calls. For backward
compatibility, explicitly requesting `lingua` or translation without `mode`
selects full mode; pass `mode='fast'` to apply cost gates to those requests too.
Full mode bypasses cost gates, **not preservation checks**. Mixed-script prose
and recognized unit/math contexts skip LLMLingua by default even in full mode:
unchanged factual paragraphs can still produce wrong answers after surrounding
context changes. `--allow-mixed-lingua` / `--allow-sensitive-lingua` (Python:
`allow_mixed_lingua=True`, `allow_sensitive_lingua=True`) are research overrides
and can reproduce unsafe behavior. The primitive compressor remains available.

`--timeout SECONDS` / `timeout_seconds=SECONDS` runs optimization in a spawned
worker, avoiding a fork of live torch/OpenMP state. The worker is terminated on
timeout; WSL process groups are cleaned up. The exact original is returned,
with `timed_out=True` and a warning. Counts are `null` if counting did not finish;
no estimate is silently substituted. An original not verified within
`max_tokens` is marked unaccepted; the CLI writes the fallback but exits nonzero,
and callers must not send it without checking. Deadlines include worker startup
and counting; OS scheduling, serialization and cleanup add overhead, so this is
not a hard real-time guarantee. Slow/non-picklable custom adapters may fall back.
CLI argument/counter preparation and imports before the API call are outside its
timer. Spawned workers have cold model caches and can consume the budget loading
models. Use a main guard in scripts that call the timed Python API.

`result.original_text` keeps an independent source snapshot even if the caller
edits candidate objects. JSON includes it only with `include_text=True`.
The CLI refuses to overwrite its source file, including resolved aliases.

The process-local result cache defaults to 128 entries, 8 MiB of serialized
values, and a 300-second TTL; entry caps also bound key/metadata overhead. It
uses LRU eviction and defensive result copies, and writes no prompts to disk.
It caches deterministic local rule results, including counter identity, input,
and all optimization settings. Loaded HF tokenizer instances have separate
identities. Custom counters must provide a stable, versioned `cache_key()` to
opt in. ML/translation results, opaque/remote counters, timed runs, failures,
and timeouts bypass this cache. Changing settings, namespace, tokenizer instance,
TTL expiry, or `clear()` invalidates results. This does not bound upstream model
or tokenizer caches.

```python
from tokopt import optimize, get_counter, OptimizationCache
cache = OptimizationCache(max_entries=64, max_bytes=4 * 1024 * 1024,
                          ttl_seconds=120)
result = optimize(text, counter=get_counter('openai:gpt-4o'),
                  mode='fast', cache=cache, cache_namespace='my-app-v1')
assert result.original_text == text
# cache=False disables result caching; cache.clear() removes retained prompts.
```

## Strategies and acceptance

| Strategy | Behavior | Limits |
|---|---|---|
| `normalize` | Removes terminal escapes, normalizes CRLF, cleans some trailing whitespace and repeated blank lines | Formatting changes; fenced/structured content is protected; general CR updates are retained |
| `logs` | Removes recognized passing tests/install progress, collapses repeated nondiagnostic lines, focuses long logs around recognized diagnostics | Lossy: unknown lines can matter; default keeps all Java frames |
| `lingua` | Local LLMLingua-2 word pruning; skips protected segments and asks the model to retain digits and critical words | Local model required; critical item gate can reject damaged output |
| `translate:XX` (Python API), `-l XX` (CLI) | Local prose translation with protected literals and a round-trip check | Experimental; back-translation does not prove correctness |
| `mixed-per-paragraph` | Tries languages independently per unstructured paragraph | May lose terminology consistency and context across paragraphs |

`auto` tries logs only if the input resembles a log; normalization is always
considered. It does not automatically enable ML or translation.

Lossy candidates must save at least `--min-savings` (default 0.10). Translation
counts the final text including `(Answer in English.)` before acceptance. Its
checks require protected literals/numbers, round-trip character similarity, and
critical words/items in the back-translation. Argos translates the prose between
literal spans because its models can damage placeholders; fragmented context
can lower translation quality. NLLB retains the placeholder path. `--no-roundtrip` produces only
rejected translation experiments. `--min-preservation` can explicitly relax
critical-item retention; keep its default 1.0 for normal use.

The character score is a simplified, case-insensitive chrF-like metric, not
standardized SacreBLEU chrF. Its default threshold has not been quality-calibrated.

Negated clauses, recognized number/unit associations and elementary math
expressions also have conservative same-language checks. Numbers with
signs/exponents, code fences, inline code, recognized paths, URLs,
quoted literals, and identifiers are checked. Recognition is pattern-based;
this is not a parser for every language or document type. Lists, headings,
tables, block quotes and indented code are conservatively left untranslated.
No check guarantees that subjects, units, ordering or intent are preserved.

If no smaller acceptable candidate exists, the original is returned.
`--max-tokens N` fails when no accepted text fits N. This limit applies to the
counter's scope; callers must count their final serialized request, including
system messages, tools and application wrappers, before sending it.

## Token counters

| Spec | Count scope |
|---|---|
| `tiktoken:o200k_base` (default attempt) | Exact encoded text length under this encoding |
| `openai:gpt-4o` | Text encoding selected by tiktoken's model registry; unknown models fail |
| `hf:Qwen/Qwen2.5-7B-Instruct` | Encoded text without special tokens; excludes chat template overhead |
| `anthropic:MODEL` | Remote single-user Messages request count, including API overhead; sends text to Anthropic |
| `heuristic` | **Rough estimate**, script-aware character ratios; unsuitable for enforcing a real model context limit |

The default falls back to a visibly named heuristic counter with a warning if
tiktoken cannot load. Specify `-c` explicitly to require a real tokenizer and
fail instead of estimating. These counts are not interchangeable. OpenAI/HF
text counts exclude chat protocol overhead; Anthropic counts a request.

## Commands

```bash
pytest 2>&1 | tokopt compress - -q
# Bash: use `set -o pipefail` if the test exit status must propagate.
tokopt compress examples/pytest_run.log -s logs -o compact.log
tokopt count examples/prompt.md -c tiktoken:o200k_base -c heuristic
tokopt compress examples/prompt.md -s lingua --mode full --rate 0.5
tokopt experiment examples/prompt.md -l fa,zh,ja --per-segment --show-text
```

Text files must be UTF-8; malformed bytes are rejected. `-` reads stdin.
`compress --json` writes metadata to stderr and optimized text to stdout.
`experiment --json --show-text` deliberately includes input-derived text.

```python
from tokopt import optimize, get_counter
result = optimize(text, counter=get_counter('openai:gpt-4o'),
                  strategies=['logs'], max_tokens=4000)
print(result.best.text)
```

## Benchmark correctness separately from agreement

```jsonl
{"id":"arithmetic","prompt":"Compute six times seven.","expected":"42","match":"exact"}
{"id":"failure","prompt_file":"pytest_run.log","question":"Which test failed?","expected":"test_checkout_total","match":"contains"}
{"id":"agreement-only","prompt_file":"prompt.md","match":"same_as_original"}
```

Paths are relative to the cases file. Case files are trusted local configuration:
`prompt_file` can read any file your account can read. Review downloaded cases
before running them, especially with a remote backend.

Runners: `openai:MODEL`, `anthropic:MODEL`, `ollama:MODEL`, `cmd:SHELL COMMAND`.
API runners can incur charges; invoke them only when you intend to pay.
For a deterministic local test, use a script reading UTF-8 stdin:

```bash
tokopt bench examples/cases.jsonl -r 'cmd:python my_local_runner.py' -c heuristic -o results.jsonl
```

Command/Ollama runners require an explicit counter. OpenAI/Anthropic CLI runners
default to the counter for their requested model. `question` is appended after
optimization. Token counts include this question and its delimiters, and the
final savings gate is rechecked. Both arms use the same runner and generation
settings; API runs remain nondeterministic and always run baseline first.

`contains`, normalized `exact`, and `regex` grade independently specified answers.
These rules are limited: `contains` may accept a negated answer containing the
expected phrase, and regex is only as good as its author. `same_as_original`
measures agreement (normalized equality or chrF >= 70), never accuracy.
Summaries report accuracy on reference-graded cases only, agreement separately,
and normalized answer changes separately from grade changes. Summaries now
include `paired_regressions` and `paired_improvements` case IDs. Both wrong and
optimized-better outcomes can therefore be represented honestly.

Reports include per-arm latency and output counts under the selected counter.
They do not yet collect provider billing usage, chat overhead for offline text
counters, repeated-trial confidence intervals, or target process RAM/VRAM.
Raw answers are omitted from saved reports unless `--include-answers` is set.
No report should be treated as safe to publish without checking its metadata.

## Security and trust

Input is data, including any embedded instructions. Compression does not remove
prompt injection. Preserve your application's system/user/tool boundaries.
The shell runner intentionally executes its configured command with your account
permissions. Never derive its command from document content, benchmark cases, or
model output. Prompt text is passed on stdin and never interpolated into a command.
Timeouts attempt to terminate the subprocess tree. Output is buffered in memory;
run untrusted workloads inside a separate OS sandbox with resource limits.

API runners/counters transmit text externally. Model/encoding downloads and
Argos package installation trust upstream repositories and their dependencies;
models are not pinned to immutable revisions. Hugging Face custom tokenizer
code is disabled, but that is not a sandbox for dependencies or weights.
NLLB's default weights have a separate noncommercial license; see the
[model card](https://huggingface.co/facebook/nllb-200-distilled-600M).
See [SECURITY.md](SECURITY.md) for disclosure and deployment limits.

## Resources and measurements

Fewer tokens may reduce dynamic KV-cache allocation for some local runtimes.
Weights do not shrink. Reserved caches, batching and runtime allocation policies
can prevent observable RAM/VRAM savings. Compression and translation themselves
consume CPU memory and time. No proportional memory saving is guaranteed.

[review/measurements.json](review/measurements.json) retains the historical
Windows estimates and adds a separate `phase2` section with real WSL tokenizer,
translation, LLMLingua, RAM, and independently graded Qwen results. The raw
`review/phase2-*.json` files distinguish accepted results from rejected model
output. See the current review for the answer regression and latency overhead.
Reproduce the historical rule measurements with:

```bash
python scripts/review_measurements.py
```

Reproduce the real local measurements (downloads model files on first use):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase2_measurements.py tokenizers
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase2_measurements.py translation
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase2_measurements.py lingua
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase2_measurements.py benchmark
# Current Phase 3 evidence (cached local models; no paid API):
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase3_ablation.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase3_deadlines.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase3_benchmark.py
```

## Tests and real integrations

```bash
python -m pip install -e '.[dev]'
pytest -v  # offline unit/contract tests; integration tests skip explicitly
# Bash; PowerShell uses $env:NAME='1' before invoking pytest:
TOKOPT_TEST_TOKENIZERS=1 pytest tests/test_integrations.py -k tiktoken -v
TOKOPT_TEST_ARGOS=1 pytest tests/test_integrations.py -k argos -v
TOKOPT_TEST_LINGUA=1 pytest tests/test_integrations.py -k lingua -v
TOKOPT_TEST_HF=1 pytest tests/test_integrations.py -k real_hf -v
TOKOPT_TEST_NLLB=1 pytest tests/test_integrations.py -k nllb -v
TOKOPT_TEST_PHASE3_MODEL=1 pytest tests/test_phase3_integrations.py -k persian -v
```

Install the relevant extras and language packs first. Explicitly enabled
integrations fail if their dependencies are missing. Integration smoke tests do
not establish semantic quality. No CI job calls paid APIs.

## Claude Code skill

Copy `skills/token-optimizer` into `~/.claude/skills/` if desired. The skill tells
Claude Code to pipe large tool output through the CLI. It does not install or
invoke a hidden hook, intercept already-read text, or optimize every prompt.
The integration was reviewed as instructions, not exercised in Claude Code.

## License and prior art

The package is MIT licensed; optional dependencies and model weights retain
their own licenses. [LLMLingua](https://github.com/microsoft/LLMLingua) supplies
the optional pruning backend. Related projects include
[token-compact](https://github.com/theosib/token-compact) and
[PromptMin](https://github.com/DelvyG/promptmin).
