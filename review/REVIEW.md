> **Note (final independent review):** the fast-mode "41.6% savings" and "no paired
> regressions" results below come from 10 log fixtures; the other 110 prompts were unchanged,
> and all 10 compressed cases were already wrong at baseline, so a regression was impossible.
> See [FINAL-REVIEW.md](FINAL-REVIEW.md) for the corrected interpretation and later fixes.

# Current review: Phase 3

**Conditionally ready for an explicitly experimental alpha with fast mode as the
default.** General answer preservation, speedup and memory-saving claims remain
unsupported. No repository was published and no paid API was called.

The Persian regression is reproducible by removing one space before a paragraph
break: unchanged factual text then yields `10` instead of `42` in pinned Qwen.
Normalization now preserves that space, and conservative ML guards avoid
mixed-script and recognized unit/math contexts. The legacy failures remain in
the evidence rather than being replaced with passing mock tests.

Final validation: **182 passed, zero failures or skips**, including 25 real
integrations. Across 120 independently answered cases, fast mode retains 69/120
exact-answer passes, reduces serialized input tokens by **41.6%**, and has no
paired regressions. The revised conservative-policy replay has no regressions
in all six arms, but the corpus informed its safeguards and is not held-out.
The first unguarded quantity tests exposed two real LLMLingua answer regressions.

The final implementation adds configurable worker deadlines, bounded private
result caching, immutable source recovery, CLI overwrite prevention, stronger
clause/unit/math checks, and code-control/CRLF preservation. A fresh-process
fast-path run processed 120 prompts in 0.052 s with
96.9 MiB peak RSS, without inference models.
The final wheel and timeout worker pass an isolated core smoke test.

Read [the complete Phase 3 assessment](PHASE3.md),
[machine summary](phase3-summary.json), [paired benchmark](phase3-benchmark.json)
and [current consolidated measurements](measurements.json). They distinguish
real initial generations, exact-input observation reuse, fresh optimization
measurements, mocks, cold startup, missing GPU measurements and remaining limits.

---

<!-- PHASE2-HISTORY -->

# Phase 2: real-world integration review

Date: 2026-11-08. **Historical Phase 2 evidence.** The Phase 1 review below is
retained as historical evidence; its missing-dependency and untested-inference
statements describe the previous Windows environment.

**Verdict: not ready for a release claiming reliable preservation of target-model
answers.** Packaging and all integrations now work locally, and actual token
savings are demonstrated. One independently graded answer regressed after an
accepted optimization. An explicitly experimental alpha is technically buildable;
it must retain the limitations and avoid semantic-preservation, speedup, or
VRAM-saving guarantees. Nothing was published and no Git repository was created.

## Environment and automated evidence

Verified Linux kernel `6.18.54.1-microsoft-standard-WSL2`, Python 3.12.3, and
`<project>/.venv/bin/python`. The installed editable
project points to this workspace. No Git metadata exists. Approximately 15 GiB
RAM is available to WSL; `torch.cuda.is_available()` is false. CPU runs use two
OpenMP/MKL/PyTorch threads. No paid API, Anthropic counter, or remote inference
endpoint was called. Downloads were free public model/tokenizer artifacts.

The existing tiktoken, Argos, language packs, torch, and OpenAI SDK were reused.
Installed only the missing LLMLingua/Transformers stack and its missing transitive
dependencies into `.venv`; no system packages were changed. Current versions:
tiktoken 0.14.0, Argos 1.11.0, LLMLingua
0.2.2, Transformers 4.57.6, torch
2.15.0, tokenizers 0.22.2. `pip check` passes.
Argos packs: en↔fa 1.5, en↔zh 1.9, en↔ja 1.1. Anthropic SDK remains unnecessary.

| Execution | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Initial WSL baseline | 107 | 0 | 23 |
| Final combined suite, every integration enabled | 147 | 0 | 0 |

Final combined elapsed time: 164.08 seconds. The original
23 opt-in cases are nine tiktoken corpus cases, nine Qwen 7B tokenizer cases,
three real Argos cases, one real LLMLingua case, and one NLLB language-matrix
case. All executed real libraries/models. Seventeen new offline regression
cases exercise the fixes below; their fake backend objects are unit tests,
**not model-quality evidence**. Original tests and features remain intact.
See [JUnit evidence](phase2-pytest.xml) and [measurements](measurements.json).

All source modules, tests, examples, configuration, README, integration skill,
and previous review artifacts were inspected. A wheel built with existing
setuptools and no build isolation, installed into a separate project-local
smoke-test environment, and returned `tokopt 0.1.0` and a valid count through its
console entry point. This verifies a wheel build/install, not an sdist build,
a clean-network resolver install, Windows execution, or an executed GitHub CI
matrix. Setuptools warns about the deprecated license-table metadata.

## Confirmed findings and fixes

| Finding | Repair and evidence |
|---|---|
| Argos models corrupt `⟦N⟧` markers; Japanese can repeat them into a very long output | Argos now keeps recognized literals/numbers outside inference and translates intervening prose fragments. Shared translation helpers use that path in forward and reverse directions. Actual outputs preserve recognized spans; regression tests cover code, paths, flags, identifiers, signs, exponents, whitespace and fences. |
| Reported chrF `ValueError` obscured placeholder loss | The exception came from restoring back-translation markers, before chrF computation. Placeholder destruction has its own rejection reason. Computed chrF and lost back-translation items/words now have separate diagnostics. No threshold was relaxed. |
| CJK-adjacent numbers were falsely treated as missing | Numeric recognition no longer requires a Unicode word boundary around `42` in `42条` / `42件`. Localized Persian digits still compare correctly. Three regression cases verify this. |
| LLMLingua forcing missed capitalized words and could deduplicate repeated critical words | Force actual spellings such as `Never`, whole recognized literals/numbers, and disable forced-token deduplication. Keep the paragraph unchanged if its force vocabulary exceeds upstream's 100-entry capacity. |
| Same-language critical-word checks covered only English | Added selected Persian lexemes and Chinese/Japanese instruction markers. These catch the tested losses but remain a limited dictionary, not comprehensive morphological or semantic analysis. |
| Explicit CPU and automatic CPU loaded separate cached classifier instances | Cache by resolved device. The cache regression proves one load; final real measurements use the repaired cache. |

The changes are in `check.py`, `optimizer.py`, `strategies/translate.py`, and
`strategies/lingua.py`; regressions are in `tests/test_phase2_regressions.py`.
The new `scripts/phase2_measurements.py` reproduces the real evidence. The
original candidate remains available, and failed candidates never replace it.

### Reproduction of the user's translation experiment

Exact original `o200k_base` counts reproduced: English 197, Persian 259,
Chinese 211, Japanese 515. The original protection path yielded only 33%, 17%,
and 17% recognized-item retention respectively and failed reverse marker
restoration. A direct tiny marker probe produced damaged Chinese markers and
long repetitive Japanese output. This establishes a real protection/backend
interaction, not a chrF arithmetic failure.

After repair, the same example measures:

| Candidate | Actual tokens | Recognized-item retention | Computed chrF | Selected? |
|---|---:|---:|---:|---|
| English | 197 | original | — | Yes |
| Persian | 257 | 100% | 71.45 | No |
| Chinese | 223 | 100% | 69.43 | No |
| Japanese | 298 | 100% | 69.86 | No |

All translations still increase tokens. Back-translation also loses `must`
(Persian), `before` (Chinese), and `must`/`never` (Japanese). The character scores
exceed 55, but the additional critical-word gate rejects them. A lexical gate
can reject legitimate synonyms or reformulations: these losses are **signals**,
not a bilingual semantic judgment. There is no reason to weaken it here since
none of these candidates saves tokens. Fragment translation protects literals
at the cost of sentence context; it does not fix underlying translation quality.

The forward/reverse matrix also tested native Persian, Chinese, Japanese and
mixed-script inputs. Every measured translated candidate preserves recognized
literal/numeric items, but **no translation candidate passed all final default
gates** in this six-input corpus. Mixed input is explicitly tested with source
`en`; Argos does not automatically identify each embedded language, so its
translations can be especially poor. Native prompts use their correct source
language. NLLB's real six-direction smoke test passed; NLLB quality, protection
under complex literals, latency and RAM were not benchmarked separately.
See [complete translation outputs and reasons](phase2-translation.json).

## Actual tokenizer measurements

These are measured encoded text lengths, not estimates or provider billing.
The Qwen 0.5B tokenizer matches the local benchmark model. Separately, the nine
existing HF integration cases used the requested Qwen 7B tokenizer without
loading the 7B inference weights. OpenAI counts use local encodings, without API
requests. Anthropic token counts remain unverified and were not approximated as
if exact. The scripts below are authored fixtures, not equal-length translations.

| Sample | o200k | cl100k | Qwen 0.5B | Heuristic estimate |
|---|---:|---:|---:|---:|
| en | 28 | 28 | 29 | 26 |
| fa | 42 | 70 | 65 | 46 |
| zh | 30 | 33 | 31 | 32 |
| ja | 38 | 51 | 39 | 53 |
| mixed | 28 | 39 | 35 | 32 |
| code | 20 | 20 | 20 | 15 |
| logs | 7738 | 7611 | 8822 | 6200 |
| json | 17 | 16 | 17 | 10 |
| markdown | 12 | 12 | 13 | 8 |

Heuristic error varies by language and structure; it cannot validate target
context limits. Counts above exclude chat wrappers. The Qwen benchmark below
also counts the **actual serialized chat template** including the appended
question. See [tokenizer data](phase2-tokenizers.json).

## Real LLMLingua-2 results

Model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, CPU,
keep rate 0.5. Each fixture repeats its language-specific instructions five
times and includes a fenced shell command. Raw output is recorded separately
from accepted output; every accepted lossy candidate passes the default exact
critical-item and known instruction-marker checks.

| Fixture | Original tokens | Raw classifier tokens | Selected tokens | Selected strategy | Warm optimizer latency |
|---|---:|---:|---:|---|---:|
| en | 154 | 109 | 108 | lingua@0.5 | 0.696 s |
| fa | 224 | 144 | 143 | lingua@0.5 | 0.751 s |
| zh | 164 | 116 | 116 | lingua@0.5 | 0.650 s |
| ja | 204 | 145 | 203 | normalize | 0.693 s |
| mixed | 154 | 115 | 115 | lingua@0.5 | 0.707 s |

All counts use real `o200k_base`. Japanese raw pruning loses the `ない`
negation marker, so it is rejected; normalization saves only a formatting token.
Fenced commands remain byte-exact. Recognition and force tokens do not establish
that every unrecognized command, unit, relationship, or instruction survives.
Selected savings are about 25–36% on the four passing fixtures, not a universal
50% reduction. The classifier's keep rate uses its own tokenization.

The first model-load/raw-inference interval was
68.33 s; subsequent raw calls were approximately
0.63–0.73 s.
This interval includes lazy model/dependency loading and is not pure inference
latency. Whole-command startup and initial downloads are not included in these
per-operation timers. Final Linux process peak RSS was
1271.3 MiB, including Python, torch and the
classifier. Earlier runs with duplicate device-cache entries peaked near 1694
MiB; this is an uncontrolled before/after observation, not a controlled memory
experiment. [Raw/selected text and resource data](phase2-lingua.json).

## Independent target-model benchmark

Real local `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, float32 CPU, greedy generation,
32-token output limit. Twenty-four authored cases specify expected answers
**before** inference: access code 42, forbidden deletion (`no`), filename
`config.yaml`, code result 9, and failed-test identification. The cases include
five languages/scripts, integrated long safety paragraphs, protected code,
translation attempts, and noisy logs. No score uses baseline agreement as its
expected answer. Exact grading normalizes case and edge punctuation only.
Arm order alternates between cases. This is one small deterministic trial, not
a randomized multilingual benchmark or evidence about larger hosted models.

| Metric | Original | Optimized |
|---|---:|---:|
| Independently correct | 12/24 (50.0%) | 14/24 (58.3%) |
| Complete text input tokens | 4135 | 2242 |
| Actual chat-template input tokens | 4831 | 2938 |
| Generated output tokens | 65 | 64 |
| Total model-generation wall time | 30.97 s | 20.55 s |

Actual input reductions: **45.8% text tokens** and
**39.2% serialized chat tokens**. Optimization took
24.23 s total (median 0.727 s/case),
so optimized inference **plus optimization** took
44.79 s versus
30.97 s baseline generation. These observations do not
establish end-to-end speedup. Model loading took
4.38 s after dependency imports;
command startup, downloads and imports are excluded from these totals. Other
integration jobs ran concurrently, so timing is observational and subject to
resource contention.

**Critical counterexample:** `fa-code` changed from correct `42` to wrong `10`
after accepted LLMLingua optimization. Three cases improved (`ja-safety`,
`mixed-safety`, `mixed-file`), but aggregate improvement does not cancel that
regression. The short factual paragraph remains protected by the compressor's
minimum-length rule; changing surrounding context alone can change model
behavior. The tiny model also answers several original safety cases wrongly.
Thus neither agreement nor this model's weak baseline proves preservation.
No generic code fix can infer arbitrary target-model answer equivalence from
literal counts. Adding an expectation-aware grader to ordinary optimization
would require independently supplied task answers and real inference.

The code case and log failure question pass both arms. Translation attempts
select normalization/fallback in these benchmark cases; **no downstream quality
benefit from an accepted translation is claimed**. Benchmark process peak RSS
was 3790.1 MiB and includes
target weights, classifier, translation libraries/caches and interpreter.
There is no isolated per-arm target-memory measurement, no CUDA/VRAM result,
and no demonstrated KV-cache or target-model memory saving.
[Per-case answers, timing and exact counts](phase2-benchmark.json).

## Reproduce and remaining release work

Use the existing virtual environment; working dependencies need no reinstall:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKOPT_TEST_TOKENIZERS=1 TOKOPT_TEST_ARGOS=1 TOKOPT_TEST_HF=1 \
TOKOPT_TEST_LINGUA=1 TOKOPT_TEST_NLLB=1 \
.venv/bin/python -m pytest -ra -p no:cacheprovider \
  --junitxml=review/phase2-pytest.xml

# Each mode is an independent process; first use may download public artifacts.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase2_measurements.py tokenizers
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase2_measurements.py translation
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase2_measurements.py lingua
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase2_measurements.py benchmark
```

`measurements.json` retains the original Phase 1 fields and adds authoritative
current data under `phase2`, including model snapshot IDs and source hashes.
The historical `scripts/review_measurements.py` still overwrites that file;
run it only if deliberately regenerating the old rule measurements, then
restore/regenerate the Phase 2 section. The new harness writes separate files
and never relabels the old heuristic log counts as exact measurements.

Before a preservation-focused release: expand independent task evaluation to
stronger local models, bilingual human review, terminology/ordering/unit tasks,
and repeated trials; investigate the Persian context-sensitivity counterexample;
measure uncontended end-to-end latency and isolated per-arm memory; lock tested
model revisions/dependency versions; verify clean installation, sdist and CI.
NLLB protection is still a placeholder path despite its successful smoke test.
Pattern-based critical-item checks, mixed-language routing, dictionary coverage,
uncalibrated simplified chrF, and correlated round-trip errors remain limits.
The earlier subprocess/log/security limitations also remain. No feature was
removed to make tests pass, no unsafe output replaced the original fallback,
and no paid or destructive system-level operation was performed.

Upstream model usage was checked against the
[Microsoft LLMLingua-2 small model card](https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank).
It documents a token-classification compressor; our real execution and independent
QA results above determine the observed behavior, not its upstream claims.

---

<!-- PHASE1-HISTORY -->

# Independent review of tokopt

Review date: 2026-11-08. Scope: every original source file, test, example,
README, pyproject and Claude Code skill, followed by targeted fixes and tests.
No GitHub repository was created, no changes were published, no paid API was
called, and no external service was modified.

## A. Executive summary

**Verdict: Not ready for GitHub under the requested verification standard.**
The core implementation is real, and the repaired offline pipeline is useful
for known log formats. The original 30 passing tests missed reproducible data
loss and misleading benchmark accounting. Those defects have regression tests
and fixes. However, installation, exact tokenizer comparisons, optional ML
inference, and target-model quality/resource benefits remain unverified here.
Publishing an explicitly experimental tool is possible after the blockers below
are resolved; the original broad preservation/resource claims were unsupported.

No simulated benchmark or fake translator result in this report is presented as
real model performance. No current feature is claimed to prove semantic fidelity.

## B. Environment and test results

- Windows 11, build 10.0.26300, x86-64; CPython 3.13.7, MSC v.1944.
- Isolated `.venv`, created from the installed Python313 executable. The PATH
  `python` alias was inaccessible and the Python launcher reported no runtimes.
- Installed offline: pip 25.2, pytest 9.0.1, pluggy 1.6.0, packaging 26.0,
  iniconfig 2.3.0, Pygments 2.19.2, colorama 0.4.6.
- setuptools, tiktoken, OpenAI/Anthropic SDKs, Argos, LLMLingua, Transformers,
  torch, tokenizers and SentencePiece were unavailable in this environment.
- The requested `pip install -e '.[openai,argos,dev]'` failed fetching
  `setuptools>=68`: outbound sockets were denied with WinError 10013. This is
  an environment/network failure, not evidence of an incompatible Argos version.
- Offline retry with `--no-index --no-build-isolation` failed because setuptools
  was absent. Cached pytest wheels enabled real pytest execution; source imports
  used `PYTHONPATH=src`. This is **not** a successful editable install.
- WSL failed with `E_ACCESSDENIED`. No Ollama executable was found; a read-only
  localhost `/api/tags` request failed with URLError. No local LLM ran.
- Doctor was run as `python -m tokopt doctor`; all optional dependencies were
  missing. The installed `tokopt` console script could not be verified because
  package installation failed. Doctor now clearly labels its limited scope.

| Check | Passed | Failed | Skipped / blocked |
|---|---:|---:|---|
| Original unchanged suite | 30 | 0 | 0 |
| New regression file before repairs | 2 | 36 | 0 |
| Final normal suite | 107 | 0 | 23 opt-in integration cases skipped |
| Explicit integration probes | 0 | 23 | All 23 blocked at missing dependency imports |
| `pip check` | Pass | 0 | Checks installed pytest stack only |
| Compileall; TOML parsing | Pass | 0 | Does not verify packaging/build |
| Python 3.9 AST parsing | 23 Python files | 0 | Syntax only; Python 3.9 runtime untested |

The 23 integrations are nine tiktoken corpus cases, nine HF tokenizer corpus
cases, three Argos language cases, one LLMLingua case, and one NLLB language
matrix case. Enabling them produces failures rather than silently skipping
missing packages. Their opt-in defaults avoid automatic large downloads.

Reproduction in this environment, using `.venv/Scripts/python.exe`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONUTF8 = '1'
python -m pytest -v
python -m tokopt doctor
python scripts/review_measurements.py
# Explicit missing-dependency probes:
$env:TOKOPT_TEST_TOKENIZERS = '1'
$env:TOKOPT_TEST_HF = '1'
$env:TOKOPT_TEST_ARGOS = '1'
$env:TOKOPT_TEST_LINGUA = '1'
$env:TOKOPT_TEST_NLLB = '1'
python -m pytest tests/test_integrations.py -v
```

Use the venv executable explicitly if it has not been activated. Raw local
command logs are retained in `.review/` and `install-review.log` (covered by
`.gitignore`); generated, nonsensitive
measurements are committed-ready files in `review/`. No Git metadata was present,
so no commit history or previously tracked secrets could be audited.

## C. Confirmed defects and disposition

All reproductions below are covered by added tests unless identified as a
documentation or dependency constraint correction. Severity reflects potential
consequences, not a formal vulnerability score.

| Severity | Files | Minimal reproduction / consequence | Disposition |
|---|---|---|---|
| High | `strategies/logs.py` | Compress `test_x PASSED WARNING: exposed` or `Collecting package ERROR: checksum mismatch`: diagnostic deleted and verification returned no loss | Fixed: diagnostics override noise; verification no longer exempts noise |
| High | `strategies/normalize.py` | `ERROR: denied\rDone` became `Done` | Fixed: general CR content retained; only narrowly recognized download progress is collapsed |
| High | `strategies/normalize.py`, `protect.py` | Blank lines and trailing spaces inside fenced Python string constants changed; indented/unclosed/long fences not reliably protected | Fixed for tested structures: broader fence protection, structured segments, Markdown hard breaks |
| High | `check.py` | `preservation('balance is -5', 'balance is 5').ok` was true | Fixed: signed/scientific numeric spans checked |
| High | `protect.py`, `check.py` | Windows/absolute paths, bare filenames and quotes unmasked; `other_user_id` could satisfy `user_id` | Fixed: expanded masking and exact critical-item multisets |
| High | `optimizer.py` | Remove `not` from a long instruction: chrF 94.18 passes default 55 gate | Mitigated: back-translation must also retain critical words/items; correlated semantic errors remain possible |
| Medium | `optimizer.py` | Forward translation works, missing reverse pack raises and aborts the run, including mixed candidates | Fixed: candidate rejected with sanitized reason; other candidates survive |
| Medium | `optimizer.py` | LLMLingua import succeeds but model loading/inference fails, aborting optimization | Fixed: rejected candidate and safe fallback; real model still unverified |
| Medium | `optimizer.py` | `strategies=['translate:xx']` silently did nothing; invalid/NaN thresholds accepted | Fixed: named translation works; unknown strategies and invalid settings fail |
| High | `counters.py` | Anthropic request count minus guessed `count('.') - 1` overhead reported as text count | Fixed: retain complete single-user request count; no undocumented subtraction; mock contract verified |
| High | `bench.py`, `cli.py` | Large appended question excluded from input savings; default OpenAI encoding used even for unrelated benchmark models | Fixed: count full prompt and recheck savings/limit; API CLI defaults use requested model, command/Ollama require explicit counter |
| High | `bench.py` | Baseline/optimized both wrong but similar counted as optimized accuracy for agreement cases | Fixed: independent expected-answer accuracy separated from agreement; wrong agreement is never accuracy |
| Medium | `bench.py` | `wrong A` versus `wrong B` counted unchanged because both had same grade | Fixed: normalized answer changes counted independently of grades |
| Medium | `bench.py` | `contains` with no expected answer accepted everything; malformed case scoring discovered after requests | Fixed: validate expected answers and scoring rules before running |
| Medium | `bench.py` | Optimized request fails and successful baseline answer/grade is discarded | Fixed: per-arm outcomes retained; failed graded arm is false |
| Medium | `bench.py` | Command failure embeds stderr in exceptions/reports; reports always contain model answers | Fixed: stderr/error details withheld; report answers opt-in; commands excluded from runner name |
| Medium | `bench.py` | Timeout only kills shell parent; descendant inference process can survive | Improved: process groups and timeout tree termination; detached workloads still require OS isolation |
| Medium | `bench.py` | API settings lack bounded output/consistent timeout controls; malformed Ollama answers not validated | Fixed for supported request shapes; same settings used across arms; live API compatibility untested |
| Medium | `strategies/logs.py` | Default Java truncation deletes middle file/line frames; timestamp-based dedupe erases diagnostic occurrences | Fixed: frames retained by default, explicit truncation still available; repeated diagnostics/timestamps retained |
| Medium | `strategies/logs.py` | Large failed pytest captured output/diffs omitted by focus filtering | Fixed for recognized failure/captured sections; arbitrary log formats remain heuristic |
| Medium | `cli.py` | Invalid UTF-8 bytes silently replaced | Fixed: explicit failure, no corrupted optimized output |
| Medium | `strategies/translate.py` | One NLLB pipeline/model copy cached per language direction | Fixed to reuse one pipeline; mock contract only, runtime memory unmeasured |
| High compatibility | `pyproject.toml` | `openai` extra lacks OpenAI SDK; NLLB translation pipeline permitted Transformers v5 which removed it | Fixed declarations; editable installation/resolution remain blocked |
| Low | `counters.py` | Nonempty short text estimates zero; fallback HF loading errors inconsistently wrapped | Fixed: nonempty estimates >=1 and consistent CounterUnavailable errors |
| Documentation | README, skill, doctor, module docs | Unqualified preservation, tokenizer scope, model sizes and proportional VRAM claims | Corrected; experimental features and data transmission made explicit |

## D. Feature verification and implementation map

`__init__.py` exports `optimize`, counter construction, checks and result types.
`__main__.py` calls `cli.main`; pyproject declares the same installed entry point.
The optimizer keeps the original candidate, always considers normalization,
optionally builds log/LLMLingua/translation candidates, then selects the smallest
passing candidate with stable nonlossy tie preference. Explicit token limits
now fail if no accepted candidate fits.

| Feature | Classification | Evidence / limit |
|---|---|---|
| Source package import; module CLI count/compress/experiment | Verified working | Original and new unit/CLI tests; installed entry point not tested |
| Normalize | Verified working for tested inputs | CR diagnostics, ANSI, Markdown and code regressions; still formatting cleanup |
| Log compression | Partially working | All eight synthetic realistic format fixtures survive; unknown long-log content may still be omitted |
| LLMLingua-2 adapter and safety fallback | Mock-tested only | Real classifier loading/inference blocked; pruning can change meaning even when keywords survive |
| Real LLMLingua reduction/time/RAM/VRAM | Not tested | No llmlingua/torch/model data |
| Argos translation EN-fa/zh/ja | Not tested | Real tests fail before translation; mock gates do not establish language quality |
| NLLB translation and shared model caching | Mock-tested only | Dependency correction based on upstream documentation; real pipeline unavailable |
| Per-segment selection and structural preservation | Mock-tested only | Gate/structure tests pass; real terminology consistency not established |
| Round-trip chrF as proof of meaning | Broken as a guarantee | It is a heuristic; correlated errors can translate back perfectly. Documentation and critical-word gate improved |
| tiktoken/HF exact text counting | Not tested with real dependencies | Nine-language/format cases each are opt-in; no real/estimated error measurements possible |
| Anthropic request counter | Mock-tested only | Corrected request-count scope; no credential/API use |
| Heuristic counter | Verified estimate implementation | Labeled rough, not accuracy-validated against real tokenizers |
| Local command benchmark/scoring/reporting | Verified working | Real deterministic Python subprocesses; no actual LLM |
| OpenAI/Claude API benchmark adapters | Mock-tested only | Comparable payloads and response contracts; no paid request |
| Ollama benchmark adapter | Mock-tested only | Malformed-response tests; localhost unavailable |
| Claude Code skill | Not tested in Claude Code | Instructions inspected/corrected; CLI pipeline mechanism is real, no automatic interception hook |
| Distribution build/editable install/GitHub CI | Not tested successfully | TOML valid, files/config added; network/build backend missing |

Log fixtures cover pytest, unittest, Gradle, npm/Jest, Docker, syslog, Java and
Python tracebacks. They are authored test data, not fresh executions of these
external tools. Edge coverage includes ANSI escape handling, Unicode,
empty/malformed text, duplicates, mixed streams, large captured sections and
100,002-line generated logs. Optional Java truncation deliberately loses frames;
long-log focus is also explicitly lossy. Relevant content outside recognized
patterns cannot be guaranteed.

Argos calls the local installed translation graph and language setup downloads
both directions. NLLB uses the default distilled 600M model through a v4 pipeline.
Both mask fragile prose spans and leave protected structures untouched. Per-
paragraph selection independently chooses each cheapest passing translation,
then counts the joined result with the answer-language hint. Cross-paragraph
terminology and discourse cannot be established by those independent checks.
The function called `chrf` is a simplified, lowercased/whitespace-normalized
character metric, not standardized SacreBLEU chrF. Its default threshold is
uncalibrated against independent human quality ratings.

## E. Token accounting and resource measurements

### Every approximate accounting path

`HeuristicCounter.count` uses ASCII/4, CJK/1, other noncombining characters/2,
rounded with a nonempty minimum. It is used explicitly by existing unit fixtures,
new local measurements/benchmarks, and `-c heuristic`. `get_counter(None)` can
fall back to it with a warning. Once selected, it feeds **all** optimizer gates,
segment comparisons, CLI candidate tables, final benchmark prompts, and output
counts. Those values are estimates throughout; counter names are preserved in
results. No model limit is assured with a heuristic counter.

LLMLingua's keep-rate uses its own classifier/tokenization; it is not a target
token count. Final candidate counts use the selected external counter. The old
Anthropic guessed-overhead subtraction was another approximation and is removed.
Offline tiktoken/HF text counts exclude model chat templates and protocol
wrappers; callers must serialize and count their own full requests. Anthropic
counts a single-user API request and is not comparable to raw text counts.

The reply hint, segmentation delimiters and final question are counted as joined
text. Token counts are not assumed additive. `--max-tokens` applies to the
selected counter's scope, including the benchmark question, and throws instead
of silently returning an oversized prompt.

Actual tiktoken/HF values are **unavailable** for English, Persian, Chinese,
Japanese, mixed text, source code, logs, JSON and Markdown. Their heuristic
values and explicit null real counts are in [tokenizer-comparison.json](tokenizer-comparison.json).
No tokenizer error rate or real token saving can be inferred from these values.

### Actual local rule-compression overhead

Each workload ran in a fresh Python subprocess. Working-set peaks came from
Windows GetProcessMemoryInfo; Python allocation peaks came from tracemalloc.
Timing includes tracing overhead and is a single observation, not a stable
production throughput estimate. Input construction/import time is outside the
timed region. RSS peaks include interpreter, input and compressor allocations.

| Lines | Actual input -> output bytes | Estimated tokens before -> after | Measured optimizer wall time | Measured process peak working set |
|---:|---:|---:|---:|---:|
| 502 | 23,472 -> 123 | 5,868 -> 31 | 0.0560 s | 26.27 MiB |
| 10,002 | 478,972 -> 125 | 119,743 -> 31 | 0.9793 s | 30.42 MiB |
| 100,002 | 4,888,972 -> 126 | 1,222,243 -> 32 | 8.5075 s | 67.35 MiB |

Estimated token savings are 99.47%, 99.97%, and 99.997% on these intentionally
noise-heavy generated fixtures. Real byte ratios are 0.0052403, 0.0002610 and
0.00002577. They are not representative averages for real logs.

Count-only subprocess observations were 0.0139/0.2176/1.9421 seconds and about
26.06/27.63/40.75 MiB peak working set. For the largest optimizer run, Python
allocation peak was 24,371,603 bytes and CPU time was 8.46875 seconds. The
compressor uses additional memory; these observations establish **no target-
model memory saving**. Peak VRAM and target-model inference were not measured.

### Actual deterministic command benchmark

Six generated cases used a local Python script with independently specified
arithmetic answer 42 for five cases. The sixth deliberately checks only
agreement. This validates benchmark mechanics, not model quality.

| Case | Baseline | Optimized | Independent result |
|---|---|---|---|
| Correct | 42 | 42 | Both correct |
| Both wrong | 41 | 41 | Both incorrect, despite agreement |
| Better | 41 | 42 | Optimized improves |
| Worse | 42 | 41 | Optimized degrades |
| Format | Answer: 42. | 42 | Both pass contains rule; answer text changes |
| Agreement only | 41 | 41 | No accuracy grade |

Results: 3/5 baseline and 3/5 optimized accuracy, 3/6 answer agreement, three
normalized answer changes, no runner errors. Estimated total inputs were 5,161
baseline versus 137 optimized (97.35% estimated reduction). Estimated output
tokens total 8 versus 6. Rule-based optimization used zero model tokens and zero
API requests. There is no provider billing/token-usage measurement.

The latest benchmark's complete six-case paired evaluation took 0.6195 seconds,
including optimization, counts and all 12 subprocess runs. Per-case means were
3.81 ms optimization, 50.17 ms baseline subprocess and 49.06 ms optimized
subprocess. Each subprocess duration includes shell/Python startup. This does
not demonstrate LLM latency improvement; optimization overhead outweighs the
small observed startup-time difference. Full CLI startup and target-model
inference costs are not measured.

Machine-readable evidence: [measurements.json](measurements.json) and
[local-benchmark.jsonl](local-benchmark.jsonl). Reproduce with
`python scripts/review_measurements.py`; `--benchmark-only` retains existing
log measurements and reruns the small local benchmark.

## F. Security findings and remaining risks

The shell runner is an intentional command-execution capability. Prompt text
containing shell operators, substitutions and Unicode was passed literally
through stdin and was not evaluated. Its configured command remains trusted
code; there is no sandbox. Timeout, stderr withholding and report redaction
were improved. Windows timeout execution was tested locally; POSIX process-group
cleanup is implemented but awaits Linux CI. Detached processes and unlimited
stdout/stderr buffering still require external resource controls.

`prompt_file` intentionally supports arbitrary local paths, including parents
and absolute paths. This is a trust boundary, not containment: a malicious case
file can select a private file for transmission to an API. Cases/backend specs
must be reviewed; this is unsuitable for an untrusted upload service. No document
content is used as a path or command automatically. No eval/exec/pickle loading
or project-managed unsafe temporary files were found in the original code.
The measurement harness uses TemporaryDirectory for its generated runner.

Remote counters/runners transmit prompt data. Reports now withhold raw answers
by default, but explicit text output and case IDs can expose caller-supplied
data. Upstream dependencies/model loaders remain trusted and versions/model
revisions are not frozen. HF tokenizer remote code is explicitly disabled.
No dependency vulnerability database scan or optional-stack installation audit
could be completed here. NLLB weights have separate license restrictions.

A bounded credential-pattern scan found no embedded credentials/private keys in
the reviewed source, tests, examples, skills and reports. The environment's
credential values were not printed or used. This is not an exhaustive secret
audit and cannot inspect nonexistent Git history. Local environments, cached
wheels, raw logs and key files are ignored for future Git use.

Compression is not a prompt-injection defense. Protected words can still form
malicious instructions and pruning can change authority boundaries. Keep model
trust boundaries outside the compressor. See [SECURITY.md](../SECURITY.md).

## G. Files changed and publication preparation

| Files | Purpose |
|---|---|
| `src/tokopt/optimizer.py` | Validation, explicit translation strategies, robust candidate rejection, stronger round trips, token limits |
| `src/tokopt/check.py` | Signed/scientific numbers, exact critical spans, broader critical words |
| `src/tokopt/protect.py` | Robust fences, structure protection, fragile literals/paths/placeholders |
| `src/tokopt/strategies/normalize.py` | Preserve CR diagnostics, code whitespace and Markdown hard breaks |
| `src/tokopt/strategies/logs.py` | Diagnostic precedence, occurrence verification, conservative frames/captured output |
| `src/tokopt/strategies/translate.py` | Reuse NLLB pipeline across directions |
| `src/tokopt/strategies/lingua.py` | Remove unsupported download/memory-size estimates |
| `src/tokopt/counters.py` | Honest Anthropic request scope, safe tokenizer loading/errors, nonzero rough estimates |
| `src/tokopt/bench.py` | Correct accuracy/agreement/changes, complete prompts, per-arm failures/timing/output counts, safer subprocess/report handling |
| `src/tokopt/cli.py` | Final token limits, explicit benchmark counter selection, strict UTF-8, honest doctor, answer-report opt-in |
| `tests/test_review_regressions.py` | Independent defect reproductions |
| `tests/test_log_formats.py` | Eight format fixtures and conservative log preservation regressions |
| `tests/test_backend_contracts.py` | Mock API/ML contracts, local process/security/error tests, known semantic limitation |
| `tests/test_integrations.py` | Explicit real tokenizer/Argos/LLMLingua/NLLB tests, separate from mocks |
| `scripts/review_measurements.py` | Reproducible real local CPU/time/working-set/byte measurements and scripted benchmark |
| `README.md`, `skills/token-optimizer/SKILL.md` | Accurate behavior, installation, trust, benchmark and experimental limits |
| `pyproject.toml` | Missing SDK, compatible ML constraints, license metadata, test markers |
| `.github/workflows/tests.yml` | Least-permission CI for core tests/builds and real tiktoken corpus, no paid APIs |
| `.gitignore`, `LICENSE`, `MANIFEST.in`, `SECURITY.md` | Publication hygiene, MIT license, source-distribution resources and trust documentation |
| `review/*` | This report and generated nonsensitive evidence |

All original tests remain unchanged. New files add 77 passing offline tests
and 23 opt-in integration cases. No features were deleted and no safety
threshold was silently relaxed. No optional integration was replaced with a
fake implementation in the package.

## H. Prioritized remaining work

1. **Before publication:** run an actual editable install and build from a clean,
   network-enabled environment, then install the resulting wheel and test its
   console script. Run the added GitHub CI matrix; it has not executed here.
2. **Before token-benefit claims:** execute tiktoken/HF multilingual comparisons,
   count the actual downstream chat templates, and run real Argos bidirectional
   packs plus LLMLingua/NLLB inference. Record dependency/model revisions, CPU
   latency, RAM and quality rejects. Test mixed-language text and technical
   preservation with real translators, not only smoke/contract tests.
3. **Before quality claims:** use independently authored tasks and expected
   answers, bilingual human review for translations, terminology/discourse tests,
   real local-model inference, and multiple counterbalanced generation runs.
   Do not aggregate agreement into accuracy. Consider custom evaluators for
   semantic correctness; contains/regex alone can be fooled by negated answers.
4. **Before memory/performance claims:** measure the same actual local runtime
   and generation settings, including output tokens, optimizer overhead, peak
   CPU/GPU allocation, cache reservation/batching and complete wall time.
5. **Hardening:** add resource caps for untrusted subprocess output/input and
   regex scoring; verify Linux timeout cleanup; evaluate immutable model pins,
   optional dependency constraints/lockfiles and upstream installation safety.
6. **Release metadata:** maintainer should confirm MIT is the intended project
   license and configure private security reporting when creating the repository.
   Model/dependency licenses remain separate. No repository URL was invented.

## I. Final verdict

**Not ready for GitHub.** The repairs and publication files are concrete and
reviewable, and the offline suite passes. The requested independent verification
of installation, real token savings, optional ML behavior and real answer-quality
or memory benefits could not be completed in this environment. The report
records those gaps explicitly rather than treating passing mocks as evidence.

Upstream facts used for review: the [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
documents its noncommercial license and removal of translation pipelines in
Transformers v5; [Transformers v4 pipeline documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/pipelines)
describes translation language arguments; [Argos documentation](https://argos-translate.readthedocs.io/en/latest/)
describes local packages and pivot translation. These references establish API
and packaging facts, not successful execution of those integrations here.
