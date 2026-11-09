> **Note (final independent review):** the fast-mode "41.6% savings" and "no paired
> regressions" results below come from 10 log fixtures; the other 110 prompts were unchanged,
> and all 10 compressed cases were already wrong at baseline, so a regression was impossible.
> See [FINAL-REVIEW.md](FINAL-REVIEW.md) for the corrected interpretation and later fixes.

# Phase 3: reliability and resource review

Date: 2026-11-08. This is the current assessment; Phase 2 and Phase 1 are
historical evidence retained in `REVIEW.md` and `measurements.json`.

**Verdict: conditionally ready for a clearly experimental public alpha with the
conservative fast default.** Packaging, local inference and regression tests
work. The observed results support scoped log/rule use. They do not support
universal answer preservation, general speedup, or target-model RAM/VRAM-saving
claims. ML overrides remain research features. Nothing was published and no
paid API or destructive system-level action was used.

## 1. Exact Persian failure and repair

A pinned local Qwen replay isolates a sufficient trigger: removing **one ASCII
space at zero-based byte/character offset 484**, immediately before the paragraph
break, changes `42` to `10`. The Persian factual paragraph is byte-identical.
The Qwen tokenizer replaces IDs `[13, 4710]` (decoded `. \n\n`) with `[382]`
(decoded `.\n\n`). This was normalization before LLMLingua, not translation,
loss of the number 42, or an arithmetic calculation in tokopt.

LLMLingua additionally removes words from the English filler. Restoring the
original paragraph separator to its pruned output still produces `10`, so
pruning also independently creates an answer-changing context. This is a
reproducible target-model sensitivity; it does not identify an internal neural
mechanism. The original Phase 2 report blamed the selected LLMLingua candidate
without separating these transformations.

| Real greedy Qwen input | Answer |
|---|---:|
| Original captured input | 42 |
| Legacy normalization: one space removed | 10 |
| Current normalization | 42 |
| Captured legacy LLMLingua output | 10 |
| Current fast default | 42 |
| Current conservative LLMLingua policy | 42 |
| Factual paragraph alone | 42 |

Repairs preserve single trailing spaces and skip mixed-script prose pruning by
default. Same-language checks now retain recognized critical clauses rather than
only word/literal multisets, protecting negation scope. Number/unit associations
and elementary expressions are also checked, including comparison, power and
floor-division operators. Unit detection occurs before identifier masking:
`MiB` otherwise looks like a PascalCase identifier and can hide its quantity.

A separate code-normalization bug was fixed: ANSI controls and CRLF inside
fenced code were being modified before segmentation. Fences are now protected
before terminal cleanup. Regression tests cover negation/scope, signed values,
units, formulas, code controls/line endings, originals, cache invalidation,
worker lifecycle and CLI safety. [Byte/token evidence](phase3-ablation.json),
[captured fixture](phase3-legacy-fixture.json), and
[first exploratory ablations](phase3-ablation-initial.json).

## 2. Fast mode, bounded results, and safe failure

- CLI compression/benchmarking defaults to `fast`: local normalization and
  detected logs, no implicit ML. Tiny cosmetic changes are declined unless
  normalization was explicitly requested. `experiment` is an explicit full
  research command.
- `balanced` optionally considers LLMLingua for large auto inputs. Fast/balanced
  cost gates require 1,024 source tokens and an optimistic 256-token reduction
  by default; thresholds are configurable. These are policy cutoffs, not
  measured payback predictions. Passing rule savings short-circuit ML.
- Python API auto/rule calls default to fast. Existing explicit ML/translation
  API requests retain full behavior unless the caller sets `mode='fast'`.
  Full bypasses cost gates, not preservation checks.
- Mixed-script prose and recognized unit/math contexts do not enter LLMLingua
  by default in any mode. `allow_mixed_lingua` / `allow_sensitive_lingua` are
  explicit research overrides; primitive inference was not removed.
- `timeout_seconds` / CLI `--timeout` uses a spawned worker rather than an
  uncancellable inference thread or fork of a live torch runtime. WSL worker
  process groups are stopped on expiration. Timed runs bypass result caching.
- The exact original is returned on timeout or operational backend failure.
  Counts remain `null` if counting did not finish, with explicit warnings.
  Unverified/oversized `max_tokens` fallbacks are marked unaccepted and the CLI
  exits nonzero. Invalid configuration still raises an error.
- `result.original_text` is an independent snapshot; caller edits to candidates
  cannot erase it. Raw-source JSON is opt-in. The CLI refuses source/output
  aliases and keeps source line endings when reading.

The private process-local result cache has LRU eviction, a 300-second TTL,
128-entry limit, and 8 MiB serialized-value limit. Keys are bounded digests of
input, settings, namespace and counter identity. Copies prevent caller mutation
from corrupting cached results. Loaded HF tokenizer instances have distinct
identities; custom counters must supply a versioned `cache_key()` to opt in.
Parameter/namespace/tokenizer changes, TTL, `clear()` and process exit invalidate
results. Opaque/remote counters, ML/translation outputs, failures and timed runs
are not result-cached; mutable backend/model revisions are not guessed. This
bounds the **result** cache, not upstream model/tokenizer caches.

## 3. Independent 120-case, six-strategy benchmark

The corpus has 12 families, ten cases each: multilingual access codes,
negations, units, arithmetic, fenced code, paths/filenames, ordering,
log failures, JSON, distractors, instruction scope and prose facts. Expected
answers are authored from the fixtures before inference; no baseline agreement
is treated as accuracy. English, Persian, Chinese, Japanese and the exact mixed
Persian failure are included. Cases are synthetic, not a sampled production
workload, and strict exact scoring can penalize valid wording or formatting.

Model: local Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, float32 CPU,
two PyTorch threads, greedy generation, 32-token limit. Actual chat-template
counts include the question and wrappers. Baseline: **69/120 (57.5%)**, with
**25,443 chat-input tokens**, 21,963
plain-text input tokens and **404 output tokens**. The tiny
model's weak baseline limits generalization to more capable models.

### Findings before the added quantity/context safeguard

The complete first run includes **244 distinct real target-model generations**.
Fast, normalization and logs have no paired regressions. LLMLingua has two:

| Case | Independent expected | Baseline | Pruned |
|---|---|---|---|
| units-3 | 512 | 512 | 4096 |
| units-4 | -5 | -5 | -20°C |

The factual paragraphs themselves survive. Changing surrounding filler is enough
to damage downstream answers. The new general unit/math guard declines pruning
of these contexts before loading the model; its tests use different quantities
and expressions. The failed results are retained in
[the unmodified first-run report](phase3-benchmark-before-sensitive-policy.json).
They are not replaced by passing mock tests or hidden by aggregate accuracy.

### Final conservative-policy replay

| Strategy | Exact correct | Paired regressions / improvements | Chat-input tokens | Input saving | Output tokens |
|---|---:|---:|---:|---:|---:|
| fast | 69/120 | 0 / 0 | 14,863 | 41.6% | 428 |
| normalize | 69/120 | 0 / 0 | 25,443 | 0.0% | 404 |
| logs | 69/120 | 0 / 0 | 14,863 | 41.6% | 428 |
| lingua | 70/120 | 0 / 1 | 21,152 | 16.9% | 378 |
| translation | 70/120 | 0 / 1 | 25,305 | 0.5% | 400 |
| combined | 71/120 | 0 / 2 | 11,777 | 53.7% | 444 |

Fast selects the original for 110 cases and logs for ten. Normalization alone
leaves all 120 unchanged. Final LLMLingua selects pruning for 93 cases and
originals for 27; translation selects one native Persian→English candidate.
Improvements are `ordering-5` for LLMLingua and `access-6` for translation;
combined selection retains both. Combined is a portfolio choosing among
independent candidates, not a newly invented chained transformation.

**This is an in-sample safeguard verification, not a held-out quality claim.**
The corpus revealed the quantity regressions and informed the guard. All 720
final selected inputs exactly match retained real observations; the replay
recomputes each optimization and reuses the recorded greedy answer only for
identical serialized inputs. It does not execute 720 fresh target generations.
Reused answers contribute zero additional measured generation time/CPU.
Per-case flags, raw answers, output counts, regressions, improvements and
original/selected text fingerprints remain available.

Reuse validation checks model revision, tokenizer files, dependency versions,
generation settings, questions, expected answers and exact prompt bytes.
[Validated observations](phase3-observations.json) add provenance metadata to the
first-run answers/timings without changing them; the original first-run file is
retained byte-for-byte. [Final 720-arm report](phase3-benchmark.json),
[120 independent cases](phase3-cases.jsonl), and
[machine summary](phase3-summary.json).

## 4. Latency, CPU, RAM and output costs

The following generation times are **measured reference observations reused for
identical inputs**. Optimization timings are from the policy verification.
Paired totals add those two quantities; they are not fresh complete-session
wall times. Imports, downloads, model loading, serialization/report I/O and
initialization before the optimizer call are excluded. Baselines ran first;
strategy order, caches and concurrent local integration jobs confound timings.
No causal or universal speedup is established.

| Strategy | Optimization s | Average optimizer CPU | Reference generation s | Paired warm compute s | Sampled peak process RSS MiB |
|---|---:|---:|---:|---:|---:|
| fast | 0.159 | 99.5% | 101.348 | 101.507 | 2722.5 |
| normalize | 0.129 | 99.4% | 155.014 | 155.143 | 2722.9 |
| logs | 0.212 | 100.3% | 101.348 | 101.561 | 2723.9 |
| lingua | 79.720 | 192.7% | 133.081 | 212.800 | 3187.2 |
| translation | 41.338 | 205.8% | 154.012 | 195.350 | 4009.0 |
| combined | 113.142 | 205.4% | 89.428 | 202.570 | 4054.3 |

Baseline reference generation is **155.014 s**.
Fast's observed/reference pairing is approximately
101.507 s; expensive arms remain slower than
baseline despite reduced input counts. These results justify the conservative
choice on this workload, not thresholds calibrated for every downstream model.
Fast output tokens increase from 404 to 428; counting them still gives roughly
40.8% total input+output token reduction here. Weight files and reserved memory
are not reduced by shortening input.

CPU percentages are process CPU seconds / wall seconds × 100, so 200% means
roughly two cores. RSS is sampled at 10 ms using psutil and includes Python,
target weights, classifier and already-loaded translation/model caches. Strategy
order affects those peaks; they do not isolate incremental optimizer or
per-arm target memory. Kernel lifetime peak is recorded separately. No CUDA
was available, VRAM is `null`, and no target-model memory saving was measured.

To remove model-loading contamination, the **final rule code** was also run in
fresh processes with real `o200k_base`, without any inference model:

| Rule path, all 120 prompts | Optimization wall | CPU time | Average CPU | Process peak RSS |
|---|---:|---:|---:|---:|
| fast | 0.0524 s | 0.0513 s | 97.9% | 96.9 MiB |
| normalize | 0.0226 s | 0.0216 s | 95.5% | 96.1 MiB |
| logs | 0.0927 s | 0.0914 s | 98.5% | 96.9 MiB |

Counter initialization is timed separately in the raw file. Final fast selected
texts exactly match the model benchmark despite the different measurement
counter. This independently verifies the final fence-only fix did not change
those selected inputs. These are optimizer-only measurements, not target
latency. [Isolated rule measurements](phase3-rule-performance.json).

The bounded result-cache probe reports a hit and separate cold/warm times;
repeated cached results remain byte-correct and storage is within bounds. This
small probe is not a universal cache-speed claim.

### Deadline and tokenizer startup evidence

Real cancellation budgets of 0.05, 0.5 and 2 seconds returned the original in
0.0517,
0.5024 and
2.0058 seconds, with no surviving
workers or child inference processes. The 50 ms case has an unknown count;
the longer cases report the completed count of 680.

An exact loaded HF snapshot transferred successfully and returned six tokens
for `Keep 42 records.`. Transferring the native tokenizer JSON avoids importing
Transformers/torch in the worker: observed snapshot-check startup changed from
76.12 s to **1.17 s**. These are two
uncontrolled observations, including filesystem/cache effects, not a general
speed ratio. Multilingual, chat-marker and code snapshot equivalence is also
covered by a real integration test. [Deadline evidence](phase3-deadlines.json).

Deadlines are not hard real-time guarantees: scheduling, trusted object
serialization and termination can add overhead. CLI preparation/counter
construction before the API call is outside its timer. Spawned ML workers have
cold model caches; a short deadline can expire during startup. Windows cleanup,
notebooks and arbitrary custom/detached subprocesses require separate validation.

## 5. Tests, packaging and alpha assessment

**182 passed, zero failures, zero skips**, in 178.24 s.
This includes 157 offline cases and **25 real integrations**: all 23 previous
backend tests plus real loaded-HF snapshot and pinned-Qwen Persian checks.
Phase 3 adds 35 cases overall. [Final JUnit report](phase3-pytest.xml).

The final wheel builds with existing setuptools, installs without optional
dependencies into a separate project-local environment, and its CLI plus spawned
timeout worker pass from outside the source tree. `pip check` passes. This is a
core wheel smoke test, not a clean-network optional-dependency resolution,
sdist verification, Windows run or executed GitHub CI matrix. Setuptools still
warns about deprecated license-table metadata. [Packaging](phase3-packaging.json).

A public alpha should keep fast mode, explicit experimental flags, the raw
original fallback and these limitations. Before stronger claims: evaluate new
held-out production tasks and stronger independent local models; test semantic
and bilingual grading; counterbalance repeated timing trials; isolate target
memory; pin dependencies/models; and execute CI/platform checks. Pattern-based
quantity, expression, script and clause checks remain incomplete and can both
miss meaning changes and reject harmless edits. Translation chrF is still an
uncalibrated heuristic. The existing log/subprocess/security limitations remain.

No source feature or model adapter was removed to make tests pass. The original
is never overwritten by CLI compression. Existing dependencies and free local
models were reused; nothing was published or billed to an API.

## 6. Reproduce

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKOPT_TEST_TOKENIZERS=1 TOKOPT_TEST_ARGOS=1 TOKOPT_TEST_HF=1 \
TOKOPT_TEST_LINGUA=1 TOKOPT_TEST_NLLB=1 TOKOPT_TEST_PHASE3_MODEL=1 \
.venv/bin/python -m pytest -ra -p no:cacheprovider \
  --junitxml=review/phase3-pytest.xml

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase3_ablation.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase3_deadlines.py
# Fresh target-model observations:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase3_benchmark.py
# Or exact-input replay against the retained validated observations:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase3_benchmark.py \
  --reuse-report review/phase3-observations.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/phase3_rule_performance.py
```

Raw modes write separate Phase 3 files; `measurements.json` retains old fields
and adds current data under `phase3`. The historical Phase 1 measurement script
still rewrites its report file; do not run it over current consolidated evidence
without preserving the attached Phase 2/3 sections. The current standalone
scripts do not automatically regenerate this narrative or summary. Model files
live in the existing user cache; no model weights are added to the project.
