# Final independent review: PromptSieve (tokopt 0.1.0)

Date: 2026-11-08. Reviewer: Claude (Opus 5.5). Scope: everything in the
`prompt-sieve-main` snapshot after Codex's Phase 1–3 work. The source was read
against `REVIEW.md`, `PHASE3.md`, the raw benchmark JSON, README, SECURITY.md,
packaging, CI and the Claude Code skill. Tests were re-run. No paid API, no
untrusted command and no publication was used.

**Verdict: ready with specific documented limitations** (experimental alpha),
after the fixes below. Before you announce it, push and confirm that CI is green.
Release it as a log/test-output compressor with experimental ML research
features. Do not present it as a general prompt optimizer that keeps answers
unchanged.

---

## 1. What Codex improved successfully

I checked each of these in the code itself, not only in the reports:

- **Normalization is now safe around code.** Fenced blocks are protected before
  ANSI/CRLF cleanup. Carriage-return lines are kept unless they are recognized
  progress updates. Single and double trailing spaces are preserved.
- **Preservation checks are much stronger.** They cover signed numbers and exponents,
  Persian/Arabic/full-width digits, number–unit pairs, elementary expressions,
  whole clauses containing negations or modal words, quoted literals, paths
  (including Windows/UNC) and identifiers. Literal placeholders can no longer
  collide with input text.
- **Translation.** Argos translates the prose between protected literals
  (Argos damaged placeholders in real tests). Round-trip checks now also require
  critical items and words in the back-translation, and translator failures
  report only the exception type.
- **Benchmark honesty.** Accuracy and agreement are now separate. Paired
  regressions and improvements are listed by case ID. The question is counted
  and the savings gate rechecked on the final prompt. Raw answers are opt-in,
  and backend errors are redacted.
- **Process handling.** The command runner uses a process group and kills the
  whole tree on timeout. API runners have timeouts and no silent retries. The
  optimization deadline uses a *spawned* worker (no fork of a live torch
  runtime), cleans up the process group, returns the exact original and
  reports unknown counts as `null` rather than inventing them.
- **Bounded cache.** It is in-memory and bounded by entries, bytes and TTL.
  It deliberately excludes ML, translation, remote counters and timed runs,
  and keys include the counter identity.
- **Evidence handling.** Failing evidence (the Persian 42→10 regression, the two
  LLMLingua unit regressions) was kept rather than replaced by passing mocks.
  The README is unusually candid about limits.

Costs of this work: `runtime.py` is written in a dense one-line style that is
hard to maintain. `review/` adds about 7.3 MB of raw JSON. The single-space
preservation rule is calibrated on one Qwen case. The clause-level check
blocks pruning or translating any sentence containing words such as *must*,
*should*, *may*, *only*, *before* or *after*. That is safe, but it removes
much of the ML strategies' usefulness on instruction-style text. None of these
are bugs.

## 2. What remains incorrect, unsafe or unverified

Fixed in this review (details in section 3):

- LLMLingua could run arbitrary code from any model repository.
- A case file could read any file and send it to a remote API.
- Log compression silently dropped the user's instructions.
- CI failed on its first run.
- Error messages were useless.
- The fast-mode benchmark claims were misleading.

Still open, documented but not fixed:

| Area | Issue |
|---|---|
| Verification | ML, translation and tiktoken paths could not be exercised here (no model/encoding downloads in this sandbox). The `trust_remote_code=False` change is covered by a contract test only. Run `TOKOPT_TEST_LINGUA=1` locally to confirm the published Microsoft models still load. They use standard BERT/XLM-R token-classification classes, so they should. |
| Platforms | Windows and GitHub CI have never actually run. |
| Logs | Inside a log body, any line far from a recognized diagnostic is still omitted when the log is long. A question about non-error content in the middle of a log can become unanswerable. The footer says how much was removed. |
| CLI | With `--max-tokens` and an unverified fallback, the CLI writes the original to stdout/`-o` and *then* exits non-zero. A pipeline without `pipefail` still forwards it. This behaviour is documented. |
| Benchmark | The original prompt always runs first, so prompt caching and warm-up bias the latency comparison. Case-authored regexes can be slow (ReDoS). Subprocess output is unbounded. |
| Models | Model revisions are not pinned; LLMLingua's loader also calls `tiktoken` at startup. |
| Data | Default counter silently degrades to the rough heuristic if tiktoken can't load (a warning is printed). |

## 3. Bugs found and fixes made

Every fix has a regression test in `tests/test_final_review.py` (19 tests).
I ran these tests against the original code: 15 fail there and 4 pass. The 4
that pass are guard tests that pin behaviour that was already correct.

1. **CI failure (your screenshot).** `ubuntu-latest` now resolves to Ubuntu
   26.04, which has no `setup-python` build of Python 3.9. Python 3.9 has also
   been end-of-life since October 2025. I raised `requires-python` to `>=3.10`,
   pinned Linux jobs to `ubuntu-24.04`, and set the matrix to 3.10/3.12/3.13
   plus Windows 3.13. The full suite passes locally on 3.10 and 3.13.
2. **Security: remote code in LLMLingua.** `llmlingua` 0.2.2 defaults
   `trust_remote_code=True` (checked in its source, `load_model`). With
   `--lingua-model some/repo`, any Python in that Hub repo would run. tokopt
   now passes `model_config={"trust_remote_code": False}`, as a fresh dict
   because upstream mutates its default argument.
3. **Security: arbitrary file read in benchmarks.** A case file's `prompt_file`
   could be absolute or `../`. A downloaded case file could therefore read
   `~/.ssh/id_rsa` and send it to OpenAI/Anthropic. Paths must now resolve
   inside the cases directory. `--allow-external-prompt-files` opts out.
4. **Silent loss of user instructions (default fast mode).** In a prompt made of
   instructions followed by a long log, only the first five lines survived. In
   my reproduction, "IMPORTANT INSTRUCTION: reply in French…" and "Do not
   suggest reinstalling." were dropped, and `verify()` still passed. Text before
   the first and after the last recognized log line is now kept verbatim and
   enforced by `verify()`. Python exception lines (`ConnectionError: …`) are
   recognized so tracebacks stay contiguous. Long logs with *no* diagnostic or
   summary are no longer cut to head+tail blindly. On Codex's 120-case corpus
   all fast-mode selected texts are still byte-identical to the recorded ones,
   so the existing evidence still applies.
5. **Opaque CLI errors.** Every error printed `error: ValueError; check input
   files…`, including "you must pass --counter" and "output equals input". A
   new `UsageError` carries tokopt's own validation messages, which never
   contain input text, and the CLI prints them. Backend exceptions are still
   redacted (there is a test for this).
6. **Cosmetic normalization in full mode.** The fast-mode fix for the Persian
   regression did not apply to `--mode full`, `experiment`, or Python calls that
   request lingua/translation. In those, a one-token whitespace edit could still
   be selected. Cosmetic savings below `min_savings` are now declined in every
   mode unless `normalize` is requested explicitly.
7. **Wrong token count on LLMLingua failure.** The rejected candidate reused a
   stale count from a previous candidate. It now reports the original count and
   0% savings.
8. **`OLLAMA_HOST` without a scheme.** Ollama's own form `127.0.0.1:11434`
   failed with "unknown url type". Bare hosts are now accepted, and non-HTTP
   schemes are rejected.
9. **Packaging and documentation.**
   - SPDX `license = "MIT"`, which removes the setuptools deprecation warning.
   - The sdist no longer ships 7 MB of raw review JSON.
   - README retitled PromptSieve, with corrected Python/CI statements and a new
     "What the evidence shows" section.
   - The skill's lingua example now needs `--mode full`, because the CLI
     applies fast-mode cost gates to `-s lingua`.
   - SECURITY.md points to GitHub private vulnerability reporting.
   - A correction note was added at the top of REVIEW.md and PHASE3.md.
   - Your local Windows path (`/mnt/c/Users/<name>/…`) was redacted to
     `<project>` in seven evidence files. Only paths changed; all JSON still
     parses.

## 4. Final automated test results

| Environment | Result |
|---|---|
| Python 3.13, offline suite | **176 passed, 25 skipped** |
| Python 3.10, offline suite | **176 passed, 25 skipped** |
| Wheel built and installed into a clean 3.10 venv outside the tree | CLI and `--timeout` worker work |

The 25 skips are the opt-in integration tests (tiktoken, Argos, LLMLingua, HF,
NLLB, pinned Qwen), which need downloads unavailable here. Codex's report of
182 passing includes them, from its own WSL run. I did not reproduce those
integration results.

## 5. Are the benchmark claims justified?

| Claim | Assessment |
|---|---|
| Fast mode: 41.6% fewer input tokens | **Measured, but misleading as a headline.** Fast changed only the 10 log fixtures, which are 44.6% of the corpus's tokens, compressing each from 1,134 to 76 chat tokens. The other 110 prompts were unchanged. Median per-prompt saving is 0%; mean per-prompt is 7.8%. The number depends entirely on the corpus mix. |
| Fast mode: zero paired regressions | **Vacuous.** 110 inputs were byte-identical to the original. All 10 changed cases were *wrong at baseline*, so a regression was impossible by definition. |
| Fast mode: 69/120, no change in accuracy | **A scoring artefact.** On all 10 compressed logs, the model answered with the right failed test (`tests/test_api.py::test_failure_N`, expected `test_failure_N`). The strict exact scorer marked these wrong. Under `contains`, accuracy on these cases goes from 0/10 to 10/10, a probable *improvement* that the report hides. One 0.5B model on synthetic fixtures, so treat this as anecdotal. |
| LLMLingua / combined: no regressions after the guard | **In-sample only**, and the report says so. The guard was designed after seeing two regressions on this corpus. The final 720-arm run reused recorded answers, with 0 fresh generations. |
| Recorded-answer replay | Correctly disclosed. It shows the final code selects the same inputs; it does not re-measure the model. |
| Held-out validation | **None exists.** |
| 0.052 s for 120 prompts, ~97 MiB RSS | Measured, optimizer-only, fresh process; plausible for regex work. It does not describe target-model latency or memory. |
| RAM/VRAM savings | **Not measured.** The README now says so plainly; earlier figures were estimates. |
| Translation saves tokens | Not supported. One of 120 cases selected a translation, a 0.5% aggregate saving. |

## 6. Recommended before public release

1. Push and confirm the CI matrix is green, including Windows.
2. Enable GitHub private vulnerability reporting (referenced in SECURITY.md).
3. Run `TOKOPT_TEST_LINGUA=1` once to confirm the `trust_remote_code=False`
   loading change against the real models.
4. If you want a benchmark headline: fix the log-case scorer (`contains` or
   node-ID aware), then run **fresh** generations on a **held-out** case set
   with a stronger model. Report per-family results, not only the aggregate.
5. Optional: alternate arm order in `tokopt bench`, pin LLMLingua model
   revisions, cap subprocess output, and stop writing unverified fallbacks
   when `--max-tokens` fails.

## 7. Release-readiness verdict

**Ready with specific documented limitations.** The default fast path is
deterministic and dependency-free. It is now guarded against the
instruction-loss case and has well-documented limits. The two security
issues found are fixed. The ML and translation features are clearly labelled
experimental, and the README no longer lets the 41.6% / zero-regression
numbers stand without context. The remaining gap is CI on GitHub itself: it has
never passed. Treat a green run as the last gate before announcing the
repository.
