# Security and data handling

This is an experimental local CLI/library, not a sandbox or injection filter.
Do not use it as a network service accepting untrusted backend specifications.

- Prompt/log text enters the shell backend only through UTF-8 stdin. The
  configured `cmd:` command itself executes through a shell with account
  permissions. Keep it trusted. Timeouts attempt process-tree termination;
  detached processes and commands controlling external services require
  independent isolation. Subprocess output is buffered without a size limit.
- `prompt_file` resolves relative to the case file and must stay inside its
  directory; absolute paths and `..` escapes are rejected unless
  `--allow-external-prompt-files` is given. Still review downloaded case files
  before running them against an API.
- API runners and the Anthropic token counter transmit input to their providers.
  Ollama uses `OLLAMA_HOST`, which can point to a remote service. This setting
  must be trusted. No backend is automatically invoked by `doctor` beyond local
  imports and tiktoken encoding loading/download.
- Standard reports omit raw answers and backend stderr. Explicit answer/text
  output and application logs can still disclose secrets. Metadata such as
  case IDs can contain secrets if the caller puts them there. Review artifacts
  before publication; redaction is not a general secret detector.
- Text may contain prompt injections. Preserve trust boundaries when sending it
  to a model. Removing words or delimiters can strengthen malicious instructions.
- Dependency/model installation executes or loads upstream software/data. Model
  revisions are not pinned. HF tokenizer and LLMLingua remote code are explicitly disabled;
  third-party loaders and weights remain part of the trust boundary. Use trusted
  sources, patched dependencies and an isolated environment.
- Regex scoring evaluates case-authored regular expressions. Malicious regexes,
  very large input, huge output and many translation candidates can exhaust CPU
  or memory. Run untrusted evaluation workloads with OS resource limits.

No credentials belong in cases, commands, reports, source files, or commits.
Use provider environment variables. `.env`, keys and local reports are ignored
by default; `.gitignore` does not remove files already committed.

To report a vulnerability, use GitHub's private vulnerability reporting on
https://github.com/HRY47/prompt-sieve (Security tab → "Report a vulnerability").
Do not include live credentials or private prompts in a public issue.

Phase 3 result caching retains source/candidate text only in bounded process
memory (TTL/LRU), never in a disk cache. Disable it with `cache=False` or clear
its cache object for sensitive sessions. This does not redact explicit JSON/text
exports or restrict third-party model caches. Deadline workers receive trusted
Python adapter objects through local multiprocessing serialization; never accept
adapter objects or serialized cache contents from untrusted inputs. WSL deadline
cleanup was measured; other platforms and detached custom subprocesses require
separate verification. The CLI preserves its source file; output paths are still
trusted local configuration rather than a filesystem security boundary.

CLI diagnostic policy:

- Dependency and backend failures show a fixed operation label and exception
  class, with actionable configuration hints. Raw exception messages, chained
  exceptions, traceback bodies, counter specifications and private file paths
  are withheld. Normal CLI errors do not save tracebacks or diagnostic files.
- Argument-parser failures do not echo rejected arguments or values; use the
  displayed command's `--help` to check valid options. Case validation identifies
  the JSONL line number rather than copying caller-supplied case IDs or paths.
- Command and counter/doctor initialization warnings are captured and replaced with static
  notices. The fallback to a rough heuristic counter remains visible.
- This policy covers PromptSieve's diagnostics. Optimized stdout, explicit text
  or answer exports, successful metadata, and independent third-party logging
  can contain input data. Review those outputs before sharing; the CLI does not
  make arbitrary text or third-party code safe to publish.

Local files and release inputs:

- Ignore rules cover local credential stores, private-key containers, targeted
  private-data directories, model-download directories, weights and generated
  caches. Public examples and historical benchmark fixtures remain visible.
  Keep tokens out of deliberately shared configuration, including `.npmrc` and
  `.env.example`. Git ignore rules do not protect already-tracked files or
  intentional forced additions, and they do not filter packaging manifests.
- Inspect and secret-scan the built wheel and source archive as well as current
  tracked files and all reachable Git history before each release. Do not add
  real credentials to tests; use clearly synthetic sentinels. Retain historical
  evidence and scope any scanner exceptions to individually verified checksums.
- CI actions are pinned to verified full commit SHAs of the existing
  major versions, with persisted checkout credentials disabled. Dependabot
  proposes Actions and Python dependency updates for review; it does not merge
  or publish releases. Review updates and test them before merging.
- Library dependency ranges remain compatible rather than frozen to one local
  ML environment. For reproducible release validation, retain tested resolver
  versions/constraints and hashes per Python/platform, and record immutable
  model revisions. Neither a version range nor a successful import is evidence
  that an upstream package/model is secure.
- Build releases from an explicitly reviewed commit and verify their source
  provenance. Security corrections to an existing public release should get a
  new patch version; do not silently move the existing public tag. Historical
  author metadata and machine paths require a separate owner-approved privacy
  decision; ordinary ignore changes do not erase public history.
