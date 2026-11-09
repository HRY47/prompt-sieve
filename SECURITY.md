# Security and data handling

This is an experimental local CLI/library, not a sandbox or injection filter.
Do not use it as a network service accepting untrusted backend specifications.

- Prompt/log text enters the shell backend only through UTF-8 stdin. The
  configured `cmd:` command itself executes through a shell with account
  permissions. Keep it trusted. Timeouts attempt process-tree termination;
  detached processes and commands controlling external services require
  independent isolation. Subprocess output is buffered without a size limit.
- Cases are trusted configuration. `prompt_file` resolves relative to the case
  file and intentionally allows absolute paths and parent directories. Never
  run an unreviewed case file against an API: it could disclose local files.
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
  revisions are not pinned. HF tokenizer remote code is explicitly disabled;
  third-party loaders and weights remain part of the trust boundary. Use trusted
  sources, patched dependencies and an isolated environment.
- Regex scoring evaluates case-authored regular expressions. Malicious regexes,
  very large input, huge output and many translation candidates can exhaust CPU
  or memory. Run untrusted evaluation workloads with OS resource limits.

No credentials belong in cases, commands, reports, source files, or commits.
Use provider environment variables. `.env`, keys and local reports are ignored
by default; `.gitignore` does not remove files already committed.

For a vulnerability, contact the repository maintainer privately using the
eventual repository's security advisory/contact channel. Do not include live
credentials or private prompts in a public issue. A repository/contact channel
has not been created by this review.

Phase 3 result caching retains source/candidate text only in bounded process
memory (TTL/LRU), never in a disk cache. Disable it with `cache=False` or clear
its cache object for sensitive sessions. This does not redact explicit JSON/text
exports or restrict third-party model caches. Deadline workers receive trusted
Python adapter objects through local multiprocessing serialization; never accept
adapter objects or serialized cache contents from untrusted inputs. WSL deadline
cleanup was measured; other platforms and detached custom subprocesses require
separate verification. The CLI preserves its source file; output paths are still
trusted local configuration rather than a filesystem security boundary.
