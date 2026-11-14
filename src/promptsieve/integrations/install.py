"""`prompt-sieve install codex|claude-code`: copy the PromptSieve skill into an
agent's skills folder (or remove it again with --uninstall).

Codex reads skills from ~/.agents/skills (user) or .agents/skills (project).
Claude Code reads ~/.claude/skills or .claude/skills; the Claude Code plugin is
the recommended route there because it also adds the automatic output hook.

Only folders this command created (marked by MARKER) are ever changed.
"""
from __future__ import annotations

import os
import secrets
import stat
from importlib.resources import files
from pathlib import Path
from typing import List

from ..errors import UsageError

SKILL_NAME = "prompt-sieve"
MARKER = ".prompt-sieve-managed"
SKILL_DIRS = {
    "codex": Path(".agents") / "skills",
    "claude-code": Path(".claude") / "skills",
}
PLUGIN_HINT = ("For automatic compression of Bash output, install the Claude Code plugin instead:\n"
               "  claude plugin marketplace add HRY47/prompt-sieve\n"
               "  claude plugin install prompt-sieve@prompt-sieve")


def skill_text() -> str:
    return files("promptsieve.integrations").joinpath("skills", SKILL_NAME, "SKILL.md").read_text(encoding="utf-8")


def destination(agent: str, scope: str, home: Path, cwd: Path) -> Path:
    if agent not in SKILL_DIRS:
        raise UsageError("unknown agent (use codex or claude-code)")
    if scope not in ("user", "project"):
        raise UsageError("scope must be user or project")
    return (home if scope == "user" else cwd) / SKILL_DIRS[agent] / SKILL_NAME


def _entry_kind(path: Path) -> str:
    """'missing', 'file' or 'other' (links, folders, devices), without following links."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return "missing"
    return "file" if stat.S_ISREG(mode) else "other"


def _write_new(path: Path, text: str) -> None:
    """Write via a fresh temporary file, then rename over the entry.

    A rename replaces the directory entry itself, so even a link that appears
    after our checks is replaced rather than followed.
    """
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def install(agent: str, scope: str = "user", home: Path | None = None, cwd: Path | None = None,
            uninstall: bool = False, force: bool = False, dry_run: bool = False) -> List[str]:
    dest = destination(agent, scope, home or Path.home(), cwd or Path.cwd())
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise UsageError("the skill destination is a link or not a folder; remove it or choose another scope")
    entries = {name: _entry_kind(dest / name) for name in ("SKILL.md", MARKER)}
    if "other" in entries.values():
        # Never write through, or delete, a link or special file, even with --force.
        raise UsageError("the skill folder contains a link or special file where SKILL.md or its "
                         "marker belongs; remove it by hand and retry")
    ours = entries[MARKER] == "file"
    if dest.exists() and not ours and not force:
        raise UsageError("a prompt-sieve skill folder that this tool did not create already exists; "
                         "pass --force to replace it")
    prefix = "would " if dry_run else ""

    if uninstall:
        if not dest.exists():
            return [f"nothing to remove at {dest}"]
        if not dry_run:
            for name in ("SKILL.md", MARKER):
                (dest / name).unlink(missing_ok=True)
            if not any(dest.iterdir()):
                dest.rmdir()
        return [f"{prefix}remove the PromptSieve skill from {dest}"]

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        _write_new(dest / "SKILL.md", skill_text())
        _write_new(dest / MARKER, "Installed by `prompt-sieve install`; safe to delete with --uninstall.\n")
    messages = [f"{prefix}install the PromptSieve skill to {dest}"]
    if agent == "codex":
        messages.append("Restart Codex if the skill does not appear. Ask it to run noisy commands as "
                        "`prompt-sieve run -- <command>`.")
    else:
        messages.append(PLUGIN_HINT)
    return messages
