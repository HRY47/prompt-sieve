"""`prompt-sieve run -- COMMAND ...`: run a command and shorten noisy output.

Works with any agent (Codex, Claude Code without the plugin, scripts):

* the command runs without a shell, with stdout and stderr merged in order;
* its exit status is returned unchanged, so test failures still fail;
* short or non-log output is passed through byte for byte;
* when output is shortened, the original bytes are saved privately first and
  the footer says where; if they cannot be saved, the original output is
  printed unshortened.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO, List, Mapping, Optional

from ..errors import UsageError
from .rawlog import save_raw
from .shrink import DEFAULT_MIN_CHARS, count_lines, footer, shrink


def run_command(argv: List[str], min_chars: int = DEFAULT_MIN_CHARS, save: bool = True,
                out: Optional[BinaryIO] = None, env: Mapping[str, str] = os.environ) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise UsageError("run needs a command, for example: prompt-sieve run -- pytest -q")
    out = out or sys.stdout.buffer
    executable = shutil.which(argv[0]) or argv[0]
    try:
        proc = subprocess.run([executable, *argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        out.write(b"prompt-sieve run: command not found\n")
        out.flush()
        return 127
    except PermissionError:
        out.write(b"prompt-sieve run: command is not executable\n")
        out.flush()
        return 126
    code = proc.returncode if proc.returncode >= 0 else 128 + (-proc.returncode)

    original = proc.stdout
    text = original.decode("utf-8", errors="replace")  # a separate copy, only for compression
    result = shrink(text, min_chars)
    raw = None
    if result.changed and save:
        raw = save_raw(original, label=Path(argv[0]).name, env=env)  # the exact bytes
        if raw is None:
            sys.stderr.write("prompt-sieve run: could not save the full output; showing it unshortened\n")
    if not result.changed or (save and raw is None):
        out.write(original)
    else:
        note = footer(raw, count_lines(text))
        if code:
            note = note[:-1] + f" Exit status {code}.]"
        out.write((result.text.rstrip("\n") + "\n" + note + "\n").encode("utf-8"))
    out.flush()
    return code
