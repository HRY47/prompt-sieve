"""Claude Code PostToolUse hook: shorten large log-like Bash output.

Claude Code sends the finished tool call as JSON on stdin. If the Bash output
is large and looks like a log (test runs, builds, installs), this prints a
``hookSpecificOutput.updatedToolOutput`` with the compressed text, so Claude
reads the short version. The command has already run; only what Claude sees
changes. The full output is saved privately and the footer names the file.

The hook fails open: on any problem it prints nothing and exits 0, and Claude
sees the original output.

Environment:
  PROMPTSIEVE_HOOK=off               disable without uninstalling
  PROMPTSIEVE_HOOK_MIN_CHARS=6000    only consider output at least this long
  PROMPTSIEVE_LOG_DIR=DIR            where full outputs are saved
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from typing import Mapping, Optional, TextIO

from .rawlog import save_raw
from .shrink import DEFAULT_MIN_CHARS, count_lines, footer, shrink

# Commands whose output Claude asked for verbatim, usually to read or edit files.
EXACT_COMMANDS = {
    "cat", "head", "tail", "less", "more", "sed", "awk", "gawk", "grep", "egrep", "fgrep", "rg", "ag",
    "diff", "cmp", "jq", "yq", "find", "fd", "ls", "tree", "wc", "stat", "file", "od", "xxd",
    "hexdump", "bat", "nl", "sort", "uniq", "cut", "echo", "printf", "env", "printenv", "tac", "strings",
}
EXACT_GIT = {"diff", "show", "log", "blame", "status", "grep", "cat-file"}

# Prefix commands that run another command: options that take a value, and how
# many plain arguments come before the wrapped command.
WRAPPERS = {
    "sudo": ({"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "--user", "--group",
              "--host", "--prompt", "--chdir", "--role", "--type", "--other-user"}, 0),
    "doas": ({"-u", "-C"}, 0),
    "command": (set(), 0),
    "builtin": (set(), 0),
    "exec": ({"-a"}, 0),
    "env": ({"-u", "-C", "--unset", "--chdir"}, 0),
    "time": ({"-f", "-o", "--format", "--output"}, 0),
    "nice": ({"-n", "--adjustment"}, 0),
    "nohup": (set(), 0),
    "stdbuf": ({"-i", "-o", "-e", "--input", "--output", "--error"}, 0),
    "unbuffer": (set(), 0),
    "ionice": ({"-c", "-n", "-p", "--class", "--classdata"}, 0),
    "timeout": ({"-s", "-k", "--signal", "--kill-after"}, 1),
    "xargs": ({"-a", "-d", "-E", "-e", "-I", "-i", "-L", "-l", "-n", "-P", "-s", "--arg-file",
               "--delimiter", "--eof", "--replace", "--max-lines", "--max-args", "--max-procs",
               "--max-chars"}, 0),
}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
SHELL_OPTIONS_WITH_VALUE = {"-O", "+O", "-o", "+o", "--rcfile", "--init-file"}
# Words this parser does not interpret. Seeing one as a command word means the
# command line is treated as uncertain, so its output is left unshortened.
UNCERTAIN = {
    "eval", "source", ".", "alias", "function", "coproc", "select",
    "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "{", "}", "[[", "]]", "!", "in",
}
_PUNCTUATION = "();<>|&\n"
_CONTROL = set(";|&()\n")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DYNAMIC = re.compile(r"[$*?\[\]~{}]")
_SHELL_TOKEN = re.compile(
    r"""[ \t\r]+|[();<>|&\n]+|(?:[^ \t\r\n'"\\();<>|&]+|'[^']*'|"(?:[^"\\]|\\[\s\S])*"|\\[\s\S])+"""
)


def _shell_tokens(command: str):
    """Yield raw words and their offsets before shlex removes quoting and spaces."""
    pos = 0
    while pos < len(command):
        match = _SHELL_TOKEN.match(command, pos)
        if match is None:
            raise ValueError("unsupported shell word or unbalanced quotes")
        raw, start = match.group(), pos
        pos = match.end()
        if raw[0] in " \t\r":
            continue
        punct = set(raw) <= set(_PUNCTUATION)
        if punct:
            word = raw
        else:
            words = shlex.split(raw, comments=False, posix=True)
            if len(words) != 1:
                raise ValueError("unsupported shell word")
            word = words[0]
        yield raw, word, start, pos, punct


def _segments(command: str):
    """Shell words grouped into simple commands, with redirections removed.

    Raises ValueError for anything this deliberately small parser does not
    model: command substitution, process substitution, comments, unbalanced
    quotes. Callers treat that as "leave the output alone".
    """
    if "$(" in command or "`" in command or "<(" in command or ">(" in command:
        raise ValueError("command substitution")
    segment, expect_target = [], False
    last_raw, last_end = "", None
    for raw, token, start, end, punct in _shell_tokens(command):
        if token.startswith("#"):
            raise ValueError("comment")
        if punct and ("<" in token or ">" in token):
            # Only an unquoted number attached to the operator is a descriptor.
            # `timeout 30 <input` keeps its duration; `timeout 30 0<input` drops 0.
            if segment and last_end == start and last_raw.isascii() and last_raw.isdigit():
                segment.pop()
            last_raw, last_end = "", None
            expect_target = True
            continue
        if punct and set(token) & _CONTROL:
            if expect_target:
                raise ValueError("redirection without a target")
            if segment:
                yield segment
            segment = []
            last_raw, last_end = "", None
            continue
        if expect_target:
            expect_target = False
            continue
        segment.append(token)
        last_raw, last_end = raw, end
    if expect_target:
        raise ValueError("redirection without a target")
    if segment:
        yield segment


def _unwrap(words):
    """Strip assignments and wrapper commands; return the words of the real command."""
    words = list(words)
    wrapped = False
    for _ in range(10):
        while words and _ASSIGN.match(words[0]):
            words.pop(0)
        if not words:
            if wrapped:
                raise ValueError("wrapper without a command")  # e.g. bare `xargs` runs echo
            return words
        name = os.path.basename(words[0])
        if name not in WRAPPERS:
            return words
        if name == "env" and any(w in ("-S", "--split-string") or w.startswith("--split-string=") for w in words):
            raise ValueError("env -S")
        with_value, positional = WRAPPERS[name]
        words.pop(0)
        wrapped = True
        while words and words[0].startswith("-") and words[0] != "-":
            option = words.pop(0)
            if option == "--":
                break
            if option in with_value and words:
                words.pop(0)
        for _ in range(positional):
            if words:
                words.pop(0)
    raise ValueError("too many wrappers")


def _shell_script(words):
    """The script of `bash [options] -c SCRIPT`, None if there is no -c.

    Raises ValueError if -c is present but the script cannot be identified.
    """
    args = iter(words[1:])
    found_c = False
    for word in args:
        if word in SHELL_OPTIONS_WITH_VALUE:
            next(args, None)
        elif word == "--":
            break
        elif word.startswith(("-", "+")) and not word.startswith("--"):
            found_c = found_c or "c" in word[1:]
        elif word.startswith("--"):
            continue
        else:
            if found_c:
                return word
            return None  # `bash script.sh`: runs a file, not a command string
    if found_c:
        script = next(args, None)
        if script is not None:
            return script
        raise ValueError("shell -c without a script")
    return None


def should_skip(command: str, _depth: int = 0) -> bool:
    """True when the output should reach Claude unchanged.

    That is the case for commands that print files or diffs verbatim, for
    output already produced by prompt-sieve, and for any command line this
    parser cannot read with confidence.
    """
    if "prompt-sieve" in command or "promptsieve" in command or _depth > 3:
        return True
    try:
        for segment in _segments(command):
            words = _unwrap(segment)
            if not words:
                continue
            if _DYNAMIC.search(words[0]):
                return True  # e.g. "$READER" build.log: the program is not known here
            name = os.path.basename(words[0])
            if name in UNCERTAIN or name in EXACT_COMMANDS:
                return True
            if name == "git" and _git_subcommand(words[1:]) in EXACT_GIT:
                return True
            if name in SHELLS:
                script = _shell_script(words)
                if script is not None and should_skip(script, _depth + 1):
                    return True
    except ValueError:
        return True  # unbalanced quotes, substitutions, unusual wrappers: leave it alone
    return False


def _git_subcommand(args):
    """First non-option word, skipping global options such as `-C dir`."""
    skip_next = False
    for word in args:
        if skip_next:
            skip_next = False
        elif word in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            skip_next = True
        elif not word.startswith("-"):
            return word
    return None


def _min_chars(env: Mapping[str, str]) -> int:
    try:
        return max(0, int(env.get("PROMPTSIEVE_HOOK_MIN_CHARS", DEFAULT_MIN_CHARS)))
    except ValueError:
        return DEFAULT_MIN_CHARS


def handle(payload: dict, env: Mapping[str, str] = os.environ) -> Optional[dict]:
    """Return the hook's JSON response, or None to leave the output alone."""
    if str(env.get("PROMPTSIEVE_HOOK", "")).lower() in ("0", "off", "false", "no"):
        return None
    if payload.get("tool_name") != "Bash" or payload.get("hook_event_name", "PostToolUse") != "PostToolUse":
        return None
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or tool_input.get("run_in_background") or should_skip(command):
        return None
    response = payload.get("tool_response")
    if not isinstance(response, dict) or response.get("isImage"):
        return None
    stdout, stderr = response.get("stdout", ""), response.get("stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None

    limit = _min_chars(env)
    new_out, new_err = shrink(stdout, limit), shrink(stderr, limit)
    if not (new_out.changed or new_err.changed):
        return None

    raw = stdout if not stderr else f"{stdout}\n----- stderr -----\n{stderr}"
    path = save_raw(raw, label="bash", env=env)
    if path is None:
        return None  # never hide output that we could not keep a full copy of
    lines = count_lines(stdout) + count_lines(stderr)
    updated = dict(response)  # keep every field Claude Code expects
    updated["stdout"] = new_out.text.rstrip("\n") + "\n" + footer(path, lines) + "\n"
    updated["stderr"] = new_err.text
    return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": updated}}


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
         env: Mapping[str, str] = os.environ) -> int:
    try:
        payload = json.loads(stdin.read())
        result = handle(payload, env) if isinstance(payload, dict) else None
        if result is not None:
            stdout.write(json.dumps(result, ensure_ascii=False))
            stdout.flush()
    except Exception:
        pass  # fail open: Claude sees the original output
    return 0
