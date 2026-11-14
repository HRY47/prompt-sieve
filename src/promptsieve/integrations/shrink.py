"""The conservative step shared by `prompt-sieve run` and the agent hooks.

Only the default fast path is used: log/test-output compression plus terminal
cleanup, no ML, no network. Output below ``min_chars``, or output that does
not look like a log, is returned unchanged, character for character.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..counters import HeuristicCounter
from ..runtime import optimize
from ..strategies.logs import looks_like_log

DEFAULT_MIN_CHARS = 6000  # roughly 1,500 tokens


@dataclass
class Shrunk:
    text: str
    changed: bool
    original_tokens: int = 0
    tokens: int = 0


def shrink(text: str, min_chars: int = DEFAULT_MIN_CHARS) -> Shrunk:
    if len(text) < min_chars or not looks_like_log(text):
        return Shrunk(text, False)
    # A fixed heuristic counter keeps this offline and fast. Log compression
    # typically removes most of the text, so rough counts are enough to decide.
    res = optimize(text, counter=HeuristicCounter(), mode="fast", cache=False)
    best = res.best
    if not best.passed or best.name == "original" or best.text == text or best.tokens is None:
        return Shrunk(text, False)
    return Shrunk(best.text, True, res.original_tokens, best.tokens)


def count_lines(text: str) -> int:
    return text.count("\n") + (0 if not text or text.endswith("\n") else 1)


def footer(raw_path, original_lines: int) -> str:
    """``raw_path`` is None only when the caller chose not to save (--no-save)."""
    if raw_path is None:
        return (f"[PromptSieve shortened this output; the full {original_lines:,}-line version was not saved "
                "(--no-save). Rerun the command without prompt-sieve to see it.]")
    return (f"[PromptSieve shortened this output; the full {original_lines:,}-line version is saved to "
            f"{raw_path}. Search it with grep before reading it whole.]")
