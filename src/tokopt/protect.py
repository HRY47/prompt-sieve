"""Split text into prose and protected segments, and mask fragile spans.

Fenced code blocks are never compressed or translated. Inside prose, inline
code, URLs, file paths and identifiers are swapped for placeholders like
``⟦0⟧`` before a lossy transform and restored afterwards. If a transform
destroys a placeholder, :func:`unmask` reports it and the candidate fails the
preservation check instead of silently losing data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_FENCE = re.compile(
    r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*(?:\n"
    r".*?(?:^[ ]{0,3}(?P=fence)[`~]*[ \t\r]*(?=\n|$)|\Z)|\Z)", re.M | re.S
)

# Order matters: longest/most specific first.
_MASK_PATTERNS = [
    r"⟦\s*\d+\s*⟧",                              # literal placeholder, prevent collisions
    r"`[^`\n]+`",                                   # inline code
    r"https?://[^\s)>\]]+",                         # URLs
    r"(?:[A-Za-z]:\\|\\\\)[^\s`\"'<>|]+",       # Windows paths / UNC
    r"(?<!\w)/?(?:[\w.-]+/)+[\w.-]+",             # relative / absolute POSIX paths
    r"\b[\w-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|toml|ini|cfg|txt|md|csv|log|java|sh|exe)\b",
    r'"[^"\n]*"|\'[^\'\n]*\'',                    # quoted literals
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",                 # emails
    r"\b[A-Za-z_]+[A-Za-z0-9]*_[A-Za-z0-9_]+\b",    # snake_case
    r"\b[a-z]+[A-Z][A-Za-z0-9]*\b",                 # camelCase
    r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b",         # PascalCase
    r"--?[A-Za-z][\w-]*",                           # CLI flags
]
_MASK = re.compile("|".join(f"(?:{p})" for p in _MASK_PATTERNS))
_PLACEHOLDER = re.compile(r"⟦\s*(\d+)\s*⟧")


@dataclass
class Segment:
    text: str
    protected: bool  # True -> never transform


def split_segments(text: str) -> List[Segment]:
    """Split into alternating prose / fenced-code segments. Prose is further
    split on blank lines so per-paragraph decisions are possible."""
    out: List[Segment] = []
    pos = 0
    for m in _FENCE.finditer(text):
        if m.start() > pos:
            out.extend(_split_paragraphs(text[pos:m.start()]))
        out.append(Segment(m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.extend(_split_paragraphs(text[pos:]))
    return out


def _split_paragraphs(chunk: str) -> List[Segment]:
    parts = re.split(r"(\n\s*\n)", chunk)
    segs = []
    for p in parts:
        if not p:
            continue
        # blank separators are "protected" so they are kept verbatim
        # Leave structured Markdown and indented code intact. Translating whole
        # lists/tables can destroy delimiters, indentation and relationships.
        structured = re.search(r"^(?:[ \t]{4}|\t|[ ]{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||(?:---|\*\*\*|___)\s*$))", p, re.M)
        segs.append(Segment(p, protected=not p.strip() or bool(structured)))
    return segs


def join_segments(segs: List[Segment]) -> str:
    return "".join(s.text for s in segs)


def mask(text: str) -> Tuple[str, List[str]]:
    saved: List[str] = []

    def repl(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"⟦{len(saved) - 1}⟧"

    return _MASK.sub(repl, text), saved


def unmask(text: str, saved: List[str]) -> Tuple[str, List[str]]:
    """Restore placeholders. Returns (text, lost_items)."""
    seen = set()

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        if i < len(saved):
            seen.add(i)
            return saved[i]
        return m.group(0)

    restored = _PLACEHOLDER.sub(repl, text)
    lost = [s for i, s in enumerate(saved) if i not in seen]
    return restored, lost
