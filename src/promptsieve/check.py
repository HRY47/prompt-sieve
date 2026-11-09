"""Meaning-preservation checks.

Two complementary checks:

* :func:`preservation` - hard facts must survive: numbers, URLs, paths,
  identifiers, inline code, fenced code blocks, quoted strings, and (for
  same-language transforms) negations / modal words like "not", "never", "must".
* :func:`chrf` - character n-gram F-score, used to compare a round-trip
  translation (EN -> X -> EN) against the original. Language-independent and
  dependency-free.

Neither proves meaning is preserved. ``prompt-sieve bench`` (task accuracy) is the
real test; these are cheap filters that catch the obvious breakage.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from .protect import _FENCE, _MASK

# Persian, Arabic-Indic and full-width digits -> ASCII, so "۲۰۲۶" == "2026".
_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹" "٠١٢٣٤٥٦٧٨٩" "０１２３４５６７８９" "٫٬",
    "0123456789" "0123456789" "0123456789" ".,",
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[+\-−]?\d+(?:[.,]\d+)*(?:[eE][+\-]?\d+)?%?(?![A-Za-z0-9_])")
# Preserve number/unit associations and complete elementary expressions. This is
# conservative pattern recognition, not an algebra or dimensional-analysis tool.
_MEASURE = re.compile(r"[+\-−]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?[ \t]*(?:mg|kg|g|km|cm|mm|mL|ml|L|ms|s|Hz|kHz|MHz|V|mV|W|kW|MB|GB|MiB|GiB|°C|°F|meters?|seconds?|minutes?|hours?|dollars?|USD|EUR|percent)(?![A-Za-z])")
_EXPRESSION = re.compile(r"(?<![\w/])[A-Za-z0-9_().]+(?:[ \t]*(?:\*\*|//|<=|>=|==|!=|[+*/^=<>−-])[ \t]*[A-Za-z0-9_().]+)+")

_QUOTED = re.compile(r'"([^"\n]{2,80})"')
_CRITICAL_WORDS = {
    "not", "no", "never", "none", "nothing", "don't", "doesn't", "didn't", "isn't",
    "aren't", "won't", "can't", "cannot", "shouldn't", "mustn't", "without",
    "must", "only", "always", "except", "unless", "before", "after",
    "should", "shall", "may", "might", "could", "required", "forbidden",
    "prohibited", "allowed", "least", "most", "exactly",
    "هرگز", "باید", "نباید", "فقط", "بدون", "نکنید", "نیست",
}


def normalize_digits(text: str) -> str:
    return text.translate(_DIGITS)


def critical_items(text: str, same_language: bool = True) -> List[str]:
    items: List[str] = []
    items += [m.group(0).strip() for m in _FENCE.finditer(text)]
    prose = _FENCE.sub(" ", text)
    items += [m.group(0) for m in _MASK.finditer(prose)]
    items += [n.replace(",", "") for n in _NUMBER.findall(normalize_digits(_MASK.sub(" ", prose)))]
    if same_language:
        items += _QUOTED.findall(prose)
        # Literal multisets cannot catch a negation attached to the wrong
        # action. Retain whole recognized negated clauses in same-language work.
        for clause in re.split(r"[;.!?。！？\n]+", prose):
            if critical_word_counts(clause):
                items.append("clause:" + re.sub(r"\s+", " ", clause).strip().lower())
        literals_removed = _MASK.sub(" ", prose)
        items += [re.sub(r"[ \t]+", "", m.group(0)) for m in _MEASURE.finditer(prose)]
        items += [re.sub(r"[ \t]+", "", m.group(0)) for m in _EXPRESSION.finditer(literals_removed)]
    return items


@dataclass
class Preservation:
    score: float                      # fraction of critical items kept (1.0 = all)
    missing: List[str] = field(default_factory=list)
    lost_critical_words: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.lost_critical_words


def preservation(original: str, candidate: str, same_language: bool = True) -> Preservation:
    want = Counter(critical_items(original, same_language))
    have_items = Counter(critical_items(candidate, same_language))
    missing: List[str] = []
    for item, n in want.items():
        have = have_items[item]
        if have < n:
            missing += [item] * (n - have)
    total = sum(want.values())
    score = 1.0 if total == 0 else 1 - len(missing) / total

    lost_words: List[str] = []
    if same_language:
        o = critical_word_counts(original)
        c = critical_word_counts(candidate)
        lost_words = sorted(w for w in o if c[w] < o[w])
    return Preservation(score, missing, lost_words)


def _words(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower())


def critical_word_counts(text: str) -> Counter:
    """Conservative EN/FA lexemes and ZH/JA markers, not a semantic parser."""
    counts = Counter(w for w in _words(text) if w in _CRITICAL_WORDS)
    # CJK prose generally lacks spaces; word boundaries cannot find these.
    for marker in ("不", "勿", "禁止", "必须", "只有", "除非", "之前", "之后",
                   "ない", "必ず", "だけ", "のみ"):
        if marker in text:
            counts[marker] = text.count(marker)
    return counts


def _ngrams(text: str, n: int) -> Counter:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Simplified chrF-like score in [0, 100], lowercasing/collapsing whitespace.

    This is not standardized SacreBLEU chrF. No threshold has been calibrated
    against translation quality; even very high scores can hide semantic errors.
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * r / (b2 * p + r)
