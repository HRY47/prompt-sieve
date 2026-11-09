"""Build candidate versions of a prompt, measure them with the target model's
tokenizer, check them, and pick the smallest one that passes.

Every lossy candidate (logs, lingua, translation) has to clear the same gate:

  1. measure the original token count
  2. transform (translation always runs locally)
  3. measure the candidate with the same counter
  4. check meaning (critical items, error lines, round-trip chrF for translations)
  5. accept only if checks pass AND savings >= ``min_savings``

If nothing passes, the (normalized) original is returned. So a 1,000-token
English prompt can never come back as 1,400 Persian tokens.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .check import chrf, preservation
from .errors import UsageError
from .counters import TokenCounter, get_counter
from .protect import Segment, join_segments, mask, split_segments, unmask
from .strategies.logs import LogCompressor, looks_like_log
from .strategies.normalize import normalize
from .strategies.translate import Translator, translate_segment, translate_text

REPLY_HINT = {"en": "\n\n(Answer in English.)"}


@dataclass
class Candidate:
    name: str
    text: str
    tokens: Optional[int]
    savings: float                 # fraction of original tokens saved (negative = bigger)
    lossy: bool
    passed: bool
    reasons: List[str] = field(default_factory=list)
    preservation: Optional[float] = None
    roundtrip_chrf: Optional[float] = None

    def to_dict(self, include_text: bool = False) -> dict:
        d = asdict(self)
        if not include_text:
            d.pop("text")
        return d


@dataclass
class OptimizeResult:
    counter: str
    original_tokens: Optional[int]
    candidates: List[Candidate]
    best: Candidate

    timed_out: bool = False
    cache_hit: bool = False
    mode: str = "full"
    warnings: List[str] = field(default_factory=list)
    _original_text: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self._original_text is None:
            self._original_text = self.candidates[0].text

    @property
    def original_text(self) -> str:
        return self._original_text

    def table(self) -> str:
        rows = [("candidate", "tokens", "saved", "preserv.", "chrF", "result")]
        for c in self.candidates:
            mark = "BEST" if c is self.best else ("pass" if c.passed else "rejected")
            rows.append((
                c.name, "unknown" if c.tokens is None else f"{c.tokens:,}", f"{c.savings:+.1%}",
                "-" if c.preservation is None else f"{c.preservation:.0%}",
                "-" if c.roundtrip_chrf is None else f"{c.roundtrip_chrf:.0f}",
                mark,
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = [f"counter: {self.counter}"]
        for i, r in enumerate(rows):
            lines.append("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())
            if i == 0:
                lines.append("  ".join("-" * w for w in widths))
        rejected = [c for c in self.candidates if not c.passed and c.reasons]
        for c in rejected:
            lines.append(f"  - {c.name}: " + "; ".join(c.reasons))
        lines.extend("warning: " + warning for warning in self.warnings)
        return "\n".join(lines)

    def to_json(self, include_text: bool = False) -> str:
        return json.dumps({
            "counter": self.counter,
            "original_tokens": self.original_tokens,
            "best": self.best.name,
            "best_tokens": self.best.tokens,
            "candidates": [c.to_dict(include_text) for c in self.candidates],
            "timed_out": self.timed_out, "cache_hit": self.cache_hit,
            "mode": self.mode, "warnings": self.warnings,
            **({"original_text": self.original_text} if include_text else {}),
        }, ensure_ascii=False, indent=2)


def _optimize(
    text: str,
    counter: Optional[TokenCounter] = None,
    strategies: Sequence[str] = ("auto",),
    langs: Iterable[str] = (),
    translator: Optional[Translator] = None,
    src_lang: str = "en",
    min_savings: float = 0.10,
    min_preservation: float = 1.0,
    min_chrf: float = 55.0,
    roundtrip: bool = True,
    per_segment: bool = False,
    lingua_rate: float = 0.5,
    lingua_model: str = "small",
    reply_hint: bool = True,
    measure_all: bool = False,
    log_options: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    mode: str = "full",
    ml_min_tokens: int = 1024,
    ml_min_saved_tokens: int = 256,
    allow_mixed_lingua: bool = False,
    allow_sensitive_lingua: bool = False,
    translation_backend: str = "argos",
    _progress=None,
) -> OptimizeResult:
    for key, value, low, high in (("min_savings", min_savings, 0, 1),
                                  ("min_preservation", min_preservation, 0, 1),
                                  ("min_chrf", min_chrf, 0, 100),
                                  ("lingua_rate", lingua_rate, 0, 1)):
        if not math.isfinite(value) or not low <= value <= high or (key == "lingua_rate" and value == 0):
            raise UsageError(f"{key} must be finite and within {low}..{high}")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 0):
        raise UsageError("max_tokens must be a nonnegative integer")
    langs = list(langs)
    for strategy in strategies:
        if strategy.startswith("translate:") and strategy.partition(":")[2]:
            langs.append(strategy.partition(":")[2])
        elif strategy not in {"auto", "logs", "normalize", "lingua"}:
            raise UsageError(f"unknown strategy: {strategy!r}")
    counter = counter or get_counter()
    orig_tokens = counter.count(text)
    if _progress:
        _progress(("count", orig_tokens, counter.name))
    cands: List[Candidate] = [Candidate("original", text, orig_tokens, 0.0, False, True)]

    def saved(n: int) -> float:
        return 0.0 if orig_tokens == 0 else 1 - n / orig_tokens

    def gate(c: Candidate) -> Candidate:
        if c.tokens > orig_tokens:
            c.passed = False
            c.reasons.append("candidate increases token count")
        if c.lossy and c.savings < min_savings:
            c.passed = False
            c.reasons.append(f"saves {c.savings:+.1%}, below the {min_savings:.0%} minimum")
        return c

    base = normalize(text)
    n = counter.count(base)
    normalized = gate(Candidate("normalize", base, n, saved(n), False, n <= orig_tokens))
    # Whitespace-only edits can flip answers (Phase 3: one removed space turned
    # Qwen's 42 into 10). Small cosmetic savings are declined in every mode
    # unless normalization was explicitly requested.
    if "normalize" not in strategies and 0 < normalized.savings < min_savings:
        normalized.passed = False
        normalized.reasons.append("cosmetic savings below the minimum; original kept")
    cands.append(normalized)

    wanted = list(strategies)
    if "auto" in wanted:
        wanted.remove("auto")
        if looks_like_log(text) and "logs" not in wanted:
            wanted.append("logs")
        if mode == "balanced" and "lingua" not in wanted:
            wanted.append("lingua")

    # -- logs ---------------------------------------------------------------
    if "logs" in wanted:
        lc = LogCompressor(**(log_options or {}))
        out = lc.apply(text)
        n = counter.count(out)
        missing = lc.verify(text, out)
        c = Candidate("logs", out, n, saved(n), True, not missing)
        if missing:
            c.reasons.append(f"dropped {len(missing)} error/summary line(s)")
        cands.append(gate(c))

    # Explicit full mode preserves the experimental ML feature. The fast and
    # balanced policies use an optimistic savings ceiling, not a promise of gain.
    if mode != "full":
        rules_sufficient = any(c.passed and c.savings >= min_savings and c.tokens < orig_tokens for c in cands)
        ml_allowed = orig_tokens >= ml_min_tokens and not rules_sufficient
        for strategy in list(wanted):
            if strategy == "lingua" and (not ml_allowed or orig_tokens * (1 - lingua_rate) < ml_min_saved_tokens):
                wanted.remove(strategy)
                cands.append(Candidate(f"lingua@{lingua_rate:g}", text, orig_tokens, 0, True, False,
                                       ["skipped: expected token savings do not justify ML overhead"]))
        if langs and (not ml_allowed or orig_tokens < ml_min_saved_tokens):
            for lang in dict.fromkeys(langs):
                cands.append(Candidate(f"translate:{lang}", text, orig_tokens, 0, True, False,
                                       ["skipped: prompt below translation cost threshold"]))
            langs = []

    # -- LLMLingua-2 ----------------------------------------------------------
    if "lingua" in wanted:
        from .strategies.lingua import LinguaCompressor

        mixed_blocked = not allow_mixed_lingua and _mixed_prose(text)
        sensitive_blocked = not allow_sensitive_lingua and _sensitive_prose(text)
        if mixed_blocked:
            ok, why = False, "mixed-language prose pruning requires allow_mixed_lingua=True"
        elif sensitive_blocked:
            ok, why = False, "unit/math context pruning requires allow_sensitive_lingua=True"
        else:
            ok, why = LinguaCompressor.available()
        lc2 = LinguaCompressor(rate=lingua_rate, model=lingua_model)
        if not ok:
            cands.append(Candidate(lc2.name, base, orig_tokens, 0.0, True, False, [why]))
        else:
            try:
                out = lc2.apply(base)
                n = counter.count(out)
                p = preservation(text, out, same_language=True)
                c = Candidate(lc2.name, out, n, saved(n), True, True, preservation=p.score)
                _apply_preservation(c, p, min_preservation)
                if not allow_mixed_lingua and _mixed_prose(text):
                    c.passed = False
                    c.reasons.append("mixed-language prose pruning requires allow_mixed_lingua=True")
                cands.append(gate(c))
            except Exception as e:
                cands.append(Candidate(lc2.name, base, orig_tokens, 0.0, True, False,
                                       [f"compression failed ({type(e).__name__})"]))

    # -- translation -----------------------------------------------------------
    langs = list(dict.fromkeys(l for l in langs if l and l != src_lang))
    if langs:
        if translator is None:
            from .strategies.translate import get_translator
            translator = get_translator(translation_backend)
        cands += _translation_candidates(
            base, text, langs, translator, src_lang, counter, orig_tokens,
            min_savings, min_preservation, min_chrf, roundtrip, per_segment,
            reply_hint, measure_all,
        )

    for c in cands:
        if c.tokens > orig_tokens:
            c.passed = False
            if "candidate increases token count" not in c.reasons:
                c.reasons.append("candidate increases token count")
        if max_tokens is not None:
            if c.tokens > max_tokens:
                c.passed = False
                c.reasons.append(f"exceeds max_tokens={max_tokens}")
    passed = [c for c in cands if c.passed and c.tokens <= orig_tokens]
    if not passed:
        raise UsageError("no acceptable candidate fits max_tokens")
    # fewest tokens wins; on ties prefer non-lossy, then the earlier candidate
    best = min(passed, key=lambda c: (c.tokens, c.lossy, cands.index(c)))
    result = OptimizeResult(counter.name, orig_tokens, cands, best)
    result.mode = mode
    return result


def _mixed_prose(text):
    import re
    from .protect import _MASK
    prose = " ".join(s.text for s in split_segments(text) if not s.protected)
    prose = _MASK.sub(" ", prose)
    return len(re.findall(r"[A-Za-z]{2,}", prose)) >= 3 and bool(re.search(r"[\u0600-\u06ff\u3040-\u30ff\u3400-\u9fff]", prose))


def _sensitive_prose(text):
    from .check import _MEASURE, _EXPRESSION
    from .protect import _MASK
    prose = " ".join(s.text for s in split_segments(text) if not s.protected)
    return bool(_MEASURE.search(prose) or _EXPRESSION.search(_MASK.sub(" ", prose)))


def _apply_preservation(c: Candidate, p, min_preservation: float) -> None:
    if p.score < min_preservation:
        c.passed = False
        c.reasons.append(f"lost {len(p.missing)} critical item(s)")
    if p.lost_critical_words:
        c.passed = False
        c.reasons.append(f"dropped meaning-critical words {p.lost_critical_words}")


def _translate_segment(seg: Segment, tr: Translator, src: str, tgt: str):
    return translate_segment(seg.text, tr, src, tgt)


class ProtectedSpanError(ValueError):
    """Translation returned an unusable placeholder sequence."""


def _roundtrip(raw_translated: str, saved: List[str], tr: Translator, src: str, tgt: str) -> str:
    if callable(getattr(tr, "translate_protected", None)):
        return translate_text(raw_translated, tr, tgt, src)[0]
    back, lost = unmask(tr.translate(raw_translated, tgt, src), saved)
    if lost:
        raise ProtectedSpanError("back-translation destroyed protected spans")
    return back


def _check_roundtrip(reference, back, min_chrf):
    # chrF alone accepts a missing "not" in an otherwise identical paragraph.
    return chrf(back, reference) >= min_chrf and preservation(reference, back, same_language=True).ok


def _translation_candidates(base, original, langs, tr, src, counter, orig_tokens,
                            min_savings, min_preservation, min_chrf, roundtrip,
                            per_segment, reply_hint, measure_all) -> List[Candidate]:
    segs = split_segments(base)
    hint = REPLY_HINT.get(src, f"\n\n(Answer in {src}.)") if reply_hint else ""
    out: List[Candidate] = []
    # per-language, per-segment translation results
    per_lang: Dict[str, List[Optional[tuple]]] = {}

    def saved(n):
        return 0.0 if orig_tokens == 0 else 1 - n / orig_tokens

    for lang in langs:
        name = f"translate:{lang}"
        try:
            results = [None if s.protected else _translate_segment(s, tr, src, lang) for s in segs]
        except Exception as e:  # missing language pack, model download failure, ...
            out.append(Candidate(name, base, orig_tokens, 0.0, True, False, [f"translator error ({type(e).__name__})"]))
            continue
        per_lang[lang] = results
        text = join_segments([Segment(r[0] if r else s.text, s.protected) for s, r in zip(segs, results)]) + hint
        n = counter.count(text)  # steps 1-3: measured with the target tokenizer
        c = Candidate(name, text, n, saved(n), True, True)
        if c.savings < min_savings:
            c.passed = False
            c.reasons.append(f"saves {c.savings:+.1%}, below the {min_savings:.0%} minimum")
            if not measure_all:
                out.append(c)
                continue
        lost = [x for r in results if r for x in r[1]]
        if lost:
            c.passed = False
            c.reasons.append(f"translator destroyed {len(lost)} protected span(s)")
        p = preservation(original, text, same_language=False)
        c.preservation = p.score
        _apply_preservation(c, p, min_preservation)
        if roundtrip:
            try:
                back = join_segments([
                    Segment(_roundtrip(r[2], r[3], tr, src, lang) if r else s.text, s.protected)
                    for s, r in zip(segs, results)
                ])
                c.roundtrip_chrf = chrf(back, base)
                if c.roundtrip_chrf < min_chrf:
                    c.passed = False
                    c.reasons.append(f"round-trip chrF check failed (chrF {c.roundtrip_chrf:.0f}, minimum {min_chrf:.0f})")
                back_preservation = preservation(base, back, same_language=True)
                if not back_preservation.ok:
                    c.passed = False
                    c.reasons.append(f"back-translation lost {len(back_preservation.missing)} critical item(s) "
                                     f"and critical words {back_preservation.lost_critical_words}")
            except ProtectedSpanError:
                c.passed = False
                c.reasons.append("back-translation destroyed protected spans; chrF not evaluated")
            except Exception as e:
                c.passed = False
                c.reasons.append(f"round-trip chrF check unavailable ({type(e).__name__})")
        else:
            c.passed = False
            c.reasons.append("unverified (round-trip check disabled)")
        out.append(c)

    if per_segment and per_lang:
        out.append(_mixed_candidate(segs, per_lang, tr, src, counter, orig_tokens,
                                    min_savings, min_chrf, roundtrip, hint, original))
    return out


def _mixed_candidate(segs, per_lang, tr, src, counter, orig_tokens, min_savings,
                     min_chrf, roundtrip, hint, original) -> Candidate:
    """Pick the cheapest passing language for each paragraph separately (the
    "translate parts of the text" idea). Paragraphs where nothing passes stay
    in the source language."""
    chosen: List[str] = []
    switched = 0
    for i, s in enumerate(segs):
        if s.protected:
            chosen.append(s.text)
            continue
        best_text, best_n = s.text, counter.count(s.text)
        floor = best_n * (1 - min_savings)
        for lang, results in per_lang.items():
            r = results[i]
            t, lost, raw, saved = r
            n = counter.count(t)
            if n > floor or lost:
                continue
            if not preservation(s.text, t, same_language=False).ok:
                continue
            if not roundtrip:
                continue
            try:
                if not _check_roundtrip(s.text, _roundtrip(raw, saved, tr, src, lang), min_chrf):
                    continue
            except Exception:
                continue
            if n < best_n:
                best_text, best_n = t, n
        switched += best_text is not s.text
        chosen.append(best_text)
    text = "".join(chosen) + (hint if switched else "")
    n = counter.count(text)
    c = Candidate("mixed-per-paragraph", text, n, 1 - n / orig_tokens if orig_tokens else 0.0, True, True)
    if not switched:
        c.passed = False
        c.reasons.append("no paragraph passed the translation gate")
    elif c.savings < min_savings:
        c.passed = False
        c.reasons.append(f"saves {c.savings:+.1%}, below the {min_savings:.0%} minimum")
    if not roundtrip:
        c.passed = False
        c.reasons.append("unverified (round-trip check disabled)")
    return c


# Public orchestration lives separately so timeout workers can call the core
# directly without recursively creating workers or sharing mutable cache state.
from .runtime import optimize
