"""Local, free translation backends for the multilingual experiment.

* ``argos`` - Argos Translate (OpenNMT models, ~100 MB per language pair,
  CPU-friendly). Run ``tokopt setup-languages fa zh ja`` once.
* ``nllb``  - Meta NLLB-200 distilled 600M via transformers (better quality,
  ~2.4 GB download, slow on CPU).

Translation runs on prose only: fenced code is untouched and inline code,
URLs, paths and identifiers are masked with placeholders, then restored.
"""
from __future__ import annotations

import re
from typing import Dict, List, Protocol, Tuple

from ..protect import _MASK, join_segments, mask, split_segments, unmask
from ..check import _NUMBER


class Translator(Protocol):
    name: str

    def translate(self, text: str, src: str, tgt: str) -> str: ...


class ArgosTranslator:
    name = "argos"

    def __init__(self):
        try:
            import argostranslate.translate as at
        except ImportError as e:
            raise RuntimeError("pip install 'tokopt[argos]'") from e
        self._at = at

    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text.strip():
            return text
        return self._at.translate(text, src, tgt)

    def translate_protected(self, text: str, src: str, tgt: str) -> str:
        """Keep literals outside Argos: its models do not preserve placeholders.

        Translating fragments trades sentence context for exact literal retention.
        Round-trip and preservation gates remain mandatory; this is not a
        guarantee that the translated prose means the same thing.
        """
        fragile = re.compile(f"(?:{_MASK.pattern})|(?:{_NUMBER.pattern})")
        parts, pos = [], 0
        for match in fragile.finditer(text):
            parts.append(self._translate_fragment(text[pos:match.start()], src, tgt))
            parts.append(match.group(0))
            pos = match.end()
        parts.append(self._translate_fragment(text[pos:], src, tgt))
        return "".join(parts)

    def _translate_fragment(self, text: str, src: str, tgt: str) -> str:
        if not text.strip() or not any(c.isalpha() for c in text):
            return text
        lead = text[:len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        return lead + self.translate(text.strip(), src, tgt).strip() + trail


# ISO 639-1 -> NLLB (FLORES-200) codes for common languages
NLLB_CODES: Dict[str, str] = {
    "en": "eng_Latn", "fa": "pes_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "ar": "arb_Arab", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
    "hi": "hin_Deva", "tr": "tur_Latn", "pt": "por_Latn", "it": "ita_Latn", "vi": "vie_Latn",
}


class NLLBTranslator:
    name = "nllb"

    def __init__(self, model: str = "facebook/nllb-200-distilled-600M", device: int = -1):
        try:
            from transformers import pipeline  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pip install 'tokopt[nllb]'") from e
        self._model, self._device = model, device
        self._pipe = None

    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text.strip():
            return text
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline(
                "translation", model=self._model, device=self._device,
                src_lang=NLLB_CODES.get(src, src), tgt_lang=NLLB_CODES.get(tgt, tgt),
            )
        return self._pipe(text, src_lang=NLLB_CODES.get(src, src),
                          tgt_lang=NLLB_CODES.get(tgt, tgt),
                          max_length=1024)[0]["translation_text"]


def get_translator(backend: str = "argos") -> Translator:
    if backend == "argos":
        return ArgosTranslator()
    if backend == "nllb":
        return NLLBTranslator()
    raise ValueError(f"unknown translation backend {backend!r} (use argos or nllb)")


def translate_text(text: str, translator: Translator, src: str, tgt: str) -> Tuple[str, List[str]]:
    """Translate prose segments, keep protected ones. Returns (text, lost_items)
    where lost_items are masked spans the translator destroyed."""
    segs = split_segments(text)
    lost: List[str] = []
    for s in segs:
        if s.protected:
            continue
        out, missing, _, _ = translate_segment(s.text, translator, src, tgt)
        lost += missing
        s.text = out
    return join_segments(segs), lost


def translate_segment(text: str, translator: Translator, src: str, tgt: str):
    """Return restored text, losses, round-trip input and placeholder values."""
    if callable(getattr(translator, "translate_protected", None)):
        out = translator.translate_protected(text, src, tgt)
        return out, [], out, []
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    masked, saved = mask(text.strip())
    raw = translator.translate(masked, src, tgt)
    out, lost = unmask(raw, saved)
    return lead + out + trail, lost, raw, saved


def install_argos_languages(langs: List[str], pivot: str = "en") -> List[str]:
    """Install pivot->lang and lang->pivot packages (both needed for round-trip checks)."""
    import argostranslate.package as ap

    ap.update_package_index()
    available = ap.get_available_packages()
    done = []
    for lang in langs:
        for a, b in ((pivot, lang), (lang, pivot)):
            pkg = next((p for p in available if p.from_code == a and p.to_code == b), None)
            if pkg is None:
                done.append(f"{a}->{b}: not available in Argos index")
                continue
            ap.install_from_path(pkg.download())
            done.append(f"{a}->{b}: installed")
    return done
