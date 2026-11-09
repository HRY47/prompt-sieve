"""LLMLingua-2 token pruning (Microsoft, https://github.com/microsoft/LLMLingua).

A small XLM-RoBERTa classifier scores every word for "keep or drop" and the
lowest-scoring words are removed. Runs locally on CPU (GPU if available).
First use downloads model weights. Download size and resident memory depend on
the model revision, precision and loader; they have not been measured here.

Fenced code blocks are never passed to the model. Numbers, negations and
other critical words are forced to stay.
"""
from __future__ import annotations

import re
from typing import Optional

from ..check import _CRITICAL_WORDS, _NUMBER, critical_word_counts
from ..protect import _MASK, join_segments, split_segments

MODELS = {
    "large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
}

_compressor_cache = {}


class LinguaCompressor:
    def __init__(self, rate: float = 0.5, model: str = "small", device: Optional[str] = None,
                 min_chars: int = 200):
        self.rate = rate
        self.model = MODELS.get(model, model)
        self.device = device
        self.min_chars = min_chars  # don't bother with tiny paragraphs
        self.name = f"lingua@{rate:g}"

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import llmlingua  # noqa: F401
            return True, ""
        except ImportError:
            return False, "pip install 'tokopt[lingua]'  (installs llmlingua + torch)"

    def _get(self):
        device = self.device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        key = (self.model, device)
        if key not in _compressor_cache:
            from llmlingua import PromptCompressor

            # llmlingua 0.2.x defaults trust_remote_code=True, which would run
            # Python shipped inside any Hub repository passed as --lingua-model.
            # The published LLMLingua-2 classifiers need no custom code. A fresh
            # dict is passed because upstream mutates its default argument.
            _compressor_cache[key] = PromptCompressor(
                model_name=self.model, use_llmlingua2=True, device_map=device,
                model_config={"trust_remote_code": False},
            )
        return _compressor_cache[key]

    def compress_one(self, text: str) -> str:
        # LLMLingua matches force tokens case-sensitively. Reserve the actual
        # spellings and whole literals, and never deduplicate forced negations.
        forced = {"\n", ".", "?", "!", ",", ":"}
        forced.update(m.group(0) for m in _MASK.finditer(text))
        forced.update(m.group(0) for m in _NUMBER.finditer(text))
        forced.update(w for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text)
                      if w.lower() in _CRITICAL_WORDS)
        forced.update(w for w in critical_word_counts(text) if w in text)
        # The upstream classifier has a finite force-token vocabulary. Keep the
        # paragraph intact instead of truncating its protected vocabulary.
        if len(forced) > 100:
            return text
        res = self._get().compress_prompt(
            text,
            rate=self.rate,
            force_tokens=sorted(forced, key=lambda w: (-len(w), w)),
            force_reserve_digit=True,
            drop_consecutive=False,
        )
        return res["compressed_prompt"]

    def apply(self, text: str) -> str:
        segs = split_segments(text)
        for s in segs:
            if not s.protected and len(s.text) >= self.min_chars:
                trailing = s.text[len(s.text.rstrip()):]
                s.text = self.compress_one(s.text) + trailing
        return join_segments(segs)
