"""Token counters for different model families.

Every counter exposes ``name`` and ``count(text) -> int``. Use
:func:`get_counter` with a spec string:

    tiktoken:o200k_base        OpenAI encoding by name (offline after first download)
    openai:gpt-4o              OpenAI encoding looked up by model name
    anthropic:claude-haiku-5-5 Anthropic token-counting API (needs ANTHROPIC_API_KEY)
    hf:Qwen/Qwen2.5-7B-Instruct  Hugging Face tokenizer (for local models)
    heuristic                  Rough script-aware estimate, no dependencies

The point of the whole project is that *only the target model's own tokenizer
gives a true answer*, so the heuristic is a last resort and says so.
"""
from __future__ import annotations

import os
import re
import unicodedata
import warnings
from typing import Dict, Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class CounterUnavailable(RuntimeError):
    """Raised when a counter's dependency, data or credentials are missing."""


class TiktokenCounter:
    def __init__(self, encoding: str = "o200k_base", model: str | None = None):
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover - depends on env
            raise CounterUnavailable("pip install tiktoken") from e
        try:
            self._enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(encoding)
        except Exception as e:  # network blocked on first download, unknown model, ...
            raise CounterUnavailable(f"tiktoken could not load encoding ({type(e).__name__})") from e
        self.name = f"openai:{model}" if model else f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class AnthropicCounter:
    """Count a complete single-user Messages request, including API overhead.

    There is no documented constant overhead to subtract. This sends the text
    to Anthropic; it is not an offline text tokenizer.
    """

    def __init__(self, model: str = "claude-haiku-5-5"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise CounterUnavailable("set ANTHROPIC_API_KEY to use the Anthropic counter")
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise CounterUnavailable("pip install anthropic") from e
        self._client = anthropic.Anthropic()
        self._model = model
        self.name = f"anthropic:{model} (single-user request)"
        self._cache: Dict[str, int] = {}

    def _raw(self, text: str) -> int:
        resp = self._client.messages.count_tokens(
            model=self._model, messages=[{"role": "user", "content": text}]
        )
        return resp.input_tokens

    def count(self, text: str) -> int:
        if not text:
            return 0
        if text not in self._cache:
            self._cache[text] = self._raw(text)
        return self._cache[text]


class HFCounter:
    def __init__(self, model_id: str):
        tok = None
        native = False
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            self._encode = lambda t: tok.encode(t, add_special_tokens=False)
        except ImportError:
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_pretrained(model_id)
                native = True
                self._encode = lambda t: tok.encode(t, add_special_tokens=False).ids
            except ImportError as e:  # pragma: no cover
                raise CounterUnavailable("pip install transformers (or tokenizers)") from e
            except Exception as e:
                raise CounterUnavailable(f"could not load HF tokenizer {model_id}: {type(e).__name__}") from e
        except Exception as e:
            raise CounterUnavailable(f"could not load HF tokenizer {model_id}: {type(e).__name__}") from e
        self.name = f"hf:{model_id}"
        self._tokenizer, self._native = tok, native

    def __getstate__(self):
        # Fast tokenizer JSON is the loaded snapshot, including added tokens.
        # Transfer it without importing transformers/torch in deadline workers.
        backend = self._tokenizer if self._native else getattr(self._tokenizer, "backend_tokenizer", None)
        if backend is not None and callable(getattr(backend, "to_str", None)):
            return {"name": self.name, "_tokenizer_json": backend.to_str()}
        return {"name": self.name, "_tokenizer": self._tokenizer, "_native": self._native}

    def __setstate__(self, state):
        if "_tokenizer_json" in state:
            from tokenizers import Tokenizer
            self.name = state["name"]
            self._tokenizer = Tokenizer.from_str(state["_tokenizer_json"])
            self._native = True
        else:
            self.__dict__.update(state)
        tok = self._tokenizer
        self._encode = (lambda t: tok.encode(t, add_special_tokens=False).ids) if self._native else (lambda t: tok.encode(t, add_special_tokens=False))

    def count(self, text: str) -> int:
        return len(self._encode(text))


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


class HeuristicCounter:
    """Very rough estimate. ASCII ~4 chars/token, CJK ~1 token/char, other
    non-Latin scripts (Persian, Arabic, Cyrillic, ...) ~2 chars/token.
    Good enough for tests and for spotting 10x differences, nothing more."""

    name = "heuristic(rough)"

    def count(self, text: str) -> int:
        ascii_chars = cjk = other = 0
        for ch in text:
            if ord(ch) < 128:
                ascii_chars += 1
            elif _CJK.match(ch):
                cjk += 1
            elif not unicodedata.category(ch).startswith("M"):
                other += 1
        est = ascii_chars / 4 + cjk * 1.0 + other / 2
        return max(1, int(round(est))) if text else 0


def get_counter(spec: str | None = None) -> TokenCounter:
    """Build a counter from a spec string. ``None`` means "best available":
    tiktoken o200k_base, falling back to the heuristic with a warning."""
    if spec is None:
        try:
            return TiktokenCounter("o200k_base")
        except CounterUnavailable as e:
            warnings.warn(f"{e}; falling back to the rough heuristic counter", stacklevel=2)
            return HeuristicCounter()

    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "heuristic":
        return HeuristicCounter()
    if kind == "tiktoken":
        return TiktokenCounter(arg or "o200k_base")
    if kind == "openai":
        return TiktokenCounter(model=arg or "gpt-4o")
    if kind in ("anthropic", "claude"):
        return AnthropicCounter(arg or "claude-haiku-5-5")
    if kind == "hf":
        if not arg:
            raise ValueError("hf counter needs a model id, e.g. hf:Qwen/Qwen2.5-7B-Instruct")
        return HFCounter(arg)
    raise ValueError(f"unknown counter spec: {spec!r}")
