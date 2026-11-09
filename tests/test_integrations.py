"""Opt-in real integrations; missing dependencies FAIL when explicitly enabled."""
import os

import pytest

pytestmark = pytest.mark.integration

SAMPLES = [
    "Never delete the production database. The limit is -5.",
    "هرگز پایگاه داده را حذف نکنید. مقدار ۵ است.",
    "请勿删除数据库。限额为五。", "データベースを削除しないでください。",
    "English فارسی 中文 日本語 123", "def f(x):\n    return x + 42\n",
    "ERROR /srv/main.py:42 connection refused\n", '{"value": -1.5, "safe": true}',
    "# Heading\n\n- `value` must be 42.\n",
]


@pytest.mark.skipif(os.getenv("PROMPTSIEVE_TEST_TOKENIZERS") != "1", reason="enable PROMPTSIEVE_TEST_TOKENIZERS=1; needs tiktoken and encoding download")
@pytest.mark.parametrize("text", SAMPLES)
def test_real_tiktoken(text):
    import tiktoken
    from promptsieve.counters import get_counter
    counter = get_counter("openai:gpt-4o")
    assert counter.count(text) == len(tiktoken.encoding_for_model("gpt-4o").encode(text, disallowed_special=()))


@pytest.mark.skipif(os.getenv("PROMPTSIEVE_TEST_ARGOS") != "1", reason="enable PROMPTSIEVE_TEST_ARGOS=1 after installing both-direction language packs")
@pytest.mark.parametrize("lang", ["fa", "zh", "ja"])
def test_real_argos(lang):
    from promptsieve.optimizer import optimize
    from promptsieve.counters import get_counter
    from promptsieve.strategies.translate import ArgosTranslator
    tr = ArgosTranslator()
    text = "Never delete the database. Keep 42 records. Read `config.yaml` and https://example.org/docs.\n\n```sh\necho safe\n```"
    translated = tr.translate(text, "en", lang)
    assert translated.strip() and translated != text
    assert tr.translate(translated, lang, "en").strip()
    result = optimize(text, counter=get_counter("tiktoken:o200k_base"), langs=[lang], translator=tr, per_segment=True, measure_all=True)
    assert result.best.tokens <= result.original_tokens
    assert all("translator error" not in " ".join(c.reasons) and "unavailable" not in " ".join(c.reasons) for c in result.candidates)


@pytest.mark.skipif(os.getenv("PROMPTSIEVE_TEST_LINGUA") != "1", reason="enable PROMPTSIEVE_TEST_LINGUA=1; needs llmlingua and model download")
def test_real_lingua():
    from promptsieve.strategies.lingua import LinguaCompressor
    from promptsieve.counters import get_counter
    text = "Please carefully review this important deployment procedure. Never delete the database. The limit is 42. " * 8
    out = LinguaCompressor(device="cpu").apply(text)
    assert out and out != text
    counter = get_counter("tiktoken:o200k_base")
    assert counter.count(out) < counter.count(text)


@pytest.mark.skipif(os.getenv("PROMPTSIEVE_TEST_HF") != "1", reason="enable PROMPTSIEVE_TEST_HF=1; needs transformers and tokenizer download")
@pytest.mark.parametrize("text", SAMPLES)
def test_real_hf(text):
    from transformers import AutoTokenizer
    from promptsieve.counters import HFCounter
    model = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    assert HFCounter(model).count(text) == len(tok.encode(text, add_special_tokens=False))


@pytest.mark.skipif(os.getenv("PROMPTSIEVE_TEST_NLLB") != "1", reason="enable PROMPTSIEVE_TEST_NLLB=1; needs torch, transformers 4.x and NLLB weights")
def test_real_nllb():
    from promptsieve.strategies.translate import NLLBTranslator
    tr = NLLBTranslator()
    original = "Never delete the database. Keep forty two records."
    for lang in ("fa", "zh", "ja"):
        out = tr.translate(original, "en", lang)
        assert out.strip() and out != original
        assert tr.translate(out, lang, "en").strip()
