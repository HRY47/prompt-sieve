"""Protection regressions; these deterministic tests do not prove model quality."""
from types import SimpleNamespace
import pytest
from promptsieve.strategies.translate import ArgosTranslator, translate_text
from promptsieve.optimizer import optimize
from promptsieve.check import chrf


def translator():
    tr = ArgosTranslator.__new__(ArgosTranslator)
    calls = []
    def translate(text, src, tgt):
        calls.append(text)
        return text.replace("Read", "Inspect").replace("Keep", "Retain")
    tr._at = SimpleNamespace(translate=translate)
    return tr, calls


def test_argos_never_receives_protected_literals(counter):
    tr, calls = translator()
    text = "Read `config.yaml` at /srv/app.py with --dry-run and user_id. Keep -5 and 1e-4."
    out, lost = translate_text(text, tr, "en", "zh")
    for literal in ("`config.yaml`", "/srv/app.py", "--dry-run", "user_id", "-5", "1e-4"):
        assert literal in out
        assert all(literal not in call for call in calls)
    assert not lost
    assert "⟦" not in "".join(calls)
    res = optimize(text, counter=counter, langs=["zh"], translator=tr, measure_all=True)
    c = next(c for c in res.candidates if c.name == "translate:zh")
    assert c.roundtrip_chrf is not None
    assert c.preservation == 1


def test_protected_translation_keeps_whitespace_and_fences():
    tr, _ = translator()
    text = "  Read `x` now.  \n\n```sh\necho safe\n```\n"
    out, lost = translate_text(text, tr, "en", "ja")
    assert out == text.replace("Read", "Inspect")
    assert not lost


@pytest.mark.parametrize("text", ["a", "hello", "فارسی", "中文", "日本語"])
def test_chrf_identical_short_and_multilingual(text):
    assert chrf(text, text) == pytest.approx(100)


@pytest.mark.parametrize("original,candidate", [
    ("Never delete data.", "Delete data."),
    ("هرگز داده را حذف نکنید", "داده را حذف کنید"),
    ("请勿删除数据。必须保留。", "请勿删除数据。保留。"),
    ("削除しないでください", "削除してください"),
])
def test_multilingual_critical_instruction_loss(original, candidate):
    from promptsieve.check import preservation
    assert not preservation(original, candidate).ok


def test_lingua_forces_actual_case_literals_and_repeated_words(monkeypatch):
    from promptsieve.strategies.lingua import LinguaCompressor
    calls = []
    def compress_prompt(text, **kw):
        calls.append(kw)
        return {"compressed_prompt": text}
    monkeypatch.setattr(LinguaCompressor, "_get", lambda self: SimpleNamespace(compress_prompt=compress_prompt))
    text = "Never delete `config.yaml`. Never remove user_id. Keep -5."
    assert LinguaCompressor().compress_one(text) == text
    assert {"Never", "`config.yaml`", "user_id", "-5"} <= set(calls[0]["force_tokens"])
    assert calls[0]["drop_consecutive"] is False


def test_lingua_force_vocabulary_overflow_keeps_source(monkeypatch):
    from promptsieve.strategies.lingua import LinguaCompressor
    def fail(self):
        raise AssertionError("must not load model for unsupported force vocabulary")
    monkeypatch.setattr(LinguaCompressor, "_get", fail)
    text = " ".join(f"`literal{i}`" for i in range(101))
    assert LinguaCompressor().compress_one(text) == text


@pytest.mark.parametrize("candidate", ["保留42条记录", "42件を保持", "۴۲ رکورد"])
def test_number_survives_cjk_adjacency_and_localized_digits(candidate):
    from promptsieve.check import preservation
    assert preservation("Keep 42 records", candidate, same_language=False).ok



def test_lingua_auto_cpu_and_explicit_cpu_share_model(monkeypatch):
    import sys
    import promptsieve.strategies.lingua as module
    calls = []
    def load(**kw):
        calls.append(kw)
        return object()
    monkeypatch.setattr(module, "_compressor_cache", {})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "llmlingua", SimpleNamespace(PromptCompressor=load))
    assert module.LinguaCompressor()._get() is module.LinguaCompressor(device="cpu")._get()
    assert len(calls) == 1
