from pathlib import Path

from promptsieve.optimizer import optimize

LOG = (Path(__file__).parent.parent / "examples" / "pytest_run.log").read_text()


def by_name(res, name):
    return next(c for c in res.candidates if c.name == name)


def test_auto_picks_logs_for_logs(counter):
    res = optimize(LOG, counter=counter)
    assert res.best.name == "logs"
    assert res.best.savings > 0.9


def test_plain_prose_without_strategies_returns_original(counter, prose):
    res = optimize(prose, counter=counter)
    assert res.best.name in ("original", "normalize")
    assert res.best.text.strip() == prose.strip()


def test_token_hungry_translation_is_rejected(counter, translator, prose):
    """The user's example: 1,000 English tokens must never become 1,400 Persian tokens."""
    res = optimize(prose, counter=counter, langs=["fa"], translator=translator)
    fa = by_name(res, "translate:fa")
    assert fa.tokens > res.original_tokens
    assert not fa.passed and "below the 10% minimum" in fa.reasons[0]
    assert res.best.name in ("original", "normalize")
    # rejected on cost before the (expensive) round-trip was attempted
    assert fa.roundtrip_chrf is None


def test_good_translation_accepted_and_items_restored(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["xx"], translator=translator)
    xx = by_name(res, "translate:xx")
    assert xx.passed, xx.reasons
    assert res.best is xx
    assert "src/payments/cart.py" in xx.text and "`Cart.total()`" in xx.text
    assert xx.text.endswith("(Answer in English.)")
    assert xx.roundtrip_chrf > 99


def test_translation_losing_numbers_is_rejected(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["bad"], translator=translator)
    bad = by_name(res, "translate:bad")
    assert not bad.passed
    assert res.best.name in ("original", "normalize")


def test_low_roundtrip_is_rejected(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["garble"], translator=translator)
    g = by_name(res, "translate:garble")
    assert not g.passed and any("chrF" in r for r in g.reasons)


def test_min_savings_threshold_is_respected(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["xx"], translator=translator, min_savings=0.999)
    assert not by_name(res, "translate:xx").passed


def test_missing_language_pack_is_reported(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["de"], translator=translator)
    assert "translator error" in by_name(res, "translate:de").reasons[0]


def test_per_segment_mixing(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["fa", "xx"], translator=translator, per_segment=True)
    mixed = by_name(res, "mixed-per-paragraph")
    assert mixed.passed, mixed.reasons
    assert "ب" not in mixed.text  # never picked the bigger language


def test_code_blocks_never_translated(counter, translator):
    t = "Explain what this function does in two sentences please, thank you very much.\n\n```python\ndef f(x):\n    return x * 2\n```\n"
    res = optimize(t, counter=counter, langs=["xx"], translator=translator)
    assert "def f(x):\n    return x * 2" in by_name(res, "translate:xx").text


def test_report_table_and_json(counter, translator, prose):
    res = optimize(prose, counter=counter, langs=["fa", "xx"], translator=translator)
    assert "BEST" in res.table() and "translate:fa" in res.table()
    assert '"best": "translate:xx"' in res.to_json()
