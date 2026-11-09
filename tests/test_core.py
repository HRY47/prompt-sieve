from pathlib import Path

from tokopt.check import chrf, critical_items, normalize_digits, preservation
from tokopt.counters import HeuristicCounter, get_counter
from tokopt.protect import join_segments, mask, split_segments, unmask
from tokopt.strategies.normalize import normalize

EX = Path(__file__).parent.parent / "examples"


def test_heuristic_counts_scripts_differently():
    c = HeuristicCounter()
    assert c.count("") == 0
    assert c.count("hello world, this is text") < c.count("سلام دنیا، این یک متن است")
    assert c.count("你好世界") == 4


def test_get_counter_specs():
    assert get_counter("heuristic").name.startswith("heuristic")


def test_mask_roundtrip_and_loss_detection():
    t = "Edit src/a/b.py and call `run()` per https://x.io/docs with --force on user_id."
    m, saved = mask(t)
    assert "src/a/b.py" not in m and len(saved) == 5
    assert unmask(m, saved) == (t, [])
    broken = m.replace("⟦0⟧", "")
    _, lost = unmask(broken, saved)
    assert lost == [saved[0]]
    # translators often add spaces inside brackets
    assert unmask(m.replace("⟦1⟧", "⟦ 1 ⟧"), saved)[0] == t


def test_segments_protect_code_blocks():
    t = "Intro text.\n\n```python\nx = 1\n```\n\nOutro."
    segs = split_segments(t)
    assert join_segments(segs) == t
    code = [s for s in segs if "x = 1" in s.text]
    assert code and code[0].protected


def test_persian_digits_normalized():
    assert normalize_digits("۲۰۲۶ و ٣٤") == "2026 و 34"
    p = preservation("Version 2026 costs 34 dollars", "نسخه ۲۰۲۶ قیمت ۳۴ دلار", same_language=False)
    assert p.ok, p


def test_preservation_catches_missing_number_and_negation():
    p = preservation("The limit is 50 and it must not exceed it.", "The limit is high and it must exceed it.")
    assert "50" in p.missing
    assert "not" in p.lost_critical_words
    assert not p.ok


def test_critical_items_include_code_blocks_and_paths():
    items = critical_items("See src/x.py\n\n```\nprint(1)\n```\n")
    assert "src/x.py" in items and any("print(1)" in i for i in items)


def test_chrf_bounds():
    assert chrf("the cat sat on the mat", "the cat sat on the mat") > 99
    assert chrf("qqq zzz", "the cat sat on the mat") < 10
    assert chrf("", "x") == 0


def test_normalize_strips_ansi_and_progress():
    t = "\x1b[32mok\x1b[0m   \nDownloading 10%\rDownloading 100%\n\n\n\nend\n"
    assert normalize(t) == "ok\nDownloading 100%\n\nend\n"
