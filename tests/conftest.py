import re
import pytest
from tokopt.counters import HeuristicCounter


class FakeTranslator:
    """Deterministic stand-in for Argos/NLLB.

    xx = perfect, very compact language (keeps placeholders + numbers)
    fa = faithful but token-hungry (non-Latin script, ~2x tokens under heuristic)
    bad = compact but drops numbers (should fail preservation)
    garble = compact, keeps items, but round-trip comes back as nonsense
    """
    name = "fake"

    def __init__(self):
        self.memory = {}
        self.calls = 0

    def translate(self, text, src, tgt):
        self.calls += 1
        if tgt == "en":  # back-translation
            return self.memory.get((src, text), "completely unrelated words")
        keep = " ".join(re.findall(r"⟦\d+⟧|\d+(?:\.\d+)?%?", text))
        if tgt == "xx":
            out = (keep + " 字") if keep else "字"
        elif tgt == "fa":
            out = re.sub(r"[A-Za-z]", "ب", text)
        elif tgt == "bad":
            out = "字"
        elif tgt == "garble":
            out = keep + " 字"
            self.memory[(tgt, out)] = "zzz qqq"
            return out
        else:
            raise RuntimeError(f"no language pack for {tgt}")
        self.memory[(tgt, out)] = text
        return out


@pytest.fixture
def counter():
    return HeuristicCounter()


@pytest.fixture
def translator():
    return FakeTranslator()


PROSE = ("Please review the change to `Cart.total()` in src/payments/cart.py carefully. "
         "The tax rate is 5% and the total must never be negative. "
         "Round each line to 2 decimal places before summing, and do not approve it unless tests exist.\n\n"
         "Second paragraph explains that books are exempt from tax in some regions and that the "
         "reviewer should check the exemption logic as well as the empty cart case.\n")


@pytest.fixture
def prose():
    return PROSE
