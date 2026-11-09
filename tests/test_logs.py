from pathlib import Path

from tokopt.strategies.logs import LogCompressor, looks_like_log

LOG = (Path(__file__).parent.parent / "examples" / "pytest_run.log").read_text()


def test_detects_log():
    assert looks_like_log(LOG)
    assert not looks_like_log("Just a short note.\nTwo lines.")


def test_pytest_log_keeps_failures_drops_passes():
    lc = LogCompressor()
    out = lc.apply(LOG)
    assert "PASSED" not in out
    assert "test_checkout_total FAILED" in out
    assert "E       AssertionError: assert 26.0 == 26.25" in out
    assert "1 failed, 411 passed" in out
    assert "WARNING  payments.tax" in out
    assert lc.last_stats.dropped["passing-test"] == 411
    assert LogCompressor.verify(LOG, out) == []
    assert len(out) < len(LOG) * 0.15


def test_dedupe_and_frames():
    lines = ["2026-11-08T10:00:0%d retrying connection" % i for i in range(6)]
    lines += ["Exception in thread main java.lang.IllegalStateException: boom"]
    lines += [f"\tat com.acme.Foo.bar{i}(Foo.java:{i})" for i in range(40)]
    out = LogCompressor(max_frames=10).apply("\n".join(lines))
    assert "[x6]" in out
    assert "more frames" in out
    assert "IllegalStateException: boom" in out


def test_focus_mode_omits_far_lines():
    body = [f"info: step {i} done" for i in range(1000)]
    body[500] = "ERROR: database connection refused"
    out = LogCompressor(max_lines=100).apply("\n".join(body))
    assert "ERROR: database connection refused" in out
    assert "lines omitted" in out
    assert "info: step 497 done" in out and "info: step 300 done" not in out


def test_verify_reports_missing_error():
    assert LogCompressor.verify("a\nERROR: x failed\nb", "a\nb") == ["ERROR: x failed"]
