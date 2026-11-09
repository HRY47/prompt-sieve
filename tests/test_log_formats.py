"""Synthetic realistic format fixtures; not output from external build tools."""
import pytest

from promptsieve.optimizer import optimize
from promptsieve.strategies.logs import LogCompressor


FORMATS = {
    "pytest": """tests/test_api.py::test_create FAILED [100%]
=================================== FAILURES ===================================
    def test_create():
>       assert create() == {'status': 201}
E       AssertionError: assert {'status': 403} == {'status': 201}
tests/test_api.py:42: AssertionError
=========================== 1 failed, 20 passed in 1.20s =======================""",
    "unittest": """FAIL: test_create (tests.test_api.ApiTest.test_create)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/app/tests/test_api.py", line 42, in test_create
    self.assertEqual(create(), 201)
AssertionError: 403 != 201
Ran 21 tests in 1.20s
FAILED (failures=1)""",
    "gradle": """> Task :test FAILED
ApiTest > testCreate() FAILED
    java.lang.AssertionError: expected 201 but was 403
        at com.example.ApiTest.testCreate(ApiTest.java:42)
21 tests completed, 1 failed
FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':test'.
> There were failing tests. See the report at: file:///app/build/reports/tests/test/index.html
BUILD FAILED in 2s""",
    "npm": """FAIL tests/api.test.js
  ● create › rejects unauthorized calls
    Expected: 201
    Received: 403
      at Object.<anonymous> (tests/api.test.js:42:5)
Test Suites: 1 failed, 2 passed, 3 total
Tests: 1 failed, 20 passed, 21 total
npm ERR! code ELIFECYCLE""",
    "docker": """#8 [build 4/4] RUN npm test
#8 1.234 Error: Cannot find module '/app/index.js'
#8 1.234     at Module._resolveFilename (node:internal/modules/cjs/loader:42:15)
Dockerfile:12
--------------------
  12 | >>> RUN npm test
--------------------
ERROR: failed to solve: process \"/bin/sh -c npm test\" did not complete successfully: exit code: 1""",
    "syslog": """Nov 08 10:00:00 host kernel: Out of memory: Killed process 1234 (worker)
Nov 08 10:00:01 host systemd[1]: worker.service: Main process exited, code=killed, status=9/KILL
Nov 08 10:00:01 host systemd[1]: worker.service: Failed with result 'signal'.""",
    "java": """Exception in thread \"main\" java.lang.IllegalStateException: database unavailable
    at com.example.Main.start(Main.java:42)
    at com.example.Main.main(Main.java:12)
Caused by: java.net.ConnectException: Connection refused
    at java.net.Socket.connect(Socket.java:633)
    ... 2 more""",
    "python": """Traceback (most recent call last):
  File \"/app/main.py\", line 42, in <module>
    connect()
  File \"/app/db.py\", line 9, in connect
    raise ConnectionError('database unavailable')
ConnectionError: database unavailable""",
}


@pytest.mark.parametrize("kind", FORMATS)
def test_preserves_complete_failure_fixture(kind, counter):
    failure = FORMATS[kind]
    noise = "\n".join(f"tests/test_noise.py::test_{i} PASSED [ 50%]" for i in range(500))
    text = noise + "\n" + failure + "\n"
    result = optimize(text, counter=counter, strategies=["logs"])
    assert failure in result.best.text
    assert result.best.tokens < result.original_tokens


def test_large_captured_failure_context(counter):
    text = "=== FAILURES ===\n" + "\n".join(f"captured data record {i}" for i in range(1000)) + "\nAssertionError: response differs"
    assert text in LogCompressor().apply(text)


@pytest.mark.parametrize("text", ["", "\n", "\x00broken\ufffd", "日本語 فارسی 😀", "ERROR: x\n" * 50])
def test_empty_unicode_malformed_duplicates(text, counter):
    result = optimize(text, counter=counter, strategies=["logs"])
    assert result.best.tokens <= result.original_tokens
    assert not LogCompressor.verify(text, result.best.text)


def test_ansi_mixed_streams():
    text = "\x1b[32mtests/a.py::test_a PASSED [100%]\x1b[0m\n\x1b[31mERROR stderr: refused\x1b[0m\nstdout result: 42\n"
    out = LogCompressor().apply(text)
    assert "ERROR stderr: refused" in out and "stdout result: 42" in out


def test_success_marker_with_substantive_content_is_not_noise():
    text = "Validation PASSED: recovered customer 42 from /srv/db"
    assert text in LogCompressor().apply(text)


def test_important_java_frame_is_never_collapsed():
    frames = [f"\tat com.acme.Foo.bar{i}(Foo.java:{i})" for i in range(40)]
    frames[20] = "\tat com.acme.FatalException.raise(Foo.java:20)"
    out = LogCompressor(max_frames=10).apply("\n".join(frames))
    assert frames[20] in out


def test_default_retains_every_java_frame():
    frames = "\n".join(f"\tat com.acme.Foo.bar{i}(Foo.java:{i})" for i in range(100))
    assert frames in LogCompressor().apply(frames)


def test_diagnostic_duplicates_retain_timestamps_and_multiplicity():
    text = "\n".join(f"2026-11-08T10:00:0{i} ERROR: refused" for i in range(5))
    out = LogCompressor().apply(text)
    assert text in out
    assert LogCompressor.verify(text, "ERROR: refused")
