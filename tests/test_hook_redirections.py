"""Preserve numeric command arguments while parsing shell redirections."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from promptsieve.integrations import claude_hook

ROOT = Path(__file__).resolve().parents[1]
LOG = (ROOT / "examples/pytest_run.log").read_text(encoding="utf-8")


def payload(command, output=LOG):
    return {
        "hook_event_name": "PostToolUse", "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_response": {"stdout": output, "stderr": "", "interrupted": False, "isImage": False},
    }


@pytest.mark.parametrize("command", [
    "timeout 30 </dev/null cat build.log",
    "nice -n 5 </dev/null cat build.log",
    "timeout '30' </dev/null cat build.log",
    "timeout 3'0'</dev/null cat build.log",
    "timeout \\30</dev/null cat build.log",
])
def test_numeric_wrapper_arguments_survive_input_redirection(command, tmp_path):
    assert claude_hook.handle(payload(command), {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}) is None


@pytest.mark.parametrize("command", [
    "timeout 30 0</dev/null cat build.log",
    "timeout 30 2>/dev/null cat build.log",
    "nice -n 5 0</dev/null cat build.log",
])
def test_actual_file_descriptor_prefixes_are_handled(command, tmp_path):
    assert claude_hook.handle(payload(command), {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}) is None


@pytest.mark.parametrize("command", [
    "timeout 30 python -m pytest 2>&1",
    "nice -n 5 pytest 2>/dev/null",
    "timeout 30 0</dev/null python -m pytest",
])
def test_noisy_commands_with_redirections_are_still_considered(command, tmp_path):
    assert claude_hook.handle(payload(command), {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell command")
def test_actual_timed_file_read_remains_exact(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(LOG, encoding="utf-8")
    command = f"timeout 30 </dev/null cat {shlex.quote(str(log))}"
    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True)
    assert proc.stdout == LOG
    assert claude_hook.handle(payload(command, proc.stdout),
                              {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}) is None
