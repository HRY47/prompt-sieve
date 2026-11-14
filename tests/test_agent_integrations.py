"""Claude Code plugin, Codex skill install, `prompt-sieve run`, and the hook."""
import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import promptsieve
from promptsieve.errors import UsageError
from promptsieve.integrations import claude_hook, rawlog
from promptsieve.integrations.install import MARKER, install, skill_text
from promptsieve.integrations.run import run_command

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "examples" / "pytest_run.log"
LOG = LOG_PATH.read_text(encoding="utf-8")
LOG_BYTES = LOG_PATH.read_bytes()
POSIX = os.name != "nt"


@pytest.fixture
def env(tmp_path):
    return {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}


def _py(code):
    return [sys.executable, "-c", code]


# -- prompt-sieve run ------------------------------------------------------------

def _print_log_and_exit(code):
    # Write exact bytes: sys.stdout.write would turn LF into CRLF on Windows.
    return _py(f"import sys; sys.stdout.buffer.write(open({str(LOG_PATH)!r}, 'rb').read()); sys.exit({code})")


def test_run_preserves_exit_status_and_shortens_logs(env):
    out = io.BytesIO()
    assert run_command(_print_log_and_exit(3), out=out, env=env) == 3
    text = out.getvalue().decode()
    assert "test_checkout_total FAILED" in text and "PASSED" not in text
    assert "Exit status 3." in text
    saved = list(Path(env["PROMPTSIEVE_LOG_DIR"]).rglob("*.log"))
    assert len(saved) == 1 and saved[0].read_bytes() == LOG_BYTES
    assert str(saved[0]) in text


def test_run_passes_short_output_through_byte_for_byte(env):
    out = io.BytesIO()
    assert run_command(_py("import sys; sys.stdout.buffer.write(b'ok\\r\\n\\xff'); sys.exit(0)"), out=out, env=env) == 0
    assert out.getvalue() == b"ok\r\n\xff"
    assert not Path(env["PROMPTSIEVE_LOG_DIR"]).exists()


def test_run_merges_stderr_and_accepts_double_dash(env):
    out = io.BytesIO()
    code = run_command(["--", *_py("import sys; print('a'); sys.stderr.write('b\\n'); sys.exit(1)")], out=out, env=env)
    assert code == 1 and b"a" in out.getvalue() and b"b" in out.getvalue()


def test_run_no_save_still_shortens(env):
    out = io.BytesIO()
    run_command(_print_log_and_exit(0), save=False, out=out, env=env)
    assert b"was not saved (--no-save)" in out.getvalue()
    assert not Path(env["PROMPTSIEVE_LOG_DIR"]).exists()


def test_run_missing_command_and_empty_command(env):
    assert run_command(["definitely-not-a-real-command-xyz"], out=io.BytesIO(), env=env) == 127
    with pytest.raises(UsageError):
        run_command(["--"], out=io.BytesIO(), env=env)


def test_cli_run_exit_status(tmp_path):
    p = subprocess.run([sys.executable, "-m", "promptsieve", "run", "--", *_py("import sys; sys.exit(5)")],
                       capture_output=True, cwd=tmp_path)
    assert p.returncode == 5


# -- Claude Code hook -----------------------------------------------------------

def _payload(command="pytest -v", stdout=LOG, stderr="", **response):
    return {"hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_input": {"command": command, "description": "Run tests"},
            "tool_response": {"stdout": stdout, "stderr": stderr, "interrupted": False, "isImage": False, **response}}


def test_hook_replaces_large_log_output(env):
    result = claude_hook.handle(_payload(returnCodeInterpretation="x"), env)
    out = result["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    updated = out["updatedToolOutput"]
    assert updated["returnCodeInterpretation"] == "x"  # unknown fields are kept
    assert updated["interrupted"] is False and updated["isImage"] is False
    assert "test_checkout_total FAILED" in updated["stdout"] and "PASSED" not in updated["stdout"]
    assert "full 471-line version is saved to" in updated["stdout"]


@pytest.mark.parametrize("command", [
    "cat build.log", "git diff", "git show HEAD", "git -C repo --no-pager diff", "tail -n 500 app.log",
    "cd repo && grep -n error build.log", "FOO=1 sed -n 1,400p f.log", "sudo cat /var/log/syslog",
    "prompt-sieve run -- pytest", "pytest 2>&1 | head -200",
])
def test_hook_leaves_exact_output_commands_alone(command, env):
    assert claude_hook.handle(_payload(command=command), env) is None


@pytest.mark.parametrize("command", ["pytest -v", "cd app && npm test", "CI=1 make test", "python -m pytest"])
def test_hook_considers_test_and_build_commands(command, env):
    assert claude_hook.handle(_payload(command=command), env) is not None


def test_hook_ignores_small_images_background_and_other_tools(env):
    assert claude_hook.handle(_payload(stdout="short\n"), env) is None
    assert claude_hook.handle(_payload(isImage=True), env) is None
    background = _payload()
    background["tool_input"]["run_in_background"] = True
    assert claude_hook.handle(background, env) is None
    other = _payload()
    other["tool_name"] = "Read"
    assert claude_hook.handle(other, env) is None
    pre = _payload()
    pre["hook_event_name"] = "PreToolUse"
    assert claude_hook.handle(pre, env) is None


def test_hook_ignores_prose_even_when_long(env):
    prose = "This is an ordinary paragraph of documentation about the project.\n" * 300
    assert claude_hook.handle(_payload(stdout=prose), env) is None


def test_hook_can_be_disabled_and_tuned(env):
    assert claude_hook.handle(_payload(), {**env, "PROMPTSIEVE_HOOK": "off"}) is None
    assert claude_hook.handle(_payload(), {**env, "PROMPTSIEVE_HOOK_MIN_CHARS": str(len(LOG) + 1)}) is None
    assert claude_hook.handle(_payload(), {**env, "PROMPTSIEVE_HOOK_MIN_CHARS": "junk"}) is not None


def test_hook_main_fails_open(monkeypatch, env):
    out = io.StringIO()
    assert claude_hook.main(io.StringIO("not json"), out, env) == 0 and out.getvalue() == ""

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(claude_hook, "shrink", boom)
    out = io.StringIO()
    assert claude_hook.main(io.StringIO(json.dumps(_payload())), out, env) == 0 and out.getvalue() == ""


def test_hook_main_writes_valid_json(env):
    out = io.StringIO()
    claude_hook.main(io.StringIO(json.dumps(_payload())), out, env)
    assert json.loads(out.getvalue())["hookSpecificOutput"]["updatedToolOutput"]["stdout"]


# -- private raw-log storage -------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_raw_logs_are_private(env):
    path = rawlog.save_raw("secret output", "x", env)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.skipif(not POSIX, reason="symlink creation needs privileges on Windows")
def test_raw_logs_refuse_symlinked_directory(tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "logs"
    link.symlink_to(real)
    assert rawlog.save_raw("x", "", {"PROMPTSIEVE_LOG_DIR": str(link)}) is None
    assert list(real.iterdir()) == []


def test_raw_logs_are_pruned(env, monkeypatch):
    monkeypatch.setattr(rawlog, "MAX_FILES", 3)
    paths = [rawlog.save_raw(str(i), "t", env) for i in range(5)]
    assert len(list(Path(env["PROMPTSIEVE_LOG_DIR"]).iterdir())) == 3
    old = paths[-1]
    # On Windows each output lives in its own private folder; age that folder.
    aged = old.parent if old.name == "output.log" else old
    os.utime(aged, (time.time() - rawlog.MAX_AGE_SECONDS - 10,) * 2)
    rawlog.save_raw("new", "t", env)
    assert not old.exists()
    (Path(env["PROMPTSIEVE_LOG_DIR"]) / "notes.txt").write_text("user file")
    rawlog._prune(Path(env["PROMPTSIEVE_LOG_DIR"]), now=time.time() + 10 ** 9)
    assert (Path(env["PROMPTSIEVE_LOG_DIR"]) / "notes.txt").exists()  # only our files are pruned


def test_raw_log_dir_defaults(tmp_path):
    assert rawlog.log_dir({"CLAUDE_PLUGIN_DATA": str(tmp_path)}) == tmp_path / "logs"
    if POSIX:
        assert rawlog.log_dir({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "prompt-sieve" / "logs"


# -- skill install (Codex / Claude Code) ---------------------------------------------

def test_install_codex_user_scope(tmp_path):
    home = tmp_path / "home"
    install("codex", "user", home=home, cwd=tmp_path)
    dest = home / ".agents" / "skills" / "prompt-sieve"
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == skill_text()
    assert (dest / MARKER).is_file()
    install("codex", "user", home=home, cwd=tmp_path)  # idempotent
    install("codex", "user", home=home, cwd=tmp_path, uninstall=True)
    assert not dest.exists()


def test_install_project_scope_for_claude_code(tmp_path):
    messages = install("claude-code", "project", home=tmp_path / "h", cwd=tmp_path)
    assert (tmp_path / ".claude" / "skills" / "prompt-sieve" / "SKILL.md").is_file()
    assert any("claude plugin install prompt-sieve@prompt-sieve" in m for m in messages)


def test_install_never_overwrites_a_foreign_skill(tmp_path):
    dest = tmp_path / ".agents" / "skills" / "prompt-sieve"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("my own skill")
    for kwargs in ({}, {"uninstall": True}):
        with pytest.raises(UsageError):
            install("codex", "user", home=tmp_path, cwd=tmp_path, **kwargs)
    assert (dest / "SKILL.md").read_text() == "my own skill"
    install("codex", "user", home=tmp_path, cwd=tmp_path, force=True)
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == skill_text()


def test_install_dry_run_writes_nothing(tmp_path):
    messages = install("codex", "user", home=tmp_path, cwd=tmp_path, dry_run=True)
    assert messages[0].startswith("would install") and not (tmp_path / ".agents").exists()


def test_skill_frontmatter_has_required_fields():
    text = skill_text()
    head = text.split("---")[1]
    assert "name: prompt-sieve" in head and "\ndescription: " in head
    assert len(head.split("description: ", 1)[1].splitlines()[0]) <= 1024


# -- Claude Code plugin manifests ------------------------------------------------------

def _json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_plugin_and_marketplace_manifests_agree():
    plugin = _json(".claude-plugin/plugin.json")
    market = _json(".claude-plugin/marketplace.json")
    assert plugin["name"] == "prompt-sieve" == market["name"]
    assert plugin["version"] == promptsieve.__version__
    entry = market["plugins"][0]
    assert entry["name"] == plugin["name"] and entry["source"] == "./"
    assert market["owner"]["name"]
    for path in plugin["skills"]:
        assert path.startswith("./") and ".." not in path
        assert (ROOT / path / "prompt-sieve" / "SKILL.md").is_file()
    assert not (ROOT / "skills").exists(), "a root skills/ folder would load a second copy of the skill"


def test_hook_config_points_at_the_bundled_launcher():
    hooks = _json("hooks/hooks.json")["hooks"]["PostToolUse"]
    assert hooks[0]["matcher"] == "Bash"
    command = hooks[0]["hooks"][0]["command"]
    assert command == '"${CLAUDE_PLUGIN_ROOT}/bin/prompt-sieve" hook claude-code'
    launcher = ROOT / "bin" / "prompt-sieve"
    assert launcher.read_text(encoding="utf-8").startswith("#!/bin/sh\n")
    if POSIX:
        assert os.access(launcher, os.X_OK)


@pytest.mark.skipif(not POSIX, reason="the launcher is a POSIX shell script")
def test_launcher_ignores_modules_planted_in_the_working_directory(tmp_path):
    (tmp_path / "json.py").write_text('raise SystemExit("planted json ran")')
    (tmp_path / "promptsieve").mkdir()
    (tmp_path / "promptsieve" / "__init__.py").write_text('raise SystemExit("planted package ran")')
    launcher = str(ROOT / "bin" / "prompt-sieve")
    version = subprocess.run([launcher, "--version"], capture_output=True, text=True, cwd=tmp_path)
    assert version.returncode == 0 and promptsieve.__version__ in version.stdout
    hook = subprocess.run([launcher, "hook", "claude-code"], input=json.dumps(_payload()), capture_output=True,
                          text=True, cwd=tmp_path, env={**os.environ, "PROMPTSIEVE_LOG_DIR": str(tmp_path / "l")})
    assert hook.returncode == 0
    assert "updatedToolOutput" in json.loads(hook.stdout)["hookSpecificOutput"]


# -- regressions from the issue #4 patch review -----------------------------------------

@pytest.mark.skipif(not POSIX, reason="symlink creation needs privileges on Windows")
@pytest.mark.parametrize("entry", ["SKILL.md", MARKER])
def test_install_does_not_follow_skill_file_symlink(tmp_path, entry):
    home = tmp_path / "home"
    install("codex", "user", home=home, cwd=tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    link = home / ".agents" / "skills" / "prompt-sieve" / entry
    link.unlink()
    link.symlink_to(victim)
    for kwargs in ({}, {"force": True}, {"uninstall": True}):
        with pytest.raises(UsageError):
            install("codex", "user", home=home, cwd=tmp_path, **kwargs)
    assert victim.read_text() == "precious" and link.is_symlink()


def test_install_rejects_a_folder_where_skill_file_belongs(tmp_path):
    dest = tmp_path / ".agents" / "skills" / "prompt-sieve"
    (dest / "SKILL.md").mkdir(parents=True)
    with pytest.raises(UsageError):
        install("codex", "user", home=tmp_path, cwd=tmp_path, force=True)


def _unsavable(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    return {"PROMPTSIEVE_LOG_DIR": str(blocker / "logs")}


def test_hook_leaves_original_when_raw_log_cannot_be_saved(tmp_path):
    assert claude_hook.handle(_payload(), _unsavable(tmp_path)) is None


def test_run_leaves_original_when_raw_log_cannot_be_saved(tmp_path, capsys):
    out = io.BytesIO()
    assert run_command(_print_log_and_exit(2), out=out, env=_unsavable(tmp_path)) == 2
    assert out.getvalue() == LOG_BYTES
    assert "showing it unshortened" in capsys.readouterr().err


@pytest.mark.parametrize("command", [
    "'cat' build.log", '"cat" build.log', "command -- cat build.log", "sudo -n cat build.log",
    "git -C repo --no-pager show HEAD",
])
def test_exact_read_commands_are_not_shortened(command, env):
    assert claude_hook.handle(_payload(command=command), env) is None


@pytest.mark.parametrize("command", [
    "\\cat f", "sudo -u root -- cat f", "env -i FOO=1 cat f", "timeout 30 cat f", "ls | xargs -n1 cat",
    "bash -lc 'cat build.log'", "pytest\ncat build.log", "FOO='x;y' cat f", "nice -n 5 tail f", "/usr/bin/cat f",
])
def test_wrapped_and_chained_exact_reads_are_skipped(command):
    assert claude_hook.should_skip(command)


@pytest.mark.parametrize("command", [
    "echo $(pytest)", "pytest `which x`", "eval pytest", "source run.sh", "pytest 'unbalanced",
    "env -S 'cat f'", "diff <(pytest) expected",
])
def test_command_lines_that_cannot_be_read_confidently_are_left_alone(command):
    assert claude_hook.should_skip(command)


@pytest.mark.parametrize("command", [
    "pytest -v 2>&1", "cd app && npm test -- --grep x", "env CI=1 pytest", "timeout 600 cargo test",
    "sh -c 'pytest -q'", "sudo make install", "FOO='a b' pytest",
])
def test_ordinary_test_and_build_commands_are_still_considered(command):
    assert not claude_hook.should_skip(command)


def test_run_saved_copy_preserves_original_bytes(env):
    code = f"import sys; sys.stdout.buffer.write(b'\\xff\\xfe cp1252: \\x93quoted\\x94\\r\\n' + open({str(LOG_PATH)!r}, 'rb').read())"
    out = io.BytesIO()
    run_command(_py(code), out=out, env=env)
    saved = list(Path(env["PROMPTSIEVE_LOG_DIR"]).rglob("*.log"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"\xff\xfe cp1252: \x93quoted\x94\r\n" + LOG_BYTES


def test_nonlog_output_with_terminal_formatting_stays_unchanged(env):
    styled = "\x1b[1mThis is an ordinary paragraph of documentation text.\x1b[0m   \n" * 200
    assert claude_hook.handle(_payload(stdout=styled), env) is None
    out = io.BytesIO()
    run_command(_py(f"import sys; sys.stdout.buffer.write({styled.encode('utf-8')!r})"), out=out, env=env)
    assert out.getvalue() == styled.encode("utf-8")


@pytest.mark.skipif(not POSIX, reason="symlink creation needs privileges on Windows")
def test_raw_log_does_not_follow_symlinked_ancestor(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "cache").symlink_to(elsewhere)
    assert rawlog.save_raw("x", "", {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "cache" / "logs")}) is None
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(not POSIX, reason="symlink creation needs privileges on Windows")
def test_raw_log_trusts_a_linked_home_directory_itself(tmp_path, monkeypatch):
    real_home = tmp_path / "data-home"
    real_home.mkdir()
    (tmp_path / "home-link").symlink_to(real_home)
    monkeypatch.setenv("HOME", str(tmp_path / "home-link"))
    path = rawlog.save_raw("x", "", {"XDG_CACHE_HOME": str(tmp_path / "home-link" / ".cache")})
    assert path is not None and (real_home / ".cache" / "prompt-sieve" / "logs").is_dir()
    (real_home / ".cache" / "prompt-sieve" / "logs").rename(real_home / "moved")
    (real_home / ".cache" / "prompt-sieve" / "logs").symlink_to(real_home / "moved")
    assert rawlog.save_raw("x", "", {"XDG_CACHE_HOME": str(tmp_path / "home-link" / ".cache")}) is None


# -- regressions from the second patch review --------------------------------------------

@pytest.mark.parametrize("command", [
    "true # explanatory comment\ncat build.log",
    "if test -f build.log; then cat build.log; fi",
    'for f in build.log; do cat "$f"; done',
    "2>/dev/null cat build.log",
    "bash -O extglob -c 'cat build.log'",
])
def test_exact_output_is_preserved_in_shell_constructs(command, env):
    assert claude_hook.handle(_payload(command=command), env) is None


@pytest.mark.parametrize("command", ['READER=cat; "$READER" build.log', "${TOOL} f", "./c*t f", "~/bin/reader f"])
def test_dynamic_executable_is_treated_as_uncertain(command, env):
    assert claude_hook.handle(_payload(command=command), env) is None


@pytest.mark.parametrize("command", [
    "{ cat f; }", "! cat f", "bash -c -- 'cat f'", "< build.log cat", "while true; do pytest; done",
    "case x in *) cat f;; esac", "bash -c", "pytest >", "sudo", "xargs",
])
def test_more_shell_constructs_are_left_alone(command):
    assert claude_hook.should_skip(command)


@pytest.mark.parametrize("command", [
    "pytest -v > out.txt 2>&1", "python -m pytest -q 2>/dev/null", "bash run_tests.sh",
    "npm test 2>&1 | tee test.log", "pytest 'tests/test a.py'", "go test ./...", "bash -o pipefail -c 'pytest -q'",
])
def test_redirections_and_scripts_still_allow_shortening(command):
    assert not claude_hook.should_skip(command)


@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_existing_log_directory_is_made_private(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    os.chmod(logs, 0o777)
    path = rawlog.save_raw("synthetic output", "t", {"PROMPTSIEVE_LOG_DIR": str(logs)})
    assert stat.S_IMODE(logs.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Windows access control lists")
@pytest.mark.parametrize("inherit_core_modules", [False, True], ids=["native-modules", "inherited-core-modules"])
def test_existing_log_directory_is_private_or_refused(tmp_path, inherit_core_modules):
    logs = tmp_path / "logs"
    logs.mkdir()
    # Grant Everyone (S-1-1-0) inheritable read access on this disposable folder only.
    subprocess.run(["icacls", str(logs), "/grant", "*S-1-1-0:(OI)(CI)R"], check=True, capture_output=True)
    path = rawlog.save_raw("synthetic output", "t", {"PROMPTSIEVE_LOG_DIR": str(logs)})
    if path is None:
        return  # refusing to save is acceptable: nothing is shortened then
    child_env = dict(os.environ)
    if inherit_core_modules:
        # Model pwsh -> Python -> powershell.exe on GitHub's Windows runner.
        # Windows PowerShell cannot load the inherited PowerShell 7 module.
        module = tmp_path / "ps7-modules" / "Microsoft.PowerShell.Security"
        module.mkdir(parents=True)
        (module / "Microsoft.PowerShell.Security.psd1").write_text(
            "@{ ModuleVersion='7.0'; PowerShellVersion='7.0'; CmdletsToExport=@('Get-Acl') }",
            encoding="utf-8")
        child_env["PSModulePath"] = str(module.parent) + os.pathsep + child_env.get("PSModulePath", "")
    for target in (path, path.parent):
        # Use .NET directly, independent of PowerShell's module search path.
        kind = "Directory" if target.is_dir() else "File"
        script = (f"[System.IO.{kind}]::GetAccessControl($env:PS_TARGET).GetAccessRules($true, $true, "
                  "[System.Security.Principal.SecurityIdentifier]) | ForEach-Object { $_.IdentityReference.Value }")
        probe = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                               env={**child_env, "PS_TARGET": str(target)},
                               capture_output=True, text=True)
        assert probe.returncode == 0, f"ACL inspection failed: {probe.stderr}"
        sids = probe.stdout.split()
        assert sids, "no access rules read"
        for broad in ("S-1-1-0", "S-1-5-11", "S-1-5-32-545"):  # Everyone, Authenticated Users, Users
            assert broad not in sids


def test_windows_storage_uses_a_fresh_private_folder_per_output(tmp_path, monkeypatch):
    """Exercises the Windows code path on any platform."""
    monkeypatch.setattr(rawlog, "_HAS_DIR_FD", False)
    env = {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}
    first = rawlog.save_raw(b"\xff one", "a", env)
    second = rawlog.save_raw("two", "a", env)
    assert first.name == second.name == "output.log" and first.parent != second.parent
    assert first.read_bytes() == b"\xff one"
    if POSIX:
        assert stat.S_IMODE(first.parent.stat().st_mode) == 0o700


def test_windows_storage_refuses_pythons_without_private_mkdir(tmp_path, monkeypatch):
    monkeypatch.setattr(rawlog, "_HAS_DIR_FD", False)
    monkeypatch.setattr(rawlog, "_is_windows", lambda: True)
    monkeypatch.setattr(rawlog, "windows_private_mkdir_supported", lambda: False)
    assert rawlog.save_raw("x", "", {"PROMPTSIEVE_LOG_DIR": str(tmp_path / "logs")}) is None
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("version,expected", [
    ((3, 10, 14), False), ((3, 10, 15), False), ((3, 11, 9), False), ((3, 11, 10), True),
    ((3, 12, 3), False), ((3, 12, 4), True), ((3, 13, 0), True), ((3, 14, 1), True),
])
def test_windows_private_mkdir_versions(version, expected):
    assert rawlog.windows_private_mkdir_supported(version) is expected
