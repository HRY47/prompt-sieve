"""Private storage for the full output that PromptSieve shortened.

Whenever an integration shows an agent less than a command printed, the
complete output is saved here first, byte for byte, so it can still be
inspected. If it cannot be saved safely, callers must show the original output
instead of a shortened one.

Files live in a per-user cache directory with owner-only permissions and are
pruned automatically. They can contain anything the command printed, including
secrets, so they never go to a shared temporary directory.

No path component below the user's home directory (or, for paths outside the
home directory, below the filesystem root) may be a symbolic link. The home
directory itself is trusted, so a system that keeps homes behind a link such
as ``/home -> /data/home`` still works. On POSIX the directory chain is walked
with ``O_NOFOLLOW`` file descriptors, so a component swapped for a link during
the walk is refused rather than followed, and the final directory must be
owned by the current user and is set to mode 700.

Windows has no equivalent descriptor API, and an existing folder may carry
inherited access rules that grant other users read access. So on Windows every
output is saved into a brand-new folder created with ``os.mkdir(path, 0o700)``,
which Python documents as giving access only to the current user and
administrators. That holds whatever the parent folder allows. Python versions
that predate this behaviour (anything before 3.11.10, 3.12.4 or 3.13) do not
save at all, so nothing is shortened. The parent folders are still checked
for links with ``lstat`` first, which cannot rule out a race by another process
running as the same user.
"""
from __future__ import annotations

import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Union

MAX_AGE_SECONDS = 7 * 24 * 3600
MAX_FILES = 200
_NAME = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}(?:-[A-Za-z0-9_.-]{1,40})?\.log$")
_PRIVATE_DIR = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}(?:-[A-Za-z0-9_.-]{1,40})?$")
_WINDOWS_FILE = "output.log"
_HAS_DIR_FD = os.name != "nt" and os.open in os.supports_dir_fd and os.mkdir in os.supports_dir_fd


def log_dir(env: Mapping[str, str] = os.environ) -> Path:
    if env.get("PROMPTSIEVE_LOG_DIR"):
        return Path(env["PROMPTSIEVE_LOG_DIR"]).expanduser()
    if env.get("CLAUDE_PLUGIN_DATA"):
        return Path(env["CLAUDE_PLUGIN_DATA"]) / "logs"
    if os.name == "nt":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "prompt-sieve" / "logs"


def _trusted_start(directory: Path):
    """Split into (trusted anchor, components that must not be links)."""
    for home in {Path(os.path.abspath(Path.home())), Path(os.path.realpath(Path.home()))}:
        try:
            rest = directory.relative_to(home)
        except ValueError:
            continue
        return home, rest.parts
    return Path(directory.anchor), directory.parts[1:]


def _file_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")[:40]
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}" + (f"-{safe}" if safe else "") + ".log"


def save_raw(data: Union[str, bytes], label: str = "", env: Mapping[str, str] = os.environ) -> Optional[Path]:
    """Save ``data`` and return its path, or None if it could not be stored safely.

    Bytes are written unchanged; text is encoded as UTF-8.
    """
    payload = data if isinstance(data, bytes) else data.encode("utf-8", errors="surrogateescape")
    directory = Path(os.path.abspath(log_dir(env)))
    name = _file_name(label)
    try:
        if _HAS_DIR_FD:
            _save_posix(directory, name, payload)
            return directory / name
        return _save_checked_paths(directory, name, payload)
    except (OSError, ValueError):
        return None


# -- POSIX: descriptor walk, no component is ever followed ------------------------

def _save_posix(directory: Path, name: str, payload: bytes) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    anchor, parts = _trusted_start(directory)
    fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=fd)
                child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise ValueError("log directory is owned by another user")
        if stat.S_IMODE(info.st_mode) != 0o700:
            os.fchmod(fd, 0o700)
        _prune_fd(fd)
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        out = os.open(name, file_flags, 0o600, dir_fd=fd)
        with os.fdopen(out, "wb") as f:
            f.write(payload)
    finally:
        os.close(fd)


def _prune_fd(fd: int, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    entries = []
    for entry in os.listdir(fd):
        if not _NAME.match(entry):
            continue
        try:
            info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            entries.append((info.st_mtime, entry))
    entries.sort(reverse=True)
    for i, (mtime, entry) in enumerate(entries):
        if i >= MAX_FILES - 1 or now - mtime > MAX_AGE_SECONDS:
            try:
                os.unlink(entry, dir_fd=fd)
            except OSError:
                pass


# -- Windows: a fresh private folder for every saved output --------------------------

def _is_windows() -> bool:
    return os.name == "nt"


def windows_private_mkdir_supported(version=sys.version_info) -> bool:
    """Whether os.mkdir(path, 0o700) restricts access on Windows (per the Python docs)."""
    v = tuple(version[:3])
    return v >= (3, 13) or (3, 12, 4) <= v < (3, 13) or (3, 11, 10) <= v < (3, 12)


def _save_checked_paths(directory: Path, name: str, payload: bytes) -> Path:
    if _is_windows() and not windows_private_mkdir_supported():
        raise ValueError("this Python cannot create private folders on Windows")
    current, parts = _trusted_start(directory)
    for part in parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or _is_junction(current) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("log directory path contains a link or a non-directory")
    _prune_private_dirs(directory)
    # A new folder created with 0o700 gets an access list for the current user
    # and administrators only, so the file inside never inherits the parent's
    # rules. mkdir fails rather than reuse a folder that already exists.
    private = directory / name[: -len(".log")]
    os.mkdir(private, 0o700)
    target = private / _WINDOWS_FILE
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(target, flags, 0o600), "wb") as f:
        f.write(payload)
    return target


def _is_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(path))


def _prune_private_dirs(directory: Path, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    entries = []
    for p in directory.iterdir():
        if not _PRIVATE_DIR.match(p.name):
            continue
        try:
            info = os.lstat(p)
        except OSError:
            continue
        if stat.S_ISDIR(info.st_mode) and not _is_junction(p):
            entries.append((info.st_mtime, p))
    entries.sort(reverse=True)
    for i, (mtime, p) in enumerate(entries):
        if i >= MAX_FILES - 1 or now - mtime > MAX_AGE_SECONDS:
            try:
                (p / _WINDOWS_FILE).unlink(missing_ok=True)
                p.rmdir()
            except OSError:
                pass


def _prune(directory: Path, now: Optional[float] = None) -> None:
    """Path-based pruning of flat log files (used by tests)."""
    now = time.time() if now is None else now
    entries = []
    for p in directory.iterdir():
        if not _NAME.match(p.name):
            continue
        try:
            info = os.lstat(p)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            entries.append((info.st_mtime, p))
    entries.sort(reverse=True)
    for i, (mtime, p) in enumerate(entries):
        if i >= MAX_FILES - 1 or now - mtime > MAX_AGE_SECONDS:
            try:
                p.unlink()
            except OSError:
                pass
