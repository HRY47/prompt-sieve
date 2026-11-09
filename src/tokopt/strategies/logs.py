"""Rule-based compressor for logs and coding-agent tool output.

Designed for the common case where a 20k-token test or build log contains a
handful of lines that matter (failures, errors, stack traces, the summary)
buried under thousands of lines of noise (passing tests, dependency
installation, progress bars, repeated lines).

Guarantee: every line matching an error/failure marker is kept verbatim.
That property is checked by :meth:`LogCompressor.verify` and enforced in the
optimizer.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .normalize import normalize

NOISE: Dict[str, re.Pattern] = {
    "passing-test": re.compile(
        r"(^\S+::\S+\s+PASSED(?:\s+\[\s*\d+%\])?\s*$"
        r"|^\s*[✓✔√]\s"
        r"|^\s*--- PASS:"
        r"|^=== (RUN|PAUSE|CONT)\s"
        r"|^ok\s+\S+\s+(\(cached\)|[\d.]+s)"
        r"|^\s*PASS\s+\S+"
        r"|^\S+\.(py|js|ts)\s+\.+\s*(\[\s*\d+%\])?\s*$"
        r"|^\[\s*OK\s*\]|^\[\s+PASSED\s+\]|^test \S+ \.\.\. ok$)"
    ),
    "dependency-install": re.compile(
        r"^\s*(Requirement already satisfied|Collecting\s|Downloading\s|Using cached\s|"
        r"Obtaining\s|Installing collected packages|Building wheel|Created wheel|Stored in directory|"
        r"npm (http|timing|sill|verb)\b|added \d+ packages|Resolving dependencies|"
        r"Downloaded\s|Compiling\s\S+\sv\d|Fetching\s|Get:\d+\s|Hit:\d+\s|Unpacking\s|Setting up\s|"
        r"Preparing to unpack|Selecting previously unselected)"
    ),
    "progress": re.compile(
        r"(^\s*\d{1,3}%\s*\||\[[=#>\-. ]{5,}\]\s*\d{1,3}%|^\s*[\d.]+\s*[kMG]?B/s|"
        r"^\s*\|[█▉▊▋▌▍▎▏ ]+\|)"
    ),
}

IMPORTANT = re.compile(
    r"(\berror\b|\berrors?:|exception|traceback|\bfail(ed|ure|ures|ing)?\b|\bfatal\b|\bpanic\b|"
    r"\bassert(ion)?\b|\bwarn(ing)?\b|denied|not found|no such file|segmentation fault|"
    r"timed? ?out|\bundefined\b|cannot|could not|unable to|^E\s{2,}|^>\s|\bERR!)",
    re.I,
)
SUMMARY = re.compile(
    r"(^=+ .*\b(passed|failed|error|errors|skipped)\b.*=+$|^Tests?:\s|^Test Suites:|"
    r"^Ran \d+ tests?|^(FAILED|OK)( \(|$)|\b\d+ (passed|failed)(,| in )|BUILD (SUCCESS|FAILURE)|"
    r"^Tests run: \d+)",
    re.I,
)
TIMESTAMP = re.compile(
    r"^\[?\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z|[+-]\d\d:?\d\d)?\]?\s*"
)
JAVA_FRAME = re.compile(r"^\s+at\s+[\w$.<>]+\(.*\)\s*$")
# Lines that were evidently produced by a program. Text before the first such
# line and after the last one (a user's instructions or question around a pasted
# log) is kept verbatim: only the log body is compressed.
LOG_LINE = re.compile(
    r"^\s*\[?(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL|SEVERE)\b"
    r"|^\s*\[?(?:trace|debug|info|notice|warn|warning|error|fatal|critical)[\]:]"
    r"|^\s*\d{1,2}:\d\d:\d\d|^\[\s*\d+(?:[./]\d+)?\s*[%\]]"
    r"|^\S+::\S+|^(?:PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s|^E\s{2,}|^>\s"
    r"|^\s*File \".*\", line \d+|^Traceback \(most recent call last\)|^\s*Caused by:"
    r"|^[=_\-*#~]{5,}|^\S+\.[A-Za-z]\w*:\d+(?::\d+)?:|^(?:error|warning|note)(?:\[\w+\])?:"
    r"|^npm (?:ERR!|WARN)|^\s*at\s"
    r"|^[A-Za-z_][\w.$]*(?:Error|Exception|Warning|Exit|Interrupt|Failure)\b(?::|$)"
)


@dataclass
class LogStats:
    dropped: Counter = field(default_factory=Counter)
    duplicates_collapsed: int = 0
    lines_omitted: int = 0
    frames_collapsed: int = 0
    important_lines: int = 0


def looks_like_log(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 20:
        return False
    hits = sum(
        1 for l in lines[:2000]
        if TIMESTAMP.match(l) or IMPORTANT.search(l) or SUMMARY.search(l)
        or any(p.search(l) for p in NOISE.values())
    )
    return hits / min(len(lines), 2000) > 0.15


class LogCompressor:
    name = "logs"

    def __init__(self, context: int = 3, head: int = 5, tail: int = 15,
                 max_lines: int = 300, max_frames: int | None = None,
                 strip_timestamps: bool = False, footer: bool = True):
        if min(context, head, tail, max_lines) < 0 or (max_frames is not None and max_frames < 2):
            raise ValueError("log limits must be nonnegative and max_frames >= 2")
        self.context, self.head, self.tail = context, head, tail
        self.max_lines, self.max_frames = max_lines, max_frames
        self.strip_timestamps, self.footer = strip_timestamps, footer
        self.last_stats = LogStats()

    # -- public -----------------------------------------------------------
    def apply(self, text: str) -> str:
        st = self.last_stats = LogStats()
        preamble, lines, epilogue = _split_surrounding_text(normalize(text).split("\n"))
        if self.strip_timestamps:
            lines = [TIMESTAMP.sub("", l) for l in lines]

        lines = self._drop_noise(lines, st)
        lines = self._collapse_frames(lines, st)
        lines = self._dedupe(lines, st)
        lines = self._focus(lines, st)

        out = "\n".join(preamble + lines + epilogue).strip("\n")
        if self.footer:
            notes = [f"{n} {k} lines" for k, n in st.dropped.most_common()]
            if st.duplicates_collapsed:
                notes.append(f"{st.duplicates_collapsed} repeated lines")
            if st.frames_collapsed:
                notes.append(f"{st.frames_collapsed} stack frames")
            if st.lines_omitted:
                notes.append(f"{st.lines_omitted} other lines far from any error")
            if notes:
                out += "\n[tokopt removed: " + ", ".join(notes) + "]"
        return out + "\n"

    @staticmethod
    def verify(original: str, compressed: str) -> List[str]:
        """Return important lines, and non-log text surrounding the log, that
        are missing from the compressed output."""
        comp = Counter(l.strip() for l in normalize(compressed).split("\n"))
        missing = []
        lines = normalize(original).split("\n")
        preamble, _, epilogue = _split_surrounding_text(lines)
        surrounding = {id(l) for l in preamble + epilogue}
        for line in lines:
            s = line.strip()
            if s and (id(line) in surrounding or IMPORTANT.search(s) or SUMMARY.search(s)):
                key = s if comp[s] else TIMESTAMP.sub("", s)
                if not comp[key]:
                    missing.append(s)
                else:
                    comp[key] -= 1
        return missing

    # -- steps ------------------------------------------------------------
    def _drop_noise(self, lines: List[str], st: LogStats) -> List[str]:
        kept = []
        for l in lines:
            cat = _noise_category(l)
            if cat and not (IMPORTANT.search(l) or SUMMARY.search(l)):
                st.dropped[cat] += 1
            else:
                kept.append(l)
        return kept

    def _collapse_frames(self, lines: List[str], st: LogStats) -> List[str]:
        out, run = [], []

        def flush():
            if self.max_frames is not None and len(run) > self.max_frames and not any(IMPORTANT.search(l) for l in run):
                keep = self.max_frames // 2
                out.extend(run[:keep])
                out.append(f"\t... {len(run) - 2 * keep} more frames ...")
                out.extend(run[-keep:])
                st.frames_collapsed += len(run) - 2 * keep
            else:
                out.extend(run)
            run.clear()

        for l in lines:
            if JAVA_FRAME.match(l):
                run.append(l)
            else:
                flush()
                out.append(l)
        flush()
        return out

    def _dedupe(self, lines: List[str], st: LogStats) -> List[str]:
        out: List[str] = []
        prev_key, count = None, 0
        for l in lines + [None]:  # sentinel
            key = TIMESTAMP.sub("", l).strip() if l is not None else None
            if l is not None and (IMPORTANT.search(l) or SUMMARY.search(l)):
                key = object()  # retain diagnostic occurrences and timestamps
            if key == prev_key and key:
                count += 1
                continue
            if count > 1:
                out[-1] = f"{out[-1]}  [x{count}]"
                st.duplicates_collapsed += count - 1
            if l is not None:
                out.append(l)
            prev_key, count = key, 1
        return out

    def _focus(self, lines: List[str], st: LogStats) -> List[str]:
        # With no recognized diagnostic or summary there is nothing to focus
        # on; keeping only the head and tail would discard the log blindly.
        if not any(IMPORTANT.search(l) or SUMMARY.search(l) for l in lines):
            return lines
        # Failed pytest sections can contain arbitrary captured output and
        # assertion diffs. Keep those sections in full, even beyond max_lines.
        if any(re.search(r"^=+\s+(?:FAILURES|ERRORS)\s+=+|^-+ Captured ", l) for l in lines):
            st.important_lines = sum(1 for l in lines if IMPORTANT.search(l))
            return lines
        if len(lines) <= self.max_lines:
            st.important_lines = sum(1 for l in lines if IMPORTANT.search(l))
            return lines
        keep: Set[int] = set(range(min(self.head, len(lines))))
        keep |= set(range(max(0, len(lines) - self.tail), len(lines)))
        in_tb = False
        for i, l in enumerate(lines):
            if l.startswith("Traceback") or l.lstrip().startswith("Caused by:"):
                in_tb = True
            if in_tb:
                keep.add(i)
                if l and not l[0].isspace() and not l.startswith(("Traceback", "Caused by")):
                    in_tb = False  # the exception line itself ends the block
                continue
            if IMPORTANT.search(l) or SUMMARY.search(l):
                st.important_lines += 1
                keep |= set(range(max(0, i - self.context), min(len(lines), i + self.context + 1)))
            elif JAVA_FRAME.match(l) or l.startswith("\t... "):
                keep.add(i)
        out, gap = [], 0
        for i, l in enumerate(lines):
            if i in keep:
                if gap:
                    out.append(f"... [{gap} lines omitted]")
                    st.lines_omitted += gap
                    gap = 0
                out.append(l)
            else:
                gap += 1
        if gap:
            out.append(f"... [{gap} lines omitted]")
            st.lines_omitted += gap
        return out


def _is_log_line(line: str) -> bool:
    return bool(TIMESTAMP.match(line) or LOG_LINE.search(line) or JAVA_FRAME.match(line)
                or SUMMARY.search(line) or _noise_category(line))


def _split_surrounding_text(lines: List[str]):
    """Split into (text before the log, log body, text after the log)."""
    idx = [i for i, l in enumerate(lines) if l.strip() and _is_log_line(l)]
    if not idx:
        return lines, [], []
    return lines[:idx[0]], lines[idx[0]:idx[-1] + 1], lines[idx[-1] + 1:]


def _noise_category(line: str):
    for cat, pat in NOISE.items():
        if pat.search(line):
            return cat
    return None


def _is_noise(line: str) -> bool:
    return _noise_category(line) is not None and not SUMMARY.search(line)
