"""Near-lossless cleanup: ANSI codes, carriage-return progress bars, trailing
spaces, runs of blank lines. Indentation is kept (it matters for code)."""
import re

from ..protect import _FENCE, join_segments, split_segments

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def normalize(text: str) -> str:
    # Terminal cleanup must not alter literal controls or line endings in code.
    parts, pos = [], 0
    for fence in _FENCE.finditer(text):
        parts.append(_normalize_region(text[pos:fence.start()]))
        parts.append(fence.group(0))
        pos = fence.end()
    parts.append(_normalize_region(text[pos:]))
    return "".join(parts)


def _normalize_region(text: str) -> str:
    text = ANSI.sub("", text)
    text = text.replace("\r\n", "\n")
    lines = []
    for line in text.split("\n"):
        parts = line.split("\r")
        # Only discard a recognized, content-free download progress update.
        # General CR output may contain a diagnostic before the final display.
        if len(parts) > 1 and all(re.fullmatch(r"Downloading \d{1,3}%", p) for p in parts):
            lines.append(parts[-1])
        else:
            lines.append("\n".join(parts))
    segs = split_segments("\n".join(lines))
    for s in segs:
        if not s.protected:
            # Preserve single trailing spaces too: the Phase 3 Persian replay
            # changes Qwen's answer from 42 to 10 if the paragraph separator
            # loses that byte. Two spaces are also a Markdown hard break.
            s.text = re.sub(r"(?m)(?<! )[ \t]{3,}$|\t+$", "", s.text)
        elif not s.text.strip():
            s.text = re.sub(r"\n{3,}", "\n\n", s.text)
    return join_segments(segs)
