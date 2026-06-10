"""Extract a display title and a short blurb from a note's markdown.

Pure functions over a list of lines — no filesystem access — so they are easy
to test in isolation.
"""

from __future__ import annotations

import re

from .config import (
    BLOCK_PREFIXES,
    GREEK,
    LEADING_NUM_RE,
    MAX_DESC_LEN,
    MIN_DESC_LEN,
    ORDERED_LIST_RE,
)


def _math_to_text(inner: str) -> str:
    """Render a *simple* inline-math symbol as plain text; drop complex math."""
    s = inner.strip()
    if s in GREEK:
        return GREEK[s]
    # \hat{y} -> y, \mathbf{x} -> x, etc.
    s = re.sub(r"\\(hat|bar|tilde|vec|mathbf|mathbb|mathcal|text|mathrm)\{([^{}]*)\}",
               r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("{", "").replace("}", "").strip()
    return s if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9,'’\- ]{0,4}", s) else ""


def _clean_inline(md: str) -> str:
    """Strip the markdown/LaTeX that shouldn't appear in a plain-text blurb."""
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)          # images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)      # links -> text
    md = re.sub(r"\$\$.+?\$\$", "", md)                    # block math -> drop
    md = re.sub(r"\$([^$]+?)\$",                            # inline math -> symbol
                lambda m: _math_to_text(m.group(1)), md)
    md = re.sub(r"[*_`]+", "", md)                          # emphasis / code ticks
    md = re.sub(r"\s+([,.;:?])", r"\1", md)                # space before punctuation
    md = re.sub(r"\(\s*\)", "", md)                         # empty parens left by drops
    md = re.sub(r"\s{2,}", " ", md)
    return md.strip()


def extract_title(lines: list[str]) -> tuple[str, int]:
    """Return (title, h1_line_index). Title is the first H1 with any leading
    "1. " numbering stripped. If there is no H1, returns ("", -1)."""
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            return LEADING_NUM_RE.sub("", line[2:].strip()), idx
    return "", -1


def _take(paragraph: list[str]) -> str | None:
    """Turn a gathered paragraph into a blurb, or None if it's too thin."""
    text = _clean_inline(" ".join(paragraph))
    if text.endswith(":"):
        # Drop a trailing lead-in clause; keep the substantive sentences.
        ends = list(re.finditer(r"[.?!]\s", text))
        text = text[:ends[-1].start() + 1] if ends else ""
    return text if len(text) >= MIN_DESC_LEN else None


def _truncate(text: str) -> str:
    if len(text) > MAX_DESC_LEN:
        return text[:MAX_DESC_LEN].rsplit(" ", 1)[0] + "…"
    return text


def extract_description(lines: list[str], start: int = 0) -> str:
    """First substantive prose paragraph after `start`, cleaned and truncated.

    Section lead-ins (paragraphs ending in ":") and tiny labels are skipped in
    favour of the next real paragraph."""
    paragraph: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith(BLOCK_PREFIXES) or ORDERED_LIST_RE.match(s):
            if paragraph and (chosen := _take(paragraph)):
                return _truncate(chosen)
            paragraph = []
            continue
        paragraph.append(s)

    if paragraph and (chosen := _take(paragraph)):
        return _truncate(chosen)
    return ""
