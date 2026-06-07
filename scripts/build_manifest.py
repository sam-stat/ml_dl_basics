#!/usr/bin/env python3
"""
Generate notes.json — the manifest that drives the website (index.html).

Scans the repo root for "note folders": any directory containing numbered
section files like `01_foo.md`, `02_bar.md`. For each note file it reads the
H1 as the title and the first prose paragraph as a short description.

Run it manually before previewing locally:

    python3 scripts/build_manifest.py

…or let the GitHub Action (.github/workflows/build-notes-manifest.yml) run it
on every push so new make-notes folders appear on the site automatically.

No third-party dependencies — standard library only.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders that contain notes but should never be scanned as note folders,
# plus anything starting with "." or "_".
IGNORE_DIRS = {"scripts", "diagrams", "helper_docs", "diagram_style_samples", "node_modules"}

# Group display order. Folders not listed here are appended alphabetically,
# so a brand-new make-notes folder still shows up with zero edits to this file.
GROUP_PRIORITY = ["neural_networks", "causal_methods"]

# Pretty-print folder slugs as group titles, with acronyms forced upper-case.
ACRONYMS = {"ml", "dl", "nn", "cnn", "rnn", "lstm", "llm", "nlp", "rl", "svm", "gan"}

NOTE_RE = re.compile(r"^\d+_.+\.md$")          # 01_foo.md
LEADING_NUM_RE = re.compile(r"^\d+[.)]?\s+")   # "1. Title" / "1) Title"


def title_from_slug(slug: str) -> str:
    words = []
    for w in slug.replace("-", "_").split("_"):
        if not w:
            continue
        words.append(w.upper() if w.lower() in ACRONYMS else w.capitalize())
    return " ".join(words)


_GREEK = {r"\tau": "τ", r"\theta": "θ", r"\sigma": "σ", r"\mu": "μ",
          r"\beta": "β", r"\gamma": "γ", r"\alpha": "α", r"\lambda": "λ",
          r"\varepsilon": "ε", r"\epsilon": "ε", r"\pi": "π", r"\rho": "ρ",
          r"\eta": "η", r"\phi": "φ", r"\delta": "δ", r"\omega": "ω"}


def _math_to_text(inner: str) -> str:
    """Render a *simple* inline-math symbol as plain text; drop complex math."""
    s = inner.strip()
    if s in _GREEK:
        return _GREEK[s]
    s = re.sub(r"\\(hat|bar|tilde|vec|mathbf|mathbb|mathcal|text|mathrm)\{([^{}]*)\}",
               r"\2", s)                       # \hat{y} -> y
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
    md = re.sub(r"\s+([,.;:?])", r"\1", md)                # space before punct
    md = re.sub(r"\(\s*\)", "", md)                         # empty parens left by drops
    md = re.sub(r"\s{2,}", " ", md)
    return md.strip()


def parse_note(path: Path) -> dict:
    title, desc = path.stem, ""
    lines = path.read_text(encoding="utf-8").splitlines()

    i = 0
    # Title = first H1.
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = LEADING_NUM_RE.sub("", line[2:].strip())
            break

    # Description = first "sentence-like" prose paragraph after the title.
    # Paragraphs that are section lead-ins (end in ":") or tiny labels are
    # skipped in favour of the next real paragraph.
    para: list[str] = []

    def take(p: list[str]) -> str | None:
        text = _clean_inline(" ".join(p))
        if text.endswith(":"):
            # Trim a trailing lead-in clause; keep the substantive sentences.
            ends = list(re.finditer(r"[.?!]\s", text))
            text = text[:ends[-1].start() + 1] if ends else ""
        return text if len(text) >= 40 else None

    for line in lines[i + 1:]:
        s = line.strip()
        if not s:
            if para:
                if (chosen := take(para)):
                    desc = chosen
                    break
                para = []
            continue
        if (s.startswith(("#", "---", "![", ">", "|", "```", "$$"))
                or s.startswith(("- ", "* ", "+ "))
                or re.match(r"^\d+[.)]\s", s)):
            if para:
                if (chosen := take(para)):
                    desc = chosen
                    break
                para = []
            continue
        para.append(s)
    else:
        if not desc and para and (chosen := take(para)):
            desc = chosen

    if len(desc) > 155:
        desc = desc[:155].rsplit(" ", 1)[0] + "…"

    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "title": title, "desc": desc}


def discover() -> list[dict]:
    sections = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if d.name.startswith((".", "_")) or d.name in IGNORE_DIRS:
            continue
        notes = sorted(f for f in d.iterdir()
                       if f.is_file() and NOTE_RE.match(f.name))
        if not notes:
            continue
        sections.append({
            "id": d.name,
            "title": title_from_slug(d.name),
            "topics": [parse_note(f) for f in notes],
        })

    order = {name: i for i, name in enumerate(GROUP_PRIORITY)}
    sections.sort(key=lambda s: (order.get(s["id"], len(order)), s["title"]))
    return sections


def main() -> None:
    manifest = {
        "generated": _dt.datetime.now(_dt.timezone.utc)
                        .replace(microsecond=0).isoformat(),
        "sections": discover(),
    }
    out = ROOT / "notes.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    n = sum(len(s["topics"]) for s in manifest["sections"])
    print(f"wrote {out.relative_to(ROOT)} — "
          f"{len(manifest['sections'])} section(s), {n} note(s)")


if __name__ == "__main__":
    main()
