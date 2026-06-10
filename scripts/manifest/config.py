"""Central configuration for the notes-manifest generator.

Every tunable lives here — paths, discovery rules, parsing thresholds — so the
rest of the package contains logic only, no scattered literals.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# config.py is scripts/manifest/config.py → repo root is two parents up.
ROOT = Path(__file__).resolve().parents[2]

# All published topic folders live under notes/.
NOTES_DIR = ROOT / "notes"

# The generated manifest, read by index.html at the repo root.
OUTPUT = ROOT / "notes.json"

# ── Discovery ────────────────────────────────────────────────────────────────
# Direct children of notes/ that are not themselves topic folders.
IGNORE_DIRS = {"diagrams", "helper_docs", "diagram_style_samples"}

# Group display order; folders not listed appear afterwards, alphabetically —
# so a brand-new make-notes folder shows up with zero edits to this file.
GROUP_PRIORITY = ["neural_networks", "causal_methods"]

# Slug words forced upper-case when building a group title from a folder name.
ACRONYMS = {"ml", "dl", "nn", "cnn", "rnn", "lstm", "llm", "nlp", "rl", "svm", "gan"}

# A note file: 01_foo.md, 02_bar.md …
NOTE_RE = re.compile(r"^\d+_.+\.md$")

# ── Parsing ──────────────────────────────────────────────────────────────────
# Leading "1. " / "1) " stripped from an H1 before it becomes a title.
LEADING_NUM_RE = re.compile(r"^\d+[.)]?\s+")

# A description must be at least this long (shorter paragraphs are lead-ins or
# tiny labels and get skipped) and is truncated on a word boundary past the max.
MIN_DESC_LEN = 40
MAX_DESC_LEN = 155

# Simple Greek-letter inline math rendered as its literal symbol in a blurb.
GREEK = {
    r"\tau": "τ", r"\theta": "θ", r"\sigma": "σ", r"\mu": "μ",
    r"\beta": "β", r"\gamma": "γ", r"\alpha": "α", r"\lambda": "λ",
    r"\varepsilon": "ε", r"\epsilon": "ε", r"\pi": "π", r"\rho": "ρ",
    r"\eta": "η", r"\phi": "φ", r"\delta": "δ", r"\omega": "ω",
}

# Line prefixes that mark a non-prose block (heading, list item, table, code,
# block math, image, quote). Used to skip past structure when hunting prose.
BLOCK_PREFIXES = ("#", "---", "![", ">", "|", "```", "$$", "- ", "* ", "+ ")

# An ordered-list item ("1. ", "2) ") — also non-prose.
ORDERED_LIST_RE = re.compile(r"^\d+[.)]\s")
