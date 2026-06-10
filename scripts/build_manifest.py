#!/usr/bin/env python3
"""
Generate notes.json — the manifest that drives the website (index.html).

Thin CLI entry point. The logic lives in the `manifest` package next to this
file (config / markdown / discovery / builder). Scans notes/ for topic folders
of numbered section files (01_foo.md …); for each it reads the H1 as the title
and the first prose paragraph as a short description.

Run it manually before previewing locally:

    python3 scripts/build_manifest.py

…or let the GitHub Action (.github/workflows/build-notes-manifest.yml) run it
on every push so new make-notes folders appear on the site automatically.

No third-party dependencies — standard library only.
"""

import sys
from pathlib import Path

# Make the sibling `manifest` package importable regardless of how this script
# is launched, then defer to its main().
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import main  # noqa: E402

if __name__ == "__main__":
    main()
