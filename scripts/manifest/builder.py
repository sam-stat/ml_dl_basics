"""Assemble notes.json and provide the CLI entry point."""

from __future__ import annotations

import datetime as dt
import json

from . import config, discovery


def build_manifest() -> dict:
    """The full manifest: a UTC timestamp plus the ordered sections."""
    return {
        "generated": dt.datetime.now(dt.timezone.utc)
                       .replace(microsecond=0).isoformat(),
        "sections": discovery.find_sections(),
    }


def write_manifest(manifest: dict) -> None:
    config.OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    manifest = build_manifest()
    write_manifest(manifest)
    n_notes = sum(len(s["topics"]) for s in manifest["sections"])
    print(f"wrote {config.OUTPUT.relative_to(config.ROOT)} — "
          f"{len(manifest['sections'])} section(s), {n_notes} note(s)")
