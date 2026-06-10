"""notes-manifest generator — scans notes/ and writes notes.json.

Public entry point: `from manifest import main`.
"""

from .builder import build_manifest, main, write_manifest

__all__ = ["main", "build_manifest", "write_manifest"]
