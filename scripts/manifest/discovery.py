"""Walk notes/ and build the ordered section/topic structure."""

from __future__ import annotations

from pathlib import Path

from . import config, markdown


def title_from_slug(slug: str) -> str:
    """neural_networks -> 'Neural Networks', cnn_guide -> 'CNN Guide'."""
    words = []
    for w in slug.replace("-", "_").split("_"):
        if not w:
            continue
        words.append(w.upper() if w.lower() in config.ACRONYMS else w.capitalize())
    return " ".join(words)


def parse_note(path: Path) -> dict:
    """Read one note file into a {file, title, desc} record."""
    lines = path.read_text(encoding="utf-8").splitlines()
    title, h1 = markdown.extract_title(lines)
    desc = markdown.extract_description(lines, h1 + 1)
    return {
        "file": str(path.relative_to(config.ROOT)).replace("\\", "/"),
        "title": title or path.stem,
        "desc": desc,
    }


def find_sections() -> list[dict]:
    """Each topic folder under notes/ becomes one section, ordered by
    GROUP_PRIORITY then alphabetically."""
    sections = []
    if not config.NOTES_DIR.is_dir():
        return sections

    for d in sorted(p for p in config.NOTES_DIR.iterdir() if p.is_dir()):
        if d.name.startswith((".", "_")) or d.name in config.IGNORE_DIRS:
            continue
        notes = sorted(f for f in d.iterdir()
                       if f.is_file() and config.NOTE_RE.match(f.name))
        if not notes:
            continue
        sections.append({
            "id": d.name,
            "title": title_from_slug(d.name),
            "topics": [parse_note(f) for f in notes],
        })

    order = {name: i for i, name in enumerate(config.GROUP_PRIORITY)}
    sections.sort(key=lambda s: (order.get(s["id"], len(order)), s["title"]))
    return sections
