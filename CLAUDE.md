# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **study-notes repository**, not a software project. The deliverables are GitHub-flavoured markdown notes with hand-authored SVG diagrams, sourced from the *Short Notes (All Topics)* material (`.pdf`/`.xlsx`). There is no build, test, or lint step — content is rendered two ways:

- **Natively on GitHub** (markdown + inline SVG + GitHub LaTeX).
- **As a single-page site** via `index.html` (a self-contained SPA served at the repo root).

`ml_dl_topics.md` is the master topic outline that drives what notes get written.

## Repository layout

- `neural_networks/` — the flagship topic folder. Numbered section files (`01_*.md` … `08_*.md`), a `README.md` index, and `diagrams/`.
- `causal_methods/` — raw source material (chat exports + PDF), *not* yet converted into the structured note format and *not* wired into the site.
- `index.html` — the GitHub Pages SPA. `.nojekyll` is present so Pages serves the underscore-named folders and raw files untouched.

## Authoring notes — `/make-notes`

`neural_networks/make-notes.md` is the **authoritative spec** for creating a new topic folder (the `/make-notes <topic>` workflow). Read it before writing or editing notes. It is the source of truth for folder structure, markdown rules, content-quality rules, and the SVG conventions below. Key points not to rediscover:

- Section files are prefixed `01_`, `02_` for ordering, but headers **inside** files carry no numbering.
- Math is GitHub-native LaTeX (`$...$` inline, `$$...$$` block).
- Notation is consistent across all notes — the canonical symbol table lives in `neural_networks/README.md` and `make-notes.md` (e.g. $W^{(\ell)}$, $a^{(\ell)}$, $z^{(\ell)}$, $\nabla_W\mathcal{L}$).
- Every section opens with a plain-English motivation before any math.

## Diagrams

Each diagram exists as three files in `diagrams/`: `<name>.tex` (TikZ source), `<name>.pdf`, and `<name>.svg`. **Markdown references the `.svg`** (`![Desc](diagrams/name.svg)`). There is no committed build script — the `.tex`→`.svg`/`.pdf` conversion is done externally, and SVGs are hand-tuned to the spec in `make-notes.md` (fixed color scheme; sub/superscripts as inline `<tspan dy=...>` offsets, never separate `<text>` elements; serif font).

## The website (`index.html`)

A dependency-free SPA: it `fetch()`es the markdown at runtime, renders with `marked`, protects/restores `$...$` around MathJax typesetting, and rewrites relative image/link paths. Because paths are relative, it must be served from the repo root.

**Critical gotcha when adding or renaming a note file:** the navigation is hardcoded in *two* places inside `index.html` and both must be updated to match:
1. The sidebar `nav-item` `<div>`s (around line 431) — `data-file` + `onclick="navigate(...)"`.
2. The JS group/card data structure (around line 481) — `{ file, tag, title, desc }` per note.

A note added to a folder but not registered in both places will exist on GitHub but be invisible/unreachable on the site.

## Workflow

- Work on the `dev` branch; open PRs against `main`.
- `.DS_Store` is gitignored.
