# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **study-notes repository**, not a software project. The deliverables are GitHub-flavoured markdown notes with hand-authored SVG diagrams, sourced from the *Short Notes (All Topics)* material (`.pdf`/`.xlsx`). There is no build, test, or lint step — content is rendered two ways:

- **Natively on GitHub** (markdown + inline SVG + GitHub LaTeX).
- **As a single-page site** via `index.html` (a self-contained SPA served at the repo root).

`ml_dl_topics.md` is the master topic outline that drives what notes get written.

## Repository layout

- `neural_networks/` — the flagship topic folder. Prefixed section files (`01_*.md` … `08_*.md`) for ordering, a `README.md` index, and `diagrams/`.
- `causal_methods/` — a second topic folder in the same structured note format (`01_*.md` … `08_*.md` + `diagrams/`), wired into the site via the manifest. `helper_docs/` holds raw chat exports (mining material, not published notes).
- `index.html` — the GitHub Pages SPA. `.nojekyll` is present so Pages serves the underscore-named folders and raw files untouched.
- `notes.json` — **generated** manifest the SPA reads (do not hand-edit). `scripts/build_manifest.py` builds it; a GitHub Action regenerates it on push to `main`.

## Authoring notes — `/make-notes`

`neural_networks/make-notes.md` is the **authoritative spec** for creating a new topic folder (the `/make-notes <topic>` workflow). Read it before writing or editing notes. It is the source of truth for folder structure, markdown rules, content-quality rules, and the SVG conventions below. Key points not to rediscover:

- Section files are prefixed `01_`, `02_` for ordering, but content carries **no numbering** — the `# H1` is the plain title (no `# 3.` prefix), and no `## 1.1`-style subheadings.
- Math is GitHub-native LaTeX (`$...$` inline, `$$...$$` block).
- Notation is consistent across all notes — the canonical symbol table lives in `neural_networks/README.md` and `make-notes.md` (e.g. $W^{(\ell)}$, $a^{(\ell)}$, $z^{(\ell)}$, $\nabla_W\mathcal{L}$).
- Every section opens with a plain-English motivation before any math.

## Diagrams

Diagrams are **hand-authored SVG** in a clean `<rect>`/`<line>`/`<text>` form, `.svg` only — the old TikZ `.tex`→`pdf2svg` pipeline has been retired (its text became un-restylable glyph paths). Spec in `make-notes.md`: fixed colour scheme; serif font; sub/superscripts as inline `<tspan baseline-shift="sub|super">` (not Unicode glyphs, which serif fonts don't shrink); and the "Style A" hand-drawn finish — one `feTurbulence`+`feDisplacementMap` roughen filter (house setting `scale="1.4"`) applied to shapes only, text stays crisp. Preview an SVG locally with `qlmanage -t -s 1100 -o /tmp <file>.svg` (it square-crops wide diagrams — crop the `viewBox` to inspect a right edge).

## The website (`index.html`)

A dependency-free SPA: it loads `notes.json`, builds the sidebar + home cards from it, then `fetch()`es each markdown file at runtime, renders with `marked`, protects/restores `$...$` around MathJax typesetting, and rewrites relative image/link paths. Because paths are relative, it must be served from the repo root.

**The navigation is data-driven — never hand-edit `index.html` to add a note.** Adding a folder of `NN_*.md` files and regenerating `notes.json` (via `scripts/build_manifest.py`, or automatically by the GitHub Action on push to `main`) is all it takes for a note to appear in both the sidebar and the home grid. Per-note title comes from the `# H1`; the card blurb is the first prose paragraph; group order is the priority list in the generator, then alphabetical.

## Workflow

- Work on the `dev` branch; open PRs against `main`.
- `.DS_Store` is gitignored.
