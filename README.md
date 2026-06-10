# ML / DL Study Notes

Structured reference notes for machine learning and deep learning — GitHub-flavoured markdown with hand-authored SVG diagrams. Not a software project: there is no build, test, or lint step.

Notes render two ways:

- **Natively on GitHub** — markdown + inline SVG + GitHub LaTeX.
- **As a single-page site** — `index.html`, a self-contained SPA served from the repo root (GitHub Pages). It reads a generated manifest (`notes.json`) and lists every topic automatically.

## Topics

All notes live under `notes/`:

- **`notes/neural_networks/`** — fundamentals through modern architectures (feedforward, training, embeddings, RNN/LSTM, CNN, transformers, agents/LLMs).
- **`notes/causal_methods/`** — causal inference and estimation methods.

Each topic is a folder of ordered section files (`01_*.md`, `02_*.md`, …) plus a `diagrams/` folder of SVGs. `ml_dl_topics.md` is the master outline of what gets written.

## Working in this repo

- Notes are authored via the **`make-notes` skill** (`.claude/skills/make-notes/SKILL.md`) — the source of truth for note structure, diagram style, and publishing.
- Work on the **`dev`** branch; open PRs against **`main`**.
- `notes.json` is **generated** — never hand-edit it or `index.html`. Run `python3 scripts/build_manifest.py`, or push to `main` and a GitHub Action rebuilds it.
- `.nojekyll` at the repo root must stay, or GitHub Pages won't serve the underscore-named topic folders.
