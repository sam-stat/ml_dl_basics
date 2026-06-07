Create structured, GitHub-hostable reference notes for a technical topic.

## How to Use

Invoked as: `/make-notes <topic>`

If no topic is provided, ask the user: "What topic should I create notes for, and what sections do you want covered?"

---

## What to Build

Create a self-contained folder `<topic_slug>/` containing:

```
<topic_slug>/
  README.md              ← index with TOC and notation table
  01_<section>.md
  02_<section>.md
  ...
  diagrams/
    <concept>.svg
    ...
```

File names use `snake_case`. Section files are prefixed `01_`, `02_` etc — the prefix is **only** for ordering (it drives both file order and the website). The markdown content carries **no numbering**: the `# H1` is the plain title (`# Word Embeddings`, never `# 3. Word Embeddings`), and no `## 1.1`-style numbered subheadings.

> **The website updates itself.** The site (`index.html`) is driven by a generated manifest (`notes.json`), built by `scripts/build_manifest.py` scanning the note folders. A new folder of `NN_*.md` files appears automatically — **never hand-edit `index.html`**. Regenerate locally with `python3 scripts/build_manifest.py`; on push to `main` a GitHub Action regenerates it for you.

---

## Planning Step

Before writing files, think through:
1. What are the logical sections for this topic? (aim for 5–10 files)
2. What diagrams are genuinely needed? (only create diagrams that add clarity — not decorative ones)
3. What is the target audience? (default: technical colleague familiar with the field but not the specific topic)

If the user specifies sections in the invocation, use those. Otherwise infer sensible sections from the topic.

---

## Markdown Formatting Rules

### Headers
- `# Title` — top of each file (matches the section name)
- `## Section` — major sections within the file
- `### Subsection` — as needed
- **No numbered prefixes** inside files — never write `## 1.1 Title`, always `## Title`
- Use `---` horizontal rules to separate major sections

### Mathematics
Use GitHub-native LaTeX:
- Inline: `$...$`
- Block: `$$...$$`

Write clean, complete notation — never abbreviate in a way that loses meaning:
- Layer index: $W^{(\ell)}$, $a^{(\ell)}$, $b^{(\ell)}$, $z^{(\ell)}$
- Gradients: $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$
- Norms: $\|x\|_2$, $\|W\|_F$
- Distributions: $\mathcal{N}(\mu, \sigma^2)$, $\mathcal{U}(a, b)$

### Tables
Use them for comparisons, notation summaries, and option lists. Keep columns tight.

### Code
Use fenced blocks with language tags for any code, pseudocode, or architecture sketches:
```python
# example
```

### Callout-style notes
Use blockquotes for key insights:
> **Key idea:** write the single most important thing to remember here.

---

## SVG Diagram Rules

Every SVG must:

1. **Use proper mathematical notation** — ALL sub/superscripts must be inline `<tspan>`s within the parent `<text>` element. **Never use separate `<text>` elements** for them, and **never rely on Unicode sub/superscript glyphs** (`₁ ⁽¹⁾`) — serif fonts render those at full height. Use `baseline-shift` with a smaller `font-size`; it positions correctly and auto-resets at the end of the tspan (no manual reset needed):

   **Subscript:**
   ```xml
   W<tspan baseline-shift="sub" font-size="8">ij</tspan> continues here
   ```

   **Superscript:**
   ```xml
   z<tspan baseline-shift="super" font-size="9">(ℓ)</tspan> continues here
   ```

   - Shifted glyph size: `font-size="8"`–`9"` (parent text usually `11`–`13`)
   - Multi-character scripts go in one tspan: `<tspan baseline-shift="sub" font-size="8">t−1</tspan>`
   - Math symbols — use Unicode directly in text content: `∂ ∑ ∈ ℝ ⊙ ⊕ σ μ ε γ β ∇ ≈ → × ̃` (combining tilde for c̃)

2. **Use serif font** — `font-family="Georgia, 'Times New Roman', serif"` — for mathematical feel

3. **Include a formula annotation** at the bottom of architecture diagrams — the key equation the diagram is illustrating

4. **Color scheme** (consistent across all diagrams):
   - Input / data: `#eef4fb` fill, `#4f86c6` stroke (blue)
   - Processing / hidden: `#edfaed` fill, `#5cb85c` stroke (green)
   - Output / loss: `#fdeaea` fill, `#d9534f` stroke (red)
   - Operations (gates, attention): `#fff3cd` fill, `#f0ad4e` stroke (amber)
   - Special (norm, residual): `#f9f0ff` fill, `#9b59b6` stroke (purple)
   - Background: `#fff`, no outer border fill

5. **Arrow markers** — define a standard `<marker>` arrowhead at the top of each SVG

6. **White background**, clean — no grid lines, no drop shadows

7. **Always** add `viewBox` and explicit `width`/`height`

8. Reference diagrams in markdown as: `![Description](diagrams/filename.svg)`

9. **Hand-drawn finish ("Style A").** Diagrams should look lightly hand-inked while staying crisp. Add one roughen filter and apply it to the *shapes only* (rectangles, lines, paths, circles) — never to `<text>`, so labels stay sharp and legible:

   ```xml
   <filter id="rough" x="-5%" y="-5%" width="110%" height="110%">
     <feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="2" seed="4" result="n"/>
     <feDisplacementMap in="SourceGraphic" in2="n" scale="1.4" xChannelSelector="R" yChannelSelector="G"/>
   </filter>
   ```

   Wrap the shapes in `<g filter="url(#rough)"> … </g>` and keep all `<text>` outside that group. Keep the serif font and the colour scheme above — the wobble alone gives the hand-drawn feel, needs no external fonts, and renders identically on GitHub, in the SPA, and standalone. **`scale="1.4"` is the house setting** — gentle waver, boxes stay clean; higher (2+) makes rectangle edges look jittery and can break thin strokes. `baseFrequency` lower = longer, gentler waves.

   Author diagrams **by hand** in this clean `<rect>`/`<line>`/`<text>` form — not as TikZ/`pdf2svg` output, whose text becomes vector glyph paths that can't be restyled or roughened cleanly.

---

## README.md Structure

```markdown
# <Topic> — Notes Index

> One-line description of what these notes cover.

## Contents

| Topic | File |
|-------|------|
| Section name | [file.md](file.md) |
...

## Notation

| Symbol | Meaning |
|--------|---------|
...
```

The Contents table carries **no leading number column** — order comes from the file prefixes, not from visible numbering.

---

## Content Quality Rules

- Every section starts with a **plain-English motivation** sentence before any math — explain *why* this concept exists, not just what it is
- Every formula is introduced, not just dropped — explain what each term means
- Tables for comparisons (e.g. which technique to use when)
- Concrete examples where the concept is non-obvious
- No filler phrases ("In this section we will...") — start directly with content
- Concise but complete — a colleague should be able to read and understand without additional resources

---

## Execution Order

1. Confirm topic and section list with the user (or infer and proceed if the topic is clear)
2. Create the directory structure
3. Write all SVG diagrams first (they are referenced by the markdown)
4. Write README.md
5. Write each section file in order
6. Verify all diagram references resolve
7. Run `python3 scripts/build_manifest.py` to register the new folder on the website (or just push to `main` — the GitHub Action does it). **Do not edit `index.html`.**
