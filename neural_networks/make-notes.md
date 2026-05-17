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

File names use `snake_case`. Section files are prefixed `01_`, `02_` etc for ordering but the markdown headers inside contain **no numbering** — just the plain title.

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

1. **Use proper mathematical notation** — ALL sub/superscripts must be inline `<tspan>` offsets within the parent `<text>` element. **Never use separate `<text>` elements** for subscripts or superscripts; manually-placed coordinates misalign due to font rendering variance.

   **Subscript pattern** (shift down, then reset):
   ```xml
   W<tspan dy="5" font-size="9">ij</tspan><tspan dy="-5" font-size="12"> continues here</tspan>
   ```

   **Superscript pattern** (shift up, then reset):
   ```xml
   z<tspan dy="-6" font-size="9">(ℓ)</tspan><tspan dy="6" font-size="12"> continues here</tspan>
   ```

   - Use **absolute pixel offsets** (not `em`): `dy="5"` for subscripts, `dy="-6"` for superscripts
   - Shifted glyph size: `font-size="9"` (use `8` for very small); always reset to parent size after
   - Multi-character scripts go in one tspan: `<tspan dy="5" font-size="9">t−1</tspan>`
   - Math symbols — use Unicode directly in text content: `∂ ∑ ∈ ℝ ⊙ ⊕ σ μ ε γ β ∇ ≈ → ×`

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

---

## README.md Structure

```markdown
# <Topic> — Notes Index

> One-line description of what these notes cover.

## Contents

| # | Topic | File |
|---|-------|------|
| 1 | Section name | [file.md](file.md) |
...

## Notation

| Symbol | Meaning |
|--------|---------|
...
```

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
