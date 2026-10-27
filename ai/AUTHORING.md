# Authoring guide — AI from Scratch

The reference implementation of everything below is **lesson 01**:
`lessons/01-vectors.html` + `workbooks/01_vectors/`. Read both fully before writing.
Copy its structure; match its tone, density and quality bar.

## Learner
A strong software engineer with rusty undergrad math. Wants *deep understanding*, not
recipes. So: intuition first → picture → precise definition/derivation → worked example
→ code. Derive things instead of asserting them, at least whenever the derivation is
short. Always say *why it matters for AI* and link forward/back to other lessons by
number ("you'll use this in lesson 22").

## Lesson page (`lessons/NN-slug.html`; NN/slug/title come from `assets/manifest.js`)
- Head: identical `<head>` block to lesson 01 (KaTeX css+js, auto-render, manifest.js,
  course.js — all synchronous, in that order). `<body data-lesson="NN">`, content in `<main>`.
- Top: `.kicker` (Part N · name), `h1`, `.lede`, `.objectives`, `<nav><ul class="toc"></ul></nav>`
  (auto-filled from every `h2[id]`).
- Body: 6–10 numbered `h2` sections with ids. Prose is plain, short sentences, no fluff.
- **Visual-first**: at least 3 interactive figures per lesson (math parts: 4+). Use
  draggable handles, sliders, step/play buttons, live readouts. Each figure has a
  caption beginning "**Try it:**" telling the learner what to do and what to notice.
  Figures must teach something, not decorate.
- Callouts: `.callout.intuition`, `.callout.key`, `.callout.warning`, `.callout.ml`
  ("Why this matters for AI").
- `.example` worked examples (≥2), with every step shown.
- `Problems` section: 6–10 `.problem` blocks mixing computation, derivation/proof and
  conceptual questions; difficulty tag (warm-up / core / stretch). Every problem
  has `details.solution`; harder ones also get `details.hint`. Use `.numcheck`
  (`data-answer="x"` or `"x,y,z"`, optional `data-tol`) for numeric answers —
  **compute every answer with Python before writing it in**.
- `Quick check`: 3–5 `.quiz` blocks (`data-answer="b"`, buttons `data-k`, `.explain`).
  Wrong options should be plausible misconceptions.
- `Summary` table, then `.workbook` box describing the workbook and `./course check NN`,
  then optional "Go deeper" references (classic, well-known sources only).
- Math: `$...$` inline, `$$...$$` display (KaTeX). Escape `<` in HTML as `&lt;` — or
  write `\lt` in math. A literal dollar sign in prose must be avoided (it opens math).
- Lesson JS goes in one `<script>` at the end of `<body>` wrapped in
  `Course.ready(() => { ... })`. Use a `{ ... }` block per figure. No external
  libraries, no network, no fetch() (pages are opened via file://).

### `course.js` API (read assets/course.js for details)
- `new Plot(selectorOrCanvas, {xmin,xmax,ymin,ymax,equal})` — world-coordinate canvas
  (`<canvas id=... width=680 height=420>` inside `.figure`).
  Draw: `grid() axes({xlabel,ylabel,step}) line ray arrow point label text path polygon
  fn(f,{color}) circle rect field(f,{cmap}) contour(f,levels) heatmap(M,{x0,x1,y0,y1,
  cmap:'seq'|'div'|'viridis',values,rowLabels,colLabels}) bars(vals,{labels,values})`.
  Interaction: `p.draggable([{x,y,color,snap,constrain}], onDrag).render(p => {...})`,
  `p.drawHandles()`, `p.onClick = (x,y) => ...`, `p.render()` to redraw, `p.setView(...)`.
  Colors: `"c1".."c6"`, `"muted"`, `"fg"`, `"accent"`, `"ok"`, `"bad"` (theme-aware).
- Controls: `Course.slider(container, {label,min,max,step,value,digits}, onInput)`,
  `Course.button(container, label, onClick)`, `Course.select(container, label, [[v,t]], onChange)`,
  `Course.animator(stepFn)` → `{start, stop, running}` (stepFn returns false to stop).
- Math: `Course.rng(seed)` → `{uniform, normal, int, choice}`; `LA.dot norm add sub scale
  matvec matmul T det2 softmax eigSym2`; `Course.fmt(v, digits)`.
- Need something the API lacks? Write a local helper inside your lesson script. Do NOT
  edit shared files (`assets/*`, `course`, `index.html`, `AUTHORING.md`).
- Figures must work in both light and dark mode (only use the named colors).

## Workbook (`workbooks/NN_slug_with_underscores/`)
- `exercises.py` — module docstring with "rules of the game"; functions/classes with
  precise docstrings, a `Hint:` where useful, and body `raise NotImplementedError`.
  Group into parts (A: from scratch, B: vectorized/applied, C: stretch).
- `solutions.py` — same API, reference implementation, clean and commented where subtle.
- `test_exercises.py` — starts with `from _impl import load; ex = load(__file__)`;
  pytest tests that check math properties (not just one magic number), edge cases,
  shapes, and numerical gradients where relevant. Seed all randomness.
  **Every test must fail on the untouched stubs**, and the full file must run in < 30 s
  on a laptop CPU (< 60 s for torch-heavy ones).
- Extra provided (non-exercise) code, e.g. data loaders or a training script, goes in
  `helpers.py` / `train.py` in the same folder; tests may import it.
- Datasets: generate synthetically in code (seeded), or use `data/tinyshakespeare.txt`
  (path relative to repo root: `os.path.join(os.path.dirname(__file__), "..", "..", "data", ...)`).
- Through Part 4 use numpy only (Part 1–3: plain Python first, then numpy). PyTorch from
  lesson 19 on. No other third-party packages (matplotlib is optional and never
  required by tests).

## Verification (required before you report done)
From the repo root:
1. `./course validate NN` for each of your workbooks → "ok" (solutions pass, stubs fail,
   0 tests passing on stubs).
2. `./course pages NN` for each lesson → "ok" (no JS errors, no KaTeX errors).
3. Visual check: screenshot and look at it.
   ```sh
   C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
   "$C" --headless=new --disable-gpu --hide-scrollbars --window-size=1100,6000 \
     --virtual-time-budget=3000 --screenshot=/tmp/aic-NN.png "file://$PWD/lessons/NN-slug.html"
   sips --cropToHeightWidth 1500 1100 --cropOffset 1500 0 /tmp/aic-NN.png --out /tmp/aic-NN-b.png; sips -Z 1000 /tmp/aic-NN-b.png
   ```
   then Read the cropped PNGs (keep them ≤1000px; crop sections rather than reading one
   huge image). Check that figures render sensibly with readable labels and nothing
   overlapping or off-canvas.
