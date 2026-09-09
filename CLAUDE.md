# CLAUDE.md — pl-review-sense

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is
A Polish-language review sentiment classifier that contrasts a classic approach
(TF-IDF + logistic regression) with fine-tuning a Polish transformer (HerBERT) on the
PolEmo 2.0 dataset, and reports when each is worth it. Portfolio project — NLP in Polish
plus modern deep learning (Hugging Face), with honest, methodical comparison.

## Architecture
```
src/pl_review_sense/
  config.py         # dataset, label scheme, paths, hyperparameters, thresholds (no I/O)
  data.py           # load PolEmo, drop 'ambiguous', map to negative/neutral/positive
  baseline.py       # TF-IDF + logistic-regression pipeline (train/predict/save/load)
  baseline_train.py # `python -m pl_review_sense.baseline_train` — train + evaluate + save
  evaluate.py       # macro-F1, per-class metrics, confusion matrix, error extraction (pure)
  stats.py          # bootstrap intervals, exact McNemar, calibration/ECE (pure)
  curves.py         # learning curve + stratified subsampling; the fit is injected (pure)
  challenge.py      # probe variants (diacritics, typos) and scoring (pure)
  challenge_set.py  # the 80 hand-written Polish probe sentences — our text, never PolEmo
  cascade.py        # risk–coverage for one model, cascade for two (pure)
  interpret.py      # heaviest coefficients per class (pure + a pipeline adapter)
  analysis.py       # `python -m pl_review_sense.analysis` — the only writer of reports/metrics/
  herbert.py        # HerBERT fine-tuning (Trainer) + `--smoke` CPU mode; torch imports guarded
  site/             # `python -m pl_review_sense.site` — Jinja template, styles.css, SVG charts
api/                # FastAPI /predict serving the baseline
notebooks/          # EDA + models notebook; self-contained Colab HerBERT notebook
tests/              # pytest
docs/               # published page, adr/, model-card.md, research/
```

## Methodology rules

The project's own commitments, each with the reason it exists — a rule whose rationale is
missing is one a later change will reasonably decide to "improve" away.

- **Macro-F1 is the headline metric**, not accuracy — the classes are imbalanced.
- **Three classes**: drop PolEmo's `ambiguous`; map `minus/zero/plus` -> negative/neutral/positive.
  Ambiguous is not a point on the negative–positive axis, so folding it into neutral corrupts
  that class. Document the drop.
- **No leakage**: fit the TF-IDF vectorizer and any encoder on **train only** (inside a Pipeline),
  evaluate on the untouched test split.
- **HerBERT trains on a GPU.** The headline run was local (GTX 1050, 4 GB, 50 min, batch split
  4×4 to fit — effective batch unchanged); `notebooks/herbert_colab.ipynb` does the same on
  Colab. The `--smoke` run proves the code path works and its numbers go to a separate file,
  because a smoke run reported as the model's result is a comparison against a model that was
  never trained.
- **I/O lives at the edges.** The modules marked `(pure)` above stay free of network and disk, so
  they are unit-tested without either; `*_train`, `analysis`, `herbert`, `api` and `site/build`
  are where the boundary is crossed.
- **Every score carries its interval; comparisons are paired.** Three decimals of macro-F1 invite
  a comparison this test set cannot support, and two models answering the same rows are compared
  with exact McNemar over their disagreements rather than by which point estimate is larger.
- **Every figure carries its n** — the site build raises `IncompleteFigure` instead of publishing
  marks whose counts are unstated.
- **The page is a function of `reports/metrics/`.** No wall clock, no dataset, no model at build
  time. Change it through `site/templates/` and rebuild; CI diffs the rebuilt page against the
  committed `docs/index.html`.
- **A panel with no evidence says so** — anything waiting on the GPU run renders as a panel naming
  what is missing and what would settle it, in place of an estimate.
- **The challenge set is ours and stays balanced.** Sentences are written for this repo, never
  PolEmo text; each cell keeps both directions of its phenomenon so no single answer can win it
  (tests enforce a 60% ceiling per cell).

## Conventions
- English for code, comments, README, commits. Conventional Commits.
- No hardcoded values — configurable things live in `config.py`.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12). Standard core install has **no torch**;
  the transformer extra is only for the HerBERT path.

## How to run
```bash
.venv/Scripts/python -m pip install -r requirements.txt      # core + dev (no torch)
pytest
python -m pl_review_sense.baseline_train                     # downloads PolEmo, trains baseline
python -m pl_review_sense.analysis                           # writes reports/metrics/*.json
python -m pl_review_sense.site                               # build the static site into docs/
uvicorn api.main:app --reload                                # POST /predict {"text": "..."}

# HerBERT: real run on Colab GPU (notebooks/herbert_colab.ipynb). Local code-path check:
.venv/Scripts/python -m pip install -e .[transformer]
python -m pl_review_sense.herbert --smoke
```

## The published page

`docs/index.html` is one of twelve surfaces held to a single specification: ten house colour tokens
with pinned per-theme values, a dark override, six card-metadata tags, a profile back-link, a
result-shaped `h1`, and — since S4 — the rule that **every figure the surface prints is a figure
a committed artifact prints**, never a rounding and never a re-derivation. The spec is
`docs/audit/0007_divergence-and-the-page-spec.md` §5 in the private portfolio index, and
`tools/pagespec` there sweeps all twelve from the submodule working trees on every push.

That checker reads HTML and CSS, so it cannot see this repository's artifacts and cannot tell an
exempt page from one nobody built tiles for. What it structurally cannot carry lives in
`tests/test_palette.py` — the other half of the carrier, and the reason **the portfolio index's**
`docs/adr/0004_what-carries-the-page-spec.md` chose one checker plus local assertions over eleven
vendored copies. *That path is in the private index, not in this repository's `docs/adr/`, which
holds `0001` and `0002` — the paragraph above already qualifies `0007` that way and this one did
not, so a reader following the argument landed on a dead path.*

## Code intelligence

Two indexes exist over this repo, and which one is reachable depends on where the session started:

- `.codegraph/` — the `codegraph_explore` MCP tool, or `codegraph explore "<question>"` from a
  shell. Returns the relevant symbols' verbatim source plus the call paths between them, so it
  usually answers a "how does X work" or "what calls Y" question in one call. The CLI ships as
  `codegraph.cmd`, so from Git Bash it needs the extension — bare `codegraph` resolves only
  where PATHEXT applies.
- `.code-review-graph/` — its MCP server is declared in **this repository's** `.mcp.json`, so it
  loads when Claude Code runs with this directory as the working directory, and is simply absent
  when the session started in the private portfolio index one level up. When its tools are
  missing the CLI still works: `uvx code-review-graph <command>`.

**Neither index has a hook**, so both are only as fresh as the last manual update — and a graph
that predates the work you are looking at will answer confidently about code that is gone.
`codegraph.cmd status` reports the index's age; `codegraph.cmd sync` brings it forward, and
`uvx code-review-graph update` does the same for the other. Check before trusting either on a
question about recent changes.

Grep, Glob and Read stay correct whenever the question is about text rather than structure, or
when neither index is available.
