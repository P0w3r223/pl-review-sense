# CLAUDE.md — pl-review-sense

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is
A Polish-language review sentiment classifier that contrasts a classic approach
(TF-IDF + logistic regression) with fine-tuning a Polish transformer (HerBERT) on the
PolEmo 2.0 dataset, and reports when each is worth it. Portfolio project A4 — NLP in Polish
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

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. No hooks installed — run `code-review-graph update` after code changes.
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
