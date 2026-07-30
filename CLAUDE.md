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
  config.py         # dataset, label scheme, paths, model hyperparameters (no I/O)
  data.py           # load PolEmo, drop 'ambiguous', map to negative/neutral/positive
  baseline.py       # TF-IDF + logistic-regression pipeline (train/predict/save/load)
  baseline_train.py # `python -m pl_review_sense.baseline_train` — train + evaluate + save
  evaluate.py       # macro-F1, per-class metrics, confusion matrix, error extraction (pure)
  herbert.py        # HerBERT fine-tuning (Trainer) + `--smoke` CPU mode; torch imports guarded
  report.py         # standalone HTML report -> docs/ (GitHub Pages)
api/                # FastAPI /predict serving the baseline
notebooks/          # EDA + models notebook; self-contained Colab HerBERT notebook
tests/              # pytest
docs/research/      # data + methodology
```

## Methodology rules (do not violate)
- **Macro-F1 is the headline metric**, not accuracy — the classes are imbalanced.
- **Three classes**: drop PolEmo's `ambiguous`; map `minus/zero/plus` -> negative/neutral/positive.
  Document the drop; never silently merge it into another class.
- **No leakage**: fit the TF-IDF vectorizer and any encoder on **train only** (inside a Pipeline),
  evaluate on the untouched test split.
- **HerBERT trains on GPU (Colab).** The local `--smoke` run only proves the code path works; its
  numbers are non-representative and never presented as the model's result.
- **Separate I/O from logic.** `data`/`baseline`/`evaluate`/`report` stay pure and unit-tested;
  network/disk live in `*_train`, `herbert`, `api`, and the report writer.

## Conventions
- English for code, comments, README, commits. Conventional Commits.
- No hardcoded values — configurable things live in `config.py`.
- Separate I/O from logic; pure functions are unit-tested.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12). Standard core install has **no torch**;
  the transformer extra is only for the HerBERT path.

## How to run
```bash
.venv/Scripts/python -m pip install -r requirements.txt      # core + dev (no torch)
pytest
python -m pl_review_sense.baseline_train                     # downloads PolEmo, trains baseline
python -m pl_review_sense.report                             # build the static site
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
