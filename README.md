# pl-review-sense

[![CI](https://github.com/P0w3r223/pl-review-sense/actions/workflows/ci.yml/badge.svg)](https://github.com/P0w3r223/pl-review-sense/actions/workflows/ci.yml)

**Polish-language review sentiment (3-class) — a classic TF-IDF baseline vs fine-tuning the
HerBERT transformer**, on PolEmo 2.0, with an honest comparison of accuracy *and* cost.

> Portfolio project A4. Demonstrates NLP in Polish and modern deep learning (Hugging Face),
> plus methodological maturity: when a simple model is already enough, and when a transformer
> earns its compute.

## What it does

- Loads **PolEmo 2.0**, drops the `ambiguous` class, and maps the rest to
  **negative / neutral / positive**.
- Trains a **TF-IDF + logistic-regression** baseline (fast, interpretable, CPU-only).
- Fine-tunes **HerBERT** (`allegro/herbert-base-cased`) on a **Colab GPU** and compares — as a
  **paired McNemar test** on the reviews the two models disagree on, not as two scores side by
  side.
- Puts a **bootstrap interval** around every score, so the third decimal is not read as a
  finding.
- Plots a **learning curve**: how much of the score is the corpus rather than the model.
- Scores an **80-sentence Polish challenge set** written for this project — negation, irony,
  contrastive pivots, and a control cell — plus variants with diacritics stripped and typos
  introduced.
- Measures **calibration and deferral**: what the baseline scores on the reviews it keeps when
  the least confident share is handed on.
- Extracts the **heaviest coefficients per class** — the model's own reasons.
- Serves the baseline via a **FastAPI `/predict`** endpoint.

## Dataset & license

**PolEmo 2.0** (CLARIN-PL) — online reviews from four domains (hotels, medicine, products,
school). Loaded via Hugging Face `datasets` (a dataset script → needs `datasets < 3`). The data
is **CC BY-NC-SA 4.0 (non-commercial)** and is **not redistributed here** — it is downloaded on
demand and cached out of git; only aggregate metrics (numbers) are committed.

## Live report

**<https://p0w3r223.github.io/pl-review-sense/>** — comparison table, confusion matrix, and
methodology (built locally from committed metrics; the data/models stay out of git/CI).

## Results (test set)

| model | accuracy | macro-F1 | 95% interval |
|-------|:--------:|:--------:|:------------:|
| TF-IDF + logistic regression | 0.940 | **0.944** | 0.926–0.961 |
| HerBERT (fine-tuned) | *pending — run `notebooks/herbert_colab.ipynb` on a GPU* | | |

Per-class baseline F1: negative 0.95, neutral 0.97, positive 0.92.

**And the finding the corpus score hides.** The same model answers **48 of 80** one-sentence
Polish reviews written for this project — 13 of 20 in a control cell of unambiguous sentiment,
11 of 20 on irony. A macro-F1 of 0.944 describes the corpus it was trained for, not the task
its name suggests. The published page leads with that, because it reframes the project's
question: a transformer is worth its compute where the text stops looking like the training
corpus, and the cheapest way to find out is to look.

Two more results worth the click: **1 200 labelled reviews** (23% of the corpus) already land
within 0.02 macro-F1 of what all 5 264 produce, and setting aside the least confident **10%**
of reviews lifts macro-F1 on the rest from 0.944 to **0.974** — even though the model's
confidence is badly calibrated (ECE 0.209) and must be used as a ranking rather than as a
probability.

## Project structure

```
src/pl_review_sense/
  config.py  data.py  baseline.py  baseline_train.py  evaluate.py  herbert.py
  stats.py       # bootstrap intervals, McNemar, calibration
  curves.py      # learning curve, stratified subsampling
  challenge.py   # the probe: variants and scoring   challenge_set.py  # the 80 sentences
  cascade.py     # risk–coverage and the two-model cascade
  interpret.py   # heaviest coefficients per class
  analysis.py    # the runner that writes reports/metrics/
  site/          # Jinja template, shared stylesheet, SVG charts, page build
api/                 # FastAPI /predict (baseline)
notebooks/           # 01_eda_and_models.ipynb + herbert_colab.ipynb (GPU)
reports/metrics/     # committed numbers — one file per question the page asks
docs/                # GitHub Pages report, ADRs, model card
tests/               # pytest
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # core + dev (no torch)
pytest

python -m pl_review_sense.baseline_train    # downloads PolEmo, trains + evaluates the baseline
python -m pl_review_sense.analysis          # intervals, curve, probe, deferral, coefficients
python -m pl_review_sense.site              # build the static site into docs/
uvicorn api.main:app --reload               # POST /predict {"text": "Świetny produkt!"}
```

**HerBERT** (needs a GPU — use `notebooks/herbert_colab.ipynb` on Colab):

```bash
pip install -e .[transformer]               # heavy: torch, transformers, sentencepiece, ...
python -m pl_review_sense.herbert --smoke   # tiny CPU run that only validates the training code
```

## Methodology highlights

- **Macro-F1**, not accuracy — the classes are imbalanced (predicting the majority scores high
  accuracy but low macro-F1).
- **Every score carries an interval.** 2 000-resample percentile bootstrap over the test rows;
  a comparison narrower than the interval is one this test set cannot make.
- **Models are compared pairwise.** Both answer the same 684 reviews, so the evidence is an
  exact McNemar test over the disagreements — which can separate models whose intervals
  overlap, and refuse to separate models whose intervals do not.
- **Three classes**: `ambiguous` is dropped (documented), never merged into another class.
- **No leakage**: the TF-IDF vectorizer is fit inside a `Pipeline` on the training fold only.
- **Validation held out**: hyperparameters are fixed config defaults — the `validation` split is
  intentionally not tuned against, so the test metrics stay an honest estimate.
- **Pinned dataset**: PolEmo is loaded at a fixed revision, so `trust_remote_code` runs an
  audited script rather than whatever upstream HEAD happens to be.
- **Honest compute cost**: measured, not asserted — training seconds, model size and throughput
  are recorded with the machine they were measured on.
- **The probe is balanced against the class prior.** Every ironic sentence has a genuine
  counterpart in the same register, so a cell cannot be won by always answering *negative*
  ([ADR 0002](docs/adr/0002_challenge-set-design.md)).
- **The page cannot drift from the numbers.** It is generated from `reports/metrics/` and CI
  fails if the committed HTML is not what the code produces
  ([ADR 0001](docs/adr/0001_page-is-a-function-of-committed-metrics.md)).

## Limitations

- **PolEmo, traditional text-level reviews** — a single Polish dataset; not a general sentiment
  API, and the probe above is the evidence for that rather than a caveat about it.
- **The probe is a diagnostic, not a benchmark.** Eighty sentences show that a gap exists; they
  do not size it.
- **HerBERT numbers come from Colab** (no GPU in CI); until that run the comparison and the
  cascade render as panels stating what is missing. The local `--smoke` run only proves the
  training code works and is never reported as the result.
- **Confidence is not a probability here** — ECE 0.209, under-confident. It ranks well enough
  for a deferral threshold, which is a different claim.
- Non-commercial data license (CC BY-NC-SA) — fine for this portfolio demo.

## Decisions

- [ADR 0001 — the published page is a function of the committed metrics](docs/adr/0001_page-is-a-function-of-committed-metrics.md)
- [ADR 0002 — measuring the phenomena instead of asserting them](docs/adr/0002_challenge-set-design.md)
- [Model card](docs/model-card.md) — what the baseline is for, and what it is not for

## License

MIT for the code — see [LICENSE](LICENSE). PolEmo 2.0 © CLARIN-PL, CC BY-NC-SA 4.0.
