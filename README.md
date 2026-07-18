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
- Fine-tunes **HerBERT** (`allegro/herbert-base-cased`) on a **Colab GPU** and compares.
- Reports **macro-F1** (headline, because the classes are imbalanced), a confusion matrix, and
  an error analysis of what each model gets wrong.
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

| model | accuracy | macro-F1 |
|-------|:--------:|:--------:|
| TF-IDF + logistic regression | 0.940 | **0.944** |
| HerBERT (fine-tuned) | *pending — run `notebooks/herbert_colab.ipynb` on a GPU* | |

Per-class baseline F1: negative 0.95, neutral 0.97, positive 0.92. The TF-IDF baseline is
already strong on this text-level task — a good reminder that a transformer is not free.

## Project structure

```
src/pl_review_sense/
  config.py  data.py  baseline.py  baseline_train.py  evaluate.py  herbert.py  report.py
api/                 # FastAPI /predict (baseline)
notebooks/           # 01_eda_and_models.ipynb + herbert_colab.ipynb (GPU)
reports/metrics/     # committed numbers (baseline.json; herbert.json after the Colab run)
docs/                # GitHub Pages report
tests/               # pytest
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # core + dev (no torch)
pytest

python -m pl_review_sense.baseline_train    # downloads PolEmo, trains + evaluates the baseline
python -m pl_review_sense.report            # build the static site into docs/
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
- **Three classes**: `ambiguous` is dropped (documented), never merged into another class.
- **No leakage**: the TF-IDF vectorizer is fit inside a `Pipeline` on the training fold only.
- **Validation held out**: hyperparameters are fixed config defaults — the `validation` split is
  intentionally not tuned against, so the test metrics stay an honest estimate.
- **Pinned dataset**: PolEmo is loaded at a fixed revision, so `trust_remote_code` runs an
  audited script rather than whatever upstream HEAD happens to be.
- **Honest compute cost**: the baseline trains in seconds on a CPU; HerBERT needs a GPU and
  minutes per epoch — reported alongside the accuracy so the trade-off is explicit.
- **Error analysis**: negation and sarcasm (double negatives, contrastive clauses) are where a
  bag-of-words model slips; the notebook surfaces concrete misclassified reviews locally.

## Limitations

- **PolEmo, traditional text-level reviews** — a single Polish dataset; not a general sentiment API.
- **HerBERT numbers come from Colab** (no GPU in CI); until that run they show as *pending*. The
  local `--smoke` run only proves the training code works and is never reported as the result.
- Non-commercial data license (CC BY-NC-SA) — fine for this portfolio demo.

## License

MIT for the code — see [LICENSE](LICENSE). PolEmo 2.0 © CLARIN-PL, CC BY-NC-SA 4.0.
