# Data & methodology — pl-review-sense

Date: 2026-07-18
Status: accepted
Author: P0w3r223 + Claude

---

## Dataset

**PolEmo 2.0** (CLARIN-PL) — a Polish sentiment corpus of online reviews from four domains:
hotels, medicine, products, and school. We use the `all_text` configuration (full reviews,
all domains) via Hugging Face `datasets`. The loader is a dataset *script*, so `datasets < 3`
is required (`trust_remote_code=True`).

**License: CC BY-NC-SA 4.0 (non-commercial).** The corpus is **not redistributed** in this
repository — it is downloaded on demand and cached outside git. Only aggregate numbers
(metrics, confusion matrix) are committed; no review text is stored here.

## Label scheme (3-class)

PolEmo's `ClassLabel` names are `zero` (neutral), `minus` (negative), `plus` (positive), and
`amb` (ambiguous). We keep **three** classes and **drop `ambiguous`**, mapping the rest to an
ordinal scheme:

| PolEmo name | our label | index |
|-------------|-----------|:-----:|
| `minus` | negative | 0 |
| `zero`  | neutral  | 1 |
| `plus`  | positive | 2 |

Dropping `ambiguous` is a deliberate, documented choice: it is not a point on the
negative–positive axis, and folding it into "neutral" would corrupt that class. After the drop
the splits are roughly train ≈ 5.3k / val ≈ 0.7k / test ≈ 0.7k, and the classes are
**imbalanced** (negative is the largest).

## Metric

**Macro-F1 is the headline metric**, not accuracy. With imbalanced classes, always predicting
the majority label yields deceptively high accuracy but low macro-F1; macro-F1 weights each
class equally, so a model must do well on the minority classes too. Logistic regression uses
`class_weight="balanced"` for the same reason.

## Models

- **Baseline** — TF-IDF (word 1–2-grams, `min_df=2`, sublinear TF) → multinomial logistic
  regression, inside a single `Pipeline` so the vectorizer is fit on the training fold only
  (no leakage). Cheap, interpretable, CPU-only; trains in seconds.
- **HerBERT** — `allegro/herbert-base-cased` fine-tuned for sequence classification
  (`num_labels=3`). Trained on a **free Colab GPU** (`notebooks/herbert_colab.ipynb`); the
  local `--smoke` mode fine-tunes on a tiny subset for one epoch only to validate the code path,
  and its numbers are never reported as the model's result.

## Honest compute cost

The baseline is essentially free (seconds, CPU, a few MB model). HerBERT needs a GPU and
minutes per epoch, and a ~0.5 GB checkpoint. On this text-level task the TF-IDF baseline is
already strong (macro-F1 ≈ 0.94), so the comparison is as much about **cost vs marginal gain**
as about raw accuracy — which is the point of the project.

## Error analysis

Misclassifications are inspected locally (in the notebook, from data that is not committed).
The lens: **negation** cues (`nie`, `bez`, `brak`, …) and **sarcasm / contrastive clauses**
(double negatives, "świetnie, znowu się zepsuło") — phenomena a bag-of-words model cannot
represent but a contextual transformer can. The committed report shows aggregate stats plus
*illustrative* sentences written by us (not PolEmo excerpts).
