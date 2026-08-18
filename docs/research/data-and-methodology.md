# Data & methodology — pl-review-sense

Date: 2026-07-18
Status: accepted
Author: P0w3r223

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
  (`num_labels=3`). Trained on a GPU — this project's run took 50 minutes on a 4 GB GTX 1050,
  with the batch split 4×4 so the optimizer still steps on the configured 16 examples;
  `notebooks/herbert_colab.ipynb` does the same on a free Colab card. The
  local `--smoke` mode fine-tunes on a tiny subset for one epoch only to validate the code path,
  and its numbers are never reported as the model's result.

## Honest compute cost

The baseline is essentially free (about 3 seconds on a CPU, a 3.2 MB model, thousands of
reviews a second). HerBERT took **50 minutes on a 4 GB GTX 1050** for four epochs over the same
5 264 reviews — roughly 850× the training time — and it wants that card to serve as well as to
train.

So the comparison is not "is the transformer better" (it is: 0.986 against 0.944, four times
fewer errors, *p* < 0.0001) but **what the difference costs and whether all of it must be
paid**. The cascade section is the answer to the second half: most of the gain is available for
a fifth of the GPU traffic.

## Uncertainty and comparison

A macro-F1 on 684 test rows is a point estimate, and the third decimal is not a finding. Two
tools, answering two different questions:

- **How tightly is one model pinned down** — a 2 000-resample percentile bootstrap over the
  test rows, seeded, so the interval is reproducible from the committed predictions. The
  baseline lands at 0.944 with a 95% interval of 0.926–0.961.
- **Are two models different** — an **exact McNemar test** over the reviews the two models
  answer differently. Both see the same rows, so this is the question overlapping intervals
  cannot settle: a model right on every review the other misses is distinguishable long before
  the intervals come apart. The exact binomial form rather than the chi-square approximation,
  which is anticonservative at the discordant counts a 684-row test set produces.

  The measured comparison: baseline 0.944 [0.926, 0.961] against HerBERT 0.986 [0.976, 0.994],
  with HerBERT right on 38 reviews the baseline misses and wrong on 7 it gets — 45 disagreements,
  *p* < 0.0001. Forty-one errors become ten.

## Learning curve

The same pipeline refit from scratch on **stratified** subsamples (150 → 4 800, five seeds
each) and scored on the untouched test split. Stratified because an unstratified draw of 150
rows from an imbalanced corpus can miss the neutral class outright, which would measure the
draw rather than the size; five seeds because one subsample per size draws a curve whose bumps
are sampling noise. The band on the published chart is the spread across seeds.

1 200 labelled reviews — 23% of the corpus — already land within 0.02 macro-F1 of what all
5 264 produce. That is the "should we fine-tune a transformer" question asked where the answer
is cheap to test.

## Challenge set

80 sentences written for this project (never PolEmo text), in four cells of twenty: `plain`
(control), `negation`, `sarcasm`, `contrast`, plus derived variants with diacritics stripped
and a typo introduced. Cells are balanced so that no single answer wins them — see
[ADR 0002](../adr/0002_challenge-set-design.md) for the design and its rationale.

Result: **48 of 80**, including 13 of 20 on the control cell. The corpus score does not carry
over to sentence-length input, which is the more useful finding about this model than the
score itself.

## Putting the headline at risk

The claim that the corpus score does not carry over to short reviews rests on sentences we
wrote, so the corpus is asked the same question independently: the test predictions are cut by
review length, into word-count buckets, with macro-F1 inside a bucket averaged over **only the
classes that occur in it** (a bucket with no neutral reviews would otherwise take a hard zero
for that class and report its own composition as the model's failure).

The answer is that PolEmo cannot settle it, and the page says so. The median test review is 119
words and just 3 of 684 are as short as the probe's sentences; across the lengths the corpus
does cover — buckets of at least 50 reviews — length moves macro-F1 by 0.011, which is inside
the interval around the score itself. The 25–49-word bucket is the reason the floor is 50 and
not 30: thirty reviews there produce 90% accuracy and 0.62 macro-F1 at the same time, because
one class contributed a handful of rows.

So the probe tests a regime the corpus barely contains — which is why it had to be written
rather than sampled — and the defensible claim is narrower than "short text breaks it".

## Reference floors

A macro-F1 quoted alone has no scale. Two floors are computed on the same test rows from the
training prior: **always the majority class** (0.221 macro-F1 at 0.496 accuracy) and **random,
matching the prior** (0.317 at 0.367). The first is the argument for the headline metric stated
as a measurement — a model that has learned nothing is right about half the time.

## Calibration and deferral

The model's confidence is **not** a probability: expected calibration error 0.209, with bins
around 0.75 confidence answered correctly every time — under-confident, which is the direction
that makes a naive "escalate anything below 0.7" rule waste the expensive model on reviews the
cheap one gets right.

It does, however, **rank** well, which is what a deferral rule needs. Setting aside the least
confident 10% raises macro-F1 on what remains to 0.974. The operating threshold is read off
that curve rather than from the confidence number.

The two-model **cascade** was left unestimated until a GPU run existed, because it depends on
how HerBERT answers those specific reviews. It now exists: routing the least-confident **20%**
to HerBERT reaches **0.980 macro-F1**, which is 85% of everything the transformer adds over the
baseline (0.944 → 0.986) for a fifth of the GPU traffic. The quoted rate follows a rule rather
than being chosen for its number — the smallest share in the table that closes at least 80% of
the gap.

Two things the cascade is not. It is not a latency win: deferred reviews still wait for the
transformer, so the tail gets slower. And the confidence it routes on is not a probability —
it works because it *ranks*, which is what the risk–coverage curve measures.

## Interpretability

The heaviest coefficients per class, extracted from the fitted pipeline. These are model
weights, not corpus excerpts. They are also the cross-check on everything above: where a
class's heaviest terms are evaluative the model has learned sentiment, and where they are
topical it has learned to recognise the subject matter — which is what a probe of short,
domain-neutral sentences then exposes.

## Error analysis

Misclassifications on PolEmo itself are inspected locally (in the notebook, from data that is
not committed), through the same lens: **negation** cues (`nie`, `bez`, `brak`, …) and
**sarcasm / contrastive clauses**. What the report publishes is the challenge-set measurement
above plus aggregate statistics — never PolEmo excerpts.
