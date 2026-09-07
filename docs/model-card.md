# Model card — pl-review-sense baseline

Date: 2026-08-18
Status: accepted
Author: Piotr Cząstkiewicz
Related to: [research/data-and-methodology.md](research/data-and-methodology.md)

---

## What it is

TF-IDF (word 1–2-grams, `min_df=2`, sublinear TF) into multinomial logistic regression
(`C=1.0`, `class_weight="balanced"`), fitted as one scikit-learn `Pipeline` so the vectorizer
never sees the test split. Three classes: negative, neutral, positive.

Trained on PolEmo 2.0 (`all_text`, revision `802e35d2`), 5 264 reviews after dropping the
`ambiguous` class. Seed 42.

> **A fine-tuned HerBERT now exists for the same task and beats this model decisively** —
> 0.986 macro-F1 against 0.944, ten errors on the test split against forty-one, *p* < 0.0001 on
> the paired test. This card still describes the baseline, because the baseline is what the
> API serves and what a cascade would put in front of the transformer. Read it as the cheap
> half of a two-model system rather than as the recommended model.

## What it is for

Classifying **long, written Polish product/service reviews** of the kind PolEmo collects —
hotels, medicine, products, school. It is a baseline against which a fine-tuned transformer is
measured, and a demonstration of how far a linear model gets on that task.

## What it is not for

- **Short text.** It answers 48 of 80 one-sentence reviews written for the published page,
  including 13 of 20 in a control cell of unambiguous sentiment. The corpus score does not
  transfer to that kind of input. Note the bound on this: PolEmo holds only 3 test reviews as
  short as those sentences, and across the lengths it does cover, length changes the score by
  0.011 — so *why* the transfer fails (length, register, or the gap between a review and a
  constructed sentence) is not established here.
- **Irony.** 11 of 20 on a cell built from ironic praise and surface-matched genuine praise —
  the level a coin reaches on a two-label cell.
- **Anything outside the four domains.** The heaviest coefficients for the neutral class are
  topical rather than evaluative, which is a description of what that class was learned from.
- **Any decision about a person.** It scores text, and its errors are not distributed evenly
  across the phenomena above.

## Measured performance

On the untouched 684-review test split:

| metric | value |
|---|---|
| macro-F1 | 0.944, 95% bootstrap interval 0.926–0.961 |
| accuracy | 0.940 |
| per-class F1 | negative 0.946 · neutral 0.970 · positive 0.916 |
| floor — always the majority class | 0.221 macro-F1 at 0.496 accuracy |
| floor — random, matching the training prior | 0.317 macro-F1 at 0.367 accuracy |

Macro-F1 is the headline metric because the classes are imbalanced. Any comparison narrower
than the interval width is one this test set cannot make. The floors are why: a model that has
learned nothing is right about half the time on accuracy.

On the project's own 80-sentence probe: `plain` 13/20, `negation` 13/20, `sarcasm` 11/20,
`contrast` 11/20. Stripping diacritics and introducing a typo change the totals little,
which is a statement about how far the score had already fallen rather than a robustness
result.

## Confidence and deferral

The model is **under-confident**: expected calibration error 0.209, with bins around 0.75
confidence answered correctly every time. Its confidence therefore must not be read as a
probability — but it *ranks* well, which is the property a deferral rule needs. Setting aside
the least confident 10% of reviews raises macro-F1 on what remains from 0.944 to 0.974; the
threshold at that point is 0.519.

That ranking is what makes this model useful despite being beaten: routing the least-confident
**20%** to HerBERT reaches 0.980 macro-F1 overall — 85% of what the transformer adds, with the
GPU serving a fifth of the traffic.

## Cost

A few seconds to train on a laptop CPU, 3.2 MB on disk, thousands of reviews a second at
inference — the published page carries the measured figures with the machine they were taken
on, because timings move with whatever else that machine is doing. This is the bill a GPU
fine-tune has to justify itself against.

## Data and licence

PolEmo 2.0 © CLARIN-PL, CC BY-NC-SA 4.0 — non-commercial, **not redistributed** in this
repository. It is downloaded on demand and cached outside git. What is committed is numbers:
metrics, intervals, coefficients, timings, and per-row predictions as label integers. The
challenge-set sentences that appear on the page are our own.

## Reproducing

```bash
python -m pl_review_sense.baseline_train   # model, metrics, predictions
python -m pl_review_sense.analysis         # intervals, curve, probe, deferral, cost
python -m pl_review_sense.site             # the published page
```
