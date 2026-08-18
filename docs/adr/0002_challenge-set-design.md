# Measuring the phenomena instead of asserting them

Date: 2026-08-18
Status: accepted
Author: P0w3r223
Related to: [0001_page-is-a-function-of-committed-metrics.md](0001_page-is-a-function-of-committed-metrics.md)

---

## Context

The report claimed that negation, sarcasm and contrastive clauses are where a bag-of-words
model slips, and illustrated the claim with three sentences that had never been scored. The
claim is standard and probably true, which is exactly why it was worth checking: a portfolio
project whose central methodological point is "measure, do not assert" cannot leave its own
most quotable paragraph unmeasured.

PolEmo text cannot be used for this. It is CC BY-NC-SA and is not redistributed by this
repository, and a probe drawn from the training domains would measure memorisation as much as
the phenomenon.

## Decision

An 80-sentence challenge set, written for this project, in four cells of twenty: `plain`
(control), `negation`, `sarcasm`, `contrast`. Three design rules make the numbers mean
something.

**Cells are balanced so no single answer can win them.** A sarcasm probe made only of ironic
praise is scored 20 of 20 by a classifier that answers *negative* to everything. Every ironic
sentence therefore has a genuine counterpart in the same enthusiastic register — down to
minimal pairs, "bateria wytrzymała całe trzy godziny" against "całe trzy dni". The negation
cell carries cues in both directions for the same reason. A test enforces that no label holds
more than 60% of any cell.

**A control cell comes first.** Without `plain`, a low score on `sarcasm` reads as a failure
at irony when it may be a failure at short text. With it, the cells are read relative to a
sentence the model has no excuse for.

**Robustness variants are derived, not written.** Missing diacritics and typos are produced by
transforming the same 80 sentences mechanically — NFD decomposition plus the `ł` stroke, and
one adjacent-character swap nearest the middle of the longest word. A hand-written "typo set"
would differ in content as well as in spelling, and the comparison would no longer isolate the
spelling. The swap position steps outward until the two characters differ, so a swap is never
a no-op that silently drops a sentence from the variant set.

**Counts, not rates.** Results are published as *k of n* with the cell size drawn as the track
behind the bar. Twenty sentences do not support a percentage, and a percentage on the same
page as 684-row test figures invites the reader to compare the two as if they were the same
kind of number. `MIN_PHENOMENON_N` marks cells that fall below the floor.

## Consequences

- The probe is scored by `analysis` and lands in `reports/metrics/challenge.json`; the page
  reads it and shows the first few misses verbatim, since the sentences are ours to publish.
- It found something the corpus score hides: the baseline is at 0.944 macro-F1 on PolEmo and
  answers 48 of 80 short review sentences correctly, including 13 of 20 in the control cell.
  The headline of the page is derived from that comparison and changes with it.
- The probe is a diagnostic, not a benchmark. Eighty sentences show that a gap exists; they do
  not size it, and the page says so.
- Adding a cell means adding twenty balanced sentences, not three illustrative ones.

## Alternatives considered

**Use a published Polish challenge set.** Better statistics and comparable numbers, at the
cost of another licence to honour and of measuring phenomena someone else chose. Worth
revisiting; the point here was to make the project's own claim checkable.

**Extract adversarial examples from PolEmo's errors.** Free to produce, and the natural next
step for error analysis — but the sentences could not be published, so the page would again be
asserting something the reader cannot see.
