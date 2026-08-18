"""Uncertainty around a score, and whether two models are actually apart.

Pure functions over label lists — no I/O, no model, no dataset. A macro-F1 printed to three
decimals invites a comparison it cannot support: on 684 test rows the difference between
0.944 and 0.951 is well inside the noise of which reviews happened to land in the split.
Everything here exists so the page can say how wide that noise is instead of implying it
away.

Two separate questions, deliberately answered by two separate tools:

* **How well is one model pinned down** — the bootstrap interval, which resamples the test
  rows and asks how much the score moves when the sample does.
* **Are two models different** — McNemar's test on the *paired* predictions, which looks
  only at the reviews the two models disagree on. Overlapping intervals do not settle this:
  both models see the same rows, and a model that is right on every row the other misses is
  distinguishable long before the intervals come apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from . import config


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range the resampling put around it."""

    point: float
    low: float
    high: float
    resamples: int
    confidence: float

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class McNemarResult:
    """The paired comparison: only the reviews the two models answered differently."""

    only_a_correct: int  # a right, b wrong
    only_b_correct: int  # b right, a wrong
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_a_correct + self.only_b_correct


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class Calibration:
    bins: List[CalibrationBin]
    expected_error: float  # ECE: mean gap between confidence and accuracy, weighted by count


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int) -> float:
    """Macro-F1 over a fixed label set, with empty classes scoring 0 rather than vanishing.

    Recomputed here from a confusion matrix rather than called out to scikit-learn per
    resample: two thousand resamples through the full metric machinery is a minute of
    wall clock for a number this is three lines of arithmetic. ``test_stats`` pins it
    against ``f1_score`` so the shortcut cannot drift away from the reference.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    counts = np.bincount(true * n_labels + pred, minlength=n_labels * n_labels)
    matrix = counts.reshape(n_labels, n_labels)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = 2 * true_positive + (matrix.sum(axis=0) - true_positive) + (
        matrix.sum(axis=1) - true_positive
    )
    per_class = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(n_labels, dtype=np.float64),
        where=denominator > 0,
    )
    return float(per_class.mean())


def bootstrap_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    confidence: float = config.CONFIDENCE_LEVEL,
    seed: int = config.RANDOM_STATE,
) -> Interval:
    """Percentile bootstrap over the test rows.

    Rows are resampled, not predictions: the quantity being estimated is "what would this
    model score on another sample of reviews like these", and that is the sampling the
    interval has to imitate. The seed is fixed so a rebuild of the page from the same
    predictions produces the same interval.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError(f"paired arrays expected; got {true.shape} and {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot bootstrap an empty test set")

    n_labels = len(config.LABEL_NAMES)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, true.size, size=(resamples, true.size))
    scores = np.array([macro_f1(true[row], pred[row], n_labels) for row in draws])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(scores, [tail, 1.0 - tail])
    return Interval(
        point=macro_f1(true, pred, n_labels),
        low=float(low),
        high=float(high),
        resamples=resamples,
        confidence=confidence,
    )


def mcnemar(
    y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]
) -> McNemarResult:
    """Exact McNemar test on the rows where the two models disagree.

    The exact binomial form rather than the chi-square approximation: the discordant count
    on a 684-row test set can easily land in the tens, where the approximation is known to
    be anticonservative — and a p-value that is too small is exactly the failure this page
    exists to avoid.
    """
    true = np.asarray(y_true, dtype=np.int64)
    a_correct = np.asarray(pred_a, dtype=np.int64) == true
    b_correct = np.asarray(pred_b, dtype=np.int64) == true
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))

    if only_a + only_b == 0:
        # The two models are right and wrong on exactly the same reviews. There is no
        # evidence of a difference, and no test to run — 1.0 states that, where a divide
        # by zero would only crash.
        return McNemarResult(only_a_correct=0, only_b_correct=0, p_value=1.0)

    from scipy.stats import binomtest

    result = binomtest(only_a, only_a + only_b, p=0.5, alternative="two-sided")
    return McNemarResult(
        only_a_correct=only_a, only_b_correct=only_b, p_value=float(result.pvalue)
    )


def calibrate(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = config.CALIBRATION_BINS,
) -> Calibration:
    """Bin predictions by their own confidence and compare each bin to how often it was right.

    A model that says 0.9 and is right nine times in ten is usable as a filter; one that
    says 0.9 and is right seven times in ten will hand on the wrong reviews when a deferral
    rule reads its confidence as a probability. That is the only reason this is measured
    here: everything in ``cascade`` rests on the confidence meaning something.
    """
    scores = np.asarray(confidences, dtype=np.float64)
    hits = np.asarray(correct, dtype=bool)
    if scores.shape != hits.shape:
        raise ValueError(f"paired arrays expected; got {scores.shape} and {hits.shape}")
    if scores.size == 0:
        return Calibration(bins=[], expected_error=0.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    # ``right=True`` with the first edge nudged open, so a confidence of exactly 1.0 lands in
    # the top bin instead of falling off the end of the last one.
    index = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, bins - 1)

    rows: List[CalibrationBin] = []
    gap_weight = 0.0
    for position in range(bins):
        selected = index == position
        count = int(selected.sum())
        if count == 0:
            rows.append(
                CalibrationBin(
                    lower=float(edges[position]),
                    upper=float(edges[position + 1]),
                    count=0,
                    mean_confidence=0.0,
                    accuracy=0.0,
                )
            )
            continue
        mean_confidence = float(scores[selected].mean())
        accuracy = float(hits[selected].mean())
        gap_weight += count * abs(accuracy - mean_confidence)
        rows.append(
            CalibrationBin(
                lower=float(edges[position]),
                upper=float(edges[position + 1]),
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return Calibration(bins=rows, expected_error=gap_weight / scores.size)
