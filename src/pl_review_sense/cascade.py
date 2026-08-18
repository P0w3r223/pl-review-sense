"""Knowing when not to answer: deferral by confidence, and what it buys.

"Which model is better" is the wrong question for a system that can run both. The cheap model
answers what it is sure about and hands the rest onward; the bill is then set by how much
gets handed on, not by which model is more accurate overall.

Two measurements, and the distinction between them is the point:

* ``risk_coverage`` needs one model. It asks what the baseline scores on the reviews it keeps
  once the least confident share is set aside. This is a statement about the baseline alone
  and can be published today.
* ``cascade`` needs two. It asks what the *combined system* scores when the set-aside share is
  answered by the second model. Until HerBERT has run on a GPU there is no second set of
  predictions, so this returns nothing and the page says why rather than drawing an estimate
  nobody measured.

Rows are set aside strictly in order of the model's own confidence, lowest first. The reported
threshold is the lowest confidence still answered, which is the number an implementer would
actually put in a config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import config, stats


@dataclass(frozen=True)
class CoveragePoint:
    """One deferral rate, seen from the cheap model's side."""

    deferral_rate: float
    deferred: int
    kept: int
    threshold: float  # the lowest confidence still answered
    macro_f1: float  # on the kept rows only
    accuracy: float


@dataclass(frozen=True)
class CascadePoint:
    """One deferral rate, seen from the combined system's side."""

    deferral_rate: float
    deferred: int
    macro_f1: float  # over the whole test set: cheap answers plus escalated ones
    accuracy: float
    escalated_share: float  # what fraction of traffic reaches the expensive model


def _order_by_confidence(confidences: Sequence[float]) -> np.ndarray:
    """Row indices from least to most confident, stably — equal confidences keep row order."""
    return np.argsort(np.asarray(confidences, dtype=np.float64), kind="stable")


def _deferred_count(total: int, rate: float) -> int:
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"deferral rate must be in [0, 1); got {rate}")
    return int(round(total * rate))


def risk_coverage(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float],
    rates: Sequence[float] = config.DEFERRAL_RATES,
) -> List[CoveragePoint]:
    """What the baseline scores on the reviews it keeps, at each deferral rate.

    A rising curve is the evidence that the model's confidence is worth anything at all: if
    setting aside the least confident tenth does not improve the score on the rest, the
    confidence carries no information and no deferral rule built on it will work either.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(confidences, dtype=np.float64)
    if not (true.shape == pred.shape == scores.shape):
        raise ValueError("y_true, y_pred and confidences must be the same length")
    if true.size == 0:
        return []

    n_labels = len(config.LABEL_NAMES)
    order = _order_by_confidence(scores)
    points: List[CoveragePoint] = []
    for rate in rates:
        deferred = _deferred_count(true.size, rate)
        kept = order[deferred:]
        if kept.size == 0:
            continue
        points.append(
            CoveragePoint(
                deferral_rate=rate,
                deferred=deferred,
                kept=int(kept.size),
                threshold=float(scores[kept].min()),
                macro_f1=stats.macro_f1(true[kept], pred[kept], n_labels),
                accuracy=float(np.mean(true[kept] == pred[kept])),
            )
        )
    return points


def cascade(
    y_true: Sequence[int],
    base_pred: Sequence[int],
    confidences: Sequence[float],
    escalated_pred: Optional[Sequence[int]],
    rates: Sequence[float] = config.DEFERRAL_RATES,
) -> List[CascadePoint]:
    """The combined system: cheap answers kept, deferred rows replaced by the second model.

    Returns an empty list when there are no second-model predictions. That is the honest
    answer to "what would the cascade score" before the expensive model has ever run — and
    an empty list is much harder to mistake for a result than a plausible-looking number.
    """
    if escalated_pred is None:
        return []

    true = np.asarray(y_true, dtype=np.int64)
    cheap = np.asarray(base_pred, dtype=np.int64)
    expensive = np.asarray(escalated_pred, dtype=np.int64)
    scores = np.asarray(confidences, dtype=np.float64)
    if not (true.shape == cheap.shape == expensive.shape == scores.shape):
        raise ValueError("all four arrays must describe the same test rows, in the same order")
    if true.size == 0:
        return []  # same answer as risk_coverage on nothing: no rows, no operating points

    n_labels = len(config.LABEL_NAMES)
    order = _order_by_confidence(scores)
    points: List[CascadePoint] = []
    for rate in rates:
        deferred = _deferred_count(true.size, rate)
        combined = cheap.copy()
        escalate = order[:deferred]
        combined[escalate] = expensive[escalate]
        points.append(
            CascadePoint(
                deferral_rate=rate,
                deferred=deferred,
                macro_f1=stats.macro_f1(true, combined, n_labels),
                accuracy=float(np.mean(true == combined)),
                escalated_share=deferred / true.size,
            )
        )
    return points
