"""How much labelled data the cheap model actually needs.

The project's question is when a transformer earns its compute. Comparing two finished
models answers it only at one point — the size of the corpus that happens to exist. The
curve answers it everywhere: if the baseline is already at 0.90 macro-F1 on 600 reviews, the
interesting decision is not which model to fine-tune but whether to label the other five
thousand.

Two decisions keep the curve honest.

**Subsamples are stratified.** A random draw of 150 rows from an imbalanced corpus can miss
the neutral class outright, and the point would then measure that draw's luck rather than
the size.

**Every size is drawn several times.** One subsample per size produces a curve whose bumps
are sampling noise; the spread across seeds is reported alongside the mean so a reader can
see which bumps mean anything.

The fitting step is injected rather than imported, so the curve can be computed for any
model — and unit-tested without training anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from . import config, stats

# (train_texts, train_labels, test_texts) -> predicted labels for the test texts.
FitPredict = Callable[[Sequence[str], Sequence[int], Sequence[str]], Sequence[int]]


@dataclass(frozen=True)
class CurvePoint:
    """One training size: the mean score across seeds, and how far the seeds spread."""

    size: int
    seeds: int
    mean_macro_f1: float
    low: float  # worst seed, not a confidence bound — the observed spread, nothing more
    high: float

    @property
    def spread(self) -> float:
        return self.high - self.low


def stratified_indices(labels: Sequence[int], size: int, seed: int) -> List[int]:
    """Pick ``size`` row indices keeping each class's share of the training set.

    Rounding is settled by handing the leftover rows to the largest classes first, so the
    result is exactly ``size`` rows and never leaves a class empty when the requested size
    could hold it.
    """
    labels = list(labels)
    if size >= len(labels):
        return list(range(len(labels)))
    if size <= 0:
        raise ValueError(f"subsample size must be positive; got {size}")

    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)

    classes = sorted(by_class)
    if size < len(classes):
        raise ValueError(
            f"subsample of {size} cannot cover {len(classes)} classes without dropping one"
        )

    exact = {label: size * len(by_class[label]) / len(labels) for label in classes}
    quota = {label: max(1, int(exact[label])) for label in classes}
    # Hand the rounding remainder to the classes that lost the most to flooring, largest
    # first, so the shortfall lands where it distorts the proportions least. Only classes with
    # rows left to give are candidates, in either direction: picking one that is already at its
    # pool size — or already down to its floor of one — would leave the total off by that much
    # and quietly return a subsample of the wrong size.
    while sum(quota.values()) < size:
        candidates = [label for label in classes if quota[label] < len(by_class[label])]
        if not candidates:
            break  # unreachable while size < len(labels); a guard, not a branch
        label = max(candidates, key=lambda name: (exact[name] - quota[name], len(by_class[name])))
        quota[label] += 1
    while sum(quota.values()) > size:
        candidates = [label for label in classes if quota[label] > 1]
        if not candidates:
            break  # unreachable: one row per class is at most `size`, checked above
        label = max(candidates, key=lambda name: (quota[name] - exact[name], len(by_class[name])))
        quota[label] -= 1

    rng = np.random.default_rng(seed)
    picked: List[int] = []
    for label in classes:
        pool = by_class[label]
        take = min(quota[label], len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[position] for position in sorted(chosen))
    return sorted(picked)


def learning_curve(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    fit_predict: FitPredict,
    sizes: Sequence[int] = config.LEARNING_CURVE_SIZES,
    seeds: Sequence[int] = config.LEARNING_CURVE_SEEDS,
) -> List[CurvePoint]:
    """Macro-F1 against training size, plus the full corpus as the final point.

    The full-corpus point is fitted once and carries a spread of zero: there is nothing to
    subsample, so repeating it with another seed would produce the same model and pretend to
    a variability the measurement does not have.
    """
    n_labels = len(config.LABEL_NAMES)
    usable = sorted({size for size in sizes if size < len(train_labels)})

    points: List[CurvePoint] = []
    for size in usable:
        scores = []
        for seed in seeds:
            index = stratified_indices(train_labels, size, seed)
            predicted = fit_predict(
                [train_texts[i] for i in index], [train_labels[i] for i in index], test_texts
            )
            scores.append(stats.macro_f1(test_labels, predicted, n_labels))
        points.append(
            CurvePoint(
                size=size,
                seeds=len(seeds),
                mean_macro_f1=float(np.mean(scores)),
                low=float(np.min(scores)),
                high=float(np.max(scores)),
            )
        )

    full = stats.macro_f1(
        test_labels, fit_predict(train_texts, train_labels, test_texts), n_labels
    )
    points.append(
        CurvePoint(
            size=len(train_labels), seeds=1, mean_macro_f1=full, low=full, high=full
        )
    )
    return points


def reaches(points: Sequence[CurvePoint], share_of_full: float) -> int | None:
    """The smallest training size whose mean already reaches ``share`` of the full-corpus score.

    Reported in ``learning_curve.json`` as a strict reading of the curve: at 99% of the full
    score it answers "where does this stop improving". The page quotes a looser one — the first
    size within a fixed macro-F1 distance of the full corpus — because the question a reader
    brings is where the remaining gap stops being worth the labelling, not where it closes.
    """
    if not points:
        return None
    target = points[-1].mean_macro_f1 * share_of_full
    for point in points:
        if point.mean_macro_f1 >= target:
            return point.size
    return None
