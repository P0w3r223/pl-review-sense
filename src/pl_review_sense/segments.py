"""How the score varies across a covariate — here, the length of the review.

The page leads with a claim built on 80 sentences we wrote: that a macro-F1 of 0.94 does not
survive a short review. The obvious objection is that the probe measures our sentences rather
than their length, and the corpus can answer it. PolEmo runs from a sentence to several hundred
words; if the score falls with length there too, the claim has corroboration from data nobody
here wrote. If it does not, the probe is measuring something else — register, domain, or the
gap between a review and a sentence — and the page has to say so instead.

Pure functions over three parallel lists. Nothing here loads the corpus; ``analysis`` supplies
the lengths, and only the aggregates it produces are ever committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from sklearn.metrics import f1_score

from . import config


@dataclass(frozen=True)
class Segment:
    """One slice of the test set, and how the model did inside it."""

    name: str
    lower: int
    upper: int | None  # None = open-ended top bucket
    n: int
    accuracy: float
    macro_f1: float
    classes_present: int

    @property
    def thin(self) -> bool:
        """Too few rows to read the score as anything but a direction."""
        return self.n < config.MIN_SEGMENT_N


def bucket_names(edges: Sequence[int] = config.LENGTH_BUCKET_EDGES) -> List[str]:
    """Human-readable ranges for the edges, in the order the buckets are reported."""
    names = [f"< {edges[0]}"]
    names += [f"{low}–{high - 1}" for low, high in zip(edges, edges[1:])]
    names.append(f"{edges[-1]}+")
    return names


def assign(value: int, edges: Sequence[int] = config.LENGTH_BUCKET_EDGES) -> int:
    """Index of the bucket a value falls in. Edges are lower-open: 25 starts the second bucket."""
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def segment_scores(
    values: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    edges: Sequence[int] = config.LENGTH_BUCKET_EDGES,
) -> List[Segment]:
    """Accuracy and macro-F1 within each bucket, plus how many rows are behind each.

    Macro-F1 inside a segment averages **only the classes that occur in it**. A bucket holding
    no neutral reviews would otherwise score a hard zero for that class and drag the average
    down by a third — reporting the segment's class composition as if it were the model's
    failure. ``classes_present`` travels with the number so the reader can see when the two
    figures are not averaging the same thing.
    """
    if not (len(values) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"parallel lists expected; got {len(values)}, {len(y_true)}, {len(y_pred)}"
        )

    names = bucket_names(edges)
    grouped: Dict[int, List[int]] = {}
    for row, value in enumerate(values):
        grouped.setdefault(assign(value, edges), []).append(row)

    segments: List[Segment] = []
    for index, name in enumerate(names):
        rows = grouped.get(index, [])
        if not rows:
            continue
        true = [y_true[row] for row in rows]
        pred = [y_pred[row] for row in rows]
        present = sorted(set(true))
        segments.append(
            Segment(
                name=name,
                lower=0 if index == 0 else edges[index - 1],
                upper=edges[index] if index < len(edges) else None,
                n=len(rows),
                accuracy=sum(t == p for t, p in zip(true, pred)) / len(rows),
                macro_f1=float(
                    f1_score(true, pred, labels=present, average="macro", zero_division=0)
                ),
                classes_present=len(present),
            )
        )
    return segments


def trend(segments: Sequence[Segment]) -> float | None:
    """The gap between the shortest and the longest segment that are thick enough to compare.

    A single number the page can lead the section with, and one that can come out negative —
    which would be the result that contradicts the headline rather than a broken measurement.
    """
    usable = [segment for segment in segments if not segment.thin]
    if len(usable) < 2:
        return None
    return usable[-1].macro_f1 - usable[0].macro_f1
