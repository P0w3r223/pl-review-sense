"""What the baseline actually keys on: the terms carrying the most weight per class.

Cheap interpretability is one of the two things a linear model has over a transformer (the
other is the compute bill), so the comparison is incomplete without showing it. These are
model coefficients, not corpus excerpts — a learned weight on the bigram "nie polecam" says
what the classifier does, and reproduces no review.

The extraction is a pure function over ``(feature names, coefficient matrix)``; the adapter
that pulls those two out of a fitted scikit-learn pipeline is the only part that knows what
a Pipeline is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from sklearn.pipeline import Pipeline

from . import baseline, config


@dataclass(frozen=True)
class Term:
    term: str
    weight: float


def top_terms(
    feature_names: Sequence[str],
    coefficients: Sequence[Sequence[float]],
    top_n: int = config.TOP_TERMS_PER_CLASS,
    labels: Sequence[str] = config.LABEL_NAMES,
) -> Dict[str, List[Term]]:
    """The ``top_n`` highest-weighted terms for each class.

    Only positive weights are ranked. In a multinomial model the most negative coefficient
    for "positive" is simply the most positive one for another class, so listing both ends
    would print the same evidence twice under two headings.
    """
    if len(coefficients) != len(labels):
        raise ValueError(f"{len(coefficients)} coefficient rows against {len(labels)} labels")

    ranked: Dict[str, List[Term]] = {}
    for label, row in zip(labels, coefficients):
        if len(row) != len(feature_names):
            raise ValueError(
                f"{len(row)} coefficients against {len(feature_names)} feature names"
            )
        # Sorted by weight, then by term, so ties do not reorder between runs and the
        # published page stays a function of the model rather than of the sort's mood.
        order = sorted(range(len(row)), key=lambda i: (-row[i], feature_names[i]))
        ranked[label] = [
            Term(term=str(feature_names[i]), weight=float(row[i]))
            for i in order[:top_n]
            if row[i] > 0
        ]
    return ranked


def from_pipeline(
    pipeline: Pipeline, top_n: int = config.TOP_TERMS_PER_CLASS
) -> Dict[str, List[Term]]:
    """Pull the vocabulary and coefficients out of the fitted baseline pipeline."""
    vectorizer = pipeline.named_steps[baseline.VECTORIZER_STEP]
    classifier = pipeline.named_steps[baseline.CLASSIFIER_STEP]
    return top_terms(list(vectorizer.get_feature_names_out()), classifier.coef_, top_n)
