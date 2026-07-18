"""Evaluation metrics, confusion matrix, and error analysis. Pure functions over label lists.

Macro-F1 is the headline metric because the classes are imbalanced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    per_class: List[ClassMetrics]
    confusion: List[List[int]]  # rows = true, cols = predicted


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> EvalResult:
    labels = list(range(len(config.LABEL_NAMES)))
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        ClassMetrics(
            label=config.LABEL_NAMES[i],
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in labels
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return EvalResult(accuracy=acc, macro_f1=macro, per_class=per_class, confusion=cm)


# --- Error analysis (used locally / in the notebook; texts are not committed) --------------
_NEGATION_WORDS = frozenset({"nie", "bez", "żaden", "żadna", "żadne", "nigdy", "ani", "brak"})


def has_negation(text: str) -> bool:
    """Rough Polish-negation cue detector — a lens for the error analysis, not a parser."""
    return any(token.strip(".,!?;:\"'()-") in _NEGATION_WORDS for token in text.lower().split())


@dataclass(frozen=True)
class ErrorExample:
    text: str
    true_label: str
    pred_label: str
    negation: bool
    length: int


def misclassified(
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    limit: int = config.TOP_N_ERRORS,
) -> List[ErrorExample]:
    """Collect misclassified examples, longest first (longer reviews expose negation/sarcasm)."""
    errors = [
        ErrorExample(
            text=text,
            true_label=config.LABEL_NAMES[t],
            pred_label=config.LABEL_NAMES[p],
            negation=has_negation(text),
            length=len(text.split()),
        )
        for text, t, p in zip(texts, y_true, y_pred)
        if t != p
    ]
    errors.sort(key=lambda e: e.length, reverse=True)
    return errors[:limit]


def negation_error_share(errors: Sequence[ErrorExample]) -> float:
    """Fraction of misclassifications whose text contains a negation cue (0.0 if none)."""
    if not errors:
        return 0.0
    return sum(1 for e in errors if e.negation) / len(errors)
