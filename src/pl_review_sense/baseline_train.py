"""Train + evaluate the TF-IDF baseline on PolEmo; persist the model, metrics, and predictions.

Run: ``python -m pl_review_sense.baseline_train``

Only label integers are written to disk — never the PolEmo review texts (CC BY-NC-SA; not
redistributed here). That is enough to reproduce the confusion matrix and metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from . import baseline, config, evaluate
from .data import load_polemo


def _write_metrics(result: evaluate.EvalResult) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "tfidf+logreg",
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "per_class": [asdict(c) for c in result.per_class],
        "confusion": result.confusion,
        "labels": list(config.LABEL_NAMES),
    }
    config.BASELINE_METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_predictions(y_true, y_pred, probabilities) -> None:
    """Per-row predictions, in test-split order, with the model's own probabilities.

    Columnar rather than a list of objects: the file is read as three parallel arrays by
    everything downstream, and repeating three keys 684 times to say so costs more than it
    explains. The probabilities are what make deferral and calibration measurable at all —
    a confidence the page never sees cannot be checked against how often it was right.

    Row order is the test split's own, which is what pairs these predictions with HerBERT's
    for the McNemar test later. Still no review text: labels and numbers only.
    """
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "tfidf+logreg",
        "labels": list(config.LABEL_NAMES),
        "true": [int(t) for t in y_true],
        "pred": [int(p) for p in y_pred],
        "proba": [[round(float(value), 6) for value in row] for row in probabilities],
    }
    config.BASELINE_PREDICTIONS_PATH.write_text(
        json.dumps(payload, indent=None), encoding="utf-8"
    )


def main() -> None:
    data = load_polemo()
    # Hyperparameters are fixed config defaults; the validation split is intentionally held out
    # (not tuned against), so the test metrics remain an honest, leakage-free estimate.
    print(f"loaded PolEmo: train={len(data.train)} val={len(data.validation)} test={len(data.test)}")

    pipe = baseline.train(data.train.texts, data.train.labels)
    proba = baseline.predict_proba(pipe, data.test.texts)
    # Predict from the probabilities rather than calling `predict` separately: the label and
    # the confidence published beside it then cannot disagree about the same review. Reading the
    # column index as the label holds because `load_polemo` refuses a training split whose labels
    # are not exactly {0, 1, 2}, so the classifier's classes are those three in that order.
    y_pred = [max(range(len(row)), key=lambda i: row[i]) for row in proba]
    result = evaluate.evaluate(data.test.labels, y_pred)
    print(f"accuracy={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}")
    for cls in result.per_class:
        print(f"  {cls.label:<9} P={cls.precision:.3f} R={cls.recall:.3f} F1={cls.f1:.3f} n={cls.support}")

    baseline.save(pipe)
    _write_metrics(result)
    _write_predictions(data.test.labels, y_pred, proba)
    print(f"saved model -> {config.BASELINE_MODEL_PATH.name}, metrics + predictions -> reports/")
    print("next: python -m pl_review_sense.analysis   # intervals, curve, probe, deferral")


if __name__ == "__main__":
    main()
