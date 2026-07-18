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


def _write_predictions(y_true, y_pred) -> None:
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"true": int(t), "pred": int(p)} for t, p in zip(y_true, y_pred)]
    config.BASELINE_PREDICTIONS_PATH.write_text(json.dumps(rows), encoding="utf-8")


def main() -> None:
    data = load_polemo()
    # Hyperparameters are fixed config defaults; the validation split is intentionally held out
    # (not tuned against), so the test metrics remain an honest, leakage-free estimate.
    print(f"loaded PolEmo: train={len(data.train)} val={len(data.validation)} test={len(data.test)}")

    pipe = baseline.train(data.train.texts, data.train.labels)
    y_pred = baseline.predict(pipe, data.test.texts)
    result = evaluate.evaluate(data.test.labels, y_pred)
    print(f"accuracy={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}")
    for cls in result.per_class:
        print(f"  {cls.label:<9} P={cls.precision:.3f} R={cls.recall:.3f} F1={cls.f1:.3f} n={cls.support}")

    baseline.save(pipe)
    _write_metrics(result)
    _write_predictions(data.test.labels, y_pred)
    print(f"saved model -> {config.BASELINE_MODEL_PATH.name}, metrics + predictions -> reports/")


if __name__ == "__main__":
    main()
