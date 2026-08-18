"""Produce every number the published page rests on, and a manifest describing the run.

Run: ``python -m pl_review_sense.analysis`` (after ``baseline_train``).

This is the only place where the measurements meet the disk. Each question the page asks
gets its own file under ``reports/metrics/``, so a panel whose evidence has not been produced
yet — the paired comparison with HerBERT, today — is missing on its own instead of holding
up the rest. The site build renders a "waiting" state for whatever is absent and never
invents a placeholder number.

Nothing here writes review text. What lands in git is intervals, counts, coefficients and
timings; the challenge-set sentences that do appear are our own, from ``challenge_set``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import (
    baseline,
    cascade,
    challenge,
    challenge_set,
    config,
    curves,
    interpret,
    stats,
)
from .data import load_polemo

# How much of the full-corpus score counts as "already there" when reporting the size the
# curve reaches first. 0.99 rather than 1.0: the last percent of a learning curve is the
# part that costs thousands of labels, which is exactly the trade-off being reported.
CURVE_TARGET_SHARE = 0.99
# Misclassified probe sentences shown on the page. Enough to see the pattern, few enough
# that the section stays a finding rather than a dump.
PROBE_EXAMPLES = 6


def _git_sha() -> str:
    """The commit the numbers were produced at, or ``unknown`` outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit LF: these files are committed and CI reads them on another OS, so "the page
    # is a function of the metrics" has to survive the crossing.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.write("\n")
    print(f"wrote {path.relative_to(config.PROJECT_ROOT).as_posix()}")


def _read(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_predictions(path: Path) -> Optional[dict]:
    """Read a columnar predictions file, refusing one whose columns disagree."""
    payload = _read(path)
    if payload is None:
        return None
    lengths = {len(payload[column]) for column in ("true", "pred") if column in payload}
    if "proba" in payload:
        lengths.add(len(payload["proba"]))
    if len(lengths) != 1:
        raise ValueError(f"{path.name}: columns of differing length {sorted(lengths)}")
    return payload


def significance(baseline_rows: dict, herbert_rows: Optional[dict]) -> dict:
    """The interval around the baseline, and the paired test against HerBERT when it exists."""
    interval = stats.bootstrap_macro_f1(baseline_rows["true"], baseline_rows["pred"])
    payload = {
        "baseline": {
            "macro_f1": interval.point,
            "low": interval.low,
            "high": interval.high,
            "resamples": interval.resamples,
            "confidence": interval.confidence,
        },
        "herbert": None,
        "mcnemar": None,
    }
    if herbert_rows is None:
        return payload

    if herbert_rows["true"] != baseline_rows["true"]:
        raise ValueError(
            "the two prediction files describe different test rows — the paired test would "
            "compare reviews with each other rather than models"
        )
    herbert_interval = stats.bootstrap_macro_f1(herbert_rows["true"], herbert_rows["pred"])
    test = stats.mcnemar(baseline_rows["true"], baseline_rows["pred"], herbert_rows["pred"])
    payload["herbert"] = {
        "macro_f1": herbert_interval.point,
        "low": herbert_interval.low,
        "high": herbert_interval.high,
        "resamples": herbert_interval.resamples,
        "confidence": herbert_interval.confidence,
    }
    payload["mcnemar"] = {
        "only_baseline_correct": test.only_a_correct,
        "only_herbert_correct": test.only_b_correct,
        "discordant": test.discordant,
        "p_value": test.p_value,
    }
    return payload


def probe(pipeline) -> dict:
    """Score the challenge set and its derived variants, and keep a few of the misses."""
    cases = list(challenge_set.CASES)
    everything = cases + challenge.derive_variants(cases)
    predicted = baseline.predict(pipeline, [case.text for case in everything])
    scores = challenge.score(everything, predicted)

    misses = [
        {
            "text": case.text,
            "phenomenon": case.phenomenon,
            "gold": config.LABEL_NAMES[case.label],
            "predicted": config.LABEL_NAMES[guess],
        }
        for case, guess in zip(everything, predicted)
        if guess != case.label
    ]
    return {
        "base_cases": len(cases),
        "derived_cases": len(everything) - len(cases),
        "min_phenomenon_n": config.MIN_PHENOMENON_N,
        "scores": [
            {
                "phenomenon": item.phenomenon,
                "correct": item.correct,
                "total": item.total,
                "note": item.note,
                "thin": item.thin,
            }
            for item in scores
        ],
        "misses_total": len(misses),
        "misses": misses[:PROBE_EXAMPLES],
    }


def deferral(baseline_rows: dict, herbert_rows: Optional[dict]) -> dict:
    """Calibration, the risk–coverage curve, and the cascade when a second model exists."""
    confidences = [max(row) for row in baseline_rows["proba"]]
    correct = [t == p for t, p in zip(baseline_rows["true"], baseline_rows["pred"])]
    calibration = stats.calibrate(confidences, correct)
    coverage = cascade.risk_coverage(
        baseline_rows["true"], baseline_rows["pred"], confidences
    )
    combined = cascade.cascade(
        baseline_rows["true"],
        baseline_rows["pred"],
        confidences,
        herbert_rows["pred"] if herbert_rows else None,
    )
    return {
        "expected_calibration_error": calibration.expected_error,
        "bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "mean_confidence": item.mean_confidence,
                "accuracy": item.accuracy,
            }
            for item in calibration.bins
        ],
        "risk_coverage": [
            {
                "deferral_rate": point.deferral_rate,
                "deferred": point.deferred,
                "kept": point.kept,
                "threshold": point.threshold,
                "macro_f1": point.macro_f1,
                "accuracy": point.accuracy,
            }
            for point in coverage
        ],
        "cascade": [
            {
                "deferral_rate": point.deferral_rate,
                "deferred": point.deferred,
                "macro_f1": point.macro_f1,
                "accuracy": point.accuracy,
                "escalated_share": point.escalated_share,
            }
            for point in combined
        ],
    }


def learning_curve(data) -> dict:
    """Macro-F1 against training size — the same pipeline, refit from scratch each time."""

    def fit_predict(train_texts, train_labels, test_texts):
        return baseline.predict(baseline.train(train_texts, train_labels), test_texts)

    points = curves.learning_curve(
        data.train.texts,
        data.train.labels,
        data.test.texts,
        data.test.labels,
        fit_predict=fit_predict,
    )
    return {
        "seeds": list(config.LEARNING_CURVE_SEEDS),
        "target_share": CURVE_TARGET_SHARE,
        "reaches_target_at": curves.reaches(points, CURVE_TARGET_SHARE),
        "points": [
            {
                "size": point.size,
                "seeds": point.seeds,
                "mean_macro_f1": point.mean_macro_f1,
                "low": point.low,
                "high": point.high,
            }
            for point in points
        ],
    }


def cost(data) -> dict:
    """What the cheap model costs, measured rather than asserted.

    Timed on whatever machine ran the analysis, and the machine is recorded with the number:
    a training time without the hardware beside it is not a measurement, and this figure
    exists precisely to be compared against a GPU hour.
    """
    started = time.perf_counter()
    pipeline = baseline.train(data.train.texts, data.train.labels)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    baseline.predict(pipeline, data.test.texts)
    predict_seconds = time.perf_counter() - started

    model_bytes = (
        config.BASELINE_MODEL_PATH.stat().st_size
        if config.BASELINE_MODEL_PATH.exists()
        else None
    )
    return {
        "train_seconds": round(train_seconds, 2),
        "train_rows": len(data.train.labels),
        "predict_rows_per_second": round(len(data.test.labels) / predict_seconds, 1),
        "model_bytes": model_bytes,
        "machine": platform.processor() or platform.machine(),
        "device": "CPU",
        "python": platform.python_version(),
    }


def manifest(data, herbert_rows: Optional[dict]) -> dict:
    import sklearn

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "dataset": {
            "name": config.DATASET,
            "config": config.DATASET_CONFIG,
            "revision": config.DATASET_REVISION,
            "dropped_class": config.DROP_LABEL_NAME,
            "license": "CC BY-NC-SA 4.0",
        },
        "splits": {
            "train": len(data.train),
            "validation": len(data.validation),
            "test": len(data.test),
        },
        "seed": config.RANDOM_STATE,
        "herbert_available": herbert_rows is not None,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }


def main() -> None:
    baseline_rows = _load_predictions(config.BASELINE_PREDICTIONS_PATH)
    if baseline_rows is None or "proba" not in baseline_rows:
        raise SystemExit(
            "no baseline predictions with probabilities — run "
            "`python -m pl_review_sense.baseline_train` first"
        )
    try:
        pipeline = baseline.load()
    except FileNotFoundError as exc:
        raise SystemExit(
            "no saved baseline model — run `python -m pl_review_sense.baseline_train` first"
        ) from exc

    herbert_rows = _load_predictions(config.HERBERT_PREDICTIONS_PATH)
    data = load_polemo()

    _write(config.SIGNIFICANCE_PATH, significance(baseline_rows, herbert_rows))
    _write(config.CHALLENGE_PATH, probe(pipeline))
    _write(config.DEFERRAL_PATH, deferral(baseline_rows, herbert_rows))
    _write(
        config.INTERPRETABILITY_PATH,
        {
            "top_n": config.TOP_TERMS_PER_CLASS,
            "per_class": {
                label: [{"term": item.term, "weight": item.weight} for item in terms]
                for label, terms in interpret.from_pipeline(pipeline).items()
            },
        },
    )
    _write(config.LEARNING_CURVE_PATH, learning_curve(data))
    _write(config.COST_PATH, cost(data))
    _write(config.MANIFEST_PATH, manifest(data, herbert_rows))

    if herbert_rows is None:
        print(
            "HerBERT predictions absent: the paired comparison and the cascade will publish "
            "as waiting panels. Run notebooks/herbert_colab.ipynb on a GPU to fill them."
        )
    print("next: python -m pl_review_sense.site   # rebuild docs/index.html")


if __name__ == "__main__":
    main()
