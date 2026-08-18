"""The published page.

Three things are worth pinning: that a panel without evidence says so instead of inventing a
number, that a figure never goes out without the counts behind it, and that the page is a
function of the committed metrics — the last is what the CI drift job rests on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pl_review_sense import config
from pl_review_sense.site import build, charts

BASELINE = {
    "model": "tfidf+logreg",
    "accuracy": 0.94,
    "macro_f1": 0.944,
    "per_class": [
        {"label": "negative", "precision": 0.94, "recall": 0.95, "f1": 0.946, "support": 339},
        {"label": "neutral", "precision": 0.99, "recall": 0.95, "f1": 0.970, "support": 118},
        {"label": "positive", "precision": 0.92, "recall": 0.92, "f1": 0.916, "support": 227},
    ],
    "confusion": [[323, 0, 16], [3, 112, 3], [18, 1, 208]],
    "labels": ["negative", "neutral", "positive"],
}

MANIFEST = {
    "generated_at": "2026-08-18T10:00:00+00:00",
    "git_sha": "abc1234",
    "dataset": {
        "name": "clarin-pl/polemo2-official",
        "config": "all_text",
        "revision": "802e35d2b12bae84bb07911d841e8f046dc2fcef",
        "dropped_class": "amb",
        "license": "CC BY-NC-SA 4.0",
    },
    "splits": {"train": 5264, "validation": 668, "test": 684},
    "seed": 42,
    "herbert_available": False,
    "versions": {"python": "3.12.10", "scikit_learn": "1.9.0"},
}

SIGNIFICANCE = {
    "baseline": {
        "macro_f1": 0.944,
        "low": 0.926,
        "high": 0.961,
        "resamples": 2000,
        "confidence": 0.95,
    },
    "herbert": None,
    "mcnemar": None,
}

PROBE = {
    "base_cases": 80,
    "derived_cases": 158,
    "min_phenomenon_n": 20,
    "scores": [
        {"phenomenon": "plain", "correct": 13, "total": 20, "note": "control", "thin": False},
        {"phenomenon": "negation", "correct": 13, "total": 20, "note": "cues", "thin": False},
        {"phenomenon": "sarcasm", "correct": 11, "total": 20, "note": "irony", "thin": False},
        {"phenomenon": "contrast", "correct": 11, "total": 20, "note": "pivot", "thin": False},
        {"phenomenon": "typos", "correct": 47, "total": 80, "note": "derived", "thin": False},
    ],
    "misses_total": 96,
    "misses": [
        {
            "text": "Hotel leży dwa przystanki od dworca.",
            "phenomenon": "plain",
            "gold": "neutral",
            "predicted": "positive",
        }
    ],
}

DEFERRAL = {
    "expected_calibration_error": 0.209,
    "bins": [
        {"lower": 0.0, "upper": 0.1, "count": 0, "mean_confidence": 0.0, "accuracy": 0.0},
        {"lower": 0.7, "upper": 0.8, "count": 171, "mean_confidence": 0.75, "accuracy": 1.0},
    ],
    "risk_coverage": [
        {
            "deferral_rate": 0.0,
            "deferred": 0,
            "kept": 684,
            "threshold": 0.376,
            "macro_f1": 0.944,
            "accuracy": 0.94,
        },
        {
            "deferral_rate": 0.1,
            "deferred": 68,
            "kept": 616,
            "threshold": 0.519,
            "macro_f1": 0.974,
            "accuracy": 0.972,
        },
    ],
    "cascade": [],
}

CURVE = {
    "seeds": [0, 1, 2, 3, 4],
    "target_share": 0.99,
    "reaches_target_at": 4800,
    "points": [
        {"size": 150, "seeds": 5, "mean_macro_f1": 0.810, "low": 0.784, "high": 0.838},
        {"size": 1200, "seeds": 5, "mean_macro_f1": 0.925, "low": 0.918, "high": 0.928},
        {"size": 5264, "seeds": 1, "mean_macro_f1": 0.944, "low": 0.944, "high": 0.944},
    ],
}

TERMS = {
    "top_n": 2,
    "per_class": {
        "negative": [{"term": "nie polecam", "weight": 4.7}, {"term": "niestety", "weight": 3.8}],
        "neutral": [{"term": "prof", "weight": 2.9}],
        "positive": [{"term": "polecam", "weight": 4.1}],
    },
}

COST = {
    "train_seconds": 2.95,
    "train_rows": 5264,
    "predict_rows_per_second": 6548.3,
    "model_bytes": 3236052,
    "machine": "Intel64 Family 6",
    "device": "CPU",
    "python": "3.12.10",
}


def _metrics_dir(tmp_path: Path, **overrides) -> Path:
    """A metrics directory holding whatever this test wants the page to have been given."""
    files = {
        config.MANIFEST_PATH.name: MANIFEST,
        config.BASELINE_METRICS_PATH.name: BASELINE,
        config.SIGNIFICANCE_PATH.name: SIGNIFICANCE,
        config.CHALLENGE_PATH.name: PROBE,
        config.DEFERRAL_PATH.name: DEFERRAL,
        config.LEARNING_CURVE_PATH.name: CURVE,
        config.INTERPRETABILITY_PATH.name: TERMS,
        config.COST_PATH.name: COST,
    }
    files.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        if payload is None:
            continue
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_page_renders_every_section_from_the_committed_metrics(tmp_path):
    html = build.render(_metrics_dir(tmp_path))

    assert "<title>" in html
    assert "0.944" in html  # the headline score
    assert "0.926" in html and "0.961" in html  # its interval
    assert "48 / 80" in html  # the probe, base cells only
    assert "CC BY-NC-SA" in html  # dataset attribution
    assert "abc1234" in html  # provenance
    assert "data:image/png" not in html, "figures are SVG, so dark mode is not a second rendering"


def test_page_says_what_is_missing_instead_of_inventing_it(tmp_path):
    html = build.render(_metrics_dir(tmp_path))

    assert html.count("card pending") == 2, "the paired comparison and the cascade are waiting"
    assert "Waiting for a GPU run" in html
    assert "McNemar" in html, "the waiting panel still explains what will settle the question"


def test_page_leads_with_the_paired_test_once_herbert_has_run(tmp_path):
    significance = {
        "baseline": SIGNIFICANCE["baseline"],
        "herbert": {
            "macro_f1": 0.961,
            "low": 0.945,
            "high": 0.975,
            "resamples": 2000,
            "confidence": 0.95,
        },
        "mcnemar": {
            "only_baseline_correct": 8,
            "only_herbert_correct": 26,
            "discordant": 34,
            "p_value": 0.0031,
        },
    }
    html = build.render(
        _metrics_dir(tmp_path, **{config.SIGNIFICANCE_PATH.name: significance})
    )

    assert "0.961" in html
    assert "Waiting for a GPU run" not in html
    assert "p = 0.0031" in html


def test_page_reports_no_difference_when_the_paired_test_finds_none(tmp_path):
    significance = {
        "baseline": SIGNIFICANCE["baseline"],
        "herbert": {
            "macro_f1": 0.947,
            "low": 0.930,
            "high": 0.963,
            "resamples": 2000,
            "confidence": 0.95,
        },
        "mcnemar": {
            "only_baseline_correct": 14,
            "only_herbert_correct": 17,
            "discordant": 31,
            "p_value": 0.72,
        },
    }
    html = build.render(
        _metrics_dir(tmp_path, **{config.SIGNIFICANCE_PATH.name: significance})
    )

    assert "not distinguishable" in html


def test_headline_changes_when_the_probe_stops_being_the_finding(tmp_path):
    strong = dict(PROBE)
    strong["scores"] = [
        {"phenomenon": row["phenomenon"], "correct": row["total"], "total": row["total"],
         "note": row["note"], "thin": False}
        for row in PROBE["scores"]
    ]

    html = build.render(_metrics_dir(tmp_path, **{config.CHALLENGE_PATH.name: strong}))

    assert "does not survive" not in html


def test_page_renders_with_nothing_but_the_baseline_and_a_manifest(tmp_path):
    bare = _metrics_dir(
        tmp_path,
        **{
            config.SIGNIFICANCE_PATH.name: None,
            config.CHALLENGE_PATH.name: None,
            config.DEFERRAL_PATH.name: None,
            config.LEARNING_CURVE_PATH.name: None,
            config.INTERPRETABILITY_PATH.name: None,
            config.COST_PATH.name: None,
        },
    )
    for name in (
        config.SIGNIFICANCE_PATH.name,
        config.CHALLENGE_PATH.name,
        config.DEFERRAL_PATH.name,
        config.LEARNING_CURVE_PATH.name,
        config.INTERPRETABILITY_PATH.name,
        config.COST_PATH.name,
    ):
        (bare / name).unlink(missing_ok=True)

    html = build.render(bare)

    assert "0.944" in html
    assert html.count("card pending") >= 4
    assert "pl_review_sense.analysis" in html, "it says how to produce what is missing"


def test_build_refuses_a_manifest_it_does_not_have(tmp_path):
    (tmp_path / config.BASELINE_METRICS_PATH.name).write_text(json.dumps(BASELINE), "utf-8")
    with pytest.raises(FileNotFoundError):
        build.render(tmp_path)


def test_build_refuses_metrics_it_does_not_have(tmp_path):
    (tmp_path / config.MANIFEST_PATH.name).write_text(json.dumps(MANIFEST), "utf-8")
    with pytest.raises(FileNotFoundError):
        build.render(tmp_path)


def test_page_is_a_function_of_the_metrics(tmp_path):
    """Two builds of the same numbers are byte-identical — what the CI drift job checks."""
    metrics = _metrics_dir(tmp_path)
    assert build.render(metrics) == build.render(metrics)


def test_build_writes_lf_endings_so_the_drift_check_survives_windows(tmp_path):
    metrics = _metrics_dir(tmp_path / "metrics")
    target = build.build(out_dir=tmp_path / "out", metrics_dir=metrics)
    assert b"\r\n" not in target.read_bytes()


def test_a_figure_without_its_counts_is_not_publishable():
    page = {"charts": {"invented": "<svg><rect></rect></svg>"}}
    with pytest.raises(build.IncompleteFigure):
        build._assert_figures_carry_counts(page)


def test_every_chart_states_the_numbers_behind_its_marks():
    assert charts.carries_counts(
        charts.bar_chart([charts.Bar(label="nie", value=5.4)], "terms")
    )
    assert charts.carries_counts(
        charts.fraction_chart([charts.Fraction(label="plain", correct=13, total=20)], "probe")
    )
    assert charts.carries_counts(
        charts.confusion_chart([[3, 1], [0, 2]], ["a", "b"], "confusion")
    )
    assert charts.carries_counts(
        charts.curve_chart(
            [charts.CurvePoint(x=100, value=0.8, low=0.7, high=0.9)],
            "curve",
            x_ticks=[100],
            y_floor=0.5,
            x_caption="rows",
        )
    )
    assert charts.carries_counts(
        charts.reliability_chart(
            [charts.ReliabilityPoint(confidence=0.8, accuracy=0.9, count=12)], "calibration"
        )
    )


def test_empty_charts_say_so_rather_than_drawing_an_empty_frame():
    assert "<svg" not in charts.bar_chart([], "terms")
    assert "<svg" not in charts.fraction_chart([], "probe")
    assert "<svg" not in charts.curve_chart([], "curve", [], 0.5, "rows")
    assert "<svg" not in charts.reliability_chart([], "calibration")


def test_chart_labels_are_escaped():
    markup = charts.bar_chart([charts.Bar(label='<script>"x"', value=1.0)], "terms")
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
