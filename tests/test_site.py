"""The published page.

Three things are worth pinning: that a panel without evidence says so instead of inventing a
number, that a figure never goes out without the counts behind it, and that the page is a
function of the committed metrics — the last is what the CI drift job rests on.
"""

from __future__ import annotations

import json
import re
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


def test_page_does_not_report_a_loss_as_a_win(tmp_path):
    """A significant difference the wrong way round is still a significant difference."""
    significance = {
        "baseline": SIGNIFICANCE["baseline"],
        "herbert": {
            "macro_f1": 0.902,
            "low": 0.881,
            "high": 0.922,
            "resamples": 2000,
            "confidence": 0.95,
        },
        "mcnemar": {
            "only_baseline_correct": 31,
            "only_herbert_correct": 9,
            "discordant": 40,
            "p_value": 0.0004,
        },
    }
    html = build.render(
        _metrics_dir(tmp_path, **{config.SIGNIFICANCE_PATH.name: significance})
    )

    assert "HerBERT reaches 0.902 against" not in html, "0.902 is not a win over 0.944"
    assert "below the baseline" in html
    assert "not distinguishable" not in html, "p = 0.0004 is a difference, just not a gain"


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


@pytest.mark.parametrize(
    "accuracy, phrase",
    [
        (1.0, "<em>under</em>-confident"),
        (0.50, "<em>over</em>-confident"),
        (0.76, "track the diagonal closely"),
    ],
)
def test_the_page_says_which_way_the_confidence_is_wrong(tmp_path, accuracy, phrase):
    """Under- and over-confidence call for opposite advice, so the prose is derived, not written."""
    deferral = dict(DEFERRAL)
    deferral["bins"] = [
        {"lower": 0.7, "upper": 0.8, "count": 171, "mean_confidence": 0.75, "accuracy": accuracy}
    ]

    html = build.render(_metrics_dir(tmp_path, **{config.DEFERRAL_PATH.name: deferral}))

    assert phrase in html


def test_the_page_names_the_training_size_that_is_already_close_enough(tmp_path):
    """1200 rows land 0.019 short of the full corpus — inside the two points that count as close."""
    html = build.render(_metrics_dir(tmp_path))

    assert "1200 labelled reviews" in html
    assert "23% of the corpus" in html  # 1200 of 5264
    assert "0.019 macro-F1" in html


def test_the_kpi_row_quotes_the_deferral_point_the_page_leads_with(tmp_path):
    html = build.render(_metrics_dir(tmp_path))

    assert "Macro-F1 on the 90% it keeps" in html
    assert "0.974" in html, "the score at the 10% deferral row, not another rate's"
    assert "least confident 68 reviews" in html


def test_a_deferral_rate_the_curve_does_not_contain_is_not_invented(tmp_path):
    thinned = dict(DEFERRAL)
    thinned["risk_coverage"] = [DEFERRAL["risk_coverage"][0]]  # only the no-deferral reference

    html = build.render(_metrics_dir(tmp_path, **{config.DEFERRAL_PATH.name: thinned}))

    assert "no deferral curve computed yet" in html


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


def test_a_figure_that_drew_nothing_is_publishable_because_it_says_so():
    """The opposite failure: an unlabelled chart must stop the build, an empty panel must not."""
    page = {"charts": {"nothing": charts.bar_chart([], "terms")}}
    build._assert_figures_carry_counts(page)


@pytest.mark.parametrize(
    "name, payload, marker",
    [
        (
            config.INTERPRETABILITY_PATH.name,
            {"top_n": 2, "per_class": {"negative": [], "neutral": [], "positive": []}},
            "Nothing to plot.",
        ),
        (
            config.CHALLENGE_PATH.name,
            {**PROBE, "scores": [], "misses": [], "misses_total": 0},
            "The probe has not been scored.",
        ),
        (
            config.LEARNING_CURVE_PATH.name,
            {**CURVE, "points": [], "reaches_target_at": None},
            "Not computed yet.",
        ),
        (
            config.DEFERRAL_PATH.name,
            {
                **DEFERRAL,
                "bins": [
                    {"lower": 0.0, "upper": 0.1, "count": 0, "mean_confidence": 0.0, "accuracy": 0.0}
                ],
            },
            "No calibration bins in the committed metrics.",
        ),
    ],
)
def test_a_panel_whose_file_arrived_empty_says_so_rather_than_failing_the_build(
    tmp_path, name, payload, marker
):
    """A measurement that produced no rows is a state to render, not a reason to stop."""
    html = build.render(_metrics_dir(tmp_path, **{name: payload}))

    assert marker in html


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


def test_confusion_cells_are_shaded_and_read_out_as_a_share_of_their_own_row():
    """Row-normalized: a matrix shaded by column would colour the largest class darkest."""
    markup = charts.confusion_chart([[3, 1], [0, 2]], ["negative", "positive"], "confusion")

    assert "75% of row" in markup and "25% of row" in markup  # row 0 holds four reviews
    assert "0% of row" in markup and "100% of row" in markup  # row 1 holds two
    assert 'fill-opacity="0.750"' in markup, "the shading is the same share as the text"


def test_confusion_shares_would_differ_if_the_matrix_were_read_by_column():
    """The guard on the row/column mix-up: this matrix normalizes differently each way."""
    markup = charts.confusion_chart([[3, 1], [0, 2]], ["negative", "positive"], "confusion")

    assert "33% of row" not in markup, "3 of 3 down the column is not 3 of 4 across the row"


def test_a_fraction_prints_its_count_before_its_percentage():
    markup = charts.fraction_chart(
        [charts.Fraction(label="plain", correct=13, total=20)], "probe"
    )
    assert "13 / 20 (65%)" in markup


def test_a_thin_cell_is_drawn_muted_so_it_is_not_read_as_a_rate():
    markup = charts.fraction_chart(
        [charts.Fraction(label="sarcasm", correct=4, total=8, muted=True)], "probe"
    )
    assert 'class="fill muted"' in markup


def test_a_cell_answered_correctly_every_time_fills_its_own_track_and_no_more():
    """The track is the cell's size, so a small cell cannot draw a bar longer than a big one."""
    markup = charts.fraction_chart(
        [
            charts.Fraction(label="small", correct=10, total=10),
            charts.Fraction(label="large", correct=40, total=80),
        ],
        "probe",
    )
    small_track, small_fill, large_track, large_fill = _rect_widths(markup)
    assert small_fill == pytest.approx(small_track), "ten of ten fills its track exactly"
    assert large_track > small_track, "the wider track is the cell holding more cases"
    assert large_fill == pytest.approx(large_track / 2)


def _rect_widths(markup: str) -> list[float]:
    """The width of every drawn rectangle, in the order the chart emitted them."""
    return [float(width) for width in re.findall(r"<rect[^>]*width=\"([\d.]+)\"", markup)]


def test_the_curve_marks_its_own_floor_and_the_last_point_it_plotted():
    markup = charts.curve_chart(
        [
            charts.CurvePoint(x=150, value=0.81, low=0.78, high=0.84),
            charts.CurvePoint(x=5264, value=0.944, low=0.944, high=0.944),
        ],
        "curve",
        x_ticks=[150, 5264],
        y_floor=0.75,
        x_caption="labelled reviews",
    )

    assert "0.75" in markup, "the cropped axis states where it starts"
    assert "0.944 at n=5 264" in markup
    assert markup.count('class="series-dot"') == 2


def test_the_curve_drops_ticks_that_would_be_drawn_over_each_other():
    markup = charts.curve_chart(
        [charts.CurvePoint(x=x, value=0.9, low=0.9, high=0.9) for x in (150, 4800, 5264)],
        "curve",
        x_ticks=[150, 4800, 5264],
        y_floor=0.5,
        x_caption="labelled reviews",
    )
    assert '>150</text>' in markup and ">4 800</text>" in markup
    assert (
        'text-anchor="middle">5 264</text>' not in markup
    ), "on a log axis the last two sizes sit a few pixels apart; two numbers there read as neither"
    assert "at n=5 264" in markup, "the point itself is still labelled"


def test_the_command_line_builds_the_page_where_it_is_told_to(tmp_path):
    """``--out``/``--metrics`` are what the CI drift job builds into a scratch directory with."""
    from pl_review_sense.site import __main__ as entry

    metrics = _metrics_dir(tmp_path / "metrics")
    out = tmp_path / "scratch"

    entry.main(["--out", str(out), "--metrics", str(metrics)])

    assert (out / "index.html").read_text(encoding="utf-8").count("0.944") >= 1


def test_the_page_renders_when_the_model_file_was_not_kept(tmp_path):
    """``analysis.cost`` writes a null model size when ``models/`` is gone; the page still builds."""
    unsaved = dict(COST, model_bytes=None)

    html = build.render(_metrics_dir(tmp_path, **{config.COST_PATH.name: unsaved}))

    assert "not measured on this machine" in html
    assert "the saved model is" not in html, "no size sentence without a size"
    assert "6548" in html, "the timings it did measure are still reported"
