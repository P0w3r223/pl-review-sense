"""Assemble the published page from the committed metrics.

The page is a *function of* ``reports/metrics/``. Nothing here measures anything, nothing
reads the dataset, and nothing consults the wall clock — the timestamp in the footer is the
one the analysis recorded. That is what lets CI rebuild the page from the committed numbers
and fail if ``docs/index.html`` disagrees with them, which is the failure mode a hand-edited
report actually has.

The page leads with a finding rather than with a summary of itself, and the finding is
derived: if the numbers stop supporting it, the headline changes with them instead of quietly
becoming false.

Panels whose evidence does not exist yet render as themselves, saying what is missing. A
section that disappears until its data arrives looks like a page that never had one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from pl_review_sense import challenge, config
from pl_review_sense.site import charts

TEMPLATE_DIR = Path(__file__).parent / "templates"
ASSET_DIR = Path(__file__).parent / "assets"

# The probe cells written by hand, as opposed to the ones derived by re-spelling them. Only
# these count towards the headline: the derived cells re-score the same sentences, and
# pooling all six would weight every sentence three times.
BASE_PHENOMENA = (challenge.PLAIN, challenge.NEGATION, challenge.SARCASM, challenge.CONTRAST)

# Below this share of its own probe the baseline is reported as not carrying over to short
# reviews. Set where "most of them" stops being a fair description of the successes.
PROBE_CARRIES_OVER = 0.80
# The deferral rate the KPI quotes. One of config.DEFERRAL_RATES, chosen as the smallest
# that is a plausible operating point rather than the flattering largest.
HEADLINE_DEFERRAL = 0.10
# How close to the full-corpus score counts as "the rest is not worth labelling", in macro-F1
# points. Two points is roughly the width of the interval around the score itself.
WITHIN_DELTA = 0.02
# How far apart the shortest and longest length segments have to be before the page calls the
# difference a direction rather than noise. Same order as the interval around the score itself.
LENGTH_TREND_DELTA = 0.02
# How much of the gap between the two models a cascade has to close before the page quotes that
# operating point. Naming the rule rather than the rate is what keeps it from being a cherry-pick.
CASCADE_GAIN_TARGET = 0.80


def _p_text(value: float) -> str:
    """A p-value a reader can act on.

    ``%.4f`` renders 3.1e-06 as "0.0000", which reads as exactly zero — a claim no test makes.
    Below the smallest value four decimals can show, the honest rendering is the bound.
    """
    return "< 0.0001" if value < 0.0001 else f"= {value:.4f}"


class IncompleteFigure(RuntimeError):
    """Raised when a figure would be published without the counts behind it."""


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    note: str


def _read(metrics_dir: Path, name: str) -> Optional[dict]:
    path = metrics_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _probe_totals(probe: Optional[dict]) -> tuple[int, int]:
    """Correct and total over the hand-written cells only."""
    if not probe:
        return 0, 0
    rows = [row for row in probe["scores"] if row["phenomenon"] in BASE_PHENOMENA]
    return sum(row["correct"] for row in rows), sum(row["total"] for row in rows)


def _headline(baseline_metrics: dict, probe: Optional[dict], significance: Optional[dict]) -> dict:
    """The claim the page leads with, derived from the numbers rather than asserted.

    Three states, in order of what the evidence supports. Once HerBERT has run, the paired
    test is the strongest thing on the page and leads. Until then the probe is: a corpus
    score that does not survive a short sentence is a more useful finding than the score
    itself. With neither, the page falls back to reporting the score and its interval.
    """
    macro = baseline_metrics["macro_f1"]
    test = (significance or {}).get("mcnemar")
    herbert = (significance or {}).get("herbert")

    if test and herbert:
        if test["p_value"] >= config.SIGNIFICANCE_LEVEL:
            return {
                "claim": "The transformer is not distinguishable from a bag of words here",
                "detail": (
                    f"HerBERT is right on {test['only_herbert_correct']} reviews the baseline "
                    f"misses and wrong on {test['only_baseline_correct']} it gets — a difference "
                    f"this test set cannot separate from chance (p {_p_text(test['p_value'])})."
                ),
            }
        # Which way the significant difference runs is read from the numbers, not assumed. This
        # string is also the <title> and the <h1>, and it fires exactly once — on the day the
        # GPU run lands. A headline that reports a loss as a win would do it at the worst
        # possible moment, and the page's whole claim is that its lead is derived.
        if herbert["macro_f1"] < macro:
            return {
                "claim": f"Fine-tuning HerBERT scored below the baseline, {herbert['macro_f1']:.3f} to {macro:.3f}",
                "detail": (
                    f"The baseline is right on {test['only_baseline_correct']} reviews HerBERT "
                    f"misses and wrong on {test['only_herbert_correct']} it gets; across "
                    f"{test['discordant']} disagreements that gap is real (p "
                    f"{_p_text(test['p_value'])}), not a rounding artefact of one test split."
                ),
            }
        return {
            "claim": f"HerBERT reaches {herbert['macro_f1']:.3f} against the baseline's {macro:.3f}",
            "detail": (
                f"It is right on {test['only_herbert_correct']} reviews the baseline misses and "
                f"wrong on {test['only_baseline_correct']} it gets; on {test['discordant']} "
                f"disagreements that is a real difference (p {_p_text(test['p_value'])}). What it "
                "costs is further down the page."
            ),
        }

    correct, total = _probe_totals(probe)
    if total and correct / total < PROBE_CARRIES_OVER:
        return {
            "claim": f"A macro-F1 of {macro:.2f} that does not survive a short review",
            "detail": (
                f"The same model answers {correct} of {total} one-sentence reviews written for "
                "this page — sentences with no rare vocabulary, in the domains it was trained "
                "on. The score belongs to the corpus, not to the task its name describes."
            ),
        }
    return {
        "claim": f"A bag of words reaches {macro:.3f} macro-F1 on Polish reviews",
        "detail": (
            "What a fine-tuned transformer adds to that is the open question this page exists "
            "to settle, and it is not settled yet."
        ),
    }


def _cascade_highlight(
    deferral: Optional[dict], significance: Optional[dict]
) -> Optional[dict]:
    """The cheapest cascade operating point that recovers most of what the transformer adds.

    Quoting a rate would be cherry-picking; the rule is stated instead — the smallest share of
    traffic sent to the GPU that closes at least ``CASCADE_GAIN_TARGET`` of the gap between the
    two models. If no rate reaches it, nothing is highlighted rather than the best of a bad set.
    """
    if not deferral or not deferral.get("cascade") or not significance:
        return None
    baseline = (significance.get("baseline") or {}).get("macro_f1")
    herbert = (significance.get("herbert") or {}).get("macro_f1")
    if baseline is None or herbert is None or herbert <= baseline:
        return None

    gap = herbert - baseline
    for point in deferral["cascade"]:
        if not point["deferral_rate"]:
            continue
        recovered = (point["macro_f1"] - baseline) / gap
        if recovered >= CASCADE_GAIN_TARGET:
            return {**point, "recovered": recovered}
    return None


def _kpis(
    baseline_metrics: dict,
    significance: Optional[dict],
    probe: Optional[dict],
    deferral: Optional[dict],
    cost: Optional[dict],
    herbert_metrics: Optional[dict] = None,
) -> List[Kpi]:
    interval = (significance or {}).get("baseline")
    herbert = (significance or {}).get("herbert")
    test = (significance or {}).get("mcnemar")

    # Once the transformer has run, it is the finding, and the four numbers at the top of the
    # page have to be the four the reader now needs: what it scores, whether the difference is
    # real, what a cascade buys, and what the whole thing cost.
    if herbert and test:
        highlight = _cascade_highlight(deferral, significance)
        test_rows = sum(row["support"] for row in baseline_metrics["per_class"])
        return [
            Kpi(
                label="Macro-F1, HerBERT",
                value=f"{herbert['macro_f1']:.3f}",
                note=(
                    f"{herbert['confidence']:.0%} interval {herbert['low']:.3f}–"
                    f"{herbert['high']:.3f}, against {baseline_metrics['macro_f1']:.3f} for the "
                    "baseline"
                ),
            ),
            Kpi(
                label="Reviews it settles either way",
                value=f"{test['only_herbert_correct']} / {test['only_baseline_correct']}",
                note=(
                    f"it is right where the baseline is wrong / wrong where the baseline is "
                    f"right, over {test['discordant']} disagreements — p {_p_text(test['p_value'])}"
                ),
            ),
            Kpi(
                label=(
                    f"Cascade at {highlight['deferral_rate']:.0%} on the GPU"
                    if highlight
                    else "Cascade"
                ),
                value=f"{highlight['macro_f1']:.3f}" if highlight else "—",
                note=(
                    f"{highlight['recovered']:.0%} of what the transformer adds, from "
                    f"{highlight['deferred']} of {test_rows} reviews"
                    if highlight
                    else "no rate recovers most of the gap"
                ),
            ),
            Kpi(
                label="What the transformer cost",
                value=(
                    f"{herbert_metrics['train_seconds'] / 60:.0f} min"
                    if herbert_metrics and herbert_metrics.get("train_seconds")
                    else "—"
                ),
                note=(
                    f"on a {herbert_metrics['device']}, against "
                    f"{cost['train_seconds']:.0f} s on a CPU"
                    if herbert_metrics and herbert_metrics.get("device") and cost
                    else "not measured"
                ),
            ),
        ]

    kpis = [
        Kpi(
            label="Macro-F1, baseline",
            value=f"{baseline_metrics['macro_f1']:.3f}",
            note=(
                f"{interval['confidence']:.0%} interval {interval['low']:.3f}–{interval['high']:.3f}"
                f" on {sum(row['support'] for row in baseline_metrics['per_class'])} test reviews"
                if interval
                else "no interval computed yet"
            ),
        )
    ]

    correct, total = _probe_totals(probe)
    kpis.append(
        Kpi(
            label="Own probe sentences",
            value=f"{correct} / {total}" if total else "—",
            note=(
                "short reviews we wrote: a control cell, negation, irony and a pivot"
                if total
                else "the probe has not been scored"
            ),
        )
    )

    kept = _deferral_point(deferral, HEADLINE_DEFERRAL)
    kpis.append(
        Kpi(
            label=f"Macro-F1 on the {1 - HEADLINE_DEFERRAL:.0%} it keeps",
            value=f"{kept['macro_f1']:.3f}" if kept else "—",
            note=(
                f"when the least confident {kept['deferred']} reviews are handed on"
                if kept
                else "no deferral curve computed yet"
            ),
        )
    )

    kpis.append(
        Kpi(
            label="To train, on a CPU",
            value=f"{cost['train_seconds']:.0f} s" if cost else "—",
            note=(
                f"{cost['model_bytes'] / 1e6:.1f} MB model, "
                f"{cost['predict_rows_per_second']:,.0f} reviews/s".replace(",", " ")
                if cost and cost.get("model_bytes")
                else "not measured on this machine"
            ),
        )
    )
    return kpis


def _deferral_point(deferral: Optional[dict], rate: float) -> Optional[dict]:
    if not deferral:
        return None
    for point in deferral["risk_coverage"]:
        if abs(point["deferral_rate"] - rate) < 1e-9:
            return point
    return None


def _calibration_direction(deferral: Optional[dict]) -> Optional[str]:
    """Which way the model's confidence is wrong, weighted by how many rows sit in each bin.

    Derived rather than written into the prose: an under-confident model and an over-confident
    one call for opposite advice, and a page that hard-codes one of them is wrong the day the
    model changes.
    """
    if not deferral:
        return None
    populated = [item for item in deferral["bins"] if item["count"]]
    if not populated:
        return None
    rows = sum(item["count"] for item in populated)
    gap = sum(item["count"] * (item["accuracy"] - item["mean_confidence"]) for item in populated)
    average = gap / rows
    if average > config.CALIBRATION_TOLERANCE:
        return "under"
    if average < -config.CALIBRATION_TOLERANCE:
        return "over"
    return "close"


def _cost_ratio(cost: Optional[dict], herbert_metrics: Optional[dict]) -> Optional[float]:
    """How many times longer the transformer took to train than the bag of words.

    The point of the whole page is accuracy *and* what it costs, and until a GPU run existed
    the second half was a description. It is a ratio of wall-clock on two different machines,
    which is exactly what someone deciding between them is comparing — the page names both.
    """
    if not cost or not herbert_metrics:
        return None
    theirs = herbert_metrics.get("train_seconds")
    ours = cost.get("train_seconds")
    if not theirs or not ours:
        return None
    return theirs / ours


def _length_reading(length: Optional[dict], probe: Optional[dict]) -> Optional[str]:
    """Whether the corpus backs the page's own headline, in one word the template branches on.

    The claim under test is that the score belongs to long reviews. ``corroborates`` when the
    score climbs with length, ``contradicts`` when it does not, and ``flat`` when the two ends
    are within the noise. Derived rather than written: a section whose prose asserts the
    outcome it is supposed to be testing would be worth nothing.
    """
    if not length or not length["segments"]:
        return None
    # The prior question, and usually the answer: a corpus that contains almost nothing as
    # short as the probe cannot settle a claim about short text in either direction. Reporting
    # a trend across the lengths it does hold would answer a different question quietly.
    if length.get("rows_at_probe_scale", 0) < length["min_segment_n"]:
        return "unanswerable"
    if length["trend"] is None:
        return None
    correct, total = _probe_totals(probe)
    probe_fell = bool(total) and correct / total < PROBE_CARRIES_OVER
    if length["trend"] > LENGTH_TREND_DELTA:
        return "corroborates" if probe_fell else "climbs"
    if length["trend"] < -LENGTH_TREND_DELTA:
        return "contradicts"
    return "flat"


def _within(curve: Optional[dict], delta: float) -> Optional[dict]:
    """The smallest training size already within ``delta`` macro-F1 of the full corpus.

    The interesting number on a learning curve is rarely where it converges — it is where the
    remaining gap stops being worth the labelling.
    """
    if not curve or not curve["points"]:
        return None
    full = curve["points"][-1]["mean_macro_f1"]
    for point in curve["points"]:
        if full - point["mean_macro_f1"] <= delta:
            return {
                "size": point["size"],
                "macro_f1": point["mean_macro_f1"],
                "gap": full - point["mean_macro_f1"],
                "share_of_corpus": point["size"] / curve["points"][-1]["size"],
            }
    return None


def _settings() -> Dict[str, str]:
    """The configuration behind every number, read out of ``config`` rather than described.

    The analogue of publishing the queries: a reader should be able to check the setting a
    figure rests on without taking a paragraph's word for it, and a value that is printed
    from the constant cannot drift away from the code the way a prose description does.
    """
    return {
        "dataset": (
            f"name        {config.DATASET}\n"
            f"config      {config.DATASET_CONFIG}\n"
            f"revision    {config.DATASET_REVISION}\n"
            f"dropped     {config.DROP_LABEL_NAME}  (ambiguous — never merged into another class)\n"
            f"labels      {', '.join(config.LABEL_NAMES)}\n"
            f"license     CC BY-NC-SA 4.0 — downloaded on demand, never redistributed here"
        ),
        "baseline": (
            f"tfidf       ngram_range={config.TFIDF_NGRAM_RANGE}  max_features="
            f"{config.TFIDF_MAX_FEATURES}\n"
            f"            min_df={config.TFIDF_MIN_DF}  sublinear_tf={config.TFIDF_SUBLINEAR_TF}\n"
            f"logreg      C={config.LOGREG_C}  max_iter={config.LOGREG_MAX_ITER}  "
            f"class_weight={config.CLASS_WEIGHT!r}\n"
            f"seed        {config.RANDOM_STATE}\n"
            "fitted inside one Pipeline, so the vectorizer never sees the test split"
        ),
        "herbert": (
            f"model       {config.HERBERT_MODEL}\n"
            f"epochs      {config.HERBERT_EPOCHS}  lr={config.HERBERT_LR}  "
            f"weight_decay={config.HERBERT_WEIGHT_DECAY}\n"
            f"batch       {config.HERBERT_BATCH_SIZE} x {config.HERBERT_GRAD_ACCUM} accumulated "
            f"= {config.HERBERT_BATCH_SIZE * config.HERBERT_GRAD_ACCUM} per optimizer step\n"
            f"max_len     {config.HERBERT_MAX_LEN}\n"
            f"seed        {config.RANDOM_STATE}\n"
            "the split batch is a memory accommodation, not a different hyperparameter: what\n"
            "reaches the optimizer is the same 16 examples a larger card would pass in one go"
        ),
        "uncertainty": (
            f"bootstrap   {config.BOOTSTRAP_RESAMPLES} resamples of the test rows, "
            f"{config.CONFIDENCE_LEVEL:.0%} percentile interval\n"
            f"paired test exact McNemar on the reviews the two models answer differently\n"
            f"seed        {config.RANDOM_STATE} — the interval is reproducible from the "
            "committed predictions"
        ),
        "learning-curve": (
            f"sizes       {', '.join(str(size) for size in config.LEARNING_CURVE_SIZES)}, "
            "then the full corpus\n"
            f"seeds       {', '.join(str(seed) for seed in config.LEARNING_CURVE_SEEDS)} per size\n"
            "sampling    stratified — an unstratified draw of 150 rows can miss the neutral "
            "class outright"
        ),
        "deferral": (
            f"rates       {', '.join(f'{rate:.0%}' for rate in config.DEFERRAL_RATES)} of the "
            "test set, least confident first\n"
            f"calibration {config.CALIBRATION_BINS} equal-width confidence bins\n"
            f"probe floor {config.MIN_PHENOMENON_N} cases before a cell is read as a rate"
        ),
        "commands": (
            "python -m pl_review_sense.baseline_train   # trains, writes metrics + predictions\n"
            "python -m pl_review_sense.analysis         # intervals, curve, probe, deferral\n"
            "python -m pl_review_sense.site             # rebuilds this page into docs/\n"
            "\n"
            "python -m pl_review_sense.herbert          # the GPU run behind the comparison\n"
            "notebooks/herbert_colab.ipynb              # the same run on a free Colab card"
        ),
    }


def gather(metrics_dir: Optional[Path] = None) -> dict:
    """Everything the template needs, read from the committed metrics."""
    metrics_dir = Path(metrics_dir or config.METRICS_DIR)

    manifest = _read(metrics_dir, config.MANIFEST_PATH.name)
    if manifest is None:
        raise FileNotFoundError(
            f"no manifest in {metrics_dir}; run `python -m pl_review_sense.analysis` first"
        )
    baseline_metrics = _read(metrics_dir, config.BASELINE_METRICS_PATH.name)
    if baseline_metrics is None:
        raise FileNotFoundError(
            f"no baseline metrics in {metrics_dir}; run "
            "`python -m pl_review_sense.baseline_train` first"
        )

    significance = _read(metrics_dir, config.SIGNIFICANCE_PATH.name)
    probe = _read(metrics_dir, config.CHALLENGE_PATH.name)
    deferral = _read(metrics_dir, config.DEFERRAL_PATH.name)
    curve = _read(metrics_dir, config.LEARNING_CURVE_PATH.name)
    length = _read(metrics_dir, config.SEGMENTS_PATH.name)
    terms = _read(metrics_dir, config.INTERPRETABILITY_PATH.name)
    cost = _read(metrics_dir, config.COST_PATH.name)
    herbert_metrics = _read(metrics_dir, config.HERBERT_METRICS_PATH.name)

    labels = list(baseline_metrics.get("labels", config.LABEL_NAMES))
    rendered = {
        "confusion": charts.confusion_chart(
            baseline_metrics["confusion"], labels, "Confusion matrix, TF-IDF baseline"
        )
    }

    if curve and curve["points"]:
        rendered["curve"] = charts.curve_chart(
            [
                charts.CurvePoint(
                    x=point["size"],
                    value=point["mean_macro_f1"],
                    low=point["low"],
                    high=point["high"],
                )
                for point in curve["points"]
            ],
            "Macro-F1 against training size",
            x_ticks=[point["size"] for point in curve["points"]],
            y_floor=_floor(min(point["low"] for point in curve["points"])),
            x_caption=(
                f"labelled reviews, log scale · band = spread over "
                f"{len(curve['seeds'])} stratified draws per size"
            ),
        )

    if length and length["segments"]:
        rendered["length"] = charts.bar_chart(
            [
                charts.Bar(
                    label=f"{segment['name']} words",
                    value=segment["macro_f1"],
                    value_text=f"{segment['macro_f1']:.3f}",
                    note=f"n={segment['n']}",
                    muted=segment["thin"],
                )
                for segment in length["segments"]
            ],
            "Macro-F1 by review length",
            unit="macro-F1",
        )

    if probe:
        rendered["probe"] = charts.fraction_chart(
            [
                # The cell's description travels to the table beside the chart rather than onto
                # the bar: "unambiguous sentiment, no negation, irony or pivot" is a definition,
                # and a definition printed at the end of a bar is a caption in the wrong place.
                charts.Fraction(
                    label=row["phenomenon"],
                    correct=row["correct"],
                    total=row["total"],
                    muted=row["thin"],
                )
                for row in probe["scores"]
            ],
            "Challenge set, answered correctly",
        )

    if deferral:
        rendered["reliability"] = charts.reliability_chart(
            [
                charts.ReliabilityPoint(
                    confidence=item["mean_confidence"],
                    accuracy=item["accuracy"],
                    count=item["count"],
                )
                for item in deferral["bins"]
                if item["count"]
            ],
            "Confidence against accuracy",
        )

    if terms:
        for label in labels:
            rendered[f"terms_{label}"] = charts.bar_chart(
                [
                    charts.Bar(
                        label=item["term"],
                        value=item["weight"],
                        value_text=f"{item['weight']:.2f}",
                    )
                    for item in terms["per_class"].get(label, [])
                ],
                f"Heaviest terms for {label}",
                # Narrow: these three sit side by side in a column each.
                width=330,
                label_width=118,
            )

    correct, total = _probe_totals(probe)
    return {
        "manifest": manifest,
        "generated": manifest["generated_at"],
        "headline": _headline(baseline_metrics, probe, significance),
        "kpis": _kpis(
            baseline_metrics, significance, probe, deferral, cost, herbert_metrics
        ),
        "cascade_highlight": _cascade_highlight(deferral, significance),
        "cascade_gain_target": CASCADE_GAIN_TARGET,
        "mcnemar_p_text": (
            _p_text((significance or {}).get("mcnemar", {}).get("p_value", 1.0))
            if (significance or {}).get("mcnemar")
            else None
        ),
        "baseline": baseline_metrics,
        "labels": labels,
        "interval": (significance or {}).get("baseline"),
        "floors": (significance or {}).get("floors"),
        "length": length,
        "length_reading": _length_reading(length, probe),
        "mcnemar": (significance or {}).get("mcnemar"),
        "herbert": (significance or {}).get("herbert"),
        "herbert_metrics": herbert_metrics,
        "curve": curve,
        "probe": probe,
        "probe_correct": correct,
        "probe_total": total,
        "deferral": deferral,
        "headline_deferral": HEADLINE_DEFERRAL,
        "headline_deferral_point": _deferral_point(deferral, HEADLINE_DEFERRAL),
        "calibration_direction": _calibration_direction(deferral),
        "curve_within": _within(curve, WITHIN_DELTA),
        "within_delta": WITHIN_DELTA,
        "terms": terms,
        "cost": cost,
        "cost_ratio": _cost_ratio(cost, herbert_metrics),
        "smoke_subset": config.SMOKE_SUBSET,
        "settings": _settings(),
        "charts": {name: Markup(markup) for name, markup in rendered.items()},
    }


def _floor(value: float) -> float:
    """Round a score down to the nearest 0.05, for an axis that starts on a readable number."""
    return max(0.0, int(value * 20) / 20)


def _assert_figures_carry_counts(page: dict) -> None:
    for name, markup in page["charts"].items():
        if not charts.carries_counts(str(markup)):
            raise IncompleteFigure(f"chart {name!r} would publish marks without their counts")


def render(metrics_dir: Optional[Path] = None) -> str:
    page = gather(metrics_dir)
    _assert_figures_carry_counts(page)
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("index.html.j2")
    return template.render(
        styles=Markup((ASSET_DIR / "styles.css").read_text(encoding="utf-8")), **page
    )


def build(out_dir: Optional[Path] = None, metrics_dir: Optional[Path] = None) -> Path:
    """Render the page and write it where GitHub Pages serves from."""
    out_dir = Path(out_dir or config.PUBLISH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    # Explicit LF, not the platform's line ending: this file is committed and CI rebuilds it
    # on another OS, so "the page is a function of the metrics" has to hold across both.
    with open(target, "w", encoding="utf-8", newline="\n") as page:
        page.write(render(metrics_dir))
    return target
