"""The runner that produces every published number.

``analysis`` is where the pure measurements meet the disk, and the shape it writes there is
the contract the page reads back. Two kinds of test therefore live here: the writers, each
checked against the module it wraps, and one round trip that produces the metrics with these
functions and renders the page from them — the only thing that notices when a key is renamed
on one side of ``reports/metrics/`` and not the other.

Nothing here downloads PolEmo or loads a saved model: the corpus is the challenge set, which
is ours and ships with the package, and every fit is on tens of rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pl_review_sense import (
    analysis,
    baseline,
    baseline_train,
    challenge,
    challenge_set,
    config,
    evaluate,
    stats,
)
from pl_review_sense.data import Dataset, Split
from pl_review_sense.site import build

# Ten rows whose errors sit exactly where the model is least sure — the arrangement every
# deferral figure is supposed to be able to see.
TRUE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
PRED = [1, 2, 0, 0, 1, 2, 0, 1, 2, 0]
CONFIDENCE = [0.40, 0.45, 0.50, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]


def _rows(true, pred, confidences) -> dict:
    """A predictions payload in the columnar shape ``baseline_train`` writes."""
    proba = []
    for label, confidence in zip(pred, confidences):
        row = [(1.0 - confidence) / 2] * len(config.LABEL_NAMES)
        row[label] = confidence
        proba.append(row)
    return {
        "model": "tfidf+logreg",
        "labels": list(config.LABEL_NAMES),
        "true": list(true),
        "pred": list(pred),
        "proba": proba,
    }


def _corpus(repeats: int = 1) -> tuple[list[str], list[int]]:
    texts = [case.text for case in challenge_set.CASES]
    labels = [case.label for case in challenge_set.CASES]
    return texts * repeats, labels * repeats


def _dataset() -> Dataset:
    """A three-class dataset small enough to fit in milliseconds, built from our own probe."""
    texts, labels = _corpus()
    train = Split(texts[:60], labels[:60])
    held_out = Split(texts[60:], labels[60:])
    return Dataset(train=train, validation=held_out, test=held_out)


# --- reading what baseline_train wrote ----------------------------------------------------


def test_missing_predictions_are_absent_rather_than_an_error(tmp_path):
    assert analysis._load_predictions(tmp_path / "nothing.json") is None


def test_predictions_whose_columns_disagree_are_refused(tmp_path):
    payload = _rows(TRUE, PRED, CONFIDENCE)
    payload["pred"] = payload["pred"][:-1]
    path = tmp_path / "baseline_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differing length"):
        analysis._load_predictions(path)


@pytest.mark.parametrize("dropped", ["true", "pred"])
def test_predictions_missing_a_column_are_named_rather_than_failing_later(tmp_path, dropped):
    payload = _rows(TRUE, PRED, CONFIDENCE)
    del payload[dropped]
    path = tmp_path / "baseline_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"no {dropped} column"):
        analysis._load_predictions(path)


def test_a_well_formed_predictions_file_is_read_back_whole(tmp_path):
    path = tmp_path / "baseline_test.json"
    path.write_text(json.dumps(_rows(TRUE, PRED, CONFIDENCE)), encoding="utf-8")

    payload = analysis._load_predictions(path)

    assert payload["true"] == TRUE
    assert len(payload["proba"]) == len(TRUE)


# --- significance -------------------------------------------------------------------------


def test_significance_reports_the_interval_and_nothing_it_cannot_compare(tmp_path):
    payload = analysis.significance(_rows(TRUE, PRED, CONFIDENCE), None, TRUE)

    interval = payload["baseline"]
    assert interval["low"] <= interval["macro_f1"] <= interval["high"]
    assert interval["resamples"] == config.BOOTSTRAP_RESAMPLES
    assert payload["herbert"] is None and payload["mcnemar"] is None


def test_significance_pairs_the_two_models_row_by_row():
    baseline_rows = _rows(TRUE, PRED, CONFIDENCE)
    herbert_rows = _rows(TRUE, TRUE, CONFIDENCE)  # right about everything

    payload = analysis.significance(baseline_rows, herbert_rows, TRUE)

    expected = stats.mcnemar(TRUE, PRED, TRUE)
    assert payload["mcnemar"]["only_herbert_correct"] == expected.only_b_correct
    assert payload["mcnemar"]["only_baseline_correct"] == expected.only_a_correct
    assert payload["mcnemar"]["discordant"] == expected.discordant
    assert payload["herbert"]["macro_f1"] == pytest.approx(1.0)


def test_significance_refuses_two_files_describing_different_test_rows():
    baseline_rows = _rows(TRUE, PRED, CONFIDENCE)
    shifted = list(TRUE[1:]) + [TRUE[0]]
    herbert_rows = _rows(shifted, shifted, CONFIDENCE)

    with pytest.raises(ValueError, match="different test rows"):
        analysis.significance(baseline_rows, herbert_rows, TRUE)


# --- deferral -----------------------------------------------------------------------------


def test_deferral_reads_the_confidence_out_of_the_probabilities():
    payload = analysis.deferral(_rows(TRUE, PRED, CONFIDENCE), None)

    assert [point["deferral_rate"] for point in payload["risk_coverage"]] == list(
        config.DEFERRAL_RATES
    )
    first = payload["risk_coverage"][0]
    assert (first["deferred"], first["kept"]) == (0, len(TRUE))
    assert first["threshold"] == pytest.approx(min(CONFIDENCE))
    assert len(payload["bins"]) == config.CALIBRATION_BINS


def test_deferral_improves_the_score_on_what_the_model_keeps():
    payload = analysis.deferral(_rows(TRUE, PRED, CONFIDENCE), None)
    scores = [point["macro_f1"] for point in payload["risk_coverage"]]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0], "the wrong rows are the least confident ones"


def test_deferral_publishes_no_cascade_until_the_second_model_exists():
    assert analysis.deferral(_rows(TRUE, PRED, CONFIDENCE), None)["cascade"] == []

    with_herbert = analysis.deferral(
        _rows(TRUE, PRED, CONFIDENCE), _rows(TRUE, TRUE, CONFIDENCE)
    )
    assert len(with_herbert["cascade"]) == len(config.DEFERRAL_RATES)
    assert with_herbert["cascade"][-1]["macro_f1"] == pytest.approx(1.0)


# --- probe --------------------------------------------------------------------------------


def test_probe_scores_the_hand_written_cases_and_their_derived_variants():
    texts, labels = _corpus()
    pipeline = baseline.train(texts, labels)

    payload = analysis.probe(pipeline)

    derived = challenge.derive_variants(challenge_set.CASES)
    assert payload["base_cases"] == len(challenge_set.CASES)
    assert payload["derived_cases"] == len(derived)
    assert payload["min_phenomenon_n"] == config.MIN_PHENOMENON_N
    assert {row["phenomenon"] for row in payload["scores"]} == {
        case.phenomenon for case in list(challenge_set.CASES) + derived
    }
    assert sum(row["total"] for row in payload["scores"]) == len(challenge_set.CASES) + len(
        derived
    )


def test_probe_keeps_a_few_misses_and_says_how_many_there_were():
    # A model trained on rotated labels answers almost every case wrongly, so the cap is
    # exercised whatever the real baseline happens to get right.
    texts, labels = _corpus()
    pipeline = baseline.train(texts, [(label + 1) % 3 for label in labels])

    payload = analysis.probe(pipeline)

    assert payload["misses_total"] > analysis.PROBE_EXAMPLES
    assert len(payload["misses"]) == analysis.PROBE_EXAMPLES
    miss = payload["misses"][0]
    assert miss["gold"] in config.LABEL_NAMES and miss["predicted"] in config.LABEL_NAMES
    assert miss["gold"] != miss["predicted"]


# --- curve, cost, manifest ----------------------------------------------------------------


def test_learning_curve_reports_the_configured_sizes_the_corpus_can_supply():
    texts, labels = _corpus(repeats=3)  # 240 rows: the 150 point fits, the rest do not
    data = Dataset(
        train=Split(texts, labels),
        validation=Split(texts[:20], labels[:20]),
        test=Split(texts[:20], labels[:20]),
    )

    payload = analysis.learning_curve(data)

    assert payload["seeds"] == list(config.LEARNING_CURVE_SEEDS)
    assert payload["target_share"] == analysis.CURVE_TARGET_SHARE
    assert [point["size"] for point in payload["points"]] == [150, len(labels)]
    assert payload["points"][0]["seeds"] == len(config.LEARNING_CURVE_SEEDS)
    assert payload["points"][-1]["low"] == payload["points"][-1]["high"]
    assert payload["reaches_target_at"] in {point["size"] for point in payload["points"]}


def test_cost_measures_this_machine_and_says_which_one(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASELINE_MODEL_PATH", tmp_path / "absent.joblib")

    payload = analysis.cost(_dataset())

    assert payload["train_seconds"] >= 0.0
    assert payload["train_rows"] == 60
    assert payload["predict_rows_per_second"] > 0
    assert payload["device"] == "CPU" and payload["machine"]
    assert payload["model_bytes"] is None, "no saved model on this machine, and it says so"


def test_cost_reports_the_size_of_the_model_that_was_saved(tmp_path, monkeypatch):
    saved = tmp_path / "baseline.joblib"
    saved.write_bytes(b"x" * 4096)
    monkeypatch.setattr(config, "BASELINE_MODEL_PATH", saved)

    assert analysis.cost(_dataset())["model_bytes"] == 4096


def test_manifest_records_the_splits_and_whether_herbert_ran():
    data = _dataset()

    without = analysis.manifest(data, None)
    with_herbert = analysis.manifest(data, _rows(TRUE, TRUE, CONFIDENCE))

    assert without["splits"] == {"train": 60, "validation": 20, "test": 20}
    assert without["dataset"]["revision"] == config.DATASET_REVISION
    assert without["dataset"]["dropped_class"] == config.DROP_LABEL_NAME
    assert without["seed"] == config.RANDOM_STATE
    assert without["herbert_available"] is False
    assert with_herbert["herbert_available"] is True


def test_written_metrics_are_utf8_json_with_lf_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "metrics" / "significance.json"

    analysis._write(target, {"claim": "wąska ścieżka"})

    assert b"\r\n" not in target.read_bytes()
    assert json.loads(target.read_text(encoding="utf-8"))["claim"] == "wąska ścieżka"


# --- the runner's own preconditions -------------------------------------------------------
#
# Each of these exits before ``load_polemo`` is reached, so none of them downloads anything.


def test_the_runner_stops_when_baseline_train_has_not_been_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASELINE_PREDICTIONS_PATH", tmp_path / "absent.json")

    with pytest.raises(SystemExit, match="baseline_train"):
        analysis.main()


def test_the_runner_stops_when_the_predictions_carry_no_probabilities(tmp_path, monkeypatch):
    without_proba = _rows(TRUE, PRED, CONFIDENCE)
    del without_proba["proba"]
    path = tmp_path / "baseline_test.json"
    path.write_text(json.dumps(without_proba), encoding="utf-8")
    monkeypatch.setattr(config, "BASELINE_PREDICTIONS_PATH", path)

    with pytest.raises(SystemExit, match="probabilities"):
        analysis.main()


def test_the_runner_stops_when_the_saved_model_is_gone(tmp_path, monkeypatch):
    path = tmp_path / "baseline_test.json"
    path.write_text(json.dumps(_rows(TRUE, PRED, CONFIDENCE)), encoding="utf-8")
    monkeypatch.setattr(config, "BASELINE_PREDICTIONS_PATH", path)

    def missing(*args, **kwargs):
        raise FileNotFoundError(config.BASELINE_MODEL_PATH)

    monkeypatch.setattr(baseline, "load", missing)

    with pytest.raises(SystemExit, match="no saved baseline model"):
        analysis.main()


# --- the round trip -----------------------------------------------------------------------


def _produce_metrics(tmp_path: Path, monkeypatch) -> Path:
    """Run the real writers over a tiny corpus and return the metrics directory they filled."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(
        config, "BASELINE_METRICS_PATH", tmp_path / config.BASELINE_METRICS_PATH.name
    )
    monkeypatch.setattr(
        config, "BASELINE_PREDICTIONS_PATH", tmp_path / config.BASELINE_PREDICTIONS_PATH.name
    )
    monkeypatch.setattr(config, "BASELINE_MODEL_PATH", tmp_path / "baseline.joblib")

    data = _dataset()
    pipeline = baseline.train(data.train.texts, data.train.labels)
    proba = baseline.predict_proba(pipeline, data.test.texts)
    y_pred = [max(range(len(row)), key=lambda i: row[i]) for row in proba]

    baseline_train._write_metrics(evaluate.evaluate(data.test.labels, y_pred))
    baseline_train._write_predictions(data.test.labels, y_pred, proba)
    # Explicitly, not through the default: ``save``'s default path was bound to the real
    # ``models/`` directory at import time and does not follow the patched config.
    baseline.save(pipeline, config.BASELINE_MODEL_PATH)

    rows = analysis._load_predictions(config.BASELINE_PREDICTIONS_PATH)
    analysis._write(
        tmp_path / config.SIGNIFICANCE_PATH.name,
        analysis.significance(rows, None, data.train.labels),
    )
    analysis._write(
        tmp_path / config.SEGMENTS_PATH.name, analysis.length_segments(data, rows)
    )
    analysis._write(tmp_path / config.CHALLENGE_PATH.name, analysis.probe(pipeline))
    analysis._write(tmp_path / config.DEFERRAL_PATH.name, analysis.deferral(rows, None))
    analysis._write(tmp_path / config.LEARNING_CURVE_PATH.name, analysis.learning_curve(data))
    analysis._write(tmp_path / config.COST_PATH.name, analysis.cost(data))
    analysis._write(tmp_path / config.MANIFEST_PATH.name, analysis.manifest(data, None))
    return tmp_path


def test_the_page_renders_from_metrics_these_writers_actually_produced(tmp_path, monkeypatch):
    """The contract between ``analysis`` and ``site``: same keys, same nesting, both sides.

    ``test_site`` renders from hand-written fixtures, which keep passing after a key is
    renamed here. This is the test that does not.
    """
    metrics = _produce_metrics(tmp_path, monkeypatch)

    html = build.render(metrics)

    baseline_metrics = json.loads(
        (metrics / config.BASELINE_METRICS_PATH.name).read_text(encoding="utf-8")
    )
    significance = json.loads(
        (metrics / config.SIGNIFICANCE_PATH.name).read_text(encoding="utf-8")
    )
    assert f"{baseline_metrics['macro_f1']:.3f}" in html
    assert f"{significance['baseline']['low']:.3f}" in html
    assert "card pending" in html, "the HerBERT panels are still waiting"
    assert "{{" not in html, "no template expression survived into the page"


def test_the_analysis_writes_no_review_text_beyond_our_own_probe(tmp_path, monkeypatch):
    """PolEmo is CC BY-NC-SA and is not redistributed — only our own sentences may be written."""
    metrics = _produce_metrics(tmp_path, monkeypatch)
    ours = {case.text for case in challenge_set.CASES}
    ours |= {variant.text for variant in challenge.derive_variants(challenge_set.CASES)}

    probe = json.loads((metrics / config.CHALLENGE_PATH.name).read_text(encoding="utf-8"))
    for miss in probe["misses"]:
        assert miss["text"] in ours

    # Everywhere else only numbers, identifiers and short descriptors are written. A PolEmo
    # review is a paragraph; the longest legitimate string here is a 40-character commit sha.
    for name in (
        config.SIGNIFICANCE_PATH.name,
        # The length cut reads every test review to measure it; only word counts come back out.
        config.SEGMENTS_PATH.name,
        config.DEFERRAL_PATH.name,
        config.LEARNING_CURVE_PATH.name,
        config.COST_PATH.name,
        config.MANIFEST_PATH.name,
        config.BASELINE_METRICS_PATH.name,
        config.BASELINE_PREDICTIONS_PATH.name,
    ):
        payload = json.loads((metrics / name).read_text(encoding="utf-8"))
        longest = max(_leaf_strings(payload), key=len, default="")
        assert len(longest) <= 80, f"{name} carries free text: {longest!r}"


def _leaf_strings(payload):
    """Every string value in a nested payload, however deeply nested."""
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _leaf_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _leaf_strings(item)
    elif isinstance(payload, str):
        yield payload
