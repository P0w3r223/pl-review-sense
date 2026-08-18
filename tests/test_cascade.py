"""Deferral by confidence: the curve that can be published today, and the one that cannot."""

from __future__ import annotations

import pytest

from pl_review_sense import cascade

# Ten rows. The model is wrong exactly where it is least confident, which is the behaviour a
# deferral rule needs and therefore the behaviour the test has to be able to see.
TRUE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
PRED = [1, 2, 0, 0, 1, 2, 0, 1, 2, 0]
CONFIDENCE = [0.35, 0.40, 0.45, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]


def test_risk_coverage_starts_from_the_whole_test_set():
    first = cascade.risk_coverage(TRUE, PRED, CONFIDENCE, rates=[0.0])[0]

    assert (first.deferred, first.kept) == (0, len(TRUE))
    assert first.threshold == pytest.approx(min(CONFIDENCE))


def test_risk_coverage_improves_as_the_least_confident_rows_are_set_aside():
    points = cascade.risk_coverage(TRUE, PRED, CONFIDENCE, rates=[0.0, 0.1, 0.3])

    assert [point.deferred for point in points] == [0, 1, 3]
    assert points[0].macro_f1 < points[1].macro_f1 < points[2].macro_f1
    assert points[2].macro_f1 == pytest.approx(1.0), "the three wrong rows were the deferred ones"


def test_risk_coverage_threshold_is_the_lowest_confidence_still_answered():
    point = cascade.risk_coverage(TRUE, PRED, CONFIDENCE, rates=[0.3])[0]
    assert point.threshold == pytest.approx(0.90)


def test_risk_coverage_rejects_arrays_that_are_not_about_the_same_rows():
    with pytest.raises(ValueError):
        cascade.risk_coverage(TRUE, PRED[:-1], CONFIDENCE)


def test_risk_coverage_of_nothing_is_nothing():
    assert cascade.risk_coverage([], [], []) == []


def test_cascade_publishes_nothing_without_a_second_model():
    assert cascade.cascade(TRUE, PRED, CONFIDENCE, None) == []


def test_cascade_replaces_only_the_deferred_rows():
    perfect = list(TRUE)  # a second model that is right about everything

    points = cascade.cascade(TRUE, PRED, CONFIDENCE, perfect, rates=[0.0, 0.3])

    assert points[0].macro_f1 < 1.0, "no deferral means the cheap model answers alone"
    assert points[1].macro_f1 == pytest.approx(1.0)
    assert points[1].escalated_share == pytest.approx(0.3)


def test_cascade_cannot_be_worse_than_the_cheap_model_when_the_second_is_perfect():
    perfect = list(TRUE)
    points = cascade.cascade(TRUE, PRED, CONFIDENCE, perfect, rates=[0.0, 0.1, 0.2, 0.3])
    scores = [point.macro_f1 for point in points]
    assert scores == sorted(scores)


def test_cascade_refuses_predictions_of_different_length():
    with pytest.raises(ValueError):
        cascade.cascade(TRUE, PRED, CONFIDENCE, TRUE[:-1])


def test_cascade_of_nothing_is_nothing():
    """No rows means no operating points — the same answer risk_coverage gives, not a crash."""
    assert cascade.cascade([], [], [], []) == []


def test_deferral_rate_of_one_is_refused():
    with pytest.raises(ValueError):
        cascade.risk_coverage(TRUE, PRED, CONFIDENCE, rates=[1.0])
    with pytest.raises(ValueError):
        cascade.risk_coverage(TRUE, PRED, CONFIDENCE, rates=[-0.1])


def test_the_deferred_rows_are_the_least_confident_ones_whatever_order_they_arrive_in():
    """Rows are set aside by confidence, not by position: the ordering is the whole mechanism."""
    reordered = list(reversed(range(len(TRUE))))
    true = [TRUE[i] for i in reordered]
    pred = [PRED[i] for i in reordered]
    confidence = [CONFIDENCE[i] for i in reordered]

    point = cascade.risk_coverage(true, pred, confidence, rates=[0.3])[0]

    assert point.macro_f1 == pytest.approx(1.0)
    assert point.threshold == pytest.approx(0.90)


def test_equally_confident_rows_are_deferred_in_the_order_they_arrived():
    """A stable sort, so two runs over the same predictions defer the same reviews."""
    confidence = [0.5] * len(TRUE)

    first = cascade.risk_coverage(TRUE, PRED, confidence, rates=[0.2])[0]
    second = cascade.risk_coverage(TRUE, PRED, confidence, rates=[0.2])[0]

    assert first == second
    assert first.deferred == 2 and first.kept == len(TRUE) - 2
