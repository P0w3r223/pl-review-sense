"""The learning curve and its stratified sampling.

The fit is injected, so the curve is tested against a stand-in model rather than by training
anything — which also means the tests check the *shape* of the measurement, which is where
the mistakes live.
"""

from __future__ import annotations

import pytest

from pl_review_sense import config, curves

# 60 rows, deliberately imbalanced the way PolEmo is: an unstratified draw of six would be
# expected to miss the smallest class outright.
LABELS = [0] * 36 + [1] * 6 + [2] * 18
TEXTS = [f"review {index}" for index in range(len(LABELS))]


def test_stratified_subsample_keeps_every_class_and_the_requested_size():
    for size in (6, 12, 30, 59):
        index = curves.stratified_indices(LABELS, size, seed=0)
        assert len(index) == size
        assert set(LABELS[i] for i in index) == {0, 1, 2}


def test_stratified_subsample_keeps_class_shares_roughly_intact():
    index = curves.stratified_indices(LABELS, 30, seed=1)
    share = sum(1 for i in index if LABELS[i] == 0) / 30
    assert share == pytest.approx(36 / 60, abs=0.1)


def test_stratified_subsample_is_reproducible_and_seed_dependent():
    assert curves.stratified_indices(LABELS, 12, seed=3) == curves.stratified_indices(
        LABELS, 12, seed=3
    )
    assert curves.stratified_indices(LABELS, 12, seed=3) != curves.stratified_indices(
        LABELS, 12, seed=4
    )


def test_stratified_subsample_returns_everything_when_asked_for_more_than_it_has():
    assert curves.stratified_indices(LABELS, 500, seed=0) == list(range(len(LABELS)))


def test_stratified_subsample_refuses_a_size_that_cannot_cover_the_classes():
    with pytest.raises(ValueError):
        curves.stratified_indices(LABELS, 2, seed=0)
    with pytest.raises(ValueError):
        curves.stratified_indices(LABELS, 0, seed=0)


def test_learning_curve_reports_one_point_per_size_plus_the_full_corpus():
    seen_sizes = []

    def fit_predict(train_texts, train_labels, test_texts):
        seen_sizes.append(len(train_labels))
        # A model that improves with data: wrong on a shrinking share of the test rows.
        wrong = max(0, 3 - len(train_labels) // 12)
        return [(2 if index < wrong else label) for index, label in enumerate(TEST_LABELS)]

    points = curves.learning_curve(
        TEXTS, LABELS, TEST_TEXTS, TEST_LABELS, fit_predict, sizes=(12, 24), seeds=(0, 1)
    )

    assert [point.size for point in points] == [12, 24, len(LABELS)]
    assert [point.seeds for point in points] == [2, 2, 1]
    assert seen_sizes == [12, 12, 24, 24, len(LABELS)]
    assert points[0].mean_macro_f1 <= points[-1].mean_macro_f1
    assert points[-1].low == points[-1].high, "the full corpus is fitted once, so it has no spread"


def test_learning_curve_drops_sizes_the_training_set_cannot_supply():
    points = curves.learning_curve(
        TEXTS,
        LABELS,
        TEST_TEXTS,
        TEST_LABELS,
        lambda train_texts, train_labels, test_texts: list(TEST_LABELS),
        sizes=(12, 6_000),
        seeds=(0,),
    )
    assert [point.size for point in points] == [12, len(LABELS)]


def test_reaches_finds_the_first_size_at_the_target_share():
    points = [
        curves.CurvePoint(size=100, seeds=2, mean_macro_f1=0.50, low=0.4, high=0.6),
        curves.CurvePoint(size=200, seeds=2, mean_macro_f1=0.90, low=0.9, high=0.9),
        curves.CurvePoint(size=400, seeds=1, mean_macro_f1=1.00, low=1.0, high=1.0),
    ]

    assert curves.reaches(points, 0.9) == 200
    assert curves.reaches(points, 0.4) == 100
    assert curves.reaches(points, 1.01) is None
    assert curves.reaches([], 0.9) is None


TEST_LABELS = [0, 1, 2, 0, 1, 2]
TEST_TEXTS = [f"test {index}" for index in range(len(TEST_LABELS))]


def test_configured_sizes_are_ordered_and_positive():
    assert list(config.LEARNING_CURVE_SIZES) == sorted(set(config.LEARNING_CURVE_SIZES))
    assert all(size > 0 for size in config.LEARNING_CURVE_SIZES)
