"""The statistics the page's claims rest on.

The macro-F1 here is a hand-rolled shortcut used two thousand times per bootstrap, so the
first test pins it against scikit-learn's implementation: the moment the two disagree, every
interval on the page is wrong in a way nothing else would catch.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import f1_score

from pl_review_sense import config, stats

_LABELS = len(config.LABEL_NAMES)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]),  # perfect
        ([0, 1, 2, 0, 1, 2], [2, 0, 1, 2, 0, 1]),  # everything wrong
        ([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]),  # mixed
        ([0, 0, 0, 0], [0, 0, 0, 0]),  # two classes never appear
        ([0, 1, 1, 2], [1, 1, 1, 1]),  # one class predicted for everything
    ],
)
def test_macro_f1_matches_scikit_learn(y_true, y_pred):
    expected = f1_score(
        y_true, y_pred, labels=list(range(_LABELS)), average="macro", zero_division=0
    )
    assert stats.macro_f1(y_true, y_pred, _LABELS) == pytest.approx(expected)


def test_bootstrap_brackets_the_point_estimate_and_is_reproducible():
    y_true = [0, 1, 2] * 40
    y_pred = [0, 1, 2] * 35 + [1, 2, 0] * 5

    first = stats.bootstrap_macro_f1(y_true, y_pred, resamples=200)
    second = stats.bootstrap_macro_f1(y_true, y_pred, resamples=200)

    assert first == second, "same predictions and seed must give the same interval"
    assert first.low <= first.point <= first.high
    assert 0.0 <= first.low < first.high <= 1.0


def test_bootstrap_rejects_mismatched_and_empty_input():
    with pytest.raises(ValueError):
        stats.bootstrap_macro_f1([0, 1, 2], [0, 1])
    with pytest.raises(ValueError):
        stats.bootstrap_macro_f1([], [])


def test_mcnemar_is_symmetric_in_which_model_is_named_first():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1]
    a = [0, 1, 2, 0, 1, 2, 1, 0]  # right on the first six
    b = [0, 1, 1, 1, 1, 2, 0, 1]  # right on a different set

    forward = stats.mcnemar(y_true, a, b)
    backward = stats.mcnemar(y_true, b, a)

    assert forward.only_a_correct == backward.only_b_correct
    assert forward.p_value == pytest.approx(backward.p_value)
    assert forward.discordant == backward.discordant


def test_mcnemar_reports_no_evidence_when_the_models_never_disagree():
    y_true = [0, 1, 2, 0]
    predictions = [0, 1, 1, 0]

    result = stats.mcnemar(y_true, predictions, predictions)

    assert result.discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_is_significant_when_one_model_wins_every_disagreement():
    y_true = [0] * 20
    a = [0] * 20  # right on everything
    b = [1] * 10 + [0] * 10  # wrong on half, right on the rest

    result = stats.mcnemar(y_true, a, b)

    assert (result.only_a_correct, result.only_b_correct) == (10, 0)
    assert result.p_value < 0.01


def test_calibration_of_a_perfectly_honest_model_has_no_error():
    # Nine tenths confidence, right nine times in ten.
    confidences = [0.9] * 100
    correct = [True] * 90 + [False] * 10

    calibration = stats.calibrate(confidences, correct)

    assert calibration.expected_error == pytest.approx(0.0, abs=1e-9)
    populated = [item for item in calibration.bins if item.count]
    assert len(populated) == 1
    assert populated[0].accuracy == pytest.approx(0.9)


def test_calibration_measures_the_gap_of_an_overconfident_model():
    confidences = [0.95] * 100
    correct = [True] * 50 + [False] * 50

    calibration = stats.calibrate(confidences, correct)

    assert calibration.expected_error == pytest.approx(0.45)


def test_calibration_puts_full_confidence_in_the_top_bin():
    calibration = stats.calibrate([1.0, 1.0], [True, False])

    populated = [item for item in calibration.bins if item.count]
    assert len(populated) == 1
    assert populated[0].upper == pytest.approx(1.0)
    assert populated[0].count == 2


def test_calibration_puts_a_confidence_on_a_bin_edge_in_the_lower_bin():
    """Bins are right-closed: 0.7 belongs to 0.6–0.7, not to the bin that starts at it."""
    calibration = stats.calibrate([0.7], [True], bins=10)

    populated = [item for item in calibration.bins if item.count]
    assert len(populated) == 1
    assert populated[0].upper == pytest.approx(0.7)


def test_calibration_spans_every_bin_it_was_asked_for_even_the_empty_ones():
    """The reliability chart drops the empty bins itself; ECE needs them to stay countable."""
    calibration = stats.calibrate([0.05, 0.95], [True, True], bins=10)

    assert len(calibration.bins) == 10
    assert [item.count for item in calibration.bins] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert calibration.bins[3].mean_confidence == 0.0, "an empty bin claims nothing"


def test_calibration_weights_the_gap_by_how_many_rows_sit_in_each_bin():
    # 90 rows claiming 0.95 and right every time; 10 claiming 0.15 and right every time.
    confidences = [0.95] * 90 + [0.15] * 10
    correct = [True] * 100

    calibration = stats.calibrate(confidences, correct)

    expected = (90 * abs(1.0 - 0.95) + 10 * abs(1.0 - 0.15)) / 100
    assert calibration.expected_error == pytest.approx(expected)


def test_calibration_of_nothing_is_no_error_rather_than_a_crash():
    calibration = stats.calibrate([], [])
    assert calibration.bins == [] and calibration.expected_error == 0.0


def test_a_wider_confidence_level_gives_a_wider_interval():
    y_true = [0, 1, 2] * 40
    y_pred = [0, 1, 2] * 35 + [1, 2, 0] * 5

    narrow = stats.bootstrap_macro_f1(y_true, y_pred, resamples=300, confidence=0.80)
    wide = stats.bootstrap_macro_f1(y_true, y_pred, resamples=300, confidence=0.99)

    assert wide.width > narrow.width
    assert wide.low <= narrow.low and narrow.high <= wide.high
    assert narrow.point == pytest.approx(wide.point), "only the tails move, not the estimate"


def test_calibration_rejects_mismatched_input():
    with pytest.raises(ValueError):
        stats.calibrate([0.5, 0.6], [True])
