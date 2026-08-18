"""Cutting the score by review length.

The section this feeds exists to put the page's own headline at risk, so the tests care most
about the ways it could quietly answer a different question: averaging over classes a segment
does not contain, or reading a trend off a segment too thin to carry one.
"""

from __future__ import annotations

import pytest

from pl_review_sense import config, segments


def test_bucket_names_run_from_open_bottom_to_open_top():
    assert segments.bucket_names((25, 50)) == ["< 25", "25–49", "50+"]


@pytest.mark.parametrize(
    "value, expected",
    [(0, 0), (24, 0), (25, 1), (49, 1), (50, 2), (199, 3), (200, 4), (10_000, 4)],
)
def test_assign_puts_a_value_on_the_edge_into_the_bucket_that_starts_there(value, expected):
    assert segments.assign(value, (25, 50, 100, 200)) == expected


def test_segment_scores_group_the_rows_and_carry_their_counts():
    lengths = [10, 12, 30, 30, 30, 300]
    y_true = [0, 1, 0, 1, 2, 2]
    y_pred = [0, 1, 0, 1, 2, 2]

    scores = segments.segment_scores(lengths, y_true, y_pred, edges=(25, 50))

    assert [item.name for item in scores] == ["< 25", "25–49", "50+"]
    assert [item.n for item in scores] == [2, 3, 1]
    assert all(item.accuracy == 1.0 for item in scores)


def test_segments_with_no_rows_are_left_out_rather_than_drawn_empty():
    scores = segments.segment_scores([10, 10], [0, 1], [0, 1], edges=(25, 50))

    assert [item.name for item in scores] == ["< 25"]


def test_macro_f1_inside_a_segment_averages_only_the_classes_it_contains():
    """A segment holding one class would otherwise take a hard zero for the two it lacks."""
    scores = segments.segment_scores([10, 10, 10], [0, 0, 0], [0, 0, 0], edges=(25,))

    assert scores[0].classes_present == 1
    assert scores[0].macro_f1 == pytest.approx(1.0), "perfect on what it holds is 1.0, not 0.33"


def test_a_segment_records_how_many_classes_stood_behind_its_average():
    scores = segments.segment_scores([10, 10], [0, 2], [0, 2], edges=(25,))
    assert scores[0].classes_present == 2


def test_a_thin_segment_is_flagged_rather_than_dropped():
    thin = segments.segment_scores([10] * 3, [0, 1, 2], [0, 1, 2], edges=(25,))[0]
    cycled = ([0, 1, 2] * config.MIN_SEGMENT_N)[: config.MIN_SEGMENT_N]
    thick = segments.segment_scores(
        [10] * config.MIN_SEGMENT_N, cycled, cycled, edges=(25,)
    )[0]

    assert thin.thin and thin.n == 3, "shown, so a reader can see what was too small"
    assert not thick.thin


def test_segment_scores_refuse_lists_that_are_not_about_the_same_rows():
    with pytest.raises(ValueError):
        segments.segment_scores([10, 20], [0], [0])


def test_trend_ignores_segments_too_thin_to_carry_one():
    """The whole point: a 30-row bucket must not set the direction of the finding."""
    thin_low = segments.Segment("< 25", 0, 25, n=6, accuracy=0.5, macro_f1=0.30,
                                classes_present=3)
    thick_low = segments.Segment("50–99", 50, 100, n=200, accuracy=0.96, macro_f1=0.96,
                                 classes_present=3)
    thick_high = segments.Segment("200+", 200, None, n=90, accuracy=0.95, macro_f1=0.95,
                                  classes_present=3)

    assert segments.trend([thin_low, thick_low, thick_high]) == pytest.approx(-0.01)


def test_trend_is_none_when_fewer_than_two_segments_are_thick_enough():
    thick = segments.Segment("50–99", 50, 100, n=200, accuracy=0.9, macro_f1=0.9,
                             classes_present=3)
    thin = segments.Segment("< 25", 0, 25, n=6, accuracy=0.5, macro_f1=0.5, classes_present=3)

    assert segments.trend([thick, thin]) is None
    assert segments.trend([]) is None


def test_trend_can_come_out_negative():
    """It has to be able to contradict the page, or the section is decoration."""
    low = segments.Segment("50–99", 50, 100, n=200, accuracy=0.9, macro_f1=0.99,
                           classes_present=3)
    high = segments.Segment("200+", 200, None, n=200, accuracy=0.9, macro_f1=0.80,
                            classes_present=3)

    assert segments.trend([low, high]) < 0
