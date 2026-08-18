"""Charts as inline SVG — pure functions from numbers to markup.

SVG rather than an embedded PNG for three reasons that matter on this page: it inherits the
reader's colour scheme, so dark mode is not a second rendering; it stays sharp at any size;
and its text is real text, so a screen reader and a search engine can both read the figures.
The previous report shipped a base64 matplotlib image, which met none of the three and made
the published page impossible to check against the numbers it was built from.

Every figure here carries the count it rests on. ``build`` refuses to publish one that does
not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Sequence

_WIDTH = 720
_ROW_HEIGHT = 26
_LABEL_WIDTH = 150
_PAD = 12

# Geometry of the plotted (rather than laid out in rows) figures.
_PLOT_HEIGHT = 250
_PLOT_LEFT = 54
_PLOT_RIGHT = 24
_PLOT_TOP = 18
_PLOT_BASELINE = 200

# Confusion matrix cells.
_CELL_WIDTH = 104
_CELL_HEIGHT = 58


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    note: str | None = None
    muted: bool = False
    value_text: str | None = None


@dataclass(frozen=True)
class Fraction:
    """``correct`` of ``total`` — a count with its ceiling drawn, never a bare percentage."""

    label: str
    correct: int
    total: int
    note: str | None = None
    muted: bool = False


@dataclass(frozen=True)
class CurvePoint:
    x: float
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class ReliabilityPoint:
    confidence: float
    accuracy: float
    count: int


def _text(value) -> str:
    return escape(str(value), quote=True)


def _thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _svg(width: int, height: int, title: str, body: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{_text(title)}" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{_text(title)}</title>{body}</svg>"
    )


def bar_chart(
    bars: Sequence[Bar],
    title: str,
    unit: str = "weight",
    width: int = _WIDTH,
    label_width: int = _LABEL_WIDTH,
) -> str:
    """Horizontal bars, each labelled with its own value.

    The frame is a parameter because these charts are laid out at two sizes: full width for a
    figure of its own, and narrow for the three per-class rankings that sit side by side. An
    SVG scaled down by CSS scales its type with it, so a 720-wide chart dropped into a 300-wide
    column arrives with four-pixel labels.
    """
    if not bars:
        return '<p class="note">Nothing to plot.</p>'

    largest = max((bar.value for bar in bars), default=1) or 1
    # Room at the right for the value. A note beside it needs roughly twice as much.
    gutter = 90 if any(bar.note for bar in bars) else 48
    plot_width = width - label_width - gutter
    height = len(bars) * _ROW_HEIGHT + _PAD * 2

    parts = []
    for index, bar in enumerate(bars):
        y = _PAD + index * _ROW_HEIGHT
        bar_width = max(1.0, bar.value / largest * plot_width)
        note = f" {bar.note}" if bar.note else ""
        parts.append(
            f'<text class="bar-label" x="{label_width - 8}" y="{y + 13}" '
            f'text-anchor="end">{_text(bar.label)}</text>'
            f'<rect class="{"bar muted" if bar.muted else "bar"}" x="{label_width}" '
            f'y="{y + 3}" width="{bar_width:.1f}" height="{_ROW_HEIGHT - 9}" rx="2"></rect>'
            f'<text class="bar-value" x="{label_width + bar_width + 6:.1f}" y="{y + 13}">'
            f"{_text(bar.value_text) if bar.value_text else _thousands(bar.value)}"
            f"{_text(note)}</text>"
        )
    return _svg(width, height, f"{title} ({unit})", "".join(parts))


def fraction_chart(items: Sequence[Fraction], title: str) -> str:
    """How many of each cell's cases were answered correctly, against the cell's own size.

    The track behind every bar is the whole cell, so a long bar cannot be read as a large
    number when it is a small cell — and the count is printed rather than a percentage,
    because twenty sentences do not support a rate.
    """
    if not items:
        return '<p class="note">The probe has not been scored.</p>'

    largest = max(item.total for item in items) or 1
    plot_width = _WIDTH - _LABEL_WIDTH - 150
    height = len(items) * (_ROW_HEIGHT + 8) + _PAD * 2

    parts = []
    for index, item in enumerate(items):
        y = _PAD + index * (_ROW_HEIGHT + 8)
        track = item.total / largest * plot_width
        filled = (item.correct / largest) * plot_width if item.total else 0.0
        share = item.correct / item.total if item.total else 0.0
        note = f" · {item.note}" if item.note else ""
        parts.append(
            f'<text class="bar-label" x="{_LABEL_WIDTH - 8}" y="{y + 13}" '
            f'text-anchor="end">{_text(item.label)}</text>'
            f'<rect class="track" x="{_LABEL_WIDTH}" y="{y + 3}" width="{track:.1f}" '
            f'height="{_ROW_HEIGHT - 9}" rx="2"></rect>'
            f'<rect class="{"fill muted" if item.muted else "fill"}" x="{_LABEL_WIDTH}" '
            f'y="{y + 3}" width="{max(1.0, filled):.1f}" height="{_ROW_HEIGHT - 9}" rx="2">'
            f"</rect>"
            f'<text class="bar-value" x="{_LABEL_WIDTH + track + 8:.1f}" y="{y + 13}">'
            f"{item.correct} / {item.total} ({share:.0%}){_text(note)}</text>"
        )
    return _svg(_WIDTH, height, f"{title} (correct of total)", "".join(parts))


def confusion_chart(matrix: Sequence[Sequence[int]], labels: Sequence[str], title: str) -> str:
    """The confusion matrix as a heatmap whose intensity is normalized within each row.

    Row-normalized, not matrix-normalized: the three classes differ in size by a factor of
    three, and shading by absolute count would colour the largest class darkest whatever its
    error rate. Each cell prints the count *and* its share of the row, so the picture and the
    number never have to be reconciled by eye.
    """
    if not matrix:
        return '<p class="note">No confusion matrix in the committed metrics.</p>'

    left = 96
    top = 46
    width = left + len(labels) * _CELL_WIDTH + 12
    height = top + len(labels) * _CELL_HEIGHT + 34

    parts = [
        f'<text class="axis" x="{left}" y="16">predicted</text>',
        f'<text class="axis" x="4" y="{top - 8}">true</text>',
    ]
    for column, label in enumerate(labels):
        parts.append(
            f'<text class="axis" x="{left + column * _CELL_WIDTH + _CELL_WIDTH / 2:.1f}" '
            f'y="{top - 8}" text-anchor="middle">{_text(label)}</text>'
        )

    for row, label in enumerate(labels):
        total = sum(matrix[row]) or 1
        y = top + row * _CELL_HEIGHT
        parts.append(
            f'<text class="bar-label" x="{left - 10}" y="{y + _CELL_HEIGHT / 2 + 4:.1f}" '
            f'text-anchor="end">{_text(label)}</text>'
        )
        for column in range(len(labels)):
            count = matrix[row][column]
            share = count / total
            x = left + column * _CELL_WIDTH
            on_fill = " on-fill" if share > 0.55 else ""
            parts.append(
                f'<rect class="cell" x="{x}" y="{y}" width="{_CELL_WIDTH - 4}" '
                f'height="{_CELL_HEIGHT - 4}" rx="3" fill-opacity="{share:.3f}"></rect>'
                f'<text class="cell-text{on_fill}" x="{x + _CELL_WIDTH / 2 - 2:.1f}" '
                f'y="{y + _CELL_HEIGHT / 2 - 2:.1f}" text-anchor="middle">{count}</text>'
                f'<text class="cell-share{on_fill}" x="{x + _CELL_WIDTH / 2 - 2:.1f}" '
                f'y="{y + _CELL_HEIGHT / 2 + 14:.1f}" text-anchor="middle">{share:.0%} of row'
                f"</text>"
            )
    return _svg(width, height - 24, title, "".join(parts))


def curve_chart(
    points: Sequence[CurvePoint],
    title: str,
    x_ticks: Sequence[float],
    y_floor: float,
    x_caption: str,
) -> str:
    """One series against a logarithmic x axis, with the seed-to-seed spread as a band.

    The x axis is logarithmic because the sizes double: spacing them evenly would draw the
    step from 2 400 to 4 800 as the same effort as the step from 150 to 300, when it is
    sixteen times the labelling.

    The y axis starts at ``y_floor`` rather than at zero, and the floor is printed on the
    axis. Anchored at zero the whole curve is a flat line in the top fifth of the frame and
    the shape it exists to show disappears; cropped without saying so it would exaggerate
    every difference. Stating the floor is the compromise that is neither.
    """
    if not points:
        return '<p class="note">The curve has not been computed.</p>'

    plot_width = _WIDTH - _PLOT_LEFT - _PLOT_RIGHT
    span = _PLOT_BASELINE - _PLOT_TOP
    log_min = math.log10(min(point.x for point in points))
    log_max = math.log10(max(point.x for point in points))
    log_range = (log_max - log_min) or 1.0
    ceiling = 1.0

    def x_of(value: float) -> float:
        return _PLOT_LEFT + (math.log10(value) - log_min) / log_range * plot_width

    def y_of(value: float) -> float:
        clamped = max(0.0, min(1.0, (value - y_floor) / (ceiling - y_floor)))
        return _PLOT_BASELINE - clamped * span

    upper = " ".join(f"{x_of(p.x):.1f},{y_of(p.high):.1f}" for p in points)
    lower = " ".join(f"{x_of(p.x):.1f},{y_of(p.low):.1f}" for p in reversed(points))
    line = " ".join(f"{x_of(p.x):.1f},{y_of(p.value):.1f}" for p in points)

    parts = [
        f'<polygon class="band" points="{upper} {lower}"></polygon>',
        f'<polyline class="series" points="{line}"></polyline>',
        f'<line class="axis-line" x1="{_PLOT_LEFT}" x2="{_PLOT_LEFT + plot_width}" '
        f'y1="{_PLOT_BASELINE}" y2="{_PLOT_BASELINE}"></line>',
        f'<text class="axis" x="{_PLOT_LEFT - 8}" y="{_PLOT_BASELINE + 4}" '
        f'text-anchor="end">{y_floor:.2f}</text>',
        f'<text class="axis" x="{_PLOT_LEFT - 8}" y="{_PLOT_TOP + 4}" text-anchor="end">'
        f"{ceiling:.2f}</text>",
    ]
    # Ticks that would collide are dropped rather than drawn over each other: on a log axis
    # the last two sizes sit a few pixels apart, and two overlapping numbers read as neither.
    drawn_at = -1e9
    for tick in sorted(x_ticks):
        position = x_of(tick)
        if position - drawn_at < 44:
            continue
        drawn_at = position
        parts.append(
            f'<text class="axis" x="{position:.1f}" y="{_PLOT_BASELINE + 18}" '
            f'text-anchor="middle">{_thousands(tick)}</text>'
        )
    for point in points:
        parts.append(
            f'<circle class="series-dot" cx="{x_of(point.x):.1f}" cy="{y_of(point.value):.1f}" '
            f'r="3.5"></circle>'
        )
    last = points[-1]
    parts.append(
        f'<text class="bar-value" x="{x_of(last.x):.1f}" y="{y_of(last.value) - 10:.1f}" '
        f'text-anchor="end">{last.value:.3f} at n={_thousands(last.x)}</text>'
    )
    parts.append(
        f'<text class="axis" x="{_PLOT_LEFT}" y="{_PLOT_HEIGHT - 6}">{_text(x_caption)}</text>'
    )
    return _svg(_WIDTH, _PLOT_HEIGHT, title, "".join(parts))


def reliability_chart(points: Sequence[ReliabilityPoint], title: str) -> str:
    """Confidence against how often that confidence was right, over the diagonal it should sit on.

    Points above the diagonal are an under-confident model — it is right more often than it
    claims. That is not a harmless direction: a deferral rule written as "hand on anything
    below 0.7" would escalate a band this model gets entirely right.
    """
    if not points:
        return '<p class="note">No calibration bins in the committed metrics.</p>'

    size = 210
    left = 46
    top = 14
    width = 420
    height = size + top + 46

    def x_of(value: float) -> float:
        return left + value * size

    def y_of(value: float) -> float:
        return top + (1.0 - value) * size

    parts = [
        f'<line class="identity" x1="{x_of(0)}" y1="{y_of(0)}" x2="{x_of(1)}" y2="{y_of(1)}">'
        f"</line>",
        f'<line class="axis-line" x1="{x_of(0)}" y1="{y_of(0)}" x2="{x_of(1)}" y2="{y_of(0)}">'
        f"</line>",
        f'<line class="axis-line" x1="{x_of(0)}" y1="{y_of(0)}" x2="{x_of(0)}" y2="{y_of(1)}">'
        f"</line>",
        f'<text class="axis" x="{x_of(0) - 8}" y="{y_of(0) + 4}" text-anchor="end">0</text>',
        f'<text class="axis" x="{x_of(0) - 8}" y="{y_of(1) + 4}" text-anchor="end">1</text>',
        f'<text class="axis" x="{x_of(0)}" y="{y_of(0) + 18}">0</text>',
        f'<text class="axis" x="{x_of(1)}" y="{y_of(0) + 18}" text-anchor="end">1</text>',
        f'<text class="axis" x="{x_of(0.5):.1f}" y="{y_of(0) + 34}" text-anchor="middle">'
        f"the model's confidence</text>",
        f'<text class="axis" x="{x_of(0) - 8}" y="{top - 2}" text-anchor="end">accuracy</text>',
    ]
    for point in points:
        parts.append(
            f'<circle class="series-dot" cx="{x_of(point.confidence):.1f}" '
            f'cy="{y_of(point.accuracy):.1f}" r="4"></circle>'
            f'<text class="bar-value" x="{x_of(point.confidence) + 8:.1f}" '
            f'y="{y_of(point.accuracy) + 4:.1f}">n={point.count}</text>'
        )
    parts.append(
        f'<text class="axis" x="{left}" y="{height - 4}">'
        f"one point per populated confidence bin; the dashed line is perfect calibration</text>"
    )
    return _svg(width, height, title, "".join(parts))


def carries_counts(markup: str) -> bool:
    """Whether a rendered figure states the counts behind it.

    Checked structurally rather than by wording: matching on a phrase would pass the day
    someone rephrases a label and silently ship an unlabelled chart.
    """
    return 'class="bar-value"' in markup or 'class="cell-text' in markup
