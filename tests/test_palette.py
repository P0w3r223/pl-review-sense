"""The stylesheet's own palette, checked against WCAG contrast.

The page is read in two colour schemes and nobody sees both at once, so the second one drifts
quietly. These tests read the variables out of `styles.css` and do the arithmetic, which is the
part of a visual review that does not need a browser — and the part that regresses when someone
lightens a colour because it looked nicer in the scheme they had open.

Thresholds are WCAG 2.2: 4.5:1 for body-sized text, 3:1 for graphical objects that carry
meaning. Marks whose value is also printed beside them still need 3:1 — a bar nobody can see is
not a redundant encoding, it is a missing one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pl_review_sense.site import build, charts

STYLESHEET = build.ASSET_DIR / "styles.css"

TEXT_MINIMUM = 4.5
GRAPHIC_MINIMUM = 3.0
# Below this the muted colour stops being distinguishable from the accent, and "muted" would
# read as "the same bar" rather than as a quieter one.
MUTED_SEPARATION = 1.4


def _variables(source: str) -> dict[str, dict[str, str]]:
    """The light and dark palettes, as the browser resolves them.

    Light is the `:root` block; dark is `:root` with the `prefers-color-scheme` block applied
    over it, which is how the stylesheet is built — dark restates only what it changes.
    """
    blocks = re.findall(r"\{([^}]*)\}", source, re.S)
    light = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", blocks[0]))
    dark = dict(light)
    for block in blocks[1:3]:
        found = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block))
        if "bg" in found:
            dark.update(found)
            break
    return {"light": light, "dark": dark}


PALETTES = _variables(STYLESHEET.read_text(encoding="utf-8"))
SCHEMES = list(PALETTES)


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _luminance(colour: tuple[float, float, float]) -> float:
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in colour]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(_rgb(first)), _luminance(_rgb(second))), reverse=True)
    return (high + 0.05) / (low + 0.05)


def composite(top: str, bottom: str, alpha: float) -> str:
    """What the browser paints when `top` is drawn at `alpha` over `bottom`."""
    a, b = _rgb(top), _rgb(bottom)
    mixed = [alpha * a[i] + (1 - alpha) * b[i] for i in range(3)]
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in mixed)


def test_both_schemes_were_found_in_the_stylesheet():
    assert set(SCHEMES) == {"light", "dark"}
    assert PALETTES["light"]["bg"] != PALETTES["dark"]["bg"], "dark mode swaps the variables"


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("ink, paper", [("text", "bg"), ("text", "surface"),
                                        ("muted", "bg"), ("muted", "surface"),
                                        ("accent", "bg")])
def test_text_colours_are_readable(scheme, ink, paper):
    palette = PALETTES[scheme]
    assert contrast(palette[ink], palette[paper]) >= TEXT_MINIMUM


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_muted_mark_is_still_a_mark(scheme):
    """A greyed bar has to be visible; its number being printed beside it is not a substitute."""
    palette = PALETTES[scheme]
    assert contrast(palette["accent-soft"], palette["bg"]) >= GRAPHIC_MINIMUM


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_muted_mark_is_still_quieter_than_a_normal_one(scheme):
    palette = PALETTES[scheme]
    assert contrast(palette["accent-soft"], palette["accent"]) >= MUTED_SEPARATION


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_bar_stands_out_from_the_page_it_is_drawn_on(scheme):
    palette = PALETTES[scheme]
    assert contrast(palette["accent"], palette["bg"]) >= GRAPHIC_MINIMUM


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_fraction_track_is_outlined_in_something_visible(scheme):
    """The track is drawn in --border, which is far under 3:1 — the outline is what carries it."""
    palette = PALETTES[scheme]
    assert "stroke: var(--muted)" in STYLESHEET.read_text(encoding="utf-8")
    assert contrast(palette["muted"], palette["bg"]) >= GRAPHIC_MINIMUM


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("share", [i / 20 for i in range(21)])
def test_every_confusion_cell_keeps_its_count_readable(scheme, share):
    """One label colour over a capped shading ramp, at every share a matrix can produce."""
    palette = PALETTES[scheme]
    cell = composite(palette["accent"], palette["bg"], share * charts._CELL_MAX_OPACITY)
    assert contrast(palette["text"], cell) >= TEXT_MINIMUM
