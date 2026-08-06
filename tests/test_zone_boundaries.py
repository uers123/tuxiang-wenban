"""Tests for SBT-style zone boundary interval extraction.

Uses synthetic images (drawn with PIL) so the tests are deterministic and
do not depend on real scans or Tesseract.
"""

from __future__ import annotations

import numpy as np
import pytest

from doc_textify.chart import _extract_zone_boundaries

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

pytestmark = pytest.mark.skipif(not HAS_PIL, reason="Pillow required")


def _make_panel_image(
    width: int = 400,
    height: int = 300,
    *,
    diagonals: list[tuple[int, int, int, int]] | None = None,
    horizontals: list[int] | None = None,
    verticals: list[int] | None = None,
    line_color: tuple[int, int, int] = (0, 0, 0),
    background: tuple[int, int, int] = (255, 255, 255),
) -> "Image.Image":
    """Create a synthetic chart panel with lines drawn on a white background."""
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in (diagonals or []):
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)
    for y in (horizontals or []):
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
    for x in (verticals or []):
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
    return img


def _panel_dict(panel_id: str = "a", width: int = 400, height: int = 300) -> dict:
    return {
        "id": panel_id,
        "x0": 0, "y0": 0, "x1": width, "y1": height,
        "y_min": 0.0, "y_max": 30.0, "x_min": 0.0, "x_max": 8.0,
    }


def test_zone_boundaries_detect_diagonals() -> None:
    """Two long diagonal lines → two zone boundary intervals."""
    img = _make_panel_image(
        diagonals=[
            (50, 250, 350, 50),    # steep diagonal (slope ~0.67)
            (60, 270, 320, 230),   # shallow-but-valid diagonal (slope ~0.15)
        ],
    )
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    # Both diagonals should survive filtering (slope in [0.15, 5.0])
    assert len(intervals) >= 2, f"expected >=2 zone boundaries, got {intervals}"
    for inv in intervals:
        assert inv["type"] == "interval"
        assert inv["panel_id"] == "a"
        assert inv["source"] == "zone_boundary"
        assert inv["class"] >= 1
        assert inv["end_depth"] >= inv["start_depth"]


def test_zone_boundaries_exclude_horizontal_grid() -> None:
    """Horizontal grid lines (slope ~0) must NOT be reported as boundaries."""
    img = _make_panel_image(
        diagonals=[(30, 260, 370, 60)],
        horizontals=[50, 100, 150, 200, 250],
    )
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    # Only the diagonal survives; the horizontals are filtered by slope.
    assert len(intervals) == 1, f"expected 1 zone boundary, got {len(intervals)}"
    span = intervals[0]["end_depth"] - intervals[0]["start_depth"]
    assert span > 0


def test_zone_boundaries_exclude_vertical_error_bars() -> None:
    """Vertical lines (slope → ∞) must NOT be reported as boundaries."""
    img = _make_panel_image(
        diagonals=[(40, 250, 340, 80)],
        verticals=[80, 160, 240, 320],
    )
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    assert len(intervals) == 1, f"expected 1 zone boundary, got {len(intervals)}"


def test_zone_boundaries_no_lines() -> None:
    """Blank panel → no zone boundaries."""
    img = _make_panel_image()
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    assert intervals == []


def test_zone_boundaries_class_ordered_by_depth() -> None:
    """Classes are assigned 1..N by ascending mean data depth."""
    img = _make_panel_image(
        diagonals=[
            (50, 250, 350, 200),   # lower boundary (large y, small depth)
            (50, 80, 350, 30),     # upper boundary (small y, large depth)
        ],
    )
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    assert len(intervals) == 2
    classes = sorted(inv["class"] for inv in intervals)
    assert classes == [1, 2]
    # Lower line (smaller depth) gets class 1
    lower = next(inv for inv in intervals if inv["class"] == 1)
    upper = next(inv for inv in intervals if inv["class"] == 2)
    assert lower["start_depth"] <= upper["start_depth"]


def test_zone_boundaries_short_fragments_dropped() -> None:
    """Short diagonal fragments (text/symbol noise) are filtered out."""
    img = _make_panel_image(
        diagonals=[
            (20, 280, 380, 40),    # long, real boundary
            (150, 155, 180, 145),  # tiny fragment (len ~31 < min)
            (200, 130, 215, 120),  # tiny fragment
        ],
    )
    panel = _panel_dict()
    intervals = _extract_zone_boundaries(img, panel, [], 300, 400)
    # The tiny fragments should not survive the min-length / cluster filters
    assert len(intervals) == 1, f"expected 1 real boundary, got {len(intervals)}"
