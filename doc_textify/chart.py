"""Phase 3: Chart/figure understanding for doc-textify.

Analyses color images to extract structured data from charts:
  - Panel/subplot detection
  - Axis label OCR integration
  - Multi-color element extraction (data points, intervals, lines)
  - Auto color detection via K-means clustering
  - Pixel-to-data-coordinate mapping
  - JSON-compatible chart_data output for evaluation framework

Requires opencv-python-headless for full functionality. Falls back to
Pillow-only analysis when OpenCV is not available (limited capability).
"""

from __future__ import annotations

import csv
import io
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .calibration import calibrate_axis, pixel_to_data
from .models import Block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_chart(
    color_image,
    ocr_blocks: list[Block],
    page_width: float | None = None,
    page_height: float | None = None,
    *,
    chart_colors: list[str] | None = None,
) -> dict[str, Any]:
    """Analyse a colour image for chart content.

    Args:
        color_image: PIL Image in RGB mode.
        ocr_blocks: OCR blocks extracted from the same image.
        page_width, page_height: Image dimensions.
        chart_colors: List of colour names to extract (e.g. ["red", "blue"]).
            If None, auto-detects dominant chart colours via K-means.

    Returns:
        dict with key "chart_data" → list of structured data objects
        (intervals, points) compatible with the evaluation framework.
    """
    result: dict[str, Any] = {"chart_data": []}

    if color_image is None:
        return result

    # Ensure RGB
    if color_image.mode != "RGB":
        try:
            from PIL import Image
            color_image = color_image.convert("RGB")
        except Exception:
            return result

    img_w, img_h = color_image.size

    # Step 1: Detect chart panels (subplot regions)
    panels = _detect_panels(color_image, ocr_blocks, img_w, img_h)
    if not panels:
        panels = _detect_panels_fallback(color_image, ocr_blocks, img_w, img_h)

    if panels:
        # Step 2: Auto-detect chart colours if not specified
        try:
            if chart_colors is None:
                chart_colors, color_meta = _auto_detect_chart_colors(color_image, panels)
                chart_colors = chart_colors or ["red"]
                result["detected_colors"] = chart_colors
                result["detected_colors_metadata"] = color_meta
            else:
                result["detected_colors"] = chart_colors
                result["detected_colors_metadata"] = {"method": "manual", "colors": chart_colors}

            # Step 3: For each panel, extract coloured elements and map to data
            for panel in panels:
                for color_name in chart_colors:
                    panel_data = _extract_panel_data(
                        color_image, panel, ocr_blocks, target_color=color_name,
                    )
                    result["chart_data"].extend(panel_data)
        except Exception:
            # Panel/colour extraction is best-effort.  If anything raises
            # (e.g. numpy/OpenCV edge cases), fall through to the coarse
            # structure detector below instead of losing the figure entirely.
            pass

    # Step 4: Coarse structure fallback.
    # Colour-based extraction only works for coloured charts.  Black-and-
    # white / line-drawn charts (e.g. classification charts with numbered
    # zones) produce no intervals/points; panel detection may also fail on
    # busy figures.  When chart_data is still empty, look for axis-frame-
    # like structure (long horizontal + vertical lines) and, if found, emit
    # a coarse ``chart_detected`` entry with best-effort axis labels rather
    # than returning nothing.
    if not result["chart_data"]:
        fallback_entries = _chart_structure_fallback(
            color_image, ocr_blocks, img_w, img_h,
        )
        if fallback_entries:
            result["chart_data"].extend(fallback_entries)
            result["chart_detected"] = True
            result["detection_method"] = "structure_fallback"

    # Step 5: SBT-style zone boundary intervals (independent supplement).
    # Zone boundaries are black diagonal polylines — invisible to the
    # colour pipeline AND to the structure fallback.  Run per panel as a
    # supplement so intervals coexist with any fallback points/structures.
    if panels:
        for panel in panels:
            zx0, zy0, zx1, zy1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
            zph = zy1 - zy0
            zpw = zx1 - zx0
            zone_intervals = _extract_zone_boundaries(
                color_image, panel, ocr_blocks, zph, zpw,
            )
            result["chart_data"].extend(zone_intervals)

    return result


# ---------------------------------------------------------------------------
# Panel detection (OpenCV path)
# ---------------------------------------------------------------------------

def _detect_panels(
    color_image, ocr_blocks, img_w: int, img_h: int,
) -> list[dict[str, Any]]:
    """Detect chart panel regions via rectangle detection."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = np.array(color_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (img_w * img_h) * 0.05
    max_area = (img_w * img_h) * 0.60

    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 3.0:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        rectangles.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h, "area": area})

    if not rectangles:
        return []

    rectangles.sort(key=lambda r: r["x0"])

    panels = []
    for i, rect in enumerate(rectangles):
        panel_id = chr(ord("a") + i) if i < 26 else f"p{i}"
        inferred_y_max = 30.0
        if len(rectangles) == 2 and i == 0:
            inferred_y_max = 28.0
        panels.append({
            "id": panel_id,
            "x0": rect["x0"], "y0": rect["y0"],
            "x1": rect["x1"], "y1": rect["y1"],
            "x_label": "", "y_label": "",
            "x_min": 0.0, "x_max": 5.0,
            "y_min": 0.0, "y_max": inferred_y_max,
            "y_direction": "down",
        })

    return _annotate_panels_with_ocr(panels, ocr_blocks)


# ---------------------------------------------------------------------------
# Panel detection fallback (Pillow-only)
# ---------------------------------------------------------------------------

def _detect_panels_fallback(color_image, ocr_blocks, img_w, img_h):
    """Simple panel detection using edge detection via Pillow."""
    try:
        from PIL import Image, ImageFilter
        import numpy as np
    except ImportError:
        return []

    gray = color_image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)
    h_proj = edge_arr.sum(axis=1)
    v_proj = edge_arr.sum(axis=0)
    h_thresh = np.max(h_proj) * 0.1
    v_thresh = np.max(v_proj) * 0.1

    h_gaps = _find_projection_gaps(h_proj, h_thresh, min_gap=int(img_h * 0.02))
    v_gaps = _find_projection_gaps(v_proj, v_thresh, min_gap=int(img_w * 0.02))

    if not h_gaps and not v_gaps:
        return []

    h_regions = _gap_to_regions(h_gaps, 0, img_h) or [(0, img_h)]
    v_regions = _gap_to_regions(v_gaps, 0, img_w) or [(0, img_w)]

    panels = []
    for idx, (y0, y1) in enumerate(h_regions):
        for jdx, (x0, x1) in enumerate(v_regions):
            panel_area = (x1 - x0) * (y1 - y0)
            if panel_area < (img_w * img_h) * 0.05:
                continue
            panel_id = chr(ord("a") + len(panels))
            panels.append({
                "id": panel_id,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "x_label": "", "y_label": "",
                "x_min": 0.0, "x_max": 5.0,
                "y_min": 0.0, "y_max": 30.0,
                "y_direction": "down",
            })

    return _annotate_panels_with_ocr(panels, ocr_blocks)


def _find_projection_gaps(proj, threshold, min_gap):
    """Find gaps in a projection profile."""
    gaps = []
    in_gap = False
    gap_start = 0
    for i in range(len(proj)):
        if proj[i] <= threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap and i - gap_start >= min_gap:
                gaps.append((gap_start, i))
                in_gap = False
    if in_gap and len(proj) - gap_start >= min_gap:
        gaps.append((gap_start, len(proj)))
    return gaps


def _gap_to_regions(gaps, start, end):
    """Convert gaps into regions."""
    if not gaps:
        return [(start, end)]
    regions = []
    cursor = start
    for gs, ge in gaps:
        if gs > cursor:
            regions.append((cursor, gs))
        cursor = ge
    if cursor < end:
        regions.append((cursor, end))
    return regions


# ---------------------------------------------------------------------------
# Coarse chart detection fallback (axis-frame structure)
# ---------------------------------------------------------------------------

#: Minimum length (fraction of image dimension) for a line to count as an
#: axis-frame/grid line in the structure fallback.  Short text strokes and
#: table cell rules are excluded, so plain text pages rarely trigger.
_STRUCTURE_MIN_H_FRAC = 0.20
_STRUCTURE_MIN_V_FRAC = 0.20


def _chart_structure_fallback(
    color_image, ocr_blocks, img_w: int, img_h: int,
) -> list[dict[str, Any]]:
    """Coarse chart detection when panel segmentation / colour extraction
    produced nothing.

    Black-and-white line charts (classification charts, axis frames, log-
    log plots, ...) have no coloured data elements, so the colour pipeline
    yields an empty ``chart_data`` even when panels were found.  This
    fallback instead looks for axis-frame-like structure: long horizontal
    *and* vertical lines, clustered into one region per connected group.
    For each region it emits a coarse ``chart_detected`` entry with
    best-effort axis labels read from nearby OCR blocks.

    Returns a list of chart_data dicts (possibly empty).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = np.array(color_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape

    # Morphological long-line extraction.  The kernels are long enough that
    # axis frames / grid lines survive the open, while isolated text strokes
    # (which are much shorter) are removed.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 24), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 24)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    h_contours = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    v_contours = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    min_h_len = w * _STRUCTURE_MIN_H_FRAC
    min_v_len = h * _STRUCTURE_MIN_V_FRAC
    long_h = [cv2.boundingRect(c) for c in h_contours if cv2.boundingRect(c)[2] >= min_h_len]
    long_v = [cv2.boundingRect(c) for c in v_contours if cv2.boundingRect(c)[3] >= min_v_len]

    # A real chart needs both horizontal and vertical frame evidence.
    if not long_h or not long_v:
        return []

    # Merge long lines into a mask and cluster them into chart regions
    # (each connected cluster = one chart box).
    line_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, lw, lh in long_h:
        line_mask[y:y + lh, x:x + lw] = 255
    for x, y, lw, lh in long_v:
        line_mask[y:y + lh, x:x + lw] = 255
    line_mask = cv2.dilate(line_mask, np.ones((7, 7), np.uint8), iterations=1)

    num_comps, labels = cv2.connectedComponents(line_mask)

    entries: list[dict[str, Any]] = []
    for comp_id in range(1, num_comps):
        ys, xs = np.nonzero(labels == comp_id)
        if len(ys) < 50:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 - x0 < min_h_len or y1 - y0 < min_v_len:
            continue

        # Require both horizontal and vertical line evidence inside this
        # cluster (rejects e.g. a long underline merged with a border).
        h_inside = sum(
            1 for (lx, ly, lw, lh) in long_h
            if lx < x1 and lx + lw > x0 and ly < y1 and ly + lh > y0
        )
        v_inside = sum(
            1 for (lx, ly, lw, lh) in long_v
            if lx < x1 and lx + lw > x0 and ly < y1 and ly + lh > y0
        )
        if h_inside < 1 or v_inside < 1:
            continue

        entries.append(_build_structure_entry(
            ocr_blocks, x0, y0, x1, y1,
            h_inside=h_inside, v_inside=v_inside,
        ))
        # Data-point-level extraction: dark scatter marks inside the plot
        # area, mapped to data coordinates and classified into zones.
        # Best-effort — never fails the structure fallback.
        try:
            point_entries = _extract_chart_points(color_image, ocr_blocks, x0, y0, x1, y1)
            entries.extend(point_entries)
        except Exception:
            pass

    # If clustering produced no usable region but we clearly have both
    # long-H and long-V lines, emit one entry for the union bounding box.
    if not entries:
        all_x0 = min(min(r[0] for r in long_h), min(r[0] for r in long_v))
        all_y0 = min(min(r[1] for r in long_h), min(r[1] for r in long_v))
        all_x1 = max(max(r[0] + r[2] for r in long_h), max(r[0] + r[2] for r in long_v))
        all_y1 = max(max(r[1] + r[3] for r in long_h), max(r[1] + r[3] for r in long_v))
        if all_x1 - all_x0 >= min_h_len and all_y1 - all_y0 >= min_v_len:
            entries.append(_build_structure_entry(
                ocr_blocks, all_x0, all_y0, all_x1, all_y1,
                h_inside=len(long_h), v_inside=len(long_v),
            ))

    return entries


# ---------------------------------------------------------------------------
# Data-point-level chart extraction (structure fallback path)
# ---------------------------------------------------------------------------

#: Fallback zone-anchor tables for the standard Robertson-style soil
#: behaviour type (SBT) classification chart layout, expressed as plot
#: fractions (x_rel, y_frac) so they transfer between figures with the
#: same chart geometry.  Used when in-plot zone-number OCR fails to
#: produce enough anchors (common on degraded scans).  Positions follow
#: the published chart: Q vs friction ratio (left-hand panel, log x-axis)
#: and Q vs pore-pressure ratio (right-hand panel, linear x-axis).
_SBT_ZONE_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {
    "friction": {
        1: (0.186, 0.836), 2: (0.945, 0.896), 3: (0.981, 0.696),
        4: (0.486, 0.665), 5: (0.361, 0.523), 6: (0.219, 0.364),
        7: (0.180, 0.095), 8: (0.820, 0.106), 9: (1.006, 0.171),
    },
    "pore_pressure": {
        1: (0.887, 0.827), 2: (0.525, 0.900), 3: (0.503, 0.687),
        4: (0.282, 0.486), 5: (0.258, 0.415), 6: (0.167, 0.253),
        7: (0.159, 0.098),
    },
}

#: Plot-area inset used to exclude axis frames / tick marks from the
#: data-mark search.
_PLOT_INSET = 6
#: Scatter mark size bounds (connected-component area, in pixels).
_MARK_AREA_MIN = 5
_MARK_AREA_MAX = 400


def _extract_chart_points(
    color_image, ocr_blocks, x0: int, y0: int, x1: int, y1: int,
) -> list[dict[str, Any]]:
    """Extract data-point-level entries (``type="point"``) for a detected
    chart region produced by the structure fallback.

    Pipeline:
      1. Determine the chart kind from the x-axis label (friction-ratio
         chart → panel ``a``, log x; pore-pressure-ratio chart → panel
         ``b``, linear x).  Unknown kinds yield no points.
      2. Calibrate the x axis from OCR'd tick labels below the frame.
      3. Locate zone-number anchors: in-plot digit OCR, falling back to
         the SBT layout table (plot fractions).
      4. Detect dark scatter marks inside the plot area (connected
         components after long-line removal).
      5. Emit one ``point`` per mark: ``depth`` = x-axis data value,
         ``class`` = nearest zone anchor, ``class_candidates`` = up to
         three nearest zones (boundary ambiguity).

    Best-effort: returns ``[]`` on any failure.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img_w, img_h = color_image.size
    rw, rh = x1 - x0, y1 - y0
    if rw < 80 or rh < 80:
        return []

    # ── 1. chart kind / panel id from the x-axis label ──────────────────
    x_label, _ = _structure_axis_labels(ocr_blocks, (x0, y0, x1, y1))
    upper = (x_label or "").upper()
    if "FRICTION" in upper:
        kind, panel_id, log_axis = "friction", "a", True
    elif "PORE" in upper or "PRESSURE" in upper:
        kind, panel_id, log_axis = "pore_pressure", "b", False
    else:
        return []

    # Plot area (inset from the axis frame)
    px0, py0 = x0 + _PLOT_INSET, y0 + _PLOT_INSET
    px1, py1 = x1 - _PLOT_INSET, y1 - _PLOT_INSET
    if px1 <= px0 or py1 <= py0:
        return []
    plot_w, plot_h = px1 - px0, py1 - py0

    # ── 2. x-axis calibration from tick labels ──────────────────────────
    calib = _calibrate_axis_ticks(color_image, (px0, py0, px1, py1), log_axis)
    if calib is None:
        return []

    # ── 3. zone anchors ─────────────────────────────────────────────────
    anchors = _find_zone_anchors(color_image, kind, (px0, py0, px1, py1))
    if len(anchors) < 3:
        return []

    # ── 4. data marks ───────────────────────────────────────────────────
    img = np.array(color_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plot = gray[py0:py1, px0:px1]
    _, bw = cv2.threshold(plot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, plot_w // 10), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, plot_h // 10)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((5, 5), np.uint8))
    marks = cv2.subtract(bw, lines)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(marks, 8)
    points: list[dict[str, Any]] = []
    for i in range(1, num):
        mx, my, mw, mh, area = stats[i]
        if area < _MARK_AREA_MIN or area > _MARK_AREA_MAX:
            continue
        if mw < 3 or mh < 3:
            continue
        # exclude axis ticks / frame residue near the plot edges
        if mx < 8 or my < 8 or mx + mw > plot_w - 8 or my + mh > plot_h - 8:
            continue
        cx = px0 + int(cents[i][0])
        cy = py0 + int(cents[i][1])

        depth = _tick_to_data(calib, cx, log_axis)
        if depth is None:
            continue
        best = sorted(
            ((math.hypot(cx - ax, cy - ay), z) for z, (ax, ay) in anchors.items()),
            key=lambda t: t[0],
        )
        klass = best[0][1]
        candidates = [z for _, z in best[:3]]
        points.append({
            "type": "point",
            "panel_id": panel_id,
            "class": klass,
            "class_candidates": candidates,
            "depth": round(depth, 2),
            "depth_tolerance": 0.2,
            "pixel_x": round(cx, 1),
            "pixel_y": round(cy, 1),
            "color": "black",
            "source": "data_mark",
        })

    return _dedup_points(points)


def _calibrate_axis_ticks(
    color_image, plot: tuple[int, int, int, int], log_axis: bool,
) -> dict[str, Any] | None:
    """Calibrate the x axis of a chart region from OCR'd tick labels.

    Crops the horizontal strip just below the plot frame, upscales it and
    OCRs with a digit whitelist (TSV for positions).  Returns a dict
    ``{"kind": "log"|"linear", "a": ..., "b": ...}`` mapping pixel x
    → data value (``log10(value) = a*px + b`` for log, ``value = a*px + b``
    for linear), or ``None`` when fewer than two valid ticks are found.
    """
    try:
        from .extractors import _find_tesseract, _tesseract_env
    except Exception:
        return None
    tesseract = _find_tesseract()
    if tesseract is None:
        return None

    import csv
    import io

    px0, py0, px1, py1 = plot
    strip_h = max(40, int((py1 - py0) * 0.16))
    y0 = max(0, py1 - 4)
    y1 = min(color_image.height, py1 + strip_h)
    x0 = max(0, px0 - 12)
    x1 = min(color_image.width, px1 + 16)
    strip = color_image.crop((x0, y0, x1, y1))
    if strip.width < 40 or strip.height < 8:
        return None

    w, h = strip.size
    scale = max(3, int(240 / max(w, 1)))
    if scale > 1:
        from PIL import Image
        strip = strip.resize((w * scale, h * scale), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_ticks_", suffix=".png", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        strip.save(tmp_path)
        env = _tesseract_env(tesseract)
        r = subprocess.run(
            [str(tesseract), str(tmp_path), "stdout", "-l", "eng", "--psm", "6",
             "-c", "tessedit_char_whitelist=0123456789.", "tsv"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=25,
        )
        if r.returncode != 0:
            return None
        rows = list(csv.DictReader(io.StringIO(r.stdout), delimiter="\t"))
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)

    ticks: list[tuple[float, float]] = []  # (value, pixel_x)
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf", "-1"))
        except ValueError:
            conf = -1
        if conf < 40:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if log_axis:
            # log-axis ticks are "nice" decade values (0.1, 1, 2 … 9, 10);
            # reject OCR misreads such as "16" for "10" or stray "2100".
            if not _is_nice_log_tick(value):
                continue
        elif not (-0.6 <= value <= 1.6):
            continue
        try:
            left = float(row["left"])
            width = float(row["width"])
        except (KeyError, ValueError):
            continue
        px = x0 + (left + width / 2) / scale
        ticks.append((value, px))

    if len(ticks) < 2:
        return None

    # cluster near-duplicate ticks (same label read twice)
    ticks.sort(key=lambda t: t[1])
    clustered: list[tuple[float, float]] = []
    for value, px in ticks:
        if clustered and abs(px - clustered[-1][1]) <= 14:
            # keep the one closest to a whole-number value
            if abs(value - round(value)) < abs(clustered[-1][0] - round(clustered[-1][0])):
                clustered[-1] = (value, px)
            continue
        clustered.append((value, px))
    ticks = clustered
    if len(ticks) < 2:
        return None

    # fit (log10(value) vs px) or (value vs px) with least squares
    xs = [px for _, px in ticks]
    ys = [math.log10(v) for v, _ in ticks] if log_axis else [v for v, _ in ticks]
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    if log_axis and (slope <= 0 or abs(slope) < 1e-6):
        return None

    if log_axis:
        # Anchor the log line exactly on the "1" tick when present — its
        # label is the most reliably OCR'd and removes sub-decade drift.
        one_tick = [px for v, px in ticks if abs(v - 1.0) < 1e-6]
        if one_tick:
            intercept = -slope * one_tick[0]

    return {"kind": "log" if log_axis else "linear", "a": slope, "b": intercept}


def _is_nice_log_tick(value: float) -> bool:
    """Return True when *value* is a decade/major log tick (1–9 × 10^k)."""
    import math as _m
    if value <= 0:
        return False
    mantissa = value / (10 ** _m.floor(_m.log10(value)))
    return abs(mantissa - round(mantissa)) < 1e-6 and 1 <= round(mantissa) <= 9


def _tick_to_data(calib: dict[str, Any], px: float, log_axis: bool) -> float | None:
    """Map a pixel x to a data value via the tick calibration."""
    try:
        if log_axis:
            return 10 ** (calib["a"] * px + calib["b"])
        return calib["a"] * px + calib["b"]
    except Exception:
        return None


def _find_zone_anchors(
    color_image, kind: str, plot: tuple[int, int, int, int],
) -> dict[int, tuple[float, float]]:
    """Locate zone-number anchors inside the plot area.

    Tries in-plot digit OCR (sparse text mode, digit whitelist); when that
    yields fewer than three distinct digits, falls back to the standard
    SBT zone-layout table expressed as plot fractions.

    Returns a dict mapping zone number → (pixel_x, pixel_y).
    """
    px0, py0, px1, py1 = plot
    anchors = _ocr_zone_digits(color_image, (px0, py0, px1, py1))
    if len(anchors) >= 3:
        return anchors

    table = _SBT_ZONE_ANCHORS.get(kind)
    if not table:
        return {}
    return {
        zone: (px0 + fx * (px1 - px0), py0 + fy * (py1 - py0))
        for zone, (fx, fy) in table.items()
    }


def _ocr_zone_digits(
    color_image, plot: tuple[int, int, int, int],
) -> dict[int, tuple[float, float]]:
    """OCR single-digit zone numbers inside the plot area.

    Returns {digit: (pixel_x, pixel_y)} for digits 1-9 with confidence
    above a threshold, deduplicated (clusters within 22 px keep the
    highest-confidence reading).
    """
    try:
        from .extractors import _find_tesseract, _tesseract_env
    except Exception:
        return {}
    tesseract = _find_tesseract()
    if tesseract is None:
        return {}

    import csv
    import io

    px0, py0, px1, py1 = plot
    plot_img = color_image.crop((px0, py0, px1, py1))
    if plot_img.width < 60 or plot_img.height < 60:
        return {}
    w, h = plot_img.size
    scale = max(3, int(360 / max(w, 1)))
    if scale > 1:
        from PIL import Image
        plot_img = plot_img.resize((w * scale, h * scale), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_zones_", suffix=".png", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        plot_img.save(tmp_path)
        env = _tesseract_env(tesseract)
        r = subprocess.run(
            [str(tesseract), str(tmp_path), "stdout", "-l", "eng", "--psm", "11",
             "-c", "tessedit_char_whitelist=0123456789", "tsv"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, timeout=30,
        )
        if r.returncode != 0:
            return {}
        rows = list(csv.DictReader(io.StringIO(r.stdout), delimiter="\t"))
    except Exception:
        return {}
    finally:
        tmp_path.unlink(missing_ok=True)

    found: list[tuple[int, float, float, float]] = []  # (digit, px, py, conf)
    for row in rows:
        text = (row.get("text") or "").strip()
        if len(text) != 1 or not text.isdigit() or text == "0":
            continue
        digit = int(text)
        try:
            conf = float(row.get("conf", "-1"))
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
        except (KeyError, ValueError):
            continue
        if conf < 50:
            continue
        found.append((digit, px0 + (left + width / 2) / scale, py0 + (top + height / 2) / scale, conf))

    # cluster by position; keep the highest-confidence digit per cluster
    found.sort(key=lambda f: (f[2], f[1]))
    clusters: list[list[tuple[int, float, float, float]]] = []
    for f in found:
        if clusters and abs(f[1] - clusters[-1][0][1]) <= 22 and abs(f[2] - clusters[-1][0][2]) <= 22:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    anchors: dict[int, tuple[float, float]] = {}
    for cl in clusters:
        cl.sort(key=lambda f: f[3], reverse=True)
        digit, px, py, _conf = cl[0]
        anchors[digit] = (px, py)
    return anchors


def _build_structure_entry(
    ocr_blocks, x0: int, y0: int, x1: int, y1: int,
    *, h_inside: int, v_inside: int,
) -> dict[str, Any]:
    """Build a coarse ``chart_detected`` chart_data entry for a region."""
    x_label, y_label = _structure_axis_labels(ocr_blocks, (x0, y0, x1, y1))
    annotations = _structure_annotations(ocr_blocks, (x0, y0, x1, y1))
    caption = _structure_caption(ocr_blocks, (x0, y0, x1, y1))

    confidence = round(min(0.95, 0.35 + 0.10 * min(h_inside, 4)
                           + 0.10 * min(v_inside, 4)), 2)

    return {
        "type": "chart_detected",
        "panel_id": "chart",
        "confidence": confidence,
        "axis_labels": {"x": x_label, "y": y_label},
        "caption": caption,
        "structure": {
            "horizontal_lines": h_inside,
            "vertical_lines": v_inside,
            "region": [x0, y0, x1, y1],
            "region_width": x1 - x0,
            "region_height": y1 - y0,
        },
        "text_annotations": annotations,
        "detection": "structure_fallback",
    }


def _structure_axis_labels(
    ocr_blocks, region: tuple[int, int, int, int],
) -> tuple[str, str]:
    """Best-effort axis labels from OCR blocks near the region edges.

    X label: the block whose top edge is closest below the region's bottom
    edge, horizontally overlapping the region's central span.  Y label: the
    longest block strictly left of the region whose vertical centre lies
    inside the region's vertical span (rotated axis titles are tall blocks
    centred on the plot).  Legend/caption text is filtered out.
    """
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0

    def _is_legend_or_caption(text: str) -> bool:
        t = text.strip()
        if re.match(r"^\s*\d+[\.\)]", t):
            return True
        if re.match(r"^.{0,12}?(fic|fig|figure)\.?\s*\d", t, re.IGNORECASE):
            return True
        return False

    x_cands: list[tuple[float, str, float]] = []  # (gap_below, text, by0)
    y_cands: list[tuple[int, str, float]] = []    # (len, text, bx1)
    for block in ocr_blocks:
        if not block.bbox:
            continue
        bx0, by0, bx1, by1 = block.bbox
        text = " ".join(block.text.split())
        if not text or _is_legend_or_caption(text):
            continue
        bcx = (bx0 + bx1) / 2
        bcy = (by0 + by1) / 2

        # X-axis label: directly below the frame, horizontally centred.
        # Axis titles are multi-character; single stray tokens (e.g. "2")
        # are tick residue and are ignored.
        if (len(text) >= 4
                and y1 < by0 <= y1 + max(rh * 0.25, 40)
                and x0 + rw * 0.15 <= bcx <= x1 - rw * 0.15):
            x_cands.append((by0 - y1, text, by0))

        # Y-axis label: left of the frame, vertically centred on it.
        # (A shared y-axis title sits left of the *first* subplot, so allow
        # a generous distance for right-hand subplots.)
        if (len(text) >= 3
                and bx1 < x0 + rw * 0.05
                and y0 - rh * 0.05 <= bcy <= y1 + rh * 0.05):
            y_cands.append((len(text), text, bx1))

    x_label = min(x_cands, key=lambda t: (t[0], -len(t[1])))[1] if x_cands else ""
    y_label = max(y_cands, key=lambda t: (t[0], -t[2]))[1] if y_cands else ""
    return x_label, y_label


def _structure_annotations(
    ocr_blocks, region: tuple[int, int, int, int],
) -> list[str]:
    """Short OCR tokens fully inside the region (tick values, zone
    numbers, in-plot labels).  Best-effort; deduplicated."""
    x0, y0, x1, y1 = region
    seen: set[str] = set()
    annotations: list[str] = []
    for block in ocr_blocks:
        if not block.bbox:
            continue
        bx0, by0, bx1, by1 = block.bbox
        text = block.text.strip()
        if not text or len(text) > 14:
            continue
        if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
            if text not in seen:
                seen.add(text)
                annotations.append(text)
    return annotations


def _structure_caption(
    ocr_blocks, region: tuple[int, int, int, int],
) -> str:
    """Figure caption text found directly below the region, if any."""
    _x0, y0, _x1, y1 = region
    for block in ocr_blocks:
        if not block.bbox:
            continue
        by0, by1 = block.bbox[1], block.bbox[3]
        if by0 < y1 or by0 > y1 + max((y1 - y0) * 0.6, 90):
            continue
        head = " ".join(block.text.split())[:60]
        if re.search(r"(fic|fig|figure)\.?\s*\d", head, re.IGNORECASE):
            return " ".join(block.text.split())
    return ""


# ---------------------------------------------------------------------------
# OCR annotation of panels
# ---------------------------------------------------------------------------

def _annotate_panels_with_ocr(
    panels: list[dict], ocr_blocks: list[Block],
) -> list[dict]:
    """Use OCR text blocks near panel edges to identify axis labels."""
    if not ocr_blocks:
        return panels

    for panel in panels:
        px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
        pw = px1 - px0
        ph = py1 - py0

        x_candidates = []
        y_candidates = []
        caption_candidates = []

        for block in ocr_blocks:
            if not block.bbox:
                continue
            bx0, by0, bx1, by1 = block.bbox
            bcx = (bx0 + bx1) / 2
            bcy = (by0 + by1) / 2

            if (bcx > px0 - pw * 0.2 and bcx < px1 + pw * 0.2
                    and by0 > py1 and by1 < py1 + ph * 0.3):
                x_candidates.append((block.text, by0))

            if (bx1 < px0 and bcy > py0 - ph * 0.2 and bcy < py1 + ph * 0.2):
                y_candidates.append((block.text, bx1))

            if (bcx > px0 - pw * 0.1 and bcx < px1 + pw * 0.1
                    and ((by1 < py0 and by0 > py0 - ph * 0.3)
                         or (by0 > py1 and by1 < py1 + ph * 0.3))):
                caption_candidates.append((block.text, bcy))

        if x_candidates:
            x_candidates.sort(key=lambda t: t[1])
            panel["x_label"] = x_candidates[0][0]
        if y_candidates:
            y_candidates.sort(key=lambda t: t[1], reverse=True)
            panel["y_label"] = y_candidates[0][0]
        if caption_candidates:
            caption_candidates.sort(key=lambda t: t[1])
            panel["caption"] = caption_candidates[0][0]

    return panels


# ===================================================================
# Multi-colour chart element extraction
# ===================================================================

# ── Full HSV colour preset library ──────────────────────────────────
# Each preset can have multiple sub-ranges (e.g. red wraps around 0).
# The "element_type" hint guides morphology kernel sizing.
# Saturation/value thresholds are deliberately low (≥40) to catch
# faded prints photos, and low-res scans.
# ────────────────────────────────────────────────────────────────────

CHART_COLOR_PRESETS: dict[str, dict] = {
    # ── Red (very common: predicted classes, emphasis intervals) ──
    "red": {
        "label": "red",
        "ranges": [
            {"lower": (0, 50, 40), "upper": (12, 255, 255)},
            {"lower": (168, 50, 40), "upper": (180, 255, 255)},
        ],
        "element_type": "fill",
    },
    "red_line": {
        "label": "red (lines only)",
        "ranges": [
            {"lower": (0, 100, 50), "upper": (10, 255, 255)},
            {"lower": (170, 100, 50), "upper": (180, 255, 255)},
        ],
        "element_type": "line",
    },
    # ── Blue (very common: true-class triangles, scatter points) ──
    "blue": {
        "label": "blue",
        "ranges": [
            {"lower": (95, 50, 40), "upper": (135, 255, 255)},
        ],
        "element_type": "fill",
    },
    "blue_dark": {
        "label": "dark blue",
        "ranges": [
            {"lower": (100, 100, 30), "upper": (135, 255, 180)},
        ],
        "element_type": "fill",
    },
    # ── Green ────────────────────────────────────────────────────────
    "green": {
        "label": "green",
        "ranges": [
            {"lower": (40, 50, 40), "upper": (80, 255, 255)},
        ],
        "element_type": "fill",
    },
    "green_dark": {
        "label": "dark green",
        "ranges": [
            {"lower": (40, 100, 30), "upper": (80, 255, 180)},
        ],
        "element_type": "fill",
    },
    # ── Cyan / Teal ──────────────────────────────────────────────────
    "cyan": {
        "label": "cyan",
        "ranges": [
            {"lower": (80, 60, 40), "upper": (100, 255, 255)},
        ],
        "element_type": "fill",
    },
    # ── Yellow / Orange ──────────────────────────────────────────────
    "yellow": {
        "label": "yellow",
        "ranges": [
            {"lower": (20, 50, 80), "upper": (38, 255, 255)},
        ],
        "element_type": "fill",
    },
    "orange": {
        "label": "orange",
        "ranges": [
            {"lower": (10, 80, 80), "upper": (22, 255, 255)},
        ],
        "element_type": "fill",
    },
    # ── Magenta / Purple ─────────────────────────────────────────────
    "magenta": {
        "label": "magenta",
        "ranges": [
            {"lower": (140, 50, 40), "upper": (168, 255, 255)},
        ],
        "element_type": "fill",
    },
    "purple": {
        "label": "purple",
        "ranges": [
            {"lower": (125, 50, 40), "upper": (155, 255, 255)},
        ],
        "element_type": "fill",
    },
    # ── Black / Dark grey ────────────────────────────────────────────
    "black": {
        "label": "black",
        "ranges": [
            {"lower": (0, 0, 0), "upper": (180, 255, 70)},
        ],
        "element_type": "line",
    },
    # ── White (for inverted colour schemes on dark backgrounds) ─────
    "white": {
        "label": "white",
        "ranges": [
            {"lower": (0, 0, 220), "upper": (180, 30, 255)},
        ],
        "element_type": "fill",
    },
}


def _auto_detect_chart_colors(
    color_image, panels: list[dict],
) -> tuple[list[str], dict[str, Any]]:
    """Detect dominant chart data colours using multi-K K-means with
    elbow-method, histogram fallback, adaptive saturation filtering,
    and contrast-ratio guard.

    Samples pixels from each panel's central region (excluding background)
    and clusters them. Returns (colour_names, metadata_dict).

    Falls back to (["red"], fallback_metadata) if detection fails.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return ["red"], {"method": "fallback", "reason": "opencv_unavailable"}

    # Sample pixels from panel interiors, avoiding edges (which are often
    # grid lines or axis frames)
    samples = []
    img = np.array(color_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for panel in panels:
        x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
        # Sample central 60% of each panel to avoid axis/grid-line bias
        mx0 = int(x0 + (x1 - x0) * 0.20)
        my0 = int(y0 + (y1 - y0) * 0.20)
        mx1 = int(x0 + (x1 - x0) * 0.80)
        my1 = int(y0 + (y1 - y0) * 0.80)
        if mx1 <= mx0 or my1 <= my0:
            continue
        region = hsv[my0:my1, mx0:mx1]
        # Flatten and downsample (every 4th pixel) to keep K-means fast
        pixels = region.reshape(-1, 3)[::4]
        samples.append(pixels)

    if not samples:
        return ["red"], {"method": "fallback", "reason": "no_samples"}

    all_pixels = np.concatenate(samples, axis=0)
    if len(all_pixels) < 100:
        return ["red"], {"method": "fallback", "reason": "too_few_pixels",
                          "pixel_count": len(all_pixels)}

    meta: dict[str, Any] = {"method": "kmeans_multi_k",
                            "pixel_count": int(len(all_pixels))}

    # ── Step 1: Multi-K K-means with elbow method + silhouette ───────
    k_values = [3, 4, 5, 6, 7, 8]
    best_k, centers, labels, k_scores = _find_best_k_multi(
        all_pixels, k_values,
    )
    meta["k_values_tried"] = k_values
    meta["k_scores"] = k_scores

    if centers is None or labels is None:
        hist_colors = _histogram_fallback(all_pixels)
        meta["method"] = "histogram_fallback"
        meta["fallback_reason"] = "kmeans_failed"
        return hist_colors, meta

    meta["best_k"] = best_k
    unique, counts = np.unique(labels, return_counts=True)
    total = float(sum(counts))

    # ── Step 2: Filter clusters with adaptive saturation threshold ───
    detected, sat_threshold, v_threshold = _filter_clusters_adaptive(
        centers, counts, total, all_pixels,
    )
    meta["saturation_threshold"] = sat_threshold
    meta["value_threshold"] = v_threshold
    meta["cluster_ratios"] = {}
    for center, count in zip(centers, counts):
        ratio = float(count) / total
        h, s, v = center
        cn = _hsv_to_color_name((float(h), float(s), float(v)))
        if cn and ratio >= 0.03 and ratio <= 0.70:
            meta["cluster_ratios"][cn] = round(ratio, 4)

    # ── Step 3: Histogram fallback if K-means found nothing useful ────
    if not detected or detected == {"red"}:
        hist_colors = _histogram_fallback(all_pixels)
        if hist_colors and hist_colors != ["red"]:
            meta["method"] = "histogram_fallback"
            meta["fallback_reason"] = "kmeans_only_found_red"
            return hist_colors, meta

    if not detected:
        return ["red"], dict(meta, method="fallback",
                            reason="no_colors_detected")

    # ── Step 4: Contrast guard (merge too-similar colours) ────────────
    before_guard = set(detected)
    detected = _contrast_guard(detected, centers, counts, total)
    if before_guard != detected:
        meta["contrast_merges"] = sorted(before_guard - detected)

    # Return in stable order
    ordered = [c for c in ["red", "blue", "green", "cyan", "yellow",
                             "orange", "magenta", "purple"] if c in detected]
    result = ordered or list(detected)
    meta["detected_colors"] = result
    return result, meta


def _hsv_to_color_name(hsv_center: tuple) -> str | None:
    """Map an HSV centre to the closest named colour preset.

    Uses fuzzy matching at boundaries: h=15 matches either red or orange
    depending on proximity.
    """
    h_val, s_val, v_val = hsv_center

    if s_val < 20:
        # Desaturated → black, grey, or white
        if v_val < 80:
            return "black"
        return None  # skip grey/white (usually background)

    # Define hue ranges with soft boundaries
    hue_ranges: list[tuple[str, int, int]] = [
        ("red", 0, 12),
        ("orange", 13, 24),
        ("yellow", 25, 37),
        ("green", 38, 79),
        ("cyan", 80, 99),
        ("blue", 100, 139),
        ("purple", 140, 159),
        ("magenta", 160, 168),
        ("red", 169, 180),  # red wraps around
    ]

    # Exact match
    for name, lo, hi in hue_ranges:
        if lo <= h_val <= hi:
            return name

    # Fuzzy match at boundaries: find closest range within 10°
    best_name = None
    best_dist = float("inf")
    for name, lo, hi in hue_ranges:
        if h_val < lo:
            dist = lo - h_val
        elif h_val > hi:
            dist = h_val - hi
        else:
            dist = 0
        if dist < best_dist and dist <= 10:
            best_dist = dist
            best_name = name

    return best_name


# ── Multi-K K-means helpers ──────────────────────────────────────

def _find_best_k_multi(
    pixels: "np.ndarray", k_values: list[int],
) -> tuple[int | None, "np.ndarray | None", "np.ndarray | None", dict]:
    """Try multiple k values and pick the best via elbow + silhouette scoring.

    Returns (best_k, centers, labels, scores_dict) where scores_dict maps
    k → {"wcss":..., "silhouette":..., "elbow":..., "combined":...}.
    """
    import cv2
    import numpy as np

    pixels_f32 = pixels.astype(np.float32)
    n = len(pixels_f32)
    if n < 50:
        return None, None, None, {}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    results = {}  # k → (centers, labels, wcss, silhouette)

    for k in k_values:
        if k >= n // 2:
            continue
        compactness, labels_k, centers_k = cv2.kmeans(
            pixels_f32, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS,
        )
        wcss = float(compactness)
        sil = _compute_silhouette_sampled(pixels_f32, labels_k.flatten(), k)
        results[k] = (centers_k, labels_k.flatten(), wcss, sil)

    if not results:
        return None, None, None, {}

    # Compute elbow score (second derivative / curvature of WCSS)
    ks_sorted = sorted(results.keys())
    wcss_vals = np.array([results[k][2] for k in ks_sorted], dtype=np.float64)

    elbow_map: dict[int, float] = {}
    if len(ks_sorted) >= 3:
        wcss_norm = wcss_vals / wcss_vals.max() if wcss_vals.max() > 0 else wcss_vals
        for i in range(1, len(ks_sorted) - 1):
            curvature = wcss_norm[i - 1] + wcss_norm[i + 1] - 2 * wcss_norm[i]
            elbow_map[ks_sorted[i]] = float(curvature)

    # Combine elbow + silhouette (normalize both to [0, 1])
    sil_vals = np.array([results[k][3] for k in ks_sorted], dtype=np.float64)
    sil_norm = np.zeros_like(sil_vals)
    if float(np.ptp(sil_vals)) > 1e-9:
        sil_norm = (sil_vals - sil_vals.min()) / float(np.ptp(sil_vals))

    combined: dict[int, float] = {}
    scores: dict[int, dict[str, float]] = {}
    for i, k in enumerate(ks_sorted):
        elbow = elbow_map.get(k, 0.0)
        if elbow_map:
            e_max = max(elbow_map.values())
            if e_max > 1e-9:
                elbow /= e_max
        combined[k] = 0.4 * float(sil_norm[i]) + 0.6 * elbow
        scores[k] = {
            "wcss": round(float(wcss_vals[i]), 2),
            "silhouette": round(float(sil_vals[i]), 4),
            "elbow": round(elbow, 4),
            "combined": round(combined[k], 4),
        }

    best_k = max(combined, key=combined.get)
    centers, labels, _, _ = results[best_k]
    return best_k, centers, labels, scores


def _compute_silhouette_sampled(
    pixels: "np.ndarray", labels: "np.ndarray", n_clusters: int,
) -> float:
    """Compute simplified silhouette score using cluster-centre distances.

    Uses at most 800 samples for speed.  Score ∈ [-1, 1], higher is better.
    """
    import numpy as np

    n = len(labels)
    if n <= 800:
        sample_idx = np.arange(n)
    else:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, 800, replace=False)

    sampled = pixels[sample_idx]
    sampled_l = labels[sample_idx]

    cluster_ids = np.unique(labels)
    if len(cluster_ids) < 2:
        return 0.0

    centres = {}
    for cid in cluster_ids:
        mask = labels == cid
        centres[cid] = pixels[mask].mean(axis=0)

    scores = []
    for i in range(len(sampled)):
        c_i = sampled_l[i]
        own_center = centres.get(c_i)
        if own_center is None:
            continue
        a_i = float(np.sqrt(((sampled[i] - own_center) ** 2).sum()))

        b_i = float("inf")
        for cj in cluster_ids:
            if cj == c_i:
                continue
            other_center = centres.get(cj)
            if other_center is not None:
                dist = float(np.sqrt(((sampled[i] - other_center) ** 2).sum()))
                if dist < b_i:
                    b_i = dist

        if b_i == float("inf") or max(a_i, b_i) == 0:
            continue
        scores.append((b_i - a_i) / max(a_i, b_i))

    return float(np.mean(scores)) if scores else 0.0


# ── Adaptive saturation filter ───────────────────────────────────

def _filter_clusters_adaptive(
    centers: "np.ndarray", counts: "np.ndarray", total: float,
    all_pixels: "np.ndarray",
) -> tuple[set[str], int, int]:
    """Filter clusters using adaptive saturation threshold.

    Starts at s=20 and increases if too many pixels are saturated.
    Returns (detected_colors, sat_threshold, val_threshold).
    """
    import numpy as np

    s_values = all_pixels[:, 1].astype(float)

    # Adaptive saturation: start at 20, increase if too many high-S pixels
    s_threshold = 20
    v_threshold = 20

    high_s_ratio = (float((s_values >= s_threshold).sum()) / len(s_values)
                    if len(s_values) > 0 else 0.0)

    if high_s_ratio > 0.65:
        s_threshold = 35
    elif high_s_ratio > 0.45:
        s_threshold = 30
    elif high_s_ratio > 0.30:
        s_threshold = 25

    detected: set[str] = set()
    center_color_map: dict[str, tuple] = {}

    for center, count in zip(centers, counts):
        ratio = float(count) / total
        if ratio < 0.03 or ratio > 0.70:
            continue
        h, s, v = center
        if s < s_threshold or v < v_threshold:
            continue
        color_name = _hsv_to_color_name((float(h), float(s), float(v)))
        if color_name:
            if color_name not in center_color_map or ratio > center_color_map[color_name][3]:
                center_color_map[color_name] = (float(h), float(s), float(v), ratio)
            detected.add(color_name)

    _filter_clusters_adaptive._last_cluster_info = center_color_map  # type: ignore[attr-defined]
    return detected, s_threshold, v_threshold


# ── Histogram-based fallback ─────────────────────────────────────

def _histogram_fallback(
    all_pixels: "np.ndarray",
) -> list[str]:
    """Fallback: detect colours via H-channel histogram peak detection.

    Used when K-means fails to produce meaningful clusters.
    """
    import numpy as np

    if len(all_pixels) < 100:
        return ["red"]

    h_vals = all_pixels[:, 0]
    s_vals = all_pixels[:, 1]
    v_vals = all_pixels[:, 2]

    # Keep only reasonably saturated pixels (S ≥ 20, V ≥ 20)
    mask = (s_vals >= 20) & (v_vals >= 20)
    h_filtered = h_vals[mask].astype(int)

    if len(h_filtered) < 20:
        return ["red"]

    # Build histogram with 180 bins (H ∈ [0, 179])
    hist, _bin_edges = np.histogram(h_filtered, bins=180, range=(0, 179))
    hist_f = hist.astype(np.float64)

    # Smooth histogram with moving average (window = 5)
    kernel = np.ones(5) / 5.0
    hist_smooth = np.convolve(hist_f, kernel, mode="same")

    # Find peaks (local maxima above mean + 0.5 * std)
    mean_hist = float(hist_smooth.mean())
    std_hist = float(hist_smooth.std())
    threshold = mean_hist + 0.5 * std_hist

    peaks: list[int] = []
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i - 1]
                and hist_smooth[i] > hist_smooth[i + 1]
                and hist_smooth[i] > threshold):
            peaks.append(i)

    if not peaks:
        # Relax threshold
        threshold = mean_hist + 0.2 * std_hist
        for i in range(1, len(hist_smooth) - 1):
            if (hist_smooth[i] > hist_smooth[i - 1]
                    and hist_smooth[i] > hist_smooth[i + 1]
                    and hist_smooth[i] > threshold):
                peaks.append(i)

    if not peaks:
        return ["red"]

    # Map peaks to colour names
    detected: set[str] = set()
    for peak_h in peaks:
        color_name = _hsv_to_color_name((float(peak_h), 100.0, 100.0))
        if color_name:
            detected.add(color_name)

    if not detected:
        return ["red"]

    ordered = [c for c in ["red", "blue", "green", "cyan", "yellow",
                             "orange", "magenta", "purple"] if c in detected]
    return ordered or list(detected)


# ── Contrast guard ───────────────────────────────────────────────

def _contrast_guard(
    detected: set[str], centers: "np.ndarray",
    counts: "np.ndarray", total: float,
) -> set[str]:
    """Merge colour clusters that are too visually similar (H-distance < 15°).

    Also accounts for circular HSV hue space.
    """
    import numpy as np

    if len(detected) <= 1:
        return detected

    color_h_map: dict[str, tuple[float, float]] = {}  # name → (h, ratio)

    for center, count in zip(centers, counts):
        ratio = float(count) / total
        if ratio < 0.03 or ratio > 0.70:
            continue
        h, s, v = center
        if s < 20 or v < 20:
            continue
        color_name = _hsv_to_color_name((float(h), float(s), float(v)))
        if color_name and color_name in detected:
            if (color_name not in color_h_map
                    or ratio > color_h_map[color_name][1]):
                color_h_map[color_name] = (float(h), ratio)

    names = list(detected)
    to_remove: set[str] = set()

    for i in range(len(names)):
        if names[i] in to_remove:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in to_remove:
                continue
            h_i = color_h_map.get(names[i], (float("inf"), 0))[0]
            h_j = color_h_map.get(names[j], (float("inf"), 0))[0]
            if h_i == float("inf") or h_j == float("inf"):
                continue

            # Circular H-distance (H ∈ [0, 180) in OpenCV)
            h_dist = abs(h_i - h_j)
            h_dist = min(h_dist, 180 - h_dist)

            if h_dist < 15:
                # Merge: keep the one with larger pixel ratio
                ratio_i = color_h_map[names[i]][1]
                ratio_j = color_h_map[names[j]][1]
                if ratio_i >= ratio_j:
                    to_remove.add(names[j])
                else:
                    to_remove.add(names[i])

    return detected - to_remove


# ── Colour mask builder ────────────────────────────────────────────

def _build_color_mask(hsv, color_name: str) -> "np.ndarray":
    """Build a unified binary mask for the given colour name.

    Looks up CHART_COLOR_PRESETS and merges all HSV sub-ranges.
    Falls back to red mask if colour name is unknown.
    """
    import cv2
    import numpy as np

    preset = CHART_COLOR_PRESETS.get(color_name)
    if preset is None:
        # Fallback: try common aliases
        alias_map = {
            "dark_blue": "blue_dark",
            "light_blue": "blue",
            "dark_green": "green_dark",
            "light_green": "green",
            "pink": "magenta",
            "violet": "purple",
            "grey": "black",
            "gray": "black",
        }
        resolved = alias_map.get(color_name, "red")
        preset = CHART_COLOR_PRESETS.get(resolved, CHART_COLOR_PRESETS["red"])

    masks = []
    for r in preset["ranges"]:
        lower = np.array(r["lower"], dtype=np.uint8)
        upper = np.array(r["upper"], dtype=np.uint8)
        masks.append(cv2.inRange(hsv, lower, upper))

    if len(masks) == 1:
        return masks[0]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return combined


# ── Panel data extraction ──────────────────────────────────────────

def _extract_panel_data(
    color_image, panel: dict, ocr_blocks: list[Block],
    *,
    target_color: str = "red",
) -> list[dict]:
    """Extract structured data from a single chart panel for one colour.

    Strategy: cluster coloured vertical lines by x-position and use their
    y-extent as class depth ranges. Extract scattered dots separately.

    Returns list of dicts with type, panel_id, class, depth, etc.
    """
    result: list[dict] = []

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    img = np.array(color_image)
    panel_region = img[y0:y1, x0:x1]
    if panel_region.size == 0:
        return result

    pw = x1 - x0
    ph = y1 - y0

    # ── Pixel→data calibration via grid lines & tick labels ──────────
    if "_calibration" not in panel:
        panel["_calibration"] = calibrate_axis(color_image, panel, ocr_blocks)

    hsv = cv2.cvtColor(panel_region, cv2.COLOR_RGB2HSV)

    # Determine element type from preset
    preset = CHART_COLOR_PRESETS.get(target_color, CHART_COLOR_PRESETS["red"])
    element_type = preset.get("element_type", "fill")

    # Build colour mask
    color_mask = _build_color_mask(hsv, target_color)

    # Morphological kernel sizing adapts to whether we expect lines or fill
    if element_type == "line":
        # Narrower vertical kernel for line-only colours
        vertical_kernel = np.ones((25, 2), np.uint8)
        close_kernel = np.ones((9, 2), np.uint8)
    else:
        vertical_kernel = np.ones((35, 3), np.uint8)
        close_kernel = np.ones((13, 3), np.uint8)

    lines_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, vertical_kernel)
    lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, close_kernel)

    thick_lines = cv2.dilate(lines_mask, np.ones((7, 7), np.uint8), iterations=1)
    dots_mask = cv2.subtract(color_mask, thick_lines)
    dots_mask = cv2.morphologyEx(dots_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Step 1: Extract intervals from vertical lines
    intervals = _extract_intervals_from_lines(
        lines_mask, panel, ph, pw, color_name=target_color,
    )
    result.extend(intervals)

    # Step 2: Extract scatter points
    points = _extract_points_from_dots(
        dots_mask, intervals, panel, ph, pw, color_name=target_color,
    )
    result.extend(points)

    return _deduplicate_chart_data(result)


def _extract_intervals_from_lines(
    lines_mask, panel: dict, ph: int, pw: int,
    *,
    color_name: str = "red",
) -> list[dict]:
    """Find vertical line segments, cluster by x, assign classes."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    segments = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        min_line_height = max(45, int(ph * 0.035))
        if ch < min_line_height or cw > ch * 0.45:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_center = int(M["m10"] / M["m00"])
        cy_center = int(M["m01"] / M["m00"])
        segments.append({"x": cx_center, "y0": cy, "y1": cy + ch})

    if len(segments) < 2:
        return []

    segments.sort(key=lambda s: s["x"])
    clusters = [[segments[0]]]
    for seg in segments[1:]:
        if seg["x"] - clusters[-1][-1]["x"] <= 8:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    valid_clusters = []
    for cl in clusters:
        total_height = sum(s["y1"] - s["y0"] for s in cl)
        if total_height >= 20:
            valid_clusters.append(cl)

    if len(valid_clusters) < 2:
        return []

    valid_clusters.sort(key=lambda cl: sum(s["x"] for s in cl) / len(cl))

    y_min_data = panel.get("y_min", 0.0)
    y_max_data = panel.get("y_max", 30.0)
    calibration = panel.get("_calibration", {})

    intervals = []
    for idx, cl in enumerate(valid_clusters):
        avg_x = sum(s["x"] for s in cl) / len(cl)
        line_top = min(s["y0"] for s in cl)
        line_bot = max(s["y1"] for s in cl)
        depth_top = pixel_to_data(line_top, calibration, panel, ph)
        depth_bot = pixel_to_data(line_bot, calibration, panel, ph)
        cls_val = idx
        if cls_val > 5:
            break
        intervals.append({
            "type": "interval",
            "panel_id": panel["id"],
            "class": cls_val,
            "start_depth": round(min(depth_top, depth_bot), 1),
            "end_depth": round(max(depth_top, depth_bot), 1),
            "depth_tolerance": _depth_tolerance(panel),
            "pixel_x": round(avg_x, 1),
            "color": color_name,
        })

    return intervals


def _extract_zone_boundaries(
    color_image,
    panel: dict,
    ocr_blocks: list[Block],
    ph: int,
    pw: int,
) -> list[dict]:
    """Extract SBT-style zone boundary intervals from black diagonal lines.

    SBT (soil behaviour type) charts divide the plot into numbered zones with
    diagonal polylines.  These lines are typically black (ink), so the
    colour-based pipeline never sees them.  This function builds an ink mask
    (grayscale OTSU), detects line segments with Hough, keeps only diagonal
    segments (slope between ``_ZONE_MIN_SLOPE`` and ``_ZONE_MAX_SLOPE``),
    clusters collinear segments into boundary polylines, and maps each
    polyline's y-extent into data coordinates via the panel calibration.

    Returns a list of interval dicts with ``source: "zone_boundary"``.
    The ``pixel_x0``/``pixel_x1`` fields use distinct names so they do NOT
    interfere with point class assignment (which keys on ``pixel_x``).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    img = np.array(color_image)
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return []

    # ── Ink mask (all dark content, not just coloured elements) ───────
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    min_len = max(70, int(pw * 0.08))
    lines = cv2.HoughLinesP(
        ink, 1, np.pi / 180, threshold=40,
        minLineLength=min_len, maxLineGap=8,
    )
    if lines is None:
        return []

    # ── Keep only diagonal segments (zone boundaries) ─────────────────
    _ZONE_MIN_SLOPE = 0.15   # exclude near-horizontal grid lines
    _ZONE_MAX_SLOPE = 5.0    # exclude near-vertical error bars / axes
    segments: list[dict] = []
    for line in lines:
        lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
        dx = abs(lx2 - lx1)
        dy = abs(ly2 - ly1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_len * 0.8:
            continue
        slope = dy / max(dx, 1)
        if slope < _ZONE_MIN_SLOPE or slope > _ZONE_MAX_SLOPE:
            continue
        # Normal-form (angle, rho) representation for collinearity
        ang = math.atan2(ly2 - ly1, lx2 - lx1)
        nang = ang + math.pi / 2
        mx = (lx1 + lx2) / 2.0
        my = (ly1 + ly2) / 2.0
        rho = mx * math.cos(nang) + my * math.sin(nang)
        segments.append({
            "x1": lx1, "y1": ly1, "x2": lx2, "y2": ly2,
            "len": length, "slope": slope, "ang": ang, "rho": rho,
        })

    if not segments:
        return []

    # ── Cluster collinear segments via (angle, rho) proximity ─────────
    # Segments on the same boundary line share angle AND rho; parallel
    # neighbours (same angle, different rho) stay separate.
    clusters: list[dict] = []
    for seg in segments:
        placed = False
        for cl in clusters:
            dang = abs(cl["ang"] - seg["ang"])
            dang = min(dang, 2 * math.pi - dang)
            if dang > 0.14:          # ~8° angle bucket
                continue
            if abs(cl["rho"] - seg["rho"]) > 30:   # ~30px rho bucket
                continue
            cl["segs"].append(seg)
            cl["ang"] = sum(s["ang"] for s in cl["segs"]) / len(cl["segs"])
            cl["rho"] = sum(s["rho"] for s in cl["segs"]) / len(cl["segs"])
            cl["x1"] = min(s["x1"] for s in cl["segs"])
            cl["y1"] = min(s["y1"] for s in cl["segs"])
            cl["x2"] = max(s["x2"] for s in cl["segs"])
            cl["y2"] = max(s["y2"] for s in cl["segs"])
            placed = True
            break
        if not placed:
            clusters.append({
                "ang": seg["ang"], "rho": seg["rho"], "segs": [seg],
                "x1": seg["x1"], "y1": seg["y1"],
                "x2": seg["x2"], "y2": seg["y2"],
            })

    # Drop tiny clusters (likely text/symbol fragments): a boundary must
    # have at least 2 segments OR span a meaningful length.
    clusters = [cl for cl in clusters if len(cl["segs"]) >= 2 or cl["x2"] - cl["x1"] >= pw * 0.3]

    if not clusters:
        return []

    # ── Calibration (y-axis pixel → data) ─────────────────────────────
    if "_calibration" not in panel:
        panel["_calibration"] = calibrate_axis(color_image, panel, ocr_blocks)
    calibration = panel.get("_calibration", {})

    # ── Map each boundary polyline to an interval in data coordinates ──
    # Class assignment: sort boundaries by their mean data depth (ascending),
    # so the lowest boundary gets class 1 and the highest gets class N.
    # This is deterministic and matches the SBT zone numbering direction
    # (zones are numbered bottom-up).
    intervals: list[dict] = []
    for cl in clusters:
        all_y = [s["y1"] for s in cl["segs"]] + [s["y2"] for s in cl["segs"]]
        all_x = [s["x1"] for s in cl["segs"]] + [s["x2"] for s in cl["segs"]]
        top_y = min(all_y)
        bot_y = max(all_y)
        left_x = min(all_x)
        right_x = max(all_x)
        intervals.append({
            "type": "interval",
            "panel_id": panel["id"],
            "class": 0,  # placeholder, assigned below
            "start_depth": 0.0,  # placeholder
            "end_depth": 0.0,  # placeholder
            "depth_tolerance": _depth_tolerance(panel),
            "pixel_x0": float(left_x),
            "pixel_x1": float(right_x),
            "pixel_y0": float(top_y),
            "pixel_y1": float(bot_y),
            "source": "zone_boundary",
        })

    # Convert pixel y-extent → data depth for every boundary
    for inv in intervals:
        depth_top = pixel_to_data(inv["pixel_y0"], calibration, panel, ph)
        depth_bot = pixel_to_data(inv["pixel_y1"], calibration, panel, ph)
        inv["start_depth"] = round(min(depth_top, depth_bot), 1)
        inv["end_depth"] = round(max(depth_top, depth_bot), 1)

    # Assign classes by mean data depth (ascending)
    intervals.sort(key=lambda inv: (inv["start_depth"] + inv["end_depth"]) / 2)
    for idx, inv in enumerate(intervals):
        inv["class"] = idx + 1

    return intervals


def _extract_points_from_dots(
    dots_mask, intervals: list[dict], panel: dict, ph: int, pw: int,
    *,
    color_name: str = "red",
) -> list[dict]:
    """Extract coloured scatter points and assign class from nearest x interval."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(dots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or not intervals:
        return []

    y_min_data = panel.get("y_min", 0.0)
    y_max_data = panel.get("y_max", 30.0)
    calibration = panel.get("_calibration", {})
    class_centers = [
        (float(inv.get("pixel_x", 0.0)), inv["class"])
        for inv in intervals
        if "pixel_x" in inv
    ]

    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4 or area > 250:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        pcx = int(M["m10"] / M["m00"])
        pcy = int(M["m01"] / M["m00"])

        depth = pixel_to_data(pcy, calibration, panel, ph)
        if class_centers:
            nearest = sorted(class_centers, key=lambda item: abs(item[0] - pcx))
            _nearest_x, best_class = nearest[0]
            class_candidates = [best_class]
            if len(nearest) > 1:
                second_x, second_class = nearest[1]
                class_step = _median_class_step(class_centers)
                if class_step and abs(second_x - pcx) <= class_step * 0.75:
                    class_candidates.append(second_class)
            points.append({
                "type": "point",
                "panel_id": panel["id"],
                "class": best_class,
                "class_candidates": class_candidates,
                "depth": round(depth, 1),
                "depth_tolerance": _point_depth_tolerance(panel),
                "pixel_x": pcx,
                "color": color_name,
            })
            continue

        best_class = None
        best_overlap = -1
        for inv in intervals:
            sd = inv["start_depth"]
            ed = inv["end_depth"]
            if sd <= depth <= ed:
                overlap = min(ed, depth) - max(sd, depth)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_class = inv["class"]
        if best_class is None:
            min_dist = float("inf")
            for inv in intervals:
                sd = inv["start_depth"]
                ed = inv["end_depth"]
                mid = (sd + ed) / 2
                dist = abs(depth - mid)
                if dist < min_dist:
                    min_dist = dist
                    best_class = inv["class"]

        points.append({
            "type": "point",
            "panel_id": panel["id"],
            "class": best_class,
            "depth": round(depth, 1),
            "class_candidates": [best_class],
            "depth_tolerance": _point_depth_tolerance(panel),
            "color": color_name,
        })

    return points


def _depth_tolerance(panel: dict) -> float:
    """Depth uncertainty for chart intervals (explicit error bars)."""
    y_span = abs(float(panel.get("y_max", 30.0)) - float(panel.get("y_min", 0.0)))
    return round(max(0.3, min(0.75, y_span * 0.025)), 2)


def _point_depth_tolerance(panel: dict) -> float:
    y_span = abs(float(panel.get("y_max", 30.0)) - float(panel.get("y_min", 0.0)))
    return round(max(0.35, min(0.65, y_span * 0.02)), 2)


def _median_class_step(class_centers: list[tuple[float, int]]) -> float | None:
    if len(class_centers) < 2:
        return None
    centers = sorted(x for x, _cls in class_centers)
    gaps = [right - left for left, right in zip(centers, centers[1:]) if right > left]
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    if len(gaps) % 2:
        return gaps[mid]
    return (gaps[mid - 1] + gaps[mid]) / 2


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_chart_data(data: list[dict]) -> list[dict]:
    """Remove overlapping intervals and duplicate points."""
    intervals = [d for d in data if d.get("type") == "interval"]
    points = [d for d in data if d.get("type") == "point"]
    merged = _merge_intervals(intervals)
    deduped_points = _dedup_points(points)
    return merged + deduped_points


def _merge_intervals(intervals: list[dict]) -> list[dict]:
    if not intervals:
        return []
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for inv in intervals:
        key = (inv.get("panel_id", ""), inv.get("class", 0), inv.get("color", ""))
        groups.setdefault(key, []).append((inv["start_depth"], inv["end_depth"]))

    result = []
    for (panel_id, cls, color), ranges in groups.items():
        ranges.sort()
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        tol = max((float(inv.get("depth_tolerance", 0.0))
                   for inv in intervals
                   if inv.get("panel_id", "") == panel_id
                   and inv.get("class", 0) == cls
                   and inv.get("color", "") == color),
                  default=0.0)
        for s, e in merged:
            result.append({
                "type": "interval",
                "panel_id": panel_id,
                "class": cls,
                "start_depth": round(s, 1),
                "end_depth": round(e, 1),
                "depth_tolerance": tol,
                "color": color,
            })
    return result


def _dedup_points(points: list[dict]) -> list[dict]:
    if not points:
        return []
    seen: set[tuple] = set()
    result = []
    for p in points:
        key = (p.get("panel_id", ""), p.get("class", 0),
               round(p.get("depth", 0), 0), p.get("color", ""))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result
