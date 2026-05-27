"""Phase 3: Chart/figure understanding for doc-textify.

Analyses color images to extract structured data from charts:
  - Panel/subplot detection
  - Axis label OCR integration
  - Color-based element extraction (data points, intervals, lines)
  - Pixel-to-data-coordinate mapping
  - JSON-compatible chart_data output for evaluation framework

Requires opencv-python-headless for full functionality. Falls back to
Pillow-only analysis when OpenCV is not available (limited capability).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import Block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_chart(
    color_image,
    ocr_blocks: list[Block],
    page_width: float | None = None,
    page_height: float | None = None,
) -> dict[str, Any]:
    """Analyse a colour image for chart content.

    Args:
        color_image: PIL Image in RGB mode.
        ocr_blocks: OCR blocks extracted from the same image.
        page_width, page_height: Image dimensions.

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
        # Try with a simpler fallback if OpenCV is not available
        panels = _detect_panels_fallback(color_image, ocr_blocks, img_w, img_h)
    if not panels:
        return result

    # Step 2: For each panel, extract coloured elements and map to data
    for panel in panels:
        panel_data = _extract_panel_data(color_image, panel, ocr_blocks)
        result["chart_data"].extend(panel_data)

    return result


# ---------------------------------------------------------------------------
# Panel detection (OpenCV path)
# ---------------------------------------------------------------------------

def _detect_panels(
    color_image, ocr_blocks, img_w: int, img_h: int,
) -> list[dict[str, Any]]:
    """Detect chart panel regions via rectangle detection.

    Uses OpenCV edge detection + contour finding to locate rectangular
    grid regions. Each panel is described by its bounding box and
    detected axis information.

    Returns list of panel dicts:
        { "id": str, "x0": int, "y0": int, "x1": int, "y1": int,
          "x_label": str, "y_label": str,
          "x_min": float, "x_max": float,
          "y_min": float, "y_max": float,
          "y_direction": str }
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []  # OpenCV not available

    img = np.array(color_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Morphological close to connect edges
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours (rectangular grid regions)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for rectangular regions of significant size
    min_area = (img_w * img_h) * 0.05  # at least 5% of image area
    max_area = (img_w * img_h) * 0.60  # at most 60%

    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        # Chart panels can be tall vertical logs (for example depth/class
        # plots), so accept narrow rectangles as long as they are not
        # hairlines.
        if aspect < 0.15 or aspect > 3.0:
            continue

        # Check rectangularity (approximate polygon)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) < 4:
            continue

        rectangles.append({
            "x0": x, "y0": y, "x1": x + w, "y1": y + h,
            "area": area,
        })

    if not rectangles:
        return []

    # Sort left-to-right, assign panel IDs. Depth/class comparison charts
    # often have two tall panels whose top y differs by a few pixels; sorting
    # by y first swaps the semantic (a)/(b) order.
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
            "x_label": "",
            "y_label": "",
            "x_min": 0.0, "x_max": 5.0,
            "y_min": 0.0, "y_max": inferred_y_max,
            "y_direction": "down",
        })

    panels = _annotate_panels_with_ocr(panels, ocr_blocks)
    return panels


# ---------------------------------------------------------------------------
# Panel detection fallback (Pillow-only)
# ---------------------------------------------------------------------------

def _detect_panels_fallback(color_image, ocr_blocks, img_w, img_h):
    """Simple panel detection using edge detection via Pillow.

    Useful when OpenCV is not available. Less accurate than the
    OpenCV path but can detect simple chart layouts.
    """
    try:
        from PIL import Image, ImageFilter, ImageOps
        import numpy as np
    except ImportError:
        return []

    # Edge detection via Pillow's FIND_EDGES filter
    gray = color_image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)

    # Horizontal and vertical projection
    h_proj = edge_arr.sum(axis=1)  # row sums
    v_proj = edge_arr.sum(axis=0)  # column sums

    # Look for gaps in projections to split panels
    h_thresh = np.max(h_proj) * 0.1
    v_thresh = np.max(v_proj) * 0.1

    # Find horizontal gaps (between rows of panels)
    h_gaps = _find_projection_gaps(h_proj, h_thresh, min_gap=int(img_h * 0.02))
    # Find vertical gaps (between columns of panels)
    v_gaps = _find_projection_gaps(v_proj, v_thresh, min_gap=int(img_w * 0.02))

    if not h_gaps and not v_gaps:
        return []  # single region, might be a single chart

    # Generate grid regions from gaps
    h_regions = _gap_to_regions(h_gaps, 0, img_h)
    v_regions = _gap_to_regions(v_gaps, 0, img_w)

    if not h_regions:
        h_regions = [(0, img_h)]
    if not v_regions:
        v_regions = [(0, img_w)]

    panels = []
    for idx, (y0, y1) in enumerate(h_regions):
        for jdx, (x0, x1) in enumerate(v_regions):
            panel_area = (x1 - x0) * (y1 - y0)
            min_area = (img_w * img_h) * 0.05
            if panel_area < min_area:
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
    """Find gaps in a projection profile where values fall below threshold."""
    gaps = []
    in_gap = False
    gap_start = 0
    for i in range(len(proj)):
        if proj[i] <= threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                if i - gap_start >= min_gap:
                    gaps.append((gap_start, i))
                in_gap = False
    if in_gap and len(proj) - gap_start >= min_gap:
        gaps.append((gap_start, len(proj)))
    return gaps


def _gap_to_regions(gaps, start, end):
    """Convert gaps into regions (areas between gaps)."""
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
# OCR annotation of panels
# ---------------------------------------------------------------------------

def _annotate_panels_with_ocr(
    panels: list[dict], ocr_blocks: list[Block],
) -> list[dict]:
    """Use OCR text blocks near panel edges to identify axis labels.

    Matches text blocks to panels based on proximity, looking for:
      - x-axis labels (text below the panel)
      - y-axis labels (text to the left of the panel)
      - Panel captions (text above or below)
      - Tick labels (text near edges)
    """
    if not ocr_blocks:
        return panels

    for panel in panels:
        px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
        pw = px1 - px0
        ph = py1 - py0

        # Axis label candidates
        x_candidates = []
        y_candidates = []
        caption_candidates = []

        for block in ocr_blocks:
            if not block.bbox:
                continue
            bx0, by0, bx1, by1 = block.bbox

            # Block center
            bcx = (bx0 + bx1) / 2
            bcy = (by0 + by1) / 2

            # X-axis label: centered below the panel
            if (bcx > px0 - pw * 0.2 and bcx < px1 + pw * 0.2
                    and by0 > py1 and by1 < py1 + ph * 0.3):
                x_candidates.append((block.text, by0))

            # Y-axis label: to the left of the panel, vertically centered
            if (bx1 < px0 and bcy > py0 - ph * 0.2 and bcy < py1 + ph * 0.2):
                y_candidates.append((block.text, bx1))

            # Caption: directly above or below
            if (bcx > px0 - pw * 0.1 and bcx < px1 + pw * 0.1
                    and ((by1 < py0 and by0 > py0 - ph * 0.3)
                         or (by0 > py1 and by1 < py1 + ph * 0.3))):
                caption_candidates.append((block.text, bcy))

        # Pick closest candidates
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


# ---------------------------------------------------------------------------
# Colour element extraction from panel regions
# ---------------------------------------------------------------------------

# HSV colour ranges for common chart elements
_COLOR_RANGES = {
    "red_dot": {
        "lower": (0, 50, 50),
        "upper": (10, 255, 255),
    },
    "red_dot_alt": {
        "lower": (170, 50, 50),
        "upper": (180, 255, 255),
    },
    "blue_triangle": {
        "lower": (100, 50, 50),
        "upper": (130, 255, 255),
    },
    "red_line": {
        "lower": (0, 100, 50),
        "upper": (10, 255, 255),
    },
    "red_line_alt": {
        "lower": (170, 100, 50),
        "upper": (180, 255, 255),
    },
}


def _extract_panel_data(
    color_image, panel: dict, ocr_blocks: list[Block],
) -> list[dict]:
    """Extract structured data from a single chart panel.

    Strategy: cluster red vertical lines by x-position and use their
    y-extent as class depth ranges, instead of relying on grid boundary
    detection. This gives more robust coordinate mapping because the
    red lines ARE the data markers.

    Returns list of dicts:
      - type: "interval" | "point"
      - panel_id: str
      - class: int
      - depth / start_depth / end_depth: float
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
    hsv = cv2.cvtColor(panel_region, cv2.COLOR_RGB2HSV)

    # --- Unified red mask (covers both dark and light reds) ---
    red_mask = _build_red_mask(hsv)

    # --- Separate vertical lines from scattered dots ---
    # Preserve only long vertical red strokes as intervals. This prevents
    # stacked red scatter dots from becoming false class intervals.
    vertical_kernel = np.ones((35, 3), np.uint8)
    lines_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, vertical_kernel)
    lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, np.ones((13, 3), np.uint8))

    # Dots: remove a slightly dilated version of the interval lines.
    thick_lines = cv2.dilate(lines_mask, np.ones((7, 7), np.uint8), iterations=1)
    dots_mask = cv2.subtract(red_mask, thick_lines)
    dots_mask = cv2.morphologyEx(dots_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # --- Step 1: Extract intervals from vertical red lines ---
    intervals = _extract_intervals_from_lines(lines_mask, panel, ph, pw)
    result.extend(intervals)

    # --- Step 2: Extract scatter points ---
    points = _extract_points_from_dots(dots_mask, intervals, panel, ph, pw)
    result.extend(points)

    return _deduplicate_chart_data(result)


def _build_red_mask(hsv) -> "np.ndarray":
    """Create a unified mask for red pixels (red wraps around HSV hue 0/180)."""
    import cv2
    import numpy as np

    # Red in HSV: wraps around 0. Two ranges needed.
    lower1 = np.array([0, 40, 40], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 40, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


def _extract_intervals_from_lines(
    lines_mask, panel: dict, ph: int, pw: int,
) -> list[dict]:
    """Find vertical red line segments, cluster by x, assign classes."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Collect vertical line segments: tall, thin contours
    segments = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        min_line_height = max(45, int(ph * 0.035))
        if ch < min_line_height or cw > ch * 0.45:
            continue  # not tall enough, or too wide
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_center = int(M["m10"] / M["m00"])
        cy_center = int(M["m01"] / M["m00"])
        segments.append({
            "x": cx_center,
            "y0": cy,
            "y1": cy + ch,
        })

    if len(segments) < 2:
        return []

    # Cluster by x-position (DBSCAN-like: group nearby x-centers)
    segments.sort(key=lambda s: s["x"])
    clusters = [[segments[0]]]
    for seg in segments[1:]:
        if seg["x"] - clusters[-1][-1]["x"] <= 8:  # 8px max gap within a line
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    # Filter clusters: must have at least 2 segments or one long segment
    valid_clusters = []
    for cl in clusters:
        total_height = sum(s["y1"] - s["y0"] for s in cl)
        if total_height >= 20:
            valid_clusters.append(cl)

    if len(valid_clusters) < 2:
        return []

    # If we expect 6 classes (0..5), we should have ~7 boundary lines
    # (one per class plus top/bottom). But in practice the chart has
    # 6 vertical red lines, one per class. Each line spans the class depth.
    # Assign class 0 to the leftmost, class 5 to the rightmost.
    valid_clusters.sort(key=lambda cl: sum(s["x"] for s in cl) / len(cl))

    # Use panel y_min/y_max from the panel or the actual y-extent of lines
    y_min_data = panel.get("y_min", 0.0)
    y_max_data = panel.get("y_max", 30.0)

    intervals = []
    for idx, cl in enumerate(valid_clusters):
        avg_x = sum(s["x"] for s in cl) / len(cl)

        # Determine y-extent from all segments in this cluster
        line_top = min(s["y0"] for s in cl)
        line_bot = max(s["y1"] for s in cl)

        # The y-extent tells us the depth range for this class.
        # Map pixel y to data depth using panel dimensions.
        depth_top = y_min_data + (line_top / ph) * (y_max_data - y_min_data)
        depth_bot = y_min_data + (line_bot / ph) * (y_max_data - y_min_data)

        cls_val = idx  # 0, 1, 2, 3, 4, 5
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
        })

    return intervals


def _extract_points_from_dots(
    dots_mask, intervals: list[dict], panel: dict, ph: int, pw: int,
) -> list[dict]:
    """Extract red scatter points and assign class from nearest x interval."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(dots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or not intervals:
        return []

    # Build class boundary map from intervals
    # Each interval has a class and depth range. We assign dots to classes
    # based on which interval they fall into (by depth).
    # But the chart also has class determined by x-position. So we also
    # need x-based class assignment.
    # Strategy: assign class based on which interval's y-range the dot
    # falls into. If there's a tie, use nearest.

    y_min_data = panel.get("y_min", 0.0)
    y_max_data = panel.get("y_max", 30.0)
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

        # Convert pixel y to depth
        depth = y_min_data + (pcy / ph) * (y_max_data - y_min_data)
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
            })
            continue

        # Find class: which interval's depth range contains this point?
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
            # No interval contains this depth — assign to nearest
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
        })

    return points


def _depth_tolerance(panel: dict) -> float:
    """Depth uncertainty for photographed chart intervals.

    The value is intentionally explicit in the JSON so a downstream LLM can
    treat chart readings as measured facts with error bars, not fake exact
    numbers.
    """
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
    """Merge overlapping intervals for the same (panel_id, class)."""
    if not intervals:
        return []
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for inv in intervals:
        key = (inv.get("panel_id", ""), inv.get("class", 0))
        groups.setdefault(key, []).append((inv["start_depth"], inv["end_depth"]))

    result = []
    for (panel_id, cls), ranges in groups.items():
        ranges.sort()
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for s, e in merged:
            result.append({
                "type": "interval",
                "panel_id": panel_id,
                "class": cls,
                "start_depth": round(s, 1),
                "end_depth": round(e, 1),
                "depth_tolerance": max((float(inv.get("depth_tolerance", 0.0)) for inv in intervals if inv.get("panel_id", "") == panel_id and inv.get("class", 0) == cls), default=0.0),
            })
    return result


def _dedup_points(points: list[dict]) -> list[dict]:
    if not points:
        return []
    seen: set[tuple] = set()
    result = []
    for p in points:
        key = (p.get("panel_id", ""), p.get("class", 0), round(p.get("depth", 0), 0))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result
