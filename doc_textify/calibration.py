"""Grid-line / tick-mark calibration for chart coordinate mapping.

Replaces linear pixel→data approximation with calibration points
extracted from the chart's own grid lines and tick labels.

Supports:
- Single y-axis (left margin ticks)            — calibrate_axis()
- Single x-axis (bottom margin ticks)           — calibrate_x_axis()
- Dual y-axis (right margin ticks)              — calibrate_dual_y_axis()
- Full-panel grid-line detection                — calibrate_from_grid_lines()
- Log-scale detection & interpolation           — _detect_log_scale() / pixel_to_data_log()
- Calibration confidence scoring                — _calibration_confidence()
"""

from __future__ import annotations

from typing import Any


# ══════════════════════════════════════════════════════════════════════
# 1. Y-Axis Calibration (left margin tick marks) — backward-compatible
# ══════════════════════════════════════════════════════════════════════


def calibrate_axis(
    color_image,
    panel: dict,
    ocr_blocks: list,
) -> dict[str, Any]:
    """Build pixel→data calibration for the y-axis of a chart panel.

    1. Detect horizontal tick/grid lines in the left margin via HoughLines.
    2. Match nearby OCR tick-label numbers to each line.
    3. Fit a calibration curve (piecewise-linear).
    4. Compute R² fit quality and confidence score.
    5. Detect log vs linear scale.

    Returns dict with:
        calibration_points: list of {"pixel_y": float, "data_value": float}
        r_squared: float (0..1)
        calibrated: bool
        confidence: float (0..1, overall calibration quality)
        log_scale: bool
        fit_params: dict with slope & intercept
    """
    result: dict[str, Any] = {
        "calibration_points": [],
        "r_squared": 0.0,
        "calibrated": False,
        "confidence": 0.0,
        "log_scale": False,
    }

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    img = np.array(color_image)
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]

    # ── Step 1: Extract tick marks from left margin ──────────────────
    # Look at the left 8% of the panel (where y-axis tick marks live)
    margin_w = max(int((px1 - px0) * 0.08), 15)
    left_margin = img[py0:py1, px0 : px0 + margin_w]
    if left_margin.size == 0:
        return result

    gray = cv2.cvtColor(left_margin, cv2.COLOR_RGB2GRAY)

    # Apply GaussianBlur for noise reduction before edge detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use OTSU thresholding (adaptive, not hardcoded)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal lines (tick marks) in the margin
    h_lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 2,
        threshold=15,
        minLineLength=max(int(margin_w * 0.3), 5),
        maxLineGap=3,
    )

    tick_ys: list[int] = []
    if h_lines is not None:
        for line in h_lines:
            lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
            # Keep only near-horizontal lines
            if abs(ly2 - ly1) < 3:
                tick_ys.append(int((ly1 + ly2) / 2))

    if not tick_ys:
        return result

    # Deduplicate nearby lines (within 5 px)
    tick_ys = _merge_nearby_lines(tick_ys, axis="y", threshold=5)
    tick_ys = sorted(set(tick_ys))

    # ── Step 2: Match OCR tick labels ────────────────────────────────
    tick_matches = _match_ocr_labels_vertical(
        ocr_blocks, px0, py0, py1, margin_w, side="left"
    )

    if len(tick_matches) < 2:
        return result

    # ── Step 3: Match each tick line to the nearest OCR label ────────
    calibration_points = _pair_ticks_to_labels(tick_ys, tick_matches)

    if len(calibration_points) < 2:
        return result

    # Sort by pixel_y ascending
    calibration_points.sort(key=lambda p: p[0])

    # ── Step 4: Fit quality check ────────────────────────────────────
    ys = np.array([p[0] for p in calibration_points])
    vals = np.array([p[1] for p in calibration_points])

    # Linear regression: vals = m * ys + b
    m_coef, b_coef = np.polyfit(ys, vals, 1)
    predicted = m_coef * ys + b_coef
    ss_res = np.sum((vals - predicted) ** 2)
    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Step 5: Log-scale detection ──────────────────────────────────
    log_scale = _detect_log_scale(calibration_points)

    # ── Step 6: Confidence score ─────────────────────────────────────
    axis_range = (float(np.min(vals)), float(np.max(vals)))
    confidence = _calibration_confidence(
        n_points=len(calibration_points),
        r_squared=r_sq,
        axis_range=axis_range,
        panel_range=(panel.get("y_min", 0.0), panel.get("y_max", 30.0)),
    )

    formatted_points = [
        {"pixel_y": round(float(p[0]), 1), "data_value": round(float(p[1]), 2)}
        for p in calibration_points
    ]

    return {
        "calibration_points": formatted_points,
        "r_squared": round(float(r_sq), 4),
        "calibrated": r_sq > 0.85,
        "confidence": round(float(confidence), 4),
        "log_scale": log_scale,
        "fit_params": {"slope": round(float(m_coef), 6), "intercept": round(float(b_coef), 2)},
    }


# ══════════════════════════════════════════════════════════════════════
# 2. X-Axis Calibration (bottom margin tick marks)
# ══════════════════════════════════════════════════════════════════════


def calibrate_x_axis(
    color_image,
    panel: dict,
    ocr_blocks: list,
) -> dict[str, Any]:
    """Build pixel→data calibration for the x-axis of a chart panel.

    Mirrors calibrate_axis() but detects vertical tick marks from the
    bottom margin (bottom 8% of panel) and matches OCR labels positioned
    below the panel.

    Returns dict with:
        calibration_points: list of {"pixel_x": float, "data_value": float}
        r_squared: float (0..1)
        calibrated: bool
        confidence: float (0..1)
        log_scale: bool
        fit_params: dict with slope & intercept
    """
    result: dict[str, Any] = {
        "calibration_points": [],
        "r_squared": 0.0,
        "calibrated": False,
        "confidence": 0.0,
        "log_scale": False,
    }

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    img = np.array(color_image)
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]

    # ── Step 1: Extract tick marks from bottom margin ─────────────────
    margin_h = max(int((py1 - py0) * 0.08), 15)
    bottom_margin = img[py1 - margin_h : py1, px0:px1]
    if bottom_margin.size == 0:
        return result

    gray = cv2.cvtColor(bottom_margin, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find vertical lines (tick marks) in the margin
    v_lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 2,
        threshold=15,
        minLineLength=max(int(margin_h * 0.3), 5),
        maxLineGap=3,
    )

    tick_xs: list[int] = []
    if v_lines is not None:
        for line in v_lines:
            lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
            # Keep only near-vertical lines
            if abs(lx2 - lx1) < 3:
                tick_xs.append(int((lx1 + lx2) / 2))

    if not tick_xs:
        return result

    # Deduplicate nearby lines (within 5 px)
    tick_xs = _merge_nearby_lines(tick_xs, axis="x", threshold=5)
    tick_xs = sorted(set(tick_xs))

    # ── Step 2: Match OCR tick labels ────────────────────────────────
    tick_matches = _match_ocr_labels_horizontal(ocr_blocks, px0, py0, py1, px1, margin_h)

    if len(tick_matches) < 2:
        return result

    # ── Step 3: Match each tick line to the nearest OCR label ────────
    calibration_points = _pair_ticks_to_labels(tick_xs, tick_matches)

    if len(calibration_points) < 2:
        return result

    # Sort by pixel_x ascending
    calibration_points.sort(key=lambda p: p[0])

    # ── Step 4: Fit quality check ────────────────────────────────────
    xs = np.array([p[0] for p in calibration_points])
    vals = np.array([p[1] for p in calibration_points])

    m_coef, b_coef = np.polyfit(xs, vals, 1)
    predicted = m_coef * xs + b_coef
    ss_res = np.sum((vals - predicted) ** 2)
    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Step 5: Log-scale detection ──────────────────────────────────
    log_scale = _detect_log_scale(calibration_points)

    # ── Step 6: Confidence score ─────────────────────────────────────
    axis_range = (float(np.min(vals)), float(np.max(vals)))
    confidence = _calibration_confidence(
        n_points=len(calibration_points),
        r_squared=r_sq,
        axis_range=axis_range,
        panel_range=(panel.get("x_min", 0.0), panel.get("x_max", 100.0)),
    )

    formatted_points = [
        {"pixel_x": round(float(p[0]), 1), "data_value": round(float(p[1]), 2)}
        for p in calibration_points
    ]

    return {
        "calibration_points": formatted_points,
        "r_squared": round(float(r_sq), 4),
        "calibrated": r_sq > 0.85,
        "confidence": round(float(confidence), 4),
        "log_scale": log_scale,
        "fit_params": {"slope": round(float(m_coef), 6), "intercept": round(float(b_coef), 2)},
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Dual Y-Axis Detection (right margin tick marks)
# ══════════════════════════════════════════════════════════════════════


def calibrate_dual_y_axis(
    color_image,
    panel: dict,
    ocr_blocks: list,
    primary_calibration: dict | None = None,
) -> dict[str, Any]:
    """Detect and calibrate a second y-axis from the right margin.

    Scans the right 8% of the panel for horizontal tick marks and matches
    OCR labels to the right of the panel. If found with significantly
    different label values from the primary calibration, produces a
    second set of calibration points.

    Args:
        color_image: PIL/Pillow image or numpy array.
        panel: dict with x0, y0, x1, y1 bounds.
        ocr_blocks: list of OCR result objects with .bbox and .text.
        primary_calibration: optional primary y-axis calibration result
            for comparison (to confirm dual-axis).

    Returns dict with:
        calibration_points_y2: list of {"pixel_y": float, "data_value": float}
        r_squared_y2: float
        calibrated_y2: bool
        is_dual_axis: bool (True if a distinct second scale was found)
        confidence_y2: float
        log_scale_y2: bool
    """
    result: dict[str, Any] = {
        "calibration_points_y2": [],
        "r_squared_y2": 0.0,
        "calibrated_y2": False,
        "is_dual_axis": False,
        "confidence_y2": 0.0,
        "log_scale_y2": False,
    }

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    img = np.array(color_image)
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
    panel_width = px1 - px0
    margin_w = max(int(panel_width * 0.08), 15)

    # ── Step 1: Extract tick marks from right margin ──────────────────
    right_margin = img[py0:py1, px1 - margin_w : px1]
    if right_margin.size == 0:
        return result

    gray = cv2.cvtColor(right_margin, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 2,
        threshold=15,
        minLineLength=max(int(margin_w * 0.3), 5),
        maxLineGap=3,
    )

    tick_ys: list[int] = []
    if h_lines is not None:
        for line in h_lines:
            lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
            if abs(ly2 - ly1) < 3:
                tick_ys.append(int((ly1 + ly2) / 2))

    if not tick_ys:
        return result

    tick_ys = _merge_nearby_lines(tick_ys, axis="y", threshold=5)
    tick_ys = sorted(set(tick_ys))

    # ── Step 2: Match OCR labels to the right of the panel ───────────
    tick_matches = _match_ocr_labels_vertical(
        ocr_blocks, px0, py0, py1, margin_w, side="right", panel_right=px1
    )

    if len(tick_matches) < 2:
        return result

    # ── Step 3: Pair ticks to labels ──────────────────────────────────
    calibration_points = _pair_ticks_to_labels(tick_ys, tick_matches)

    if len(calibration_points) < 2:
        return result

    calibration_points.sort(key=lambda p: p[0])

    # ── Step 4: Fit quality ──────────────────────────────────────────
    ys = np.array([p[0] for p in calibration_points])
    vals = np.array([p[1] for p in calibration_points])

    m_coef, b_coef = np.polyfit(ys, vals, 1)
    predicted = m_coef * ys + b_coef
    ss_res = np.sum((vals - predicted) ** 2)
    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Step 5: Check if this is truly a different scale ──────────────
    is_dual = True
    if primary_calibration and primary_calibration.get("calibration_points"):
        primary_vals = {p["data_value"] for p in primary_calibration["calibration_points"]}
        secondary_vals = {p[1] for p in calibration_points}
        # If >50% of values overlap, it's probably the same axis duplicated
        overlap = primary_vals & secondary_vals
        if len(overlap) > 0.5 * min(len(primary_vals), len(secondary_vals)):
            is_dual = False
        # Also check if the value range is substantially different
        p_min = min(primary_vals)
        p_max = max(primary_vals)
        s_min = min(secondary_vals)
        s_max = max(secondary_vals)
        if (abs(p_min - s_min) < 0.01 * max(abs(p_min), 1) and
                abs(p_max - s_max) < 0.01 * max(abs(p_max), 1)):
            is_dual = False

    if not is_dual:
        return result

    # ── Step 6: Log-scale detection & confidence ──────────────────────
    log_scale = _detect_log_scale(calibration_points)
    axis_range = (float(np.min(vals)), float(np.max(vals)))
    confidence = _calibration_confidence(
        n_points=len(calibration_points),
        r_squared=r_sq,
        axis_range=axis_range,
    )

    formatted_points = [
        {"pixel_y": round(float(p[0]), 1), "data_value": round(float(p[1]), 2)}
        for p in calibration_points
    ]

    return {
        "calibration_points_y2": formatted_points,
        "r_squared_y2": round(float(r_sq), 4),
        "calibrated_y2": r_sq > 0.85,
        "is_dual_axis": True,
        "confidence_y2": round(float(confidence), 4),
        "log_scale_y2": log_scale,
        "fit_params_y2": {
            "slope": round(float(m_coef), 6),
            "intercept": round(float(b_coef), 2),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Full-Panel Grid-Line Calibration
# ══════════════════════════════════════════════════════════════════════


def calibrate_from_grid_lines(
    color_image,
    panel: dict,
    ocr_blocks: list,
) -> dict[str, Any]:
    """Detect calibration from grid lines across the ENTIRE panel.

    Uses Canny edge detection + HoughLinesP across the full panel area
    (not just margins) to find both horizontal and vertical grid lines.
    Tries multiple Canny threshold pairs and selects the best one.

    Grid lines are typically lighter and thinner than margin tick marks,
    so adaptive threshold tuning helps find them reliably.

    Returns dict with:
        calibration_points_y: list of {"pixel_y": float, "data_value": float}
        calibration_points_x: list of {"pixel_x": float, "data_value": float}
        r_squared_y, r_squared_x: float
        calibrated_y, calibrated_x: bool
        log_scale_y, log_scale_x: bool
        confidence_y, confidence_x: float
        threshold_used: (low, high) Canny thresholds that worked best
    """
    result: dict[str, Any] = {
        "calibration_points_y": [],
        "calibration_points_x": [],
        "r_squared_y": 0.0,
        "r_squared_x": 0.0,
        "calibrated_y": False,
        "calibrated_x": False,
        "log_scale_y": False,
        "log_scale_x": False,
        "confidence_y": 0.0,
        "confidence_x": 0.0,
        "threshold_used": None,
    }

    try:
        import cv2
        import numpy as np
    except ImportError:
        return result

    img = np.array(color_image)
    px0, py0, px1, py1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]

    # ── Step 1: Extract full panel ───────────────────────────────────
    panel_img = img[py0:py1, px0:px1]
    if panel_img.size == 0:
        return result

    gray = cv2.cvtColor(panel_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Step 2: Adaptive Canny threshold tuning ───────────────────────
    threshold_pairs = [(50, 150), (30, 90), (20, 60), (15, 45)]
    best_pair = (50, 150)
    best_score = float("inf")
    best_h_lines = None
    best_v_lines = None
    target_min = 3
    target_max = 15

    for low, high in threshold_pairs:
        edges = cv2.Canny(gray, low, high)

        h_lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 2, threshold=20,
            minLineLength=max(int((px1 - px0) * 0.2), 20), maxLineGap=10,
        )
        v_lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 2, threshold=20,
            minLineLength=max(int((py1 - py0) * 0.2), 20), maxLineGap=10,
        )

        # Filter for near-horizontal / near-vertical
        h_filtered = []
        if h_lines is not None:
            for line in h_lines:
                lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
                if abs(ly2 - ly1) < 3:
                    h_filtered.append((lx1, ly1, lx2, ly2))

        v_filtered = []
        if v_lines is not None:
            for line in v_lines:
                lx1, ly1, lx2, ly2 = [int(v) for v in line.ravel()[:4]]
                if abs(lx2 - lx1) < 3:
                    v_filtered.append((lx1, ly1, lx2, ly2))

        h_count = len(h_filtered)
        v_count = len(v_filtered)

        # Perfect: both axes within target range
        if target_min <= h_count <= target_max and target_min <= v_count <= target_max:
            best_pair = (low, high)
            best_h_lines = h_filtered
            best_v_lines = v_filtered
            best_score = 0
            break

        # Score: L1 penalty for being outside [target_min, target_max]
        h_penalty = max(0, target_min - h_count) * 2 + max(0, h_count - target_max)
        v_penalty = max(0, target_min - v_count) * 2 + max(0, v_count - target_max)
        score = h_penalty + v_penalty

        if score < best_score:
            best_score = score
            best_pair = (low, high)
            best_h_lines = h_filtered
            best_v_lines = v_filtered

    result["threshold_used"] = best_pair

    if not best_h_lines and not best_v_lines:
        return result

    # ── Step 3: Extract pixel positions & deduplicate ─────────────────
    h_ys: list[int] = []
    if best_h_lines:
        for x1, y1, x2, y2 in best_h_lines:
            h_ys.append(int((y1 + y2) / 2))
    h_ys = _merge_nearby_lines(h_ys, axis="y", threshold=5)
    h_ys = sorted(set(h_ys))

    v_xs: list[int] = []
    if best_v_lines:
        for x1, y1, x2, y2 in best_v_lines:
            v_xs.append(int((x1 + x2) / 2))
    v_xs = _merge_nearby_lines(v_xs, axis="x", threshold=5)
    v_xs = sorted(set(v_xs))

    # ── Step 4: Match OCR labels ─────────────────────────────────────
    # Y-axis labels (near left edge, within panel y-range)
    margin_w = max(int((px1 - px0) * 0.08), 15)
    y_label_matches = _match_ocr_labels_vertical(
        ocr_blocks, px0, py0, py1, margin_w, side="left"
    )

    # X-axis labels (below panel, within panel x-range)
    margin_h = max(int((py1 - py0) * 0.08), 15)
    x_label_matches = _match_ocr_labels_horizontal(
        ocr_blocks, px0, py0, py1, px1, margin_h
    )

    # ── Step 5: Calibrate y-axis from grid lines ─────────────────────
    if len(h_ys) >= 2 and len(y_label_matches) >= 2:
        y_points = _pair_ticks_to_labels(h_ys, y_label_matches)
        if len(y_points) >= 2:
            y_points.sort(key=lambda p: p[0])
            ys_arr = np.array([p[0] for p in y_points])
            vals_arr = np.array([p[1] for p in y_points])

            m_y, b_y = np.polyfit(ys_arr, vals_arr, 1)
            pred_y = m_y * ys_arr + b_y
            ss_res_y = np.sum((vals_arr - pred_y) ** 2)
            ss_tot_y = np.sum((vals_arr - np.mean(vals_arr)) ** 2)
            r_sq_y = 1.0 - ss_res_y / ss_tot_y if ss_tot_y > 0 else 0.0

            result["calibration_points_y"] = [
                {"pixel_y": round(float(p[0]), 1), "data_value": round(float(p[1]), 2)}
                for p in y_points
            ]
            result["r_squared_y"] = round(float(r_sq_y), 4)
            result["calibrated_y"] = r_sq_y > 0.85
            result["log_scale_y"] = _detect_log_scale(y_points)
            result["fit_params_y"] = {
                "slope": round(float(m_y), 6),
                "intercept": round(float(b_y), 2),
            }

            ax_range = (float(np.min(vals_arr)), float(np.max(vals_arr)))
            result["confidence_y"] = round(float(_calibration_confidence(
                n_points=len(y_points), r_squared=r_sq_y, axis_range=ax_range,
            )), 4)

    # ── Step 6: Calibrate x-axis from grid lines ─────────────────────
    if len(v_xs) >= 2 and len(x_label_matches) >= 2:
        x_points = _pair_ticks_to_labels(v_xs, x_label_matches)
        if len(x_points) >= 2:
            x_points.sort(key=lambda p: p[0])
            xs_arr = np.array([p[0] for p in x_points])
            vals_arr = np.array([p[1] for p in x_points])

            m_x, b_x = np.polyfit(xs_arr, vals_arr, 1)
            pred_x = m_x * xs_arr + b_x
            ss_res_x = np.sum((vals_arr - pred_x) ** 2)
            ss_tot_x = np.sum((vals_arr - np.mean(vals_arr)) ** 2)
            r_sq_x = 1.0 - ss_res_x / ss_tot_x if ss_tot_x > 0 else 0.0

            result["calibration_points_x"] = [
                {"pixel_x": round(float(p[0]), 1), "data_value": round(float(p[1]), 2)}
                for p in x_points
            ]
            result["r_squared_x"] = round(float(r_sq_x), 4)
            result["calibrated_x"] = r_sq_x > 0.85
            result["log_scale_x"] = _detect_log_scale(x_points)
            result["fit_params_x"] = {
                "slope": round(float(m_x), 6),
                "intercept": round(float(b_x), 2),
            }

            ax_range = (float(np.min(vals_arr)), float(np.max(vals_arr)))
            result["confidence_x"] = round(float(_calibration_confidence(
                n_points=len(x_points), r_squared=r_sq_x, axis_range=ax_range,
            )), 4)

    return result


# ══════════════════════════════════════════════════════════════════════
# 5. Pixel-to-Data Conversion (updated with axis parameter)
# ══════════════════════════════════════════════════════════════════════


def pixel_to_data(
    pixel_value: float,
    calibration: dict[str, Any],
    panel: dict,
    panel_dimension: int,
    axis: str = "y",
) -> float:
    """Convert a pixel coordinate (panel-local) to a data value.

    Uses calibration points with piecewise-linear interpolation if available,
    otherwise falls back to linear mapping from panel metadata.

    Args:
        pixel_value: pixel coordinate (y for axis="y", x for axis="x").
        calibration: dict from calibrate_axis / calibrate_x_axis / calibrate_from_grid_lines.
        panel: dict with bounds and data range (y_min/y_max or x_min/x_max).
        panel_dimension: panel height for y-axis, panel width for x-axis.
        axis: "y" (default, backward-compatible) or "x".

    Returns:
        Interpolated data value.
    """
    if axis == "x":
        # Look for x-axis calibration (from calibrate_x_axis or calibrate_from_grid_lines)
        points = calibration.get("calibration_points", [])
        if not points:
            # Also try calibrate_from_grid_lines naming
            points = calibration.get("calibration_points_x", [])
        if not points or not calibration.get("calibrated", False):
            if not calibration.get("calibrated_x", False) and not calibration.get("calibrated", False):
                x_min = panel.get("x_min", 0.0)
                x_max = panel.get("x_max", 100.0)
                return x_min + (pixel_value / max(panel_dimension, 1)) * (x_max - x_min)

        py = sorted(points.copy(), key=lambda p: p.get("pixel_x", p.get("pixel_y", 0)))
        pixel_key = "pixel_x" if "pixel_x" in py[0] else "pixel_y"

        # Extrapolation low
        if pixel_value <= py[0][pixel_key]:
            if len(py) >= 2:
                return _interpolate_generic(pixel_value, py[0], py[1],
                                            pixel_key=pixel_key)
            return py[0]["data_value"]

        # Extrapolation high
        if pixel_value >= py[-1][pixel_key]:
            if len(py) >= 2:
                return _interpolate_generic(pixel_value, py[-2], py[-1],
                                            pixel_key=pixel_key)
            return py[-1]["data_value"]

        # Interpolation
        for i in range(len(py) - 1):
            if py[i][pixel_key] <= pixel_value <= py[i + 1][pixel_key]:
                return _interpolate_generic(pixel_value, py[i], py[i + 1],
                                            pixel_key=pixel_key)

        return py[-1]["data_value"]
    else:
        # y-axis (default, backward-compatible)
        points = calibration.get("calibration_points", [])
        if not points or not calibration.get("calibrated"):
            y_min = panel.get("y_min", 0.0)
            y_max = panel.get("y_max", 30.0)
            return y_min + (pixel_value / max(panel_dimension, 1)) * (y_max - y_min)

        py = sorted(points.copy(), key=lambda p: p["pixel_y"])

        # Extrapolation low
        if pixel_value <= py[0]["pixel_y"]:
            if len(py) >= 2:
                return _interpolate_generic(pixel_value, py[0], py[1],
                                            pixel_key="pixel_y")
            return py[0]["data_value"]

        # Extrapolation high
        if pixel_value >= py[-1]["pixel_y"]:
            if len(py) >= 2:
                return _interpolate_generic(pixel_value, py[-2], py[-1],
                                            pixel_key="pixel_y")
            return py[-1]["data_value"]

        # Interpolation
        for i in range(len(py) - 1):
            if py[i]["pixel_y"] <= pixel_value <= py[i + 1]["pixel_y"]:
                return _interpolate_generic(pixel_value, py[i], py[i + 1],
                                            pixel_key="pixel_y")

        return py[-1]["data_value"]


def pixel_to_data_log(
    pixel_value: float,
    calibration: dict[str, Any],
    panel: dict,
    panel_dimension: int,
    axis: str = "y",
) -> float:
    """Convert pixel coordinate to data value using log-scale interpolation.

    Performs piecewise-linear interpolation in log10 space, then converts
    back. Falls back to pixel_to_data() if no calibration points available.

    Args:
        pixel_value: pixel coordinate (y for axis="y", x for axis="x").
        calibration: dict with calibration points.
        panel: dict with bounds.
        panel_dimension: panel height (y) or width (x).
        axis: "y" or "x".

    Returns:
        Interpolated data value on the original (non-log) scale.
    """
    import math

    # Determine which calibration points to use
    if axis == "x":
        points = calibration.get("calibration_points", [])
        if not points:
            points = calibration.get("calibration_points_x", [])
        pixel_key = "pixel_x" if points and "pixel_x" in points[0] else "pixel_y"
    else:
        points = calibration.get("calibration_points", [])
        pixel_key = "pixel_y"

    if len(points) < 2:
        return pixel_to_data(pixel_value, calibration, panel, panel_dimension, axis)

    # Build log-space points
    py = sorted(points.copy(), key=lambda p: p[pixel_key])
    log_points: list[dict] = []
    for p in py:
        val = max(p["data_value"], 1e-10)
        log_points.append({
            "pixel": p[pixel_key],
            "log_value": math.log10(val),
        })

    # Piecewise-linear interpolation in log space
    if pixel_value <= log_points[0]["pixel"]:
        if len(log_points) >= 2:
            log_val = _interpolate_xy(
                pixel_value,
                log_points[0]["pixel"], log_points[0]["log_value"],
                log_points[1]["pixel"], log_points[1]["log_value"],
            )
        else:
            log_val = log_points[0]["log_value"]
    elif pixel_value >= log_points[-1]["pixel"]:
        if len(log_points) >= 2:
            log_val = _interpolate_xy(
                pixel_value,
                log_points[-2]["pixel"], log_points[-2]["log_value"],
                log_points[-1]["pixel"], log_points[-1]["log_value"],
            )
        else:
            log_val = log_points[-1]["log_value"]
    else:
        found = False
        for i in range(len(log_points) - 1):
            if log_points[i]["pixel"] <= pixel_value <= log_points[i + 1]["pixel"]:
                log_val = _interpolate_xy(
                    pixel_value,
                    log_points[i]["pixel"], log_points[i]["log_value"],
                    log_points[i + 1]["pixel"], log_points[i + 1]["log_value"],
                )
                found = True
                break
        if not found:
            log_val = log_points[-1]["log_value"]

    return 10.0 ** log_val


# ══════════════════════════════════════════════════════════════════════
# 6. Internal Helpers
# ══════════════════════════════════════════════════════════════════════


def _parse_tick_number(text: str) -> float | None:
    """Parse a numeric value from an OCR tick label string.

    Handles common OCR confusions: 'O'→'0', 'l'→'1', '·'→'.', etc.
    """
    import re

    t = text.strip()
    # Clean common OCR confusions
    t = t.replace("O", "0").replace("o", "0").replace("l", "1")
    t = t.replace("·", ".").replace(",", ".").replace(" ", "")

    # Try direct float parse
    try:
        return float(t)
    except ValueError:
        pass

    # Try extracting a number pattern
    m = re.search(r"[-]?\d+\.?\d*", t)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None


def _merge_nearby_lines(
    positions: list[int],
    axis: str = "y",
    threshold: int = 5,
) -> list[int]:
    """Deduplicate HoughLinesP results by merging positions within threshold pixels.

    Lines detected by HoughLinesP often appear as multiple nearby segments
    for the same visual line. This merges clusters within `threshold` px
    into single averaged positions.

    Args:
        positions: list of pixel positions (y for horizontal lines, x for vertical).
        axis: "y" or "x".
        threshold: merge distance in pixels.

    Returns:
        Deduplicated list of pixel positions.
    """
    if not positions:
        return []

    sorted_pos = sorted(positions)
    merged: list[int] = []
    cluster: list[int] = [sorted_pos[0]]

    for pos in sorted_pos[1:]:
        if pos - cluster[-1] <= threshold:
            cluster.append(pos)
        else:
            merged.append(int(sum(cluster) / len(cluster)))
            cluster = [pos]

    merged.append(int(sum(cluster) / len(cluster)))
    return merged


def _interpolate_generic(
    x: float,
    p0: dict[str, float],
    p1: dict[str, float],
    pixel_key: str = "pixel_y",
) -> float:
    """Linear interpolation between two calibration points (generic key)."""
    x0, y0 = p0[pixel_key], p0["data_value"]
    x1, y1 = p1[pixel_key], p1["data_value"]
    if x1 == x0:
        return (y0 + y1) / 2
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _interpolate_xy(
    x: float, x0: float, y0: float, x1: float, y1: float,
) -> float:
    """Linear interpolation between two (x,y) points."""
    if x1 == x0:
        return (y0 + y1) / 2
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _detect_log_scale(
    calibration_points: list[tuple[float, float]],
    tolerance: float = 0.15,
) -> bool:
    """Detect if calibration data values follow a logarithmic progression.

    A log scale is indicated when the *ratio* between consecutive values
    is roughly constant, rather than the *difference*.

    Args:
        calibration_points: sorted list of (pixel, data_value) pairs.
        tolerance: max allowed coefficient of variation for ratios.

    Returns:
        True if log scale detected.
    """
    if len(calibration_points) < 3:
        return False

    import numpy as np

    vals = np.array([p[1] for p in calibration_points])

    # Need all positive values for log scale
    if np.any(vals <= 0):
        return False

    # For log scale: ratio between consecutive values should be roughly constant
    ratios = vals[1:] / vals[:-1]
    # For linear scale: difference should be roughly constant
    diffs = np.diff(vals)

    # Coefficient of variation (CV) = std / mean — lower is more constant
    cv_ratio = float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else float("inf")
    cv_diff = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else float("inf")

    # Log scale: ratios are more consistent than differences
    return cv_ratio < cv_diff and cv_ratio < tolerance


def _calibration_confidence(
    n_points: int,
    r_squared: float,
    axis_range: tuple[float, float],
    panel_range: tuple[float, float] | None = None,
) -> float:
    """Compute a composite calibration confidence score (0..1).

    Factors:
        - Number of calibration points (more → higher)
        - R² fit quality
        - Data range coverage (how much of the panel's data range is covered)

    Args:
        n_points: number of matched tick-label pairs.
        r_squared: linear regression R² (0..1).
        axis_range: (min_val, max_val) of calibration data values.
        panel_range: (data_min, data_max) from panel metadata, if known.

    Returns:
        Confidence score 0..1.
    """
    import numpy as np

    # Point-count score: 2→0.2, 5→0.6, 10→0.9, 15+→1.0
    point_score = min(1.0, np.log(max(n_points, 1)) / np.log(15))

    # R² score: direct mapping (clamped)
    r2_score = max(0.0, min(1.0, r_squared))

    # Coverage score: how much of the data range is spanned
    if panel_range and panel_range[1] > panel_range[0]:
        panel_span = panel_range[1] - panel_range[0]
        cal_span = axis_range[1] - axis_range[0]
        coverage = min(1.0, cal_span / max(panel_span, 1e-6))
        # Bonus for covering >50% of the panel range
        if coverage >= 0.9:
            cov_score = 1.0
        elif coverage >= 0.5:
            cov_score = 0.5 + (coverage - 0.5)
        else:
            cov_score = coverage
    else:
        # Without panel metadata: just check if we have good spread
        if n_points >= 3:
            cov_score = 0.7
        else:
            cov_score = 0.3

    # Weighted combination
    return float(np.clip(
        0.35 * point_score + 0.40 * r2_score + 0.25 * cov_score,
        0.0, 1.0,
    ))


def _match_ocr_labels_vertical(
    ocr_blocks: list,
    px0: int, py0: int, py1: int,
    margin_w: int,
    side: str = "left",
    panel_right: int | None = None,
) -> list[tuple[float, float, str]]:
    """Find OCR numeric labels near the y-axis margin.

    For side="left": labels must be left of px0 + margin_w*2.
    For side="right": labels must be right of panel_right - margin_w*2.

    Returns list of (local_y, parsed_value, original_text).
    """
    tick_matches: list[tuple[float, float, str]] = []

    for block in ocr_blocks:
        if not block.bbox:
            continue
        bx0, by0, bx1, by1 = block.bbox
        text = block.text.strip()

        # Must be within panel y-range (with some tolerance)
        if by0 < py0 - 10 or by1 > py1 + 10:
            continue

        if side == "left":
            # Labels must be near the left edge of the panel
            if bx1 > px0 + margin_w * 2:
                continue
        else:
            # Labels must be near the right edge of the panel
            ref_right = panel_right if panel_right else px0 + margin_w
            if bx0 < ref_right - margin_w * 2:
                continue

        parsed = _parse_tick_number(text)
        if parsed is None:
            continue

        block_cy = (by0 + by1) / 2
        local_y = block_cy - py0  # panel-local y

        tick_matches.append((local_y, parsed, text))

    return tick_matches


def _match_ocr_labels_horizontal(
    ocr_blocks: list,
    px0: int, py0: int, py1: int, px1: int,
    margin_h: int,
) -> list[tuple[float, float, str]]:
    """Find OCR numeric labels near the x-axis (below bottom edge).

    Labels must be below the panel bottom edge, within the margin zone,
    and horizontally within the panel x-range.

    Returns list of (local_x, parsed_value, original_text).
    """
    tick_matches: list[tuple[float, float, str]] = []

    for block in ocr_blocks:
        if not block.bbox:
            continue
        bx0, by0, bx1, by1 = block.bbox
        text = block.text.strip()

        # Must be at or below the bottom edge, within a reasonable margin
        if by0 < py1 - 5 or by0 > py1 + margin_h * 3:
            continue

        # Must be within panel x-range (with some tolerance)
        if bx0 < px0 - 20 or bx1 > px1 + 20:
            continue

        parsed = _parse_tick_number(text)
        if parsed is None:
            continue

        block_cx = (bx0 + bx1) / 2
        local_x = block_cx - px0  # panel-local x

        tick_matches.append((local_x, parsed, text))

    return tick_matches


def _pair_ticks_to_labels(
    tick_positions: list[int],
    label_matches: list[tuple[float, float, str]],
    max_distance: int = 25,
) -> list[tuple[float, float]]:
    """Match tick mark pixel positions to the nearest OCR label.

    Each label is used at most once (greedy nearest-neighbor).

    Args:
        tick_positions: list of pixel positions (y or x) of detected tick lines.
        label_matches: list of (local_position, parsed_value, original_text).
        max_distance: maximum allowed distance in pixels for a valid match.

    Returns:
        list of (pixel_position, data_value) calibration point pairs.
    """
    calibration_points: list[tuple[float, float]] = []
    matched_label_indices: set[int] = set()

    for tp in tick_positions:
        best_match = None
        best_dist = float("inf")
        for idx, (label_pos, value, _text) in enumerate(label_matches):
            if idx in matched_label_indices:
                continue
            dist = abs(label_pos - tp)
            if dist < best_dist:
                best_dist = dist
                best_match = (tp, value, idx)

        if best_match and best_dist < max_distance:
            calibration_points.append((float(best_match[0]), float(best_match[1])))
            matched_label_indices.add(best_match[2])

    return calibration_points
