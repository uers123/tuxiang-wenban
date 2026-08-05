"""Table structure recovery for doc-textify.

Detects table regions from image/PDF content, reconstructs cell grids
from visual separator lines, maps OCR text into cells, and outputs
structured Markdown tables plus JSON cell arrays.

Three strategies:
  1. PDF-native: pdfplumber table extraction (most reliable for native PDFs)
  2. Line-based: detect horizontal + vertical separator lines → cell grid
  3. Gap-based: detect text column alignment gaps (fallback)
"""

from __future__ import annotations

from typing import Any

from .models import Block

# ── Optional pdfplumber import ──────────────────────────────────────────
try:
    import pdfplumber  # noqa: F401
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def extract_tables_from_pdf_page(pdf_page) -> list[Block]:
    """Extract tables from a native PDF page using pdfplumber.

    This is the most reliable method for PDFs with embedded table data
    (not scanned images).  It uses pdfplumber's table-finding algorithm
    which analyses vector-drawing commands and text positions.

    Args:
        pdf_page: A ``pdfplumber.page.Page`` object.

    Returns:
        List of Block objects with type="table".  Falls back to an
        empty list when pdfplumber is not installed.
    """
    if not _HAS_PDFPLUMBER:
        return []

    import pdfplumber

    table_blocks: list[Block] = []

    # pdfplumber built-in table extraction
    tables = pdf_page.find_tables()

    for tbl in tables:
        cells_raw = tbl.extract()  # list[list[str | None]]
        if not cells_raw or not cells_raw[0]:
            continue

        # Build our cell grid format
        # pdfplumber cells include header heuristics via table-settings;
        # we pass the raw grid through our own renderer for consistency
        cells: list[list[dict[str, Any]]] = []
        for row in cells_raw:
            row_cells: list[dict[str, Any]] = []
            for text in row:
                row_cells.append({
                    "text": (text or "").replace("\n", " ").strip(),
                    "blocks": [],
                    "bbox": list(tbl.bbox) if tbl.bbox is not None else [],
                })
            cells.append(row_cells)

        markdown = _cells_to_markdown(cells)
        if not markdown:
            continue

        bbox = tbl.bbox
        if bbox is None:
            bbox_tuple = (0.0, 0.0, float(pdf_page.width or 0), float(pdf_page.height or 0))
        else:
            bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        table_blocks.append(Block(
            type="table",
            text=markdown,
            bbox=bbox_tuple,
            confidence=90.0,
            engine="pdfplumber-table",
            metadata={
                "cells": cells,
                "rows": len(cells),
                "cols": len(cells[0]) if cells else 0,
                "method": "pdfplumber",
            },
        ))

    return table_blocks


def extract_tables_from_image(
    color_image,
    ocr_blocks: list[Block],
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[Block]:
    """Detect and extract table blocks from an image.

    Finds table regions via line detection or column-gap alignment,
    then assigns OCR text to grid cells.

    Returns list of Block objects with type="table" and metadata
    containing "cells" (grid structure) and "rows" (text per cell).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _detect_tables_by_gaps(ocr_blocks, page_width)

    img = np.array(color_image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Invert: table lines are usually dark on light background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Detect horizontal and vertical lines ─────────────────────────
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    # Combine to find table bounding boxes
    table_mask = cv2.add(horiz_lines, vert_lines)
    kernel = np.ones((5, 5), np.uint8)
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)
    table_mask = cv2.erode(table_mask, kernel, iterations=1)

    # Find table contours
    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    table_blocks: list[Block] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter: must be a reasonable table size
        if w < 60 or h < 30:
            continue

        # ── Extract cell grid for this table region ──────────────────
        region_horiz = horiz_lines[y : y + h, x : x + w]
        region_vert = vert_lines[y : y + h, x : x + w]

        row_lines = _find_line_positions(region_horiz, axis="h")
        col_lines = _find_line_positions(region_vert, axis="v")

        if len(row_lines) < 2 or len(col_lines) < 2:
            continue  # not a proper table

        # Build cell grid (global coordinates)
        rows = [y + r for r in sorted(row_lines)]
        cols = [x + c for c in sorted(col_lines)]

        # ── Detect merged cells from line structure ───────────────────
        merged_cells = _detect_merged_cells(
            region_horiz, region_vert, row_lines, col_lines,
        )

        # ── Assign OCR text to cells (IoU-based) ──────────────────────
        cells = _build_cell_grid(rows, cols, ocr_blocks, merged_cells)

        # ── Render as Markdown table ──────────────────────────────────
        markdown = _cells_to_markdown(cells)
        if not markdown:
            continue

        table_block = Block(
            type="table",
            text=markdown,
            bbox=(float(x), float(y), float(x + w), float(y + h)),
            confidence=85.0,
            engine="table-extractor",
            metadata={
                "cells": cells,
                "rows": len(rows) - 1,
                "cols": len(cols) - 1,
                "bbox": [x, y, x + w, y + h],
                "merged_cells": merged_cells if merged_cells else {},
            },
        )
        table_blocks.append(table_block)

    # Fallback: gap-based detection for tables without visible lines
    if not table_blocks:
        return _detect_tables_by_gaps(ocr_blocks, page_width)

    return table_blocks


# ═══════════════════════════════════════════════════════════════════════
# Line detection helpers
# ═══════════════════════════════════════════════════════════════════════


def _find_line_positions(
    region: "np.ndarray", axis: str = "h",
) -> list[int]:
    """Find line positions from a binary line mask.

    Args:
        region: Binary mask (non-zero = line pixel).
        axis: "h" for horizontal lines, "v" for vertical.

    Returns:
        Sorted list of pixel offsets for each detected line.
    """
    import numpy as np

    if axis == "h":
        projection = region.sum(axis=1)  # row sums
    else:
        projection = region.sum(axis=0)  # column sums

    if projection.max() == 0:
        return []

    threshold = projection.max() * 0.25
    positions = np.where(projection > threshold)[0]

    if len(positions) == 0:
        return []

    # Cluster nearby positions
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= 5:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [int(np.mean(cl)) for cl in clusters]


# ═══════════════════════════════════════════════════════════════════════
# Merged cell detection
# ═══════════════════════════════════════════════════════════════════════


def _detect_merged_cells(
    region_horiz: "np.ndarray",
    region_vert: "np.ndarray",
    row_lines: list[int],
    col_lines: list[int],
) -> dict[tuple[int, int], dict[str, int]]:
    """Detect cells that span multiple rows or columns.

    Analyses the binary line images at each grid intersection to
    determine whether cell boundaries are missing, indicating a
    merged cell.

    Returns:
        Dict mapping ``(row_index, col_index)`` to
        ``{"rowspan": int, "colspan": int}``.  Only entries for
        cells that actually merge are included.
    """
    import numpy as np

    nr = len(row_lines)
    nc = len(col_lines)
    if nr < 2 or nc < 2:
        return {}

    # Precompute existence of horizontal line segments at each row
    # boundary between each pair of adjacent columns.
    # h_line_exists[row_boundary_idx][col_idx] = bool
    h_line_exists: list[list[bool]] = []
    for ri in range(nr):
        row_segments: list[bool] = []
        r = row_lines[ri]
        # Sample a narrow strip around this row line
        r0 = max(0, r - 1)
        r1 = min(region_horiz.shape[0], r + 2)
        for ci in range(nc - 1):
            c0 = max(0, col_lines[ci] + 1)
            c1 = min(region_horiz.shape[1], col_lines[ci + 1] - 1)
            if c1 <= c0:
                row_segments.append(False)
                continue
            strip = region_horiz[r0:r1, c0:c1]
            row_segments.append(bool(np.any(strip > 0)))
        h_line_exists.append(row_segments)

    # Precompute existence of vertical line segments at each column
    # boundary between each pair of adjacent rows.
    # v_line_exists[col_boundary_idx][row_idx] = bool
    v_line_exists: list[list[bool]] = []
    for ci in range(nc):
        col_segments: list[bool] = []
        c = col_lines[ci]
        c0 = max(0, c - 1)
        c1 = min(region_vert.shape[1], c + 2)
        for ri in range(nr - 1):
            r0 = max(0, row_lines[ri] + 1)
            r1 = min(region_vert.shape[0], row_lines[ri + 1] - 1)
            if r1 <= r0:
                col_segments.append(False)
                continue
            strip = region_vert[r0:r1, c0:c1]
            col_segments.append(bool(np.any(strip > 0)))
        v_line_exists.append(col_segments)

    merged: dict[tuple[int, int], dict[str, int]] = {}
    covered: set[tuple[int, int]] = set()

    for ri in range(nr - 1):
        for ci in range(nc - 1):
            if (ri, ci) in covered:
                continue

            # --- compute rowspan: how many rows this cell extends downward ---
            rowspan = 1
            # We extend downward as long as the bottom-row boundary lacks
            # a vertical-line segment at this column's left edge
            while ri + rowspan < nr - 1:
                # Check: does the vertical line at column `ci` continue
                # through row boundary ri+rowspan?
                if ci < nc and (
                    ci >= len(v_line_exists)
                    or ri + rowspan - 1 >= len(v_line_exists[ci])
                ):
                    break
                if v_line_exists[ci][ri + rowspan - 1]:
                    break  # vertical line exists → separate row
                rowspan += 1

            # --- compute colspan: how many columns this cell extends rightward ---
            colspan = 1
            while ci + colspan < nc - 1:
                # Check: does the horizontal line at row `ri` continue
                # through column boundary ci+colspan?
                if ri < len(h_line_exists) and (
                    ci + colspan - 1 < len(h_line_exists[ri])
                ):
                    if h_line_exists[ri][ci + colspan - 1]:
                        break  # horizontal line exists → separate column
                colspan += 1

            if rowspan > 1 or colspan > 1:
                merged[(ri, ci)] = {"rowspan": rowspan, "colspan": colspan}
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if dr == 0 and dc == 0:
                            continue
                        covered.add((ri + dr, ci + dc))

    return merged


# ═══════════════════════════════════════════════════════════════════════
# Cell grid construction
# ═══════════════════════════════════════════════════════════════════════


def _compute_overlap_area(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> float:
    """Compute the intersection (overlap) area of two axis-aligned rectangles."""
    ox0 = max(ax0, bx0)
    oy0 = max(ay0, by0)
    ox1 = min(ax1, bx1)
    oy1 = min(ay1, by1)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    return float((ox1 - ox0) * (oy1 - oy0))


def _compute_bbox_area(x0: float, y0: float, x1: float, y1: float) -> float:
    return float((x1 - x0) * (y1 - y0))


def _build_cell_grid(
    rows: list[int],
    cols: list[int],
    ocr_blocks: list[Block],
    merged_cells: dict[tuple[int, int], dict[str, int]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Build a 2D cell grid and assign OCR text using overlap-area scoring.

    Instead of the old centre-point heuristic, blocks are assigned to
    the cell that shares the largest overlap area
    (intersection / block-area).  This handles multi-line text blocks
    and tall cells more accurately.

    Empty cells are always represented in the output.

    Returns:
        list[row][col] of dicts: { "text": str, "blocks": [...] }
    """
    merged_cells = merged_cells or {}
    nr = len(rows) - 1
    nc = len(cols) - 1

    # Precompute cell bounding boxes
    cell_bboxes: list[list[tuple[float, float, float, float]]] = []
    for ri in range(nr):
        row_cells: list[tuple[float, float, float, float]] = []
        for ci in range(nc):
            # Apply merged-cell expansion
            span = merged_cells.get((ri, ci))
            if span is not None:
                cell_x0 = float(cols[ci])
                cell_y0 = float(rows[ri])
                cell_x1 = float(cols[ci + span["colspan"]])
                cell_y1 = float(rows[ri + span["rowspan"]])
            else:
                cell_x0 = float(cols[ci])
                cell_y0 = float(rows[ri])
                cell_x1 = float(cols[ci + 1])
                cell_y1 = float(rows[ri + 1])
            row_cells.append((cell_x0, cell_y0, cell_x1, cell_y1))
        cell_bboxes.append(row_cells)

    # ── Assign blocks to best-matching cell via overlap ratio ─────
    # cell_assignments[ri][ci] = list of (block_index, text)
    cell_assignments: list[list[list[tuple[int, str]]]] = [
        [[] for _ in range(nc)] for _ in range(nr)
    ]

    for idx, block in enumerate(ocr_blocks):
        if not block.bbox:
            continue
        bx0, by0, bx1, by1 = block.bbox
        block_area = _compute_bbox_area(bx0, by0, bx1, by1)
        if block_area <= 0:
            continue

        best_cell: tuple[int, int] | None = None
        best_ratio = -1.0

        for ri in range(nr):
            for ci in range(nc):
                cell_box = cell_bboxes[ri][ci]
                overlap = _compute_overlap_area(
                    bx0, by0, bx1, by1,
                    cell_box[0], cell_box[1], cell_box[2], cell_box[3],
                )
                ratio = overlap / block_area  # how much of the block is inside
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = (ri, ci)

        # Assign if at least 10% of the block overlaps the cell
        if best_cell is not None and best_ratio >= 0.10:
            ri, ci = best_cell
            cell_assignments[ri][ci].append((idx, block.text.strip()))

    # ── Build output cell grid (including empty cells) ───────────
    cells: list[list[dict[str, Any]]] = []
    for ri in range(nr):
        row_cells: list[dict[str, Any]] = []
        for ci in range(nc):
            assignments = cell_assignments[ri][ci]
            cell_text = " ".join(t for _, t in assignments) if assignments else ""
            cell_block_indices = [idx for idx, _ in assignments]
            cell_box = cell_bboxes[ri][ci]
            row_cells.append({
                "text": cell_text,
                "blocks": cell_block_indices,
                "bbox": [cell_box[0], cell_box[1], cell_box[2], cell_box[3]],
            })
        cells.append(row_cells)

    return cells


def _cells_to_markdown(cells: list[list[dict]]) -> str:
    """Convert a cell grid to a Markdown table string.

    Empty cells are rendered as blank entries so that column
    alignment is preserved.
    """
    if not cells or not cells[0]:
        return ""

    n_cols = len(cells[0])
    lines: list[str] = []

    # Header row
    header = "| " + " | ".join(
        _escape_pipe(cells[0][c].get("text", "")) for c in range(n_cols)
    ) + " |"
    lines.append(header)

    # Separator
    sep = "|" + "|".join(" --- " for _ in range(n_cols)) + "|"
    lines.append(sep)

    # Data rows
    for row in cells[1:]:
        line = "| " + " | ".join(
            _escape_pipe(row[c].get("text", "") if c < len(row) else "")
            for c in range(n_cols)
        ) + " |"
        lines.append(line)

    return "\n".join(lines)


def _escape_pipe(text: str) -> str:
    return text.replace("|", "\\|").strip()


# ═══════════════════════════════════════════════════════════════════════
# Gap-based fallback (improved)
# ═══════════════════════════════════════════════════════════════════════


def _detect_tables_by_gaps(
    ocr_blocks: list[Block],
    page_width: float | None = None,
) -> list[Block]:
    """Fallback: detect table structure from text column alignments.

    Improved over the original simple row-count heuristic:

    * Column-coordinate clustering – groups blocks whose x-start
      positions cluster together (real table columns), instead of
      just comparing row block counts.
    * Adaptive row grouping – tolerance scales with estimated font
      size instead of a fixed 8 px threshold.
    * Header-row detection – the first row is flagged as a header
      when its formatting (e.g. bold, higher conf) differs.
    * Empty cells are represented explicitly.
    """
    if len(ocr_blocks) < 4:
        return []

    # Collect blocks with bboxes
    positioned = [b for b in ocr_blocks if b.bbox]
    if len(positioned) < 4:
        return []

    # ── Estimate font size for adaptive thresholds ──────────────────
    heights = [b.bbox[3] - b.bbox[1] for b in positioned]
    median_height = sorted(heights)[len(heights) // 2] if heights else 10.0
    row_gap_tolerance = max(4.0, median_height * 0.6)   # ~0.6× font height
    col_merge_tolerance = max(6.0, median_height * 0.5)  # ~0.5× font height

    # ── Group into rows by y-position (adaptive tolerance) ──────────
    positioned.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    rows: list[list[Block]] = [[positioned[0]]]
    for b in positioned[1:]:
        # Compare with the median y-centre of the last row
        last_centres = [
            (bl.bbox[1] + bl.bbox[3]) / 2 for bl in rows[-1]
        ]
        last_median_y = sorted(last_centres)[len(last_centres) // 2]
        this_centre = (b.bbox[1] + b.bbox[3]) / 2
        if abs(this_centre - last_median_y) <= row_gap_tolerance:
            rows[-1].append(b)
        else:
            rows.append([b])

    # Need at least 3 rows to form a table
    if len(rows) < 3:
        return []

    # ── Cluster column coordinates across all blocks ────────────────
    # Collect x-start positions of all blocks
    all_x_starts = sorted({b.bbox[0] for b in positioned})
    col_clusters = _cluster_1d(all_x_starts, tolerance=col_merge_tolerance)
    # Each cluster becomes a column; use mean x-start as column anchor
    col_anchors = sorted(int(sum(c) / len(c)) for c in col_clusters)

    n_cols = len(col_anchors)
    if n_cols < 2:
        return []

    # ── Filter: enough rows with ≥2 aligned blocks ──────────────────
    table_rows: list[list[Block]] = []
    for row in rows:
        if len(row) >= 2:
            table_rows.append(row)

    if len(table_rows) < 3:
        return []

    # ── Build cell grid with explicit column assignment ─────────────
    cells: list[list[dict[str, Any]]] = []
    all_row_bboxes: list[list[float]] = []

    for row in table_rows:
        # Assign each block to the nearest column anchor
        col_texts: dict[int, list[str]] = {ci: [] for ci in range(n_cols)}
        col_bboxes: dict[int, list[float]] = {ci: [] for ci in range(n_cols)}

        for block in row:
            bx0 = block.bbox[0]
            bx_centre = (bx0 + block.bbox[2]) / 2
            # Find nearest anchor
            best_ci = 0
            best_dist = abs(bx_centre - col_anchors[0])
            for ci_idx, anchor in enumerate(col_anchors[1:], start=1):
                dist = abs(bx_centre - anchor)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci_idx
            col_texts[best_ci].append(block.text.strip())
            # Collect block bboxes for per-cell bbox estimation
            col_bboxes[best_ci].extend(list(block.bbox))

        row_cells: list[dict[str, Any]] = []
        for ci in range(n_cols):
            texts = col_texts[ci]
            bboxes = col_bboxes[ci]
            if bboxes:
                cell_x0 = min(bboxes[0::4])
                cell_y0 = min(bboxes[1::4])
                cell_x1 = max(bboxes[2::4])
                cell_y1 = max(bboxes[3::4])
                cell_bbox = [cell_x0, cell_y0, cell_x1, cell_y1]
            else:
                cell_bbox = [float(col_anchors[ci]), 0.0, 0.0, 0.0]
            row_cells.append({
                "text": " ".join(texts),
                "blocks": [],
                "bbox": cell_bbox,
            })
        cells.append(row_cells)

    # ── Detect header row ───────────────────────────────────────────
    if len(cells) >= 2:
        is_header = _is_header_row(cells[0], cells[1:])
        if is_header:
            # Mark header metadata
            for cell in cells[0]:
                cell["_header"] = True

    markdown = _cells_to_markdown(cells)
    if not markdown:
        return []

    # Compute overall bounding box
    all_bboxes = [b.bbox for b in positioned if b.bbox]
    if all_bboxes:
        x0 = min(b[0] for b in all_bboxes)
        y0 = min(b[1] for b in all_bboxes)
        x1 = max(b[2] for b in all_bboxes)
        y1 = max(b[3] for b in all_bboxes)
    else:
        x0 = y0 = x1 = y1 = 0.0

    return [Block(
        type="table",
        text=markdown,
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        confidence=70.0,
        engine="table-gap-detector",
        metadata={
            "cells": cells,
            "rows": len(cells),
            "cols": n_cols,
            "method": "gap_alignment",
            "col_anchors": col_anchors,
        },
    )]


def _cluster_1d(
    values: list[float],
    tolerance: float,
) -> list[list[float]]:
    """Cluster sorted 1-D values with a distance tolerance.

    Returns a list of clusters, each a list of values belonging
    to that cluster.
    """
    if not values:
        return []
    clusters: list[list[float]] = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def _is_header_row(
    header_cells: list[dict[str, Any]],
    data_rows: list[list[dict[str, Any]]],
) -> bool:
    """Heuristic: does the first row look like a table header?

    Compares the first row against the remaining rows on two axes:

    1. **Capitalisation ratio** – header cells tend to be
       title-cased or all-caps more often than data cells.
    2. **Text-length distribution** – headers are usually shorter
       and more uniform than data rows.
    """
    if not data_rows:
        return False

    # Capitalisation heuristic
    def _cap_ratio(text: str) -> float:
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        return sum(1 for c in alpha if c.isupper()) / len(alpha)

    header_caps = [_cap_ratio(c.get("text", "")) for c in header_cells]
    data_caps: list[float] = []
    for row in data_rows:
        for c in row:
            data_caps.append(_cap_ratio(c.get("text", "")))

    avg_header_cap = sum(header_caps) / max(1, len(header_caps))
    avg_data_cap = sum(data_caps) / max(1, len(data_caps))

    # Header is likely when cap ratio is significantly higher
    return avg_header_cap > 0.5 and avg_header_cap > avg_data_cap * 1.5
