"""Phase 2: Enhanced layout analysis for doc-textify.

Provides post-OCR layout refinement:
  - Header/footer/page-number detection and filtering
  - Position-based block reclassification (centered title detection)
  - Formula block detection (math-symbol density + visual isolation)
  - Figure/table caption association
  - Reading order improvement for complex layouts
  - Block type consistency enforcement
  - Vertical CJK text detection and flagging
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import Block, Page

if TYPE_CHECKING:
    import PIL.Image

# ── CJK character range for language detection ─────────────────────────
_CJK_RE = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'
    r'\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'
)


def _is_cjk(text: str) -> bool:
    """Check if *text* has significant CJK character ratio (>30%)."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    cjk_count = len(_CJK_RE.findall(stripped))
    return cjk_count / len(stripped) > 0.30


# ---------------------------------------------------------------------------
# Page-level layout enhancement pipeline
# ---------------------------------------------------------------------------

def enhance_page_layout(
    page: Page,
    color_image: "PIL.Image.Image | None" = None,
) -> Page:
    """Run the full layout enhancement pipeline on a single page.

    Steps:
        1. Detect and reclassify formula blocks (FIRST, so display math is
           not pre-empted into header/footer/title by position rules).
        2. Mark headers, footers, and page numbers based on y-position.
        3. Reclassify centered short blocks as titles.
        4. Enforce type consistency.
        5. Detect vertical CJK text.
        6. Detect two-column layouts and reorder blocks into column-major
           reading order (left column top→bottom, then right column).
    """
    if not page.blocks or page.height is None:
        return page

    blocks = list(page.blocks)

    # Step 1: Formula block detection (before position classification so
    # formulas at the top/bottom of a page are not misread as title/header/footer)
    blocks = detect_formula_blocks(page, color_image).blocks

    # Step 2: Position-based classification
    blocks = _classify_by_position(blocks, page.height)

    # Step 3: Reclassify centered text as title
    blocks = _reclassify_titles(blocks, page.width)

    # Step 4: Enforce type consistency
    blocks = _enforce_type_consistency(blocks)

    # Step 5: Vertical CJK text detection
    blocks = _detect_vertical_text_region(blocks)

    # Step 6: Two-column layout detection & reading-order fix
    if page.width:
        splits = _detect_columns(blocks, page.width)
        if splits:
            blocks = _reorder_two_column(blocks, splits[0])
            if blocks:
                blocks[0].metadata["page_layout"] = "two_column"
                blocks[0].metadata["column_split_x"] = round(splits[0], 1)
                # Blocks are now in column-major reading order; keep it.
                page._order_finalized = True

    page.blocks = blocks
    return page


# ---------------------------------------------------------------------------
# Two-column layout detection
# ---------------------------------------------------------------------------

def _detect_columns(blocks: list[Block], page_width: float) -> list[float] | None:
    """Detect column split x-positions from block x-centers.

    Returns a list of split x positions (e.g. ``[page_width / 2]`` for a
    two-column page) when the page's blocks clearly cluster into distinct
    columns, otherwise ``None`` (single-column layout — the common case).

    Algorithm: sort all block x-centers and find the largest gap between
    consecutive centers.  The layout is treated as multi-column only when
    that gap exceeds 15% of the page width AND at least 3 blocks sit on
    each side of the split.  This keeps single-column documents untouched.

    Args:
        blocks: The page's blocks.
        page_width: Page width in the same units as block bboxes.

    Returns:
        ``None`` for single-column layouts, otherwise a list of split
        x positions (one entry per detected column boundary).
    """
    if not page_width or page_width <= 0:
        return None
    centers = []
    for b in blocks:
        if b.bbox is None:
            continue
        width = b.bbox[2] - b.bbox[0]
        # Full-width blocks (figures, captions, tables) and running
        # headers/footers are not column members — ignore them when
        # searching for the column gutter.
        if b.type in ("header", "footer") or b.metadata.get("role") == "page_number":
            continue
        if width >= page_width * 0.5:
            continue
        centers.append((b.bbox[0] + b.bbox[2]) / 2.0)
    centers.sort()
    if len(centers) < 6:
        return None

    best_gap = 0.0
    best_split: float | None = None
    for i in range(len(centers) - 1):
        gap = centers[i + 1] - centers[i]
        if gap > best_gap:
            best_gap = gap
            best_split = (centers[i] + centers[i + 1]) / 2.0

    # Conservative: the gap must be a real column gutter (>15% of width)
    if best_split is None or best_gap <= page_width * 0.15:
        return None
    left = sum(1 for c in centers if c < best_split)
    right = len(centers) - left
    if left < 3 or right < 3:
        return None
    return [best_split]


def _reorder_two_column(blocks: list[Block], split_x: float) -> list[Block]:
    """Reorder *blocks* into two-column reading order.

    All blocks whose center is left of *split_x* come first (top→bottom),
    then all blocks at or right of *split_x* (top→bottom).  Blocks without
    a bbox are appended last.  Block-level ordering within each column is
    preserved by (y, x).
    """
    left = sorted(
        (
            b for b in blocks
            if b.bbox is not None and (b.bbox[0] + b.bbox[2]) / 2.0 < split_x
        ),
        key=lambda b: (b.bbox[1], b.bbox[0]),
    )
    right = sorted(
        (
            b for b in blocks
            if b.bbox is not None and (b.bbox[0] + b.bbox[2]) / 2.0 >= split_x
        ),
        key=lambda b: (b.bbox[1], b.bbox[0]),
    )
    rest = [b for b in blocks if b.bbox is None]
    return left + right + rest


# ---------------------------------------------------------------------------
# Position-based classification
# ---------------------------------------------------------------------------

_PAGE_HEADER_ZONE = 0.06    # top 6% of page → possible header
_PAGE_FOOTER_ZONE = 0.94   # below 94% → possible footer
_NUMBER_WIDTH_RATIO = 0.08 # page numbers are usually narrow (<8% of page width)


def _classify_by_position(blocks: list[Block], page_height: float) -> list[Block]:
    """Classify blocks as header/footer/body based on their y-position."""
    for block in blocks:
        if not block.bbox:
            continue
        y0, y1 = block.bbox[1], block.bbox[3]
        top_ratio = y0 / page_height
        bottom_ratio = y1 / page_height

        # Page number detection: at bottom, short, numeric
        if bottom_ratio > _PAGE_FOOTER_ZONE:
            width_ratio = (block.bbox[2] - block.bbox[0]) / page_height
            if width_ratio < _NUMBER_WIDTH_RATIO and _looks_numeric(block.text):
                block.type = "footer"
                block.metadata["role"] = "page_number"
            elif block.type not in ("figure", "table", "formula", "placeholder"):
                block.type = "footer"

        # Header zone
        elif top_ratio < _PAGE_HEADER_ZONE and block.type not in ("figure", "table", "title", "formula", "placeholder"):
            # Only mark as header if it looks like a running header
            # (short text, not a title)
            if len(block.text.strip()) < 60 and block.type != "heading":
                block.type = "header"

    return blocks


def _looks_numeric(text: str) -> bool:
    """Check if text is primarily a number (page number)."""
    stripped = text.strip().replace("-", "").replace("—", "")
    if not stripped:
        return False
    digit_count = sum(1 for c in stripped if c.isdigit())
    return digit_count > 0 and digit_count >= len(stripped) * 0.5


# ---------------------------------------------------------------------------
# Title detection from position
# ---------------------------------------------------------------------------

def _reclassify_titles(blocks: list[Block], page_width: float | None) -> list[Block]:
    """Reclassify blocks as 'title' if they are centered at the top of the page."""
    if not page_width or page_width <= 0:
        return blocks

    for block in blocks:
        if not block.bbox or block.type not in ("heading", "paragraph"):
            continue
        x0, y0, x1 = block.bbox[0], block.bbox[1], block.bbox[2]
        block_center = (x0 + x1) / 2
        page_center = page_width / 2
        center_offset = abs(block_center - page_center) / page_width

        # Centered (within 15% of page center) AND near top (within first 20%)
        if center_offset < 0.15 and y0 < page_width * 0.20:
            if block.type == "heading":
                block.type = "title"
            elif len(block.text.strip()) <= 80:
                block.type = "title"

    return blocks


# ---------------------------------------------------------------------------
# Type consistency enforcement
# ---------------------------------------------------------------------------

def _enforce_type_consistency(blocks: list[Block]) -> list[Block]:
    """Enforce consistent block types based on content patterns."""
    for block in blocks:
        text = block.text.strip()

        # List detection
        if block.type in ("paragraph", "heading") and _looks_like_list_item(text):
            block.type = "list"

        # Uncertain text with high confidence → promote to paragraph
        if block.type == "uncertain" and block.confidence and block.confidence >= 60:
            block.type = "paragraph"

    return blocks


# ---------------------------------------------------------------------------
# Vertical CJK text detection
# ---------------------------------------------------------------------------

def _detect_vertical_text_region(blocks: list[Block]) -> list[Block]:
    """Detect and flag blocks that likely contain vertical CJK text.

    Vertical text typically appears in tall, narrow blocks (height >> width)
    where the content is predominantly CJK characters.  These blocks are
    marked with ``metadata["text_direction"] = "vertical"`` and a warning
    suggesting re-OCR with ``--psm 5`` for better results.

    Args:
        blocks: The list of blocks to scan.

    Returns:
        The same list (mutated in-place for detected blocks).
    """
    for block in blocks:
        if not block.bbox:
            continue
        x0, y0, x1, y1 = block.bbox
        width = x1 - x0
        height = y1 - y0

        # Skip blocks that are too small to meaningfully analyse
        if width <= 0 or height <= 0 or (width < 5 and height < 5):
            continue

        # Tall-and-narrow: height significantly greater than width
        # (vertical text blocks are often ≥2× taller than wide)
        if height <= width * 1.5:
            continue

        text = block.text.strip()
        if not text or len(text) < 2:
            continue

        # Check for CJK character dominance in this candidate
        cjk_chars = len(_CJK_RE.findall(text))
        total = len(text)
        if total > 0 and cjk_chars / total > 0.40:
            block.metadata["text_direction"] = "vertical"
            block.metadata["vertical_text_warning"] = (
                "Vertical CJK text detected (block is %.1fx taller than wide). "
                "Consider re-OCR with --psm 5 for better vertical text results. "
                "Current OCR output may have incorrect character ordering."
                % (height / max(width, 1))
            )

    return blocks


def _looks_like_list_item(text: str) -> bool:
    """Detect list items: numbered, bulleted, or checkbox patterns."""
    patterns = [
        r"^\d+[.、)]\s+",          # "1.", "1、", "1)"
        r"^[-*+•·]\s+",           # "- ", "* ", "+ ", "• "
        r"^\[\s*x?\s*\]\s+",       # "[ ] " or "[x] "
        r"^[\(（]\d+[\)）]\s+",    # "(1) ", "（1）"
        r"^[A-Z][.、)]\s+",        # "A.", "A、"
    ]
    return any(re.match(p, text) for p in patterns)


# ── Formula block detection ────────────────────────────────────────────

# Unicode codepoint sets for math detection
_GREEK_LOWER = frozenset(range(0x03B1, 0x03C9 + 1))    # α … ω
_GREEK_UPPER = frozenset(range(0x0391, 0x03A9 + 1))    # Α … Ω
_MATH_OPS_CODEPOINTS: frozenset[int] = frozenset(
    ord(c) for c in (
        "∑∏∫√∂∇∈∉∋∌∝∞∟∠∥∶∷∼≈≅≠≡≤≥≪≫≺≻≼≽⊂⊃⊆⊇⊕⊖⊗⊘±×÷"
        "→⇒⇐⇔↔↦↩↪↶↷⇀⇁⇉⇊⇋⇌⇍⇎⇏⇐⇑⇒⇓⇔⇕⇖⇗⇘⇙⇚⇛"
    )
)
_SUBSCRIPTS = frozenset(
    ord(c) for c in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ"
)
_SUPERSCRIPTS = frozenset(
    ord(c) for c in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"
)
_MATH_FUNC_RE = re.compile(
    r"\b(sin|cos|tan|cot|sec|csc|log|ln|lg|exp|lim|max|min|det|arg|dim|gcd|inf|sup|Pr|argmax|argmin|sgn|mod|asin|acos|atan)\b",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(r".{2,}\s*=\s*.+")
# Math bracketed expressions: f(x), g(y, z), (a + b)^2, etc.
_BRACKETED_EXPR_RE = re.compile(r"[a-zA-Z]\s*\([^)]+\)")
_FRACTION_RE = re.compile(r"\d+\s*/\s*\d+")


def _count_math_chars(text: str) -> int:
    """Count math-related Unicode characters in *text*."""
    count = 0
    for ch in text:
        cp = ord(ch)
        if (
            cp in _GREEK_LOWER
            or cp in _GREEK_UPPER
            or cp in _MATH_OPS_CODEPOINTS
            or cp in _SUBSCRIPTS
            or cp in _SUPERSCRIPTS
        ):
            count += 1
    return count


def _math_text_score(text: str) -> float:
    """Return a 0..1 score indicating how likely *text* is a math formula.

    Considers:
      - Unicode math symbols (Greek, operators, sub/superscript)
      - ASCII math density (^, _, =, +, -, *, /, <, >, |, etc.)
      - Math function keywords (sin, cos, log, …)
      - Equation-like pattern (contains = with non-trivial LHS/RHS)
    """
    stripped = text.strip()
    if not stripped or len(stripped) < 3:
        return 0.0

    total = len(stripped)
    score = 0.0

    # 1) Unicode math symbols (strong signal — 3× weight)
    math_chars = _count_math_chars(stripped)
    score += min(math_chars / max(total, 1), 1.0) * 0.45

    # 2) ASCII math operators in isolation (^, _, ±, ×, ÷, etc.)
    ascii_math = sum(
        1 for c in stripped
        if c in "^_±×÷∞∂∇∫∑√∏¬∧∨⊕⊗"
    )
    score += min(ascii_math / max(total, 1), 0.3) * 0.20

    # 3) ASCII equation delimiters (dense = / + - pattern)
    eq_density = sum(
        1 for c in stripped
        if c in "=+-*/<>|~"
    ) / max(total, 1)
    if 0.08 <= eq_density <= 0.40:
        score += 0.15

    # 4) Math function keywords
    if _MATH_FUNC_RE.search(stripped):
        score += 0.10

    # 5) Equation pattern
    if _EQUATION_RE.search(stripped):
        score += 0.10

    # 6) Bracketed math expressions: f(x), g(y, z), etc.
    if _BRACKETED_EXPR_RE.search(stripped):
        score += 0.10

    # 7) Equation with variable tokens: "qt = qc + (1 - a) u2".
    # Measurement lines ("qc = 0.9 MPa") are excluded earlier by the
    # unit-token guard, so an "=" here is strong display-math evidence.
    if re.search(r"[=≈]", stripped) and re.search(
        r"\b[A-Za-z]{1,3}\d\b|\b[A-Z][a-z]{1,2}\b", stripped
    ):
        score += 0.20

    return min(score, 1.0)


# Unit tokens used to keep measurement lines ("qc = 0.9 MPa") out of
# formula-fragment classification — they are body text, not display math.
_UNIT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(MPa|kPa|GPa|Pa|kN|kN/m2|kN/m²|mm|cm|km|m/s|km/h|bars?|psi|°C|°F|‰|%|m|kg|g|t)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)


def _is_prose_like(text: str) -> bool:
    """Return True when *text* reads like running prose.

    Prose with an inline equation ("friction sleeve (i.e., b = 1.0).",
    "... (Au = u2 - u0) have high positive ...") is body text, not a
    display formula.  2+ real lowercase words (excluding math keywords
    such as "log"/"sin") are a reliable prose signal.
    """
    lower_words = [
        w for w in re.findall(r"[A-Za-z]+", text)
        if len(w) >= 3 and w.islower() and not _MATH_FUNC_RE.fullmatch(w)
    ]
    return len(lower_words) >= 2


def _formula_fragment_score(text: str) -> float:
    """Score short OCR formula fragments on a 0..1 scale.

    Scanned formulas often OCR into tiny fragments ("AREA =a", "g,",
    "u2", "Ql", "4]") that carry very little math-symbol density, so
    :func:`_math_text_score` misses them.  This scorer looks for
    *structural* math signals instead:

      - equation signs ("=", "≈") — a strong signal
      - single-letter fragments with trailing punctuation ("g,", "q.")
      - math operators mixed with letters/digits ("×", "÷", "±", "−", "%")
      - variable-like tokens ("u2", "Qc", "Ic", "qt", "fs")
      - fraction / bracket-junk patterns from fragmented OCR

    Only short text (< 40 chars) is considered.  Measurement lines with
    digits + units ("qc = 0.9 MPa, fs = 72 kPa") are rejected outright.

    Returns ``0.0`` when the text is not a convincing formula fragment.
    """
    stripped = text.strip()
    if not stripped or len(stripped) >= 40:
        return 0.0

    # Measurement lines are body text, never formulas
    if re.search(r"\d", stripped) and _UNIT_TOKEN_RE.search(stripped):
        return 0.0

    # Prose guard: 2+ real lowercase words (excluding math keywords such as
    # "log"/"sin") means this is running text with an inline equation
    # ("friction sleeve (i.e., b = 1.0)."), not a display formula.
    if _is_prose_like(stripped):
        return 0.0

    signals = 0
    strong = 0

    # 1) Equation sign (strong): "AREA =a", "q = 0.9"
    if re.search(r"[=≈]", stripped):
        strong += 1
        signals += 1

    # 2) Single-letter fragment with trailing punctuation: "g,", "q.", "u,"
    if re.fullmatch(r"[A-Za-z][,.;]?", stripped):
        signals += 2

    # 3) Math operators mixed with letters/digits
    if re.search(r"[×÷±−]", stripped):
        signals += 2
    if re.search(r"[=/%~]", stripped) and re.search(r"[A-Za-z0-9]", stripped):
        signals += 1

    # 4) Variable-like tokens: "u2", "qt", "Qc", "Ic", "Bq", "Fs", "Ql"
    if re.search(r"\b[A-Za-z]{1,3}\d\b", stripped):
        signals += 1
    if re.search(r"\b[A-Z]{1,2}[a-z]{1,2}\b", stripped):
        signals += 1
    if re.search(r"\b[A-Z]{2,3}\b", stripped):
        signals += 1

    # 5) Fraction-like "a/b" or "3/4"
    if re.search(r"\d+\s*/\s*\d+", stripped):
        signals += 1

    # 6) Bracket junk from fragmented OCR: "4]", "a)", "g,"
    if re.search(r"(^|[\s(])\d*[a-zA-Z]?[\]\)][\s,;]?$", stripped):
        signals += 1

    if strong >= 1 and signals >= 1:
        return min(0.6 + signals * 0.1, 1.0)
    if signals >= 2:
        return min(0.5 + signals * 0.1, 1.0)
    return 0.0


def _mixed_scripts(text: str) -> bool:
    """Return True when *text* mixes Latin, digits, and math in proximity."""
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    has_digit = bool(re.search(r"\d", text))
    has_greek = any(
        ord(c) in _GREEK_LOWER or ord(c) in _GREEK_UPPER for c in text
    )
    has_math = any(
        ord(c) in _MATH_OPS_CODEPOINTS
        or c in "^_=+-*/<>|~±×÷"
        for c in text
    )
    # Need at least two different script categories
    categories = sum([has_latin, has_digit, has_greek, has_math])
    return categories >= 2


def detect_formula_blocks(
    page: Page,
    color_image: "PIL.Image.Image | None" = None,
) -> Page:
    """Detect and reclassify formula blocks on a single page.

    Scans ``paragraph``, ``uncertain``, and ``heading`` blocks for
    high density of math symbols (Greek letters, operators, subscripts/
    superscripts, fraction-like patterns) combined with visual isolation
    cues (larger-than-average vertical gaps above and below).

    Matching blocks are reclassified to ``type="formula"`` in-place.

    Args:
        page: The page whose blocks should be scanned.
        color_image: Optional PIL image (reserved for future visual
            analysis; currently unused).

    Returns:
        The same *page* instance (mutated in-place).
    """
    if not page.blocks or page.height is None or page.height <= 0:
        return page
    page.blocks = _detect_formula_blocks(list(page.blocks), page.height)
    return page


def _detect_formula_blocks(
    blocks: list[Block],
    page_height: float,
) -> list[Block]:
    """Reclassify candidate blocks as ``"formula"`` when they show strong
    evidence of being display math.

    Detection uses two complementary signals:

    1. **Text-content signal** (``_math_text_score``): high density of
       math symbols, Greek letters, equation patterns.
    2. **Layout-isolation signal**: blocks that are vertically separated
       from their neighbours by larger-than-average gaps, which is
       typical for display math.

    A block is reclassified when the combined score exceeds a threshold.
    """
    if not blocks or page_height is None or page_height <= 0:
        return blocks

    # Only consider paragraph/uncertain/heading blocks (not already
    # classified as table, figure, formula, header, footer, placeholder).
    candidates = [
        (i, b)
        for i, b in enumerate(blocks)
        if b.type in ("paragraph", "uncertain", "heading")
        and b.bbox is not None
    ]
    if not candidates:
        return blocks

    # ── Compute average vertical gap between consecutive blocks ──
    sorted_by_y = sorted(blocks, key=lambda b: b.bbox[1] if b.bbox else 0.0)
    gaps: list[float] = []
    for a, b in zip(sorted_by_y, sorted_by_y[1:]):
        if a.bbox and b.bbox:
            gap = b.bbox[1] - a.bbox[3]
            if gap > 0:
                gaps.append(gap)
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

    # ── Score each candidate ──
    for idx, block in candidates:
        assert block.bbox is not None
        text = block.text.strip()
        if not text:
            continue

        # Text-based math score (0..1)
        math_score = _math_text_score(text)

        # Body-text measurement lines ("qc = 0.9 MPa, fs = 72 kPa") are
        # never display formulas, regardless of their math-ish content.
        if re.search(r"\d", text) and _UNIT_TOKEN_RE.search(text):
            continue

        # Prose with an inline equation ("... (Au = u2 - u0) ...") is
        # body text, not a display formula.
        if _is_prose_like(text):
            continue

        # Short formula-fragment score (0..1) — catches scanned formulas
        # that OCR into tiny symbol-heavy fragments
        frag_score = _formula_fragment_score(text)

        # Layout isolation score: how much larger is the vertical gap
        # above AND below this block compared to the average?
        above_gap = 0.0
        below_gap = 0.0
        y0, y1 = block.bbox[1], block.bbox[3]

        # Find closest block above
        closest_above_y = 0.0
        for b in reversed(sorted_by_y):
            if b.bbox and b is not block and b.bbox[3] < y0:
                closest_above_y = b.bbox[3]
                break
        above_gap = y0 - closest_above_y if closest_above_y > 0 else y0

        # Find closest block below
        closest_below_y = page_height
        for b in sorted_by_y:
            if b.bbox and b is not block and b.bbox[1] > y1:
                closest_below_y = b.bbox[1]
                break
        below_gap = closest_below_y - y1 if closest_below_y < page_height else (page_height - y1)

        iso_score = 0.0
        if avg_gap > 0:
            # Reward gaps that are 1.5× the average or larger
            above_ratio = min(above_gap / avg_gap, 3.0)
            below_ratio = min(below_gap / avg_gap, 3.0)
            iso_score = (above_ratio + below_ratio) / 6.0  # 0..1 range

        # Threshold: strong fragment evidence, OR strong text signal, OR
        # moderate text + isolation
        if (
            frag_score >= 0.6
            or math_score >= 0.45
            or (math_score >= 0.25 and iso_score >= 0.30)
        ):
            blocks[idx].type = "formula"
            blocks[idx].metadata["formula_detection"] = {
                "math_score": round(math_score, 3),
                "isolation_score": round(iso_score, 3),
                "fragment_score": round(frag_score, 3),
            }

    return blocks


# ---------------------------------------------------------------------------
# Page metadata extraction helpers
# ---------------------------------------------------------------------------

def extract_page_metadata(page: Page) -> dict:
    """Extract summary metadata from a page.

    Returns dict with:
      - block_count: total blocks
      - has_tables: bool
      - has_figures: bool
      - has_headers: bool
      - has_footers: bool
      - text_length: total character count
    """
    return {
        "block_count": len(page.blocks),
        "has_tables": any(b.type == "table" for b in page.blocks),
        "has_figures": any(b.type == "figure" for b in page.blocks),
        "has_headers": any(b.type == "header" for b in page.blocks),
        "has_footers": any(b.type == "footer" for b in page.blocks),
        "text_length": sum(len(b.text) for b in page.blocks),
    }
