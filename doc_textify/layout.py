"""Phase 2: Enhanced layout analysis for doc-textify.

Provides post-OCR layout refinement:
  - Header/footer/page-number detection and filtering
  - Position-based block reclassification (centered title detection)
  - Figure/table caption association
  - Reading order improvement for complex layouts
  - Block type consistency enforcement
"""

from __future__ import annotations

from .models import Block, Page


# ---------------------------------------------------------------------------
# Page-level layout enhancement pipeline
# ---------------------------------------------------------------------------

def enhance_page_layout(page: Page) -> Page:
    """Run the full layout enhancement pipeline on a single page.

    Steps:
        1. Mark headers, footers, and page numbers based on y-position.
        2. Reclassify centered short blocks as titles.
        3. Associate nearby text with figures/tables.
        4. Sort blocks into final reading order.
    """
    if not page.blocks or page.height is None:
        return page

    blocks = list(page.blocks)

    # Step 1: Position-based classification
    blocks = _classify_by_position(blocks, page.height)

    # Step 2: Reclassify centered text as title
    blocks = _reclassify_titles(blocks, page.width)

    # Step 3: Enforce type consistency
    blocks = _enforce_type_consistency(blocks)

    page.blocks = blocks
    return page


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
            elif block.type not in ("figure", "table", "placeholder"):
                block.type = "footer"

        # Header zone
        elif top_ratio < _PAGE_HEADER_ZONE and block.type not in ("figure", "table", "title", "placeholder"):
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


def _looks_like_list_item(text: str) -> bool:
    """Detect list items: numbered, bulleted, or checkbox patterns."""
    import re
    patterns = [
        r"^\d+[.、)]\s+",          # "1.", "1、", "1)"
        r"^[-*+•·]\s+",           # "- ", "* ", "+ ", "• "
        r"^\[\s*x?\s*\]\s+",       # "[ ] " or "[x] "
        r"^[\(（]\d+[\)）]\s+",    # "(1) ", "（1）"
        r"^[A-Z][.、)]\s+",        # "A.", "A、"
    ]
    return any(re.match(p, text) for p in patterns)


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
