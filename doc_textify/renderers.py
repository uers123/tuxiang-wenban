from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Block, Document


def render_markdown(document: Document) -> str:
    parts: list[str] = []
    for warning in document.warnings:
        parts.append(f"> [warning] {warning}")
        parts.append("")

    for page in document.pages:
        parts.append(f"# Page {page.number}")
        parts.append("")
        for warning in page.warnings:
            parts.append(f"> [warning] {warning}")
            parts.append("")
        for block in page.ordered_blocks():
            rendered = _render_block_markdown(block, page.number)
            if rendered:
                parts.append(rendered)
                parts.append("")
    return "\n".join(parts).strip() + "\n"


def render_text(document: Document) -> str:
    parts: list[str] = []
    for warning in document.warnings:
        parts.append(f"[warning] {warning}")
        parts.append("")

    for page in document.pages:
        parts.append(f"Page {page.number}")
        parts.append("=" * (len(parts[-1])))
        for warning in page.warnings:
            parts.append(f"[warning] {warning}")
        for block in page.ordered_blocks():
            rendered = _render_block_text(block, page.number)
            if rendered:
                parts.append(rendered)
        parts.append("")
    return "\n\n".join(parts).strip() + "\n"


def render_json(document: Document) -> str:
    return json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n"


def render_llm_text(document: Document) -> str:
    """Render compact text protocol v2 with token-optimized format for LLMs."""
    parts: list[str] = [
        "DOC_TEXTIFY_LLM_PROTOCOL v2",
        "---",
        f"src: {document.source.name}",
        f"pgs: {len(document.pages)}",
    ]
    lang = _detect_document_lang(document)
    parts.append(f"lang: {lang}")
    if document.warnings:
        parts.append("warnings: " + " | ".join(document.warnings))
    parts.append("---")

    for page in document.pages:
        size = ""
        if page.width and page.height:
            size = f" size={round(page.width)}x{round(page.height)}"
        parts.append("")
        parts.append(f"[page {page.number}{size}]")
        if page.warnings:
            parts.append("page_warnings: " + " | ".join(page.warnings))

        blocks = page.ordered_blocks()
        prev_type: str | None = None
        for block in blocks:
            text = " ".join(block.text.split())

            if block.type == "table" and text:
                parts.append(_render_table_deepseek(block))
                prev_type = "table"
                continue

            if block.type == "figure" and block.metadata.get("chart_data"):
                parts.append(_render_chart_data_deepseek(block.metadata["chart_data"]))
                if text:
                    parts.append(f"fig_note→ {text}")
                prev_type = "figure"
                continue

            if block.type == "formula":
                formula_text = text or ""
                if not _formula_is_latex(formula_text):
                    from .formula_ocr import reconstruct_formula_text
                    formula_text = reconstruct_formula_text(formula_text)
                parts.append(f"$${formula_text}$$")
                prev_type = "formula"
                continue

            if not text and block.type not in {"figure", "table", "formula"}:
                continue

            confidence = ""
            if block.confidence is not None:
                confidence = f" conf={round(block.confidence, 1)}"

            # Use ¶ for paragraph breaks between consecutive text-like blocks
            text_like = {"paragraph", "title", "heading", "list", "uncertain", "placeholder", "header", "footer"}
            if prev_type is not None and prev_type in text_like and block.type in text_like:
                parts.append("¶")

            parts.append(f"{block.type}{confidence}→ {text}")
            prev_type = block.type

    return "\n".join(parts).strip() + "\n"


def render_llm_text_v1(document: Document) -> str:
    """Render the legacy v1 compact text protocol (backward compatible)."""
    parts: list[str] = [
        "DOC_TEXTIFY_LLM_PROTOCOL v1",
        f"source: {document.source.name}",
        f"pages: {len(document.pages)}",
    ]
    if document.warnings:
        parts.append("warnings: " + " | ".join(document.warnings))

    for page in document.pages:
        size = ""
        if page.width and page.height:
            size = f" size={round(page.width)}x{round(page.height)}"
        parts.append("")
        parts.append(f"[page {page.number}{size}]")
        if page.warnings:
            parts.append("page_warnings: " + " | ".join(page.warnings))

        for block in page.ordered_blocks():
            text = " ".join(block.text.split())
            if block.type == "figure" and block.metadata.get("chart_data"):
                parts.extend(_render_chart_data_llm(block.metadata["chart_data"]))
                if text:
                    parts.append(f"figure_note: {text}")
                continue
            if not text and block.type not in {"figure", "table", "formula"}:
                continue
            confidence = ""
            if block.confidence is not None:
                confidence = f" conf={round(block.confidence, 1)}"
            parts.append(f"{block.type}{confidence}: {text}")

    return "\n".join(parts).strip() + "\n"


def _detect_document_lang(document: Document) -> str:
    """Detect language(s) in the document. Returns codes like 'en', 'zh', 'zh+en'."""
    has_cjk = False
    has_latin = False
    for page in document.pages:
        for block in page.ordered_blocks():
            for ch in block.text:
                if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af":
                    has_cjk = True
                elif ch.isascii() and ch.isalpha():
                    has_latin = True
                if has_cjk and has_latin:
                    return "zh+en"
    if has_cjk:
        return "zh"
    return "en"


def write_outputs(
    document: Document,
    output_dir: Path,
    *,
    output_format: str,
    rag_ready: bool = False,
    chunk_size: int = 2000,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = document.source.stem
    written: dict[str, Path] = {}

    if output_format in {"md", "both", "all"}:
        path = output_dir / f"{stem}.md"
        path.write_text(render_markdown(document), encoding="utf-8")
        written["markdown"] = path

    if output_format in {"txt", "both", "all"}:
        path = output_dir / f"{stem}.txt"
        path.write_text(render_text(document), encoding="utf-8")
        written["text"] = path

    if output_format in {"llm", "all"}:
        path = output_dir / f"{stem}.llm.txt"
        path.write_text(render_llm_text(document), encoding="utf-8")
        written["llm_text"] = path

    if output_format == "llm-v1":
        path = output_dir / f"{stem}.llm.txt"
        path.write_text(render_llm_text_v1(document), encoding="utf-8")
        written["llm_text"] = path

    if output_format == "deepseek":
        path = output_dir / f"{stem}.ds.txt"
        path.write_text(render_deepseek_llm(document), encoding="utf-8")
        written["deepseek_text"] = path

    if rag_ready:
        path = output_dir / f"{stem}.rag.txt"
        path.write_text(render_rag_ready(document, chunk_size=chunk_size), encoding="utf-8")
        written["rag_ready"] = path

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(render_json(document), encoding="utf-8")
    written["json"] = json_path
    return written


def _render_block_markdown(block: Block, page_number: int) -> str:
    text = block.text.strip()
    if block.type == "title":
        return f"# {text}"
    if block.type == "heading":
        return f"## {text}"
    if block.type == "list":
        return text
    if block.type == "table":
        return text or _placeholder("Table", block, page_number)
    if block.type == "figure":
        # Check for chart_data in metadata
        chart_data = block.metadata.get("chart_data") if block.metadata else None
        if chart_data:
            return _render_chart_data_markdown(chart_data, block, page_number)
        return _placeholder("Figure", block, page_number)
    if block.type == "formula":
        formula_text = text or ""
        if not _formula_is_latex(formula_text):
            from .formula_ocr import reconstruct_formula_text
            formula_text = reconstruct_formula_text(formula_text)
        return f"$${formula_text}$$"
    if block.type == "uncertain":
        return f"[uncertain: {text}]"
    if block.type == "placeholder":
        return f"[placeholder: {text}]"
    if block.type in {"header", "footer"}:
        return f"[{block.type}: {text}]"
    return text


def _render_chart_data_markdown(chart_data: list[dict], block: Block, page_number: int) -> str:
    """Render chart_data as structured Markdown tables."""
    parts = [f"[Figure: page={page_number}, bbox={_bbox(block)}]"]

    # Group by panel_id
    panels: dict[str, dict] = {}
    for item in chart_data:
        pid = item.get("panel_id", "?")
        if pid not in panels:
            panels[pid] = {"intervals": [], "points": []}
        if item.get("type") == "interval":
            panels[pid]["intervals"].append(item)
        elif item.get("type") == "point":
            panels[pid]["points"].append(item)

    for pid in sorted(panels.keys()):
        data = panels[pid]
        parts.append(f"\n### Panel {pid}")

        if data["intervals"]:
            parts.append("\n| Class | Start Depth | End Depth |")
            parts.append("| --- | ---: | ---: |")
            for inv in sorted(data["intervals"], key=lambda x: (x["class"], x["start_depth"])):
                cls = inv.get("class", "?")
                sd = inv.get("start_depth", "?")
                ed = inv.get("end_depth", "?")
                tol = inv.get("depth_tolerance")
                suffix = f" +/- {tol}" if tol else ""
                parts.append(f"| {cls} | {sd}{suffix} | {ed}{suffix} |")

        if data["points"]:
            parts.append("\n| Class | Depth |")
            parts.append("| --- | ---: |")
            for pt in sorted(data["points"], key=lambda x: (x["class"], x["depth"])):
                cls = pt.get("class", "?")
                dp = pt.get("depth", "?")
                candidates = pt.get("class_candidates")
                if candidates and candidates != [cls]:
                    cls = "/".join(str(item) for item in candidates)
                tol = pt.get("depth_tolerance")
                suffix = f" +/- {tol}" if tol else ""
                parts.append(f"| {cls} | {dp}{suffix} |")

    caption = block.text.strip()
    if caption:
        parts.append(f"\n*{caption}*")

    return "\n".join(parts) + "\n"


def _render_chart_data_llm(chart_data: list[dict]) -> list[str]:
    panels: dict[str, dict[str, list[dict]]] = {}
    for item in chart_data:
        pid = str(item.get("panel_id", "?"))
        panels.setdefault(pid, {"intervals": [], "points": []})
        if item.get("type") == "interval":
            panels[pid]["intervals"].append(item)
        elif item.get("type") == "point":
            panels[pid]["points"].append(item)

    lines = ["chart_data:"]
    for pid in sorted(panels):
        data = panels[pid]
        lines.append(f"  panel {pid}:")
        intervals = sorted(data["intervals"], key=lambda x: (x.get("class", 0), x.get("start_depth", 0)))
        if intervals:
            compact = [
                _format_interval_llm(item)
                for item in intervals
            ]
            lines.append("    intervals: " + "; ".join(compact))
        points = sorted(data["points"], key=lambda x: (x.get("class", 0), x.get("depth", 0)))
        if points:
            compact = [_format_point_llm(item) for item in points]
            lines.append("    points: " + "; ".join(compact))
    return lines


def _format_interval_llm(item: dict) -> str:
    tol = item.get("depth_tolerance")
    suffix = f" +/- {tol}" if tol else ""
    return f"class {item.get('class')} depth {item.get('start_depth')}-{item.get('end_depth')}{suffix}"


def _format_point_llm(item: dict) -> str:
    cls = item.get("class")
    candidates = item.get("class_candidates")
    if candidates and candidates != [cls]:
        cls = "/".join(str(candidate) for candidate in candidates)
    tol = item.get("depth_tolerance")
    suffix = f" +/- {tol}" if tol else ""
    return f"class {cls} depth {item.get('depth')}{suffix}"


# ── DeepSeek tokenizer-optimised helpers ────────────────────────────


def _render_table_deepseek(block: Block) -> str:
    """Convert a markdown table block into compact TSV-like format.

    Pipe/space/dash characters that burn tokens are replaced by a single
    tab delimiter per cell — a single token in most tokenizers.
    """
    text = block.text.strip()
    lines = text.split("\n")
    rows: list[list[str]] = []

    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # Skip markdown separator rows (e.g. |---|---|)
            if all(c.replace("-", "").replace(":", "").replace(" ", "") == "" for c in cells):
                continue
            rows.append(cells)

    if not rows:
        return f"[TABLE]{chr(10)}{text}{chr(10)}[/TABLE]"

    cols = max(len(r) for r in rows)
    result = f"[TABLE rows={len(rows)} cols={cols}]{chr(10)}"
    result += "\n".join("\t".join(r) for r in rows)
    result += f"{chr(10)}[/TABLE]"
    return result


def _render_chart_data_deepseek(chart_data: list[dict]) -> str:
    """Render chart data in ultra-compact DeepSeek-optimised notation.

    Produces one line per panel, e.g.:
        [CHART a] red: 0-5m=1, 5-12m=2; blue: pts(1:3.2m, 1:8.7m, 2:15.3m)
    """
    panels: dict[str, dict[str, list[dict]]] = {}
    for item in chart_data:
        pid = str(item.get("panel_id", "?"))
        panels.setdefault(pid, {"intervals": [], "points": []})
        if item.get("type") == "interval":
            panels[pid]["intervals"].append(item)
        elif item.get("type") == "point":
            panels[pid]["points"].append(item)

    lines: list[str] = []
    for pid in sorted(panels):
        data = panels[pid]
        class_parts: list[str] = []

        # Group intervals by class
        classes: dict[str, list[dict]] = {}
        for inv in data["intervals"]:
            cls = str(inv.get("class", "?"))
            classes.setdefault(cls, []).append(inv)

        for cls in sorted(classes):
            invs = sorted(classes[cls], key=lambda x: (x.get("start_depth", 0),))
            segs: list[str] = []
            for inv in invs:
                sd = inv.get("start_depth", "?")
                ed = inv.get("end_depth", "?")
                tol = inv.get("depth_tolerance")
                tol_s = f" ±{tol}" if tol else ""
                segs.append(f"{sd}-{ed}{tol_s}")
            class_parts.append(f"{cls}: {', '.join(segs)}")

        # Points
        if data["points"]:
            pts = sorted(data["points"], key=lambda x: (x.get("depth", 0),))
            pt_strs: list[str] = []
            for pt in pts:
                pcls = pt.get("class", "?")
                candidates = pt.get("class_candidates")
                if candidates and candidates != [pcls]:
                    pcls = "/".join(str(c) for c in candidates)
                dp = pt.get("depth", "?")
                tol = pt.get("depth_tolerance")
                tol_s = f" ±{tol}" if tol else ""
                pt_strs.append(f"{pcls}:{dp}{tol_s}")
            class_parts.append(f"pts({', '.join(pt_strs)})")

        lines.append(f"[CHART {pid}] {'; '.join(class_parts)}")

    return "\n".join(lines)


def _render_block_deepseek(block: Block, page_number: int) -> str:
    """Render a single block in DeepSeek-optimised format."""
    text = " ".join(block.text.split())

    if block.type == "title":
        return f"# {text}"
    if block.type == "heading":
        return f"## {text}"
    if block.type == "table":
        return _render_table_deepseek(block)
    if block.type == "figure":
        chart_data = block.metadata.get("chart_data") if block.metadata else None
        if chart_data:
            result = _render_chart_data_deepseek(chart_data)
            if text:
                result += f"\nfig_note→ {text}"
            return result
        if text:
            return f"[FIG] {text}"
        return ""
    if block.type == "formula":
        formula_text = text or ""
        if not _formula_is_latex(formula_text):
            from .formula_ocr import reconstruct_formula_text
            formula_text = reconstruct_formula_text(formula_text)
        return f"$${formula_text}$$"
    if block.type in {"header", "footer"}:
        return f"[{block.type}] {text}"
    if not text and block.type not in {"figure", "table", "formula"}:
        return ""
    return text


# ── DeepSeek-optimized LLM-text format ──────────────────────────────


def render_deepseek_llm(document: Document) -> str:
    """Render a token-optimized compact text format tuned for DeepSeek models.

    Optimizations over the generic LLM protocol v2:
    - YAML-like compact metadata header (src/pgs/lang)
    - TSV tables saving pipe/dash/space tokens
    - Ultra-compact chart notation: [CHART a] class: range; pts(...)
    - Formula blocks in $$...$$ (native LaTeX support)
    - Pilcrow (¶) paragraph separator — single token in DeepSeek
    - Arrow (→) key-value separator instead of ": "
    """
    parts: list[str] = []

    # YAML-like frontmatter
    parts.append("---")
    parts.append(f"src: {document.source.name}")
    parts.append(f"pgs: {len(document.pages)}")
    lang = _detect_document_lang(document)
    parts.append(f"lang: {lang}")
    parts.append("---")

    for page in document.pages:
        parts.append("")
        parts.append(f"[page {page.number}]")

        blocks = page.ordered_blocks()
        prev_type: str | None = None
        for block in blocks:
            rendered = _render_block_deepseek(block, page.number)
            if not rendered:
                continue

            # Insert ¶ between consecutive text-like blocks
            text_like = {"paragraph", "title", "heading", "list", "uncertain", "placeholder"}
            if prev_type is not None and prev_type in text_like and block.type in text_like:
                parts.append("¶")

            parts.append(rendered)
            prev_type = block.type

    return "\n".join(parts).strip() + "\n"


# ── RAG-ready helpers ──────────────────────────────────────────────


def _get_context_before(blocks: list[Block], index: int) -> str:
    """Return the preceding text sentence as context for a data block."""
    for j in range(index - 1, -1, -1):
        b = blocks[j]
        if b.type in {"paragraph", "list", "heading", "title"}:
            text = " ".join(b.text.split())
            if text:
                return text[:200]
    return ""


def _get_context_after(blocks: list[Block], index: int) -> str:
    """Return the following text sentence as context for a data block."""
    for j in range(index + 1, len(blocks)):
        b = blocks[j]
        if b.type in {"paragraph", "list", "heading", "title"}:
            text = " ".join(b.text.split())
            if text:
                return text[:200]
    return ""


def _fix_heading_hierarchy(heading_stack: list[int], block_type: str) -> str:
    """Ensure consistent # → ## → ### hierarchy, flattening broken levels."""
    if block_type == "title":
        level = 1
    elif block_type == "heading":
        level = min(len(heading_stack) + 2, 3)
    else:
        return ""

    # Trim stack to maintain hierarchy
    trimmed = [l for l in heading_stack if l < level]
    trimmed.append(level)
    heading_stack.clear()
    heading_stack.extend(trimmed)

    return "#" * level


# ── RAG-ready output ───────────────────────────────────────────────


def render_rag_ready(document: Document, chunk_size: int = 2000) -> str:
    """Render RAG-optimized output with semantic chunk markers.

    Features:
    - <!-- CHUNK --> boundary markers at heading / character-limit breaks
    - <!-- src_page=N --> metadata in every chunk
    - [CONTEXT] windows around tables and chart data blocks
    - Consistent # → ## → ### title hierarchy
    - DeepSeek-optimized content format underneath
    """
    parts: list[str] = []

    # YAML frontmatter
    parts.append("---")
    parts.append(f"src: {document.source.name}")
    parts.append(f"pgs: {len(document.pages)}")
    lang = _detect_document_lang(document)
    parts.append(f"lang: {lang}")
    parts.append("---")

    chunk_char_count = 0
    heading_stack: list[int] = []

    for page in document.pages:
        parts.append("")
        parts.append(f"<!-- src_page={page.number} -->")
        parts.append(f"[page {page.number}]")

        blocks = page.ordered_blocks()
        prev_type: str | None = None

        for i, block in enumerate(blocks):
            rendered = _render_block_deepseek(block, page.number)
            if not rendered:
                continue

            rendered_len = len(rendered)

            # Fix title/heading hierarchy
            if block.type in {"title", "heading"}:
                prefix = _fix_heading_hierarchy(heading_stack, block.type)
                # Rewrite rendered line with correct heading level
                text_content = " ".join(block.text.split())
                rendered = f"{prefix} {text_content}"
                rendered_len = len(rendered)

                # Chunk boundary at headings
                if chunk_char_count > 0:
                    parts.append("<!-- CHUNK -->")
                    chunk_char_count = 0

            # Character-limit chunk boundary
            if chunk_char_count > 0 and chunk_char_count + rendered_len > chunk_size:
                parts.append("<!-- CHUNK -->")
                chunk_char_count = 0

            is_data_block = block.type in {"table", "figure"}

            if is_data_block:
                # Context window: preceding sentence
                ctx_before = _get_context_before(blocks, i)
                if ctx_before:
                    ctx_line = f"[CONTEXT] {ctx_before}"
                    parts.append(ctx_line)
                    chunk_char_count += len(ctx_line)

                parts.append(rendered)
                chunk_char_count += rendered_len

                # Context window: following sentence
                ctx_after = _get_context_after(blocks, i)
                if ctx_after:
                    ctx_line = f"[CONTEXT] {ctx_after}"
                    parts.append(ctx_line)
                    chunk_char_count += len(ctx_line)
            else:
                # Insert ¶ between consecutive text-like blocks
                text_like = {"paragraph", "title", "heading", "list", "uncertain", "placeholder"}
                if prev_type is not None and prev_type in text_like and block.type in text_like:
                    parts.append("¶")
                    chunk_char_count += 1

                parts.append(rendered)
                chunk_char_count += rendered_len

            prev_type = block.type

    return "\n".join(parts).strip() + "\n"


def _render_block_text(block: Block, page_number: int) -> str:
    text = block.text.strip()
    if block.type == "figure":
        chart_data = block.metadata.get("chart_data") if block.metadata else None
        if chart_data:
            return "\n".join(_render_chart_data_llm(chart_data))
        return _placeholder("Figure", block, page_number)
    if block.type == "formula":
        return f"[Formula: page={page_number}, bbox={_bbox(block)}, text={text or 'not detected'}]"
    if block.type == "uncertain":
        return f"[uncertain: {text}]"
    if block.type == "placeholder":
        return f"[placeholder: {text}]"
    if block.type in {"header", "footer"}:
        return f"[{block.type}: {text}]"
    return text


def _placeholder(label: str, block: Block, page_number: int) -> str:
    caption = block.text.strip() or "not detected"
    return f"[{label}: page={page_number}, bbox={_bbox(block)}, caption={caption}]"


def _formula_is_latex(text: str) -> bool:
    """Cheap check for already-reconstructed LaTeX formula text."""
    if not text:
        return False
    t = text.strip()
    if t.startswith("[formula]"):
        return False
    return bool(re.search(r"\\[A-Za-z]+|_\{|_", t))


def _bbox(block: Block) -> str:
    if not block.bbox:
        return "unknown"
    return ",".join(str(round(value, 2)) for value in block.bbox)
