from __future__ import annotations

import json
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
    """Render a compact text protocol for downstream text-only LLMs."""
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


def write_outputs(document: Document, output_dir: Path, *, output_format: str) -> dict[str, Path]:
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
        return f"[Formula: page={page_number}, bbox={_bbox(block)}, text={text or 'not detected'}]"
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


def _bbox(block: Block) -> str:
    if not block.bbox:
        return "unknown"
    return ",".join(str(round(value, 2)) for value in block.bbox)
