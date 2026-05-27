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


def write_outputs(document: Document, output_dir: Path, *, output_format: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = document.source.stem
    written: dict[str, Path] = {}

    if output_format in {"md", "both"}:
        path = output_dir / f"{stem}.md"
        path.write_text(render_markdown(document), encoding="utf-8")
        written["markdown"] = path

    if output_format in {"txt", "both"}:
        path = output_dir / f"{stem}.txt"
        path.write_text(render_text(document), encoding="utf-8")
        written["text"] = path

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


def _render_block_text(block: Block, page_number: int) -> str:
    text = block.text.strip()
    if block.type == "figure":
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
