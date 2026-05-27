from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Block, Document, Page


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def extract_document(
    source: Path,
    *,
    lang: str = "eng",
    force_ocr: bool = False,
    min_confidence: float = 45.0,
) -> Document:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source, force_ocr=force_ocr)
    if suffix in IMAGE_EXTENSIONS:
        return extract_image(source, lang=lang, min_confidence=min_confidence)

    raise ValueError(f"Unsupported input type: {source.suffix}")


def extract_pdf(source: Path, *, force_ocr: bool = False) -> Document:
    document = Document(source=source, metadata={"input_type": "pdf"})
    if force_ocr:
        page = Page(number=1)
        page.blocks.append(
            Block(
                type="placeholder",
                text="OCR for scanned PDFs requires a PDF page renderer backend.",
                engine="pdf-placeholder",
                metadata={
                    "recommended_backends": ["PyMuPDF", "Poppler/pdftoppm", "Docling", "MinerU"],
                },
            )
        )
        page.warnings.append("PDF OCR requested, but no renderer backend is bundled in this MVP.")
        document.pages.append(page)
        document.warnings.append("Scanned PDF OCR was not performed.")
        return document

    try:
        return _extract_native_pdf_with_pypdf(source)
    except ImportError:
        document.warnings.append("pypdf is not installed; native PDF text extraction is unavailable.")
    except Exception as exc:  # noqa: BLE001 - preserve failure in sidecar for diagnostics
        document.warnings.append(f"Native PDF extraction failed: {exc}")

    page = Page(number=1)
    page.blocks.append(
        Block(
            type="placeholder",
            text="No native PDF text could be extracted. Install pypdf for digital PDFs or add an OCR backend for scanned PDFs.",
            engine="pdf-placeholder",
        )
    )
    document.pages.append(page)
    return document


def extract_image(source: Path, *, lang: str = "eng", min_confidence: float = 45.0) -> Document:
    document = Document(source=source, metadata={"input_type": "image", "ocr_language": lang})
    page = Page(number=1)

    dimensions = _image_dimensions(source)
    if dimensions:
        page.width, page.height = dimensions

    tesseract = shutil.which("tesseract")
    if not tesseract:
        page.blocks.append(
            Block(
                type="figure",
                text="Image OCR was not performed because Tesseract is not installed or not on PATH.",
                bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)),
                engine="image-placeholder",
                metadata={"source_image": str(source)},
            )
        )
        page.warnings.append("Install Tesseract to OCR images without a vision LLM.")
        document.pages.append(page)
        document.warnings.append("Image OCR backend unavailable.")
        return document

    processed = _preprocess_image(source)
    try:
        blocks = _ocr_tesseract_tsv(processed, tesseract=tesseract, lang=lang, min_confidence=min_confidence)
    finally:
        if processed != source:
            processed.unlink(missing_ok=True)

    if blocks:
        page.blocks.extend(blocks)
    else:
        page.blocks.append(
            Block(
                type="uncertain",
                text="No reliable text was detected in this image.",
                bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)),
                confidence=0.0,
                engine="tesseract",
            )
        )
        page.warnings.append("OCR completed but returned no reliable text.")

    document.pages.append(page)
    return document


def _extract_native_pdf_with_pypdf(source: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(source))
    document = Document(source=source, metadata={"input_type": "pdf", "engine": "pypdf"})

    for index, pdf_page in enumerate(reader.pages, start=1):
        width = height = None
        try:
            box = pdf_page.mediabox
            width = float(box.width)
            height = float(box.height)
        except Exception:
            pass

        page = Page(number=index, width=width, height=height)
        blocks = _extract_positioned_pdf_blocks(pdf_page, page_height=height)
        if not blocks:
            text = pdf_page.extract_text() or ""
            blocks = _text_to_blocks(text, engine="pypdf")
        if blocks:
            page.blocks.extend(blocks)
        else:
            page.blocks.append(
                Block(
                    type="placeholder",
                    text="No native text layer found on this page. It may be scanned and require OCR.",
                    engine="pypdf",
                )
            )
            page.warnings.append("No native text extracted from this page.")
        page.blocks.extend(_pdf_image_placeholders(pdf_page, index))
        document.pages.append(page)

    if not document.pages:
        document.warnings.append("PDF contains no pages.")
    return document


def _extract_positioned_pdf_blocks(pdf_page: object, *, page_height: float | None) -> list[Block]:
    spans: list[tuple[float, float, float, str]] = []

    def visitor(text: str, _cm: object, tm: object, _font: object, font_size: float) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        try:
            x = float(tm[4])
            raw_y = float(tm[5])
        except Exception:
            return
        y = (page_height - raw_y) if page_height else raw_y
        spans.append((x, y, float(font_size or 0), cleaned))

    try:
        pdf_page.extract_text(visitor_text=visitor)
    except TypeError:
        return []
    except Exception:
        return []

    if not spans:
        return []

    lines: list[list[tuple[float, float, float, str]]] = []
    for span in sorted(spans, key=lambda item: (round(item[1] / 4) * 4, item[0])):
        if not lines:
            lines.append([span])
            continue
        previous_y = lines[-1][-1][1]
        if abs(span[1] - previous_y) <= 4:
            lines[-1].append(span)
        else:
            lines.append([span])

    blocks: list[Block] = []
    for line in lines:
        text = " ".join(item[3] for item in sorted(line, key=lambda item: item[0])).strip()
        if not text:
            continue
        x0 = min(item[0] for item in line)
        y0 = min(item[1] for item in line)
        font_size = max(item[2] for item in line)
        approx_width = max(len(text) * max(font_size, 8) * 0.55, 1)
        bbox = (x0, y0, x0 + approx_width, y0 + max(font_size, 8))
        block_type = "heading" if font_size >= 14 and len(text) <= 120 else _classify_text_block(text, 1)
        blocks.append(Block(type=block_type, text=text, bbox=bbox, confidence=100.0, engine="pypdf"))

    return _merge_text_lines(blocks)


def _pdf_image_placeholders(pdf_page: object, page_number: int) -> list[Block]:
    images = []
    try:
        image_items = getattr(pdf_page, "images", [])
        for idx, _image in enumerate(image_items, start=1):
            images.append(
                Block(
                    type="figure",
                    text=f"Embedded image {idx}; visual content not described without OCR/caption extraction.",
                    engine="pypdf",
                    metadata={"page": page_number, "image_index": idx},
                )
            )
    except Exception:
        return []
    return images


def _text_to_blocks(text: str, *, engine: str) -> list[Block]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks: list[Block] = []
    paragraphs = re.split(r"\n\s*\n+", normalized)
    for raw in paragraphs:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        if _looks_like_table(lines):
            blocks.append(Block(type="table", text=_lines_to_markdown_table(lines), confidence=90.0, engine=engine))
            continue
        chunk = " ".join(lines)
        block_type = _classify_text_block(chunk, len(lines))
        blocks.append(Block(type=block_type, text=chunk, confidence=100.0, engine=engine))
    return blocks


def _classify_text_block(text: str, line_count: int) -> str:
    if line_count == 1 and len(text) <= 90 and not text.endswith((".", "。", "?", "？", "!", "！", ":", "：")):
        return "heading"
    if re.match(r"^(\d+\.|[-*+])\s+", text):
        return "list"
    return "paragraph"


def _looks_like_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    split_lines = [_split_table_line(line) for line in lines]
    usable = [cells for cells in split_lines if len(cells) >= 2]
    if len(usable) < 2:
        return False
    widths = {len(cells) for cells in usable}
    return len(widths) <= 2


def _split_table_line(line: str) -> list[str]:
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t") if cell.strip()]
    return [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]


def _lines_to_markdown_table(lines: list[str]) -> str:
    rows = [_split_table_line(line) for line in lines]
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    header = rows[0]
    separator = ["---"] * column_count
    body = rows[1:]
    table_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |" for row in table_rows)


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _merge_text_lines(blocks: list[Block]) -> list[Block]:
    if not blocks:
        return []

    merged: list[Block] = []
    current = blocks[0]
    for block in blocks[1:]:
        if _same_text_paragraph(current, block):
            current = _merge_blocks(current, block)
        else:
            merged.append(current)
            current = block
    merged.append(current)
    return merged


def _same_text_paragraph(left: Block, right: Block) -> bool:
    if left.type != "paragraph" or right.type != "paragraph":
        return False
    if not left.bbox or not right.bbox:
        return False
    vertical_gap = right.bbox[1] - left.bbox[3]
    return 0 <= vertical_gap <= 12 and abs(left.bbox[0] - right.bbox[0]) <= 16


def _image_dimensions(source: Path) -> tuple[float, float] | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(source) as image:
            return float(image.width), float(image.height)
    except Exception:
        return None


def _preprocess_image(source: Path) -> Path:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return source

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.point(lambda px: 255 if px > 180 else 0, mode="1")

        temp = tempfile.NamedTemporaryFile(prefix="doc_textify_", suffix=".png", delete=False)
        temp.close()
        output = Path(temp.name)
        image.save(output)
        return output


def _ocr_tesseract_tsv(
    image_path: Path,
    *,
    tesseract: str,
    lang: str,
    min_confidence: float,
) -> list[Block]:
    command = [tesseract, str(image_path), "stdout", "-l", lang, "--psm", "6", "tsv"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Tesseract OCR failed.")

    rows = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    line_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        conf = _safe_float(row.get("conf"), default=-1.0)
        if conf < min_confidence:
            continue
        key = (row.get("block_num", "0"), row.get("par_num", "0"), row.get("line_num", "0"))
        line_groups.setdefault(key, []).append(row)

    blocks: list[Block] = []
    for rows_for_line in line_groups.values():
        words = [(row.get("text") or "").strip() for row in rows_for_line]
        text = " ".join(word for word in words if word)
        if not text:
            continue
        lefts = [_safe_float(row.get("left")) for row in rows_for_line]
        tops = [_safe_float(row.get("top")) for row in rows_for_line]
        rights = [_safe_float(row.get("left")) + _safe_float(row.get("width")) for row in rows_for_line]
        bottoms = [_safe_float(row.get("top")) + _safe_float(row.get("height")) for row in rows_for_line]
        confs = [_safe_float(row.get("conf"), default=0.0) for row in rows_for_line]
        confidence = sum(confs) / len(confs) if confs else None
        block_type = "uncertain" if confidence is not None and confidence < min_confidence + 10 else "paragraph"
        blocks.append(
            Block(
                type=block_type,
                text=text,
                bbox=(min(lefts), min(tops), max(rights), max(bottoms)),
                confidence=confidence,
                engine="tesseract",
            )
        )

    return _merge_nearby_lines(blocks)


def _merge_nearby_lines(blocks: list[Block]) -> list[Block]:
    ordered = sorted(blocks, key=lambda block: (block.bbox[1] if block.bbox else 0, block.bbox[0] if block.bbox else 0))
    if not ordered:
        return []

    merged: list[Block] = []
    current = ordered[0]
    for block in ordered[1:]:
        if _same_paragraph(current, block):
            current = _merge_blocks(current, block)
        else:
            merged.append(current)
            current = block
    merged.append(current)
    return merged


def _same_paragraph(left: Block, right: Block) -> bool:
    if not left.bbox or not right.bbox:
        return False
    left_x0, _left_y0, left_x1, left_y1 = left.bbox
    right_x0, right_y0, _right_x1, right_y1 = right.bbox
    vertical_gap = right_y0 - left_y1
    avg_height = ((left_y1 - left.bbox[1]) + (right_y1 - right_y0)) / 2
    horizontal_overlap = min(left_x1, right.bbox[2]) - max(left_x0, right_x0)
    return 0 <= vertical_gap <= max(10, avg_height * 1.6) and horizontal_overlap > 0


def _merge_blocks(left: Block, right: Block) -> Block:
    assert left.bbox and right.bbox
    left_conf = left.confidence if left.confidence is not None else 0.0
    right_conf = right.confidence if right.confidence is not None else 0.0
    confidence = (left_conf + right_conf) / 2
    return Block(
        type="uncertain" if left.type == "uncertain" or right.type == "uncertain" else "paragraph",
        text=f"{left.text} {right.text}",
        bbox=(
            min(left.bbox[0], right.bbox[0]),
            min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]),
            max(left.bbox[3], right.bbox[3]),
        ),
        confidence=confidence,
        engine=left.engine or right.engine,
    )


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
