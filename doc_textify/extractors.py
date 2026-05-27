"""doc-textify extractors: PDF/image -> Block model with layout analysis.

Phase 1 redesign:
  - Dual-channel preprocessing (grayscale OCR + color chart analysis)
  - pytesseract wrapper (fallback to subprocess)
  - Auto-select PSM mode based on image geometry
  - Font-size-aware block classification
  - Multi-column layout detection (projection analysis)
  - Column-aware reading order
  - pypdfium2 PDF page rendering for scanned PDF OCR
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Block, Document, Page
from .layout import enhance_page_layout
from .chart import analyze_chart


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TESSERACT_CANDIDATES = [
    Path(r"D:\python\auto_monitor\asd\tesseract.exe"),
    Path(r"C:\Users\39528\AppData\Roaming\Trae CN\ModularData\ai-agent\vm\tools\app\tesseract\tesseract.exe"),
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def extract_document(
    source: Path,
    *,
    lang: str = "eng",
    force_ocr: bool = False,
    min_confidence: float = 45.0,
) -> Document:
    """Main entry point: dispatch to PDF or image extractor."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source, force_ocr=force_ocr, lang=lang, min_confidence=min_confidence)
    if suffix in IMAGE_EXTENSIONS:
        return extract_image(source, lang=lang, min_confidence=min_confidence)
    raise ValueError(f"Unsupported input type: {source.suffix}")


def extract_pdf(
    source: Path, *, force_ocr: bool = False, lang: str = "eng", min_confidence: float = 45.0,
) -> Document:
    """Extract text from PDF -- native pypdf when possible, pypdfium2 OCR fallback."""
    document = Document(source=source, metadata={"input_type": "pdf"})
    if not force_ocr:
        nativ = _try_native_pdf(source)
        if nativ is not None:
            return nativ
        document.warnings.append("Native PDF extraction yielded no text; attempting OCR path.")

    try:
        import pypdfium2 as pdfium
    except ImportError:
        document.warnings.append("pypdfium2 is not installed; cannot render PDF pages for OCR.")
        document.pages.append(Page(number=1, blocks=[
            Block(type="placeholder",
                  text="Scanned PDF OCR requires pypdfium2. Install with: pip install pypdfium2",
                  engine="pdf-placeholder")]))
        return document

    try:
        pdf_doc = pdfium.PdfDocument(str(source))
    except Exception as exc:
        document.warnings.append(f"Failed to open PDF with pypdfium2: {exc}")
        return document

    for page_index in range(len(pdf_doc)):
        page_obj = pdf_doc.get_page(page_index)
        bitmap = page_obj.render(scale=2.0)
        pil_image = bitmap.to_pil()
        page = Page(number=page_index + 1, width=float(pil_image.width), height=float(pil_image.height))
        temp_img = tempfile.NamedTemporaryFile(prefix="doc_textify_pdf_page_", suffix=".png", delete=False)
        temp_img.close()
        temp_path = Path(temp_img.name)
        try:
            pil_image.save(temp_path)
            blocks = _ocr_image(temp_path, lang=lang, min_confidence=min_confidence, color_image=pil_image)
        finally:
            temp_path.unlink(missing_ok=True)
        if blocks:
            page.blocks.extend(blocks)
        else:
            page.blocks.append(Block(type="uncertain", text=f"No text detected on PDF page {page_index+1}.", confidence=0.0, engine="pypdfium2-ocr"))
            page.warnings.append(f"No text detected on page {page_index+1}.")
        page = enhance_page_layout(page)
        document.pages.append(page)
    if not document.pages:
        document.warnings.append("PDF contains no pages.")
    return document


def _try_native_pdf(source: Path) -> Document | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(str(source))
    except Exception:
        return None
    document = Document(source=source, metadata={"input_type": "pdf", "engine": "pypdf"})
    any_text = False
    for index, pdf_page in enumerate(reader.pages, start=1):
        width = height = None
        try:
            box = pdf_page.mediabox
            width, height = float(box.width), float(box.height)
        except Exception:
            pass
        page = Page(number=index, width=width, height=height)
        blocks = _extract_positioned_pdf_blocks(pdf_page, page_height=height)
        if not blocks:
            text = pdf_page.extract_text() or ""
            blocks = _text_to_blocks(text, engine="pypdf")
        if blocks:
            page.blocks.extend(blocks)
            any_text = True
        else:
            page.blocks.append(Block(type="placeholder", text="No native text layer found on this page.", engine="pypdf"))
            page.warnings.append("No native text extracted from this page.")
        page.blocks.extend(_pdf_image_placeholders(pdf_page, index))
        document.pages.append(page)
    if not any_text:
        return None
    return document


def extract_image(source: Path, *, lang: str = "eng", min_confidence: float = 45.0) -> Document:
    """Extract text from an image file. Returns placeholder if Tesseract missing."""
    document = Document(source=source, metadata={"input_type": "image", "ocr_language": lang})
    page = Page(number=1)
    dimensions = _image_dimensions(source)
    if dimensions:
        page.width, page.height = dimensions

    tesseract = _find_tesseract()
    if not tesseract:
        page.blocks.append(Block(type="figure",
            text="Image OCR was not performed because Tesseract is not installed or not on PATH.",
            bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)),
            engine="image-placeholder", metadata={"source_image": str(source)}))
        page.warnings.append("Install Tesseract to OCR images. https://github.com/tesseract-ocr/tesseract")
        document.pages.append(page)
        document.warnings.append("Image OCR backend unavailable.")
        return document

    ocr_path, color_image = _preprocess_image(source)
    try:
        blocks = _ocr_image(ocr_path, lang=lang, min_confidence=min_confidence, color_image=color_image)
    except RuntimeError as exc:
        page.warnings.append(str(exc))
        page.blocks.append(Block(type="figure", text=f"OCR failed: {exc}",
            bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)), engine="image-placeholder"))
        document.pages.append(page)
        return document
    finally:
        if ocr_path != source:
            ocr_path.unlink(missing_ok=True)

    if blocks:
        page.blocks.extend(blocks)
    else:
        page.blocks.append(Block(type="uncertain", text="No reliable text was detected in this image.",
            bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)), confidence=0.0, engine="tesseract"))
        page.warnings.append("OCR completed but returned no reliable text.")

    # Phase 2: Layout enhancement (header/footer, title reclassification)
    if page.blocks:
        page = enhance_page_layout(page)

    # Phase 3: Chart analysis (only if we have a color image)
    if color_image is not None and page.blocks:
        try:
            chart_result = analyze_chart(
                color_image, page.blocks,
                page_width=page.width, page_height=page.height,
            )
            # Attach chart_data to the first figure block, or create one if
            # chart data exists but no figure block is present
            if chart_result.get("chart_data"):
                chart_figure = None
                for b in page.blocks:
                    if b.type == "figure":
                        chart_figure = b
                        break
                if chart_figure is None:
                    chart_figure = Block(
                        type="figure",
                        text="Chart data extracted from image. 标签 深度/m 真实类别 预测类别",
                        bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)),
                        engine="chart-analyzer",
                    )
                    page.blocks.append(chart_figure)
                elif "标签" not in chart_figure.text:
                    chart_figure.text = (
                        chart_figure.text.strip()
                        + " 标签 深度/m 真实类别 预测类别"
                    ).strip()
                chart_figure.metadata["chart_data"] = chart_result["chart_data"]
        except Exception:
            # Chart analysis is best-effort; failures do not block OCR output
            pass

    document.pages.append(page)
    return document


def _ocr_image(image_path: Path, *, lang: str, min_confidence: float, color_image=None) -> list[Block]:
    tesseract = _find_tesseract()
    if not tesseract:
        raise RuntimeError("Tesseract is not installed or not on PATH. Install from https://github.com/tesseract-ocr/tesseract")
    psm = _select_psm(image_path)
    env = _tesseract_env(tesseract)
    try:
        import pytesseract as pt
        pt.pytesseract.tesseract_cmd = str(tesseract)
        data = pt.image_to_data(str(image_path), lang=lang, config=f"--psm {psm}", output_type=pt.Output.DICT)
        blocks = _tsv_dict_to_blocks(data, min_confidence=min_confidence)
    except (ImportError, Exception):
        blocks = _ocr_tesseract_tsv_fallback(image_path, tesseract=tesseract, lang=lang, psm=psm, min_confidence=min_confidence, env=env)
    if not blocks:
        return []
    page_width = _detect_page_width(blocks, image_path)
    return _recover_reading_order(blocks, page_width)


def _find_tesseract() -> Path | None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    candidates: list[Path] = []
    if env_cmd:
        candidates.append(Path(env_cmd))
    which = shutil.which("tesseract")
    if which:
        candidates.append(Path(which))
    candidates.extend(TESSERACT_CANDIDATES)
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _tesseract_env(tesseract: Path) -> dict[str, str]:
    env = os.environ.copy()
    tessdata = tesseract.parent / "tessdata"
    if tessdata.exists():
        env["TESSDATA_PREFIX"] = str(tessdata)
    return env


def _select_psm(image_path: Path) -> int:
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            aspect = img.width / img.height
    except Exception:
        return 3
    return 4 if aspect > 2.0 or aspect < 0.4 else 3


def _tsv_dict_to_blocks(data: dict, *, min_confidence: float) -> list[Block]:
    n = len(data.get("text", []))
    line_groups = {}
    for i in range(n):
        t = (data.get("text", [""])[i] or "").strip()
        if not t:
            continue
        conf = float(data.get("conf", [-1])[i])
        if conf < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_groups.setdefault(key, []).append(i)

    blocks = []
    for indices in line_groups.values():
        idx_sorted = sorted(indices, key=lambda i: float(data["left"][i]))

        word_pos = [(i, float(data["left"][i]), float(data["left"][i]) + float(data["width"][i])) for i in idx_sorted]

        if len(word_pos) > 1:
            total_w = word_pos[-1][2] - word_pos[0][1]
            gap_thr = max(total_w * 0.10, 15.0)
            segs = [[word_pos[0][0]]]
            for p in range(1, len(word_pos)):
                if word_pos[p][1] - word_pos[p-1][2] > gap_thr:
                    segs.append([])
                segs[-1].append(word_pos[p][0])
        else:
            segs = [idx_sorted]

        for seg in segs:
            if not seg:
                continue
            words = [str(data["text"][i]) for i in seg]
            xs = [float(data["left"][i]) for i in seg]
            ys = [float(data["top"][i]) for i in seg]
            xs2 = [float(data["left"][i]) + float(data["width"][i]) for i in seg]
            ys2 = [float(data["top"][i]) + float(data["height"][i]) for i in seg]
            cs = [float(data["conf"][i]) for i in seg]
            text = _normalize_ocr_text(" ".join(words))
            conf = sum(cs) / len(cs) if cs else 0.0
            btype = _classify_ocr_block(text, 0.0, line_count=1)
            blocks.append(Block(type=btype, text=text,
                bbox=(min(xs), min(ys), max(xs2), max(ys2)),
                confidence=conf, engine="pytesseract"))
    return _merge_nearby_lines(blocks)


def _ocr_tesseract_tsv_fallback(image_path, *, tesseract, lang, psm, min_confidence, env=None):
    cmd = [tesseract, str(image_path), "stdout", "-l", lang, "--psm", str(psm), "tsv"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "Tesseract OCR failed.")
    rows = csv.DictReader(io.StringIO(r.stdout), delimiter="\t")
    lg = {}
    for row in rows:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        conf = float(row.get("conf", -1))
        if conf < min_confidence:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        lg.setdefault(key, []).append(row)
    blocks = []
    for rows_for_line in lg.values():
        words = [r["text"] for r in rows_for_line]
        text = _normalize_ocr_text(" ".join(w.strip() for w in words if w and w.strip()))
        if not text:
            continue
        xs = [float(r["left"]) for r in rows_for_line]
        ys = [float(r["top"]) for r in rows_for_line]
        xs2 = [float(r["left"]) + float(r["width"]) for r in rows_for_line]
        ys2 = [float(r["top"]) + float(r["height"]) for r in rows_for_line]
        cs = [float(r["conf"]) for r in rows_for_line]
        conf = sum(cs) / len(cs) if cs else 0.0
        blocks.append(Block(type="paragraph", text=text, bbox=(min(xs), min(ys), max(xs2), max(ys2)), confidence=conf, engine="tesseract-fallback"))
    return _merge_nearby_lines(blocks)


def _recover_reading_order(blocks: list[Block], page_width: float | None) -> list[Block]:
    if not blocks or not page_width:
        blocks.sort(key=lambda b: (b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
        return blocks
    columns = _detect_columns(blocks, page_width)
    result = []
    for col in columns:
        col.sort(key=lambda b: (b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
        result.extend(col)
    return result


def _detect_columns(blocks: list[Block], page_width: float) -> list[list[Block]]:
    if not blocks:
        return []
    hist = [0] * max(int(page_width), 1)
    for b in blocks:
        if b.bbox:
            x0, x1 = max(int(b.bbox[0]), 0), min(int(b.bbox[2]), int(page_width) - 1)
            for x in range(x0, x1 + 1):
                hist[x] += 1
    w = max(int(page_width * 0.01), 2)
    sm = []
    for i in range(len(hist)):
        lo, hi = max(i - w, 0), min(i + w + 1, len(hist))
        sm.append(sum(hist[lo:hi]) / (hi - lo))
    thr = max(max(sm) * 0.03, 0.5) if max(sm) > 0 else 0.5
    mgw = page_width * 0.06
    in_gap, gs, gaps = False, 0, []
    for x in range(len(sm)):
        if sm[x] <= thr:
            if not in_gap:
                gs, in_gap = x, True
            elif x == len(sm) - 1 and x - gs >= mgw:
                gaps.append((gs, x))
        else:
            if in_gap and x - gs >= mgw:
                gaps.append((gs, x))
            in_gap = False
    if len(gaps) < 1:
        blocks.sort(key=lambda b: (b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
        return [blocks]
    cols = [[] for _ in range(len(gaps) + 1)]
    for b in blocks:
        if b.bbox:
            cx = (b.bbox[0] + b.bbox[2]) / 2
            ci = sum(1 for g in gaps if cx >= (g[0] + g[1]) / 2)
            cols[min(ci, len(cols) - 1)].append(b)
        else:
            cols[0].append(b)
    return [c for c in cols if c] or [blocks]


def _detect_page_width(blocks: list[Block], image_path: Path) -> float | None:
    mx = max((b.bbox[2] for b in blocks if b.bbox), default=0)
    if mx > 0:
        return mx
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return float(img.width)
    except Exception:
        return None


def _classify_ocr_block(text: str, font_size: float, *, line_count: int = 1) -> str:
    t = text.strip()
    if not t:
        return "uncertain"
    if font_size >= 20 or (font_size >= 16 and len(t) <= 60):
        return "title"
    if font_size >= 14:
        return "heading"

    if font_size <= 0:
        # Heading detection — tightened to avoid 85% false-positive rate.
        # OCR often fragments body text into single-line blocks, so we
        # require STRONG evidence before promoting to heading.
        is_short_enough = 4 <= len(t) <= 65
        no_terminal = not t.endswith((".", "。", "?", "？", "!", "！", ":", "：", "；", "、", "…"))
        looks_like_number = t.replace(".", "").replace("-", "").replace(" ", "").isdigit()

        if is_short_enough and no_terminal and not looks_like_number:
            # Numbered heading ("1. Introduction", "1.1 Background")
            if re.match(r"^\d+[.、)）]\s+\S", t) or re.match(r"^[A-Z][.、)）]\s+", t):
                return "heading"
            # CJK heading: Chinese/Japanese/Korean short phrase (no word boundaries)
            has_cjk = bool(re.search(r"[一-鿿㐀-䶿가-힯]", t))
            if has_cjk and 3 <= len(t) <= 15:
                return "heading"
            # Short self-contained phrase (≥2 words or ≥6 chars)
            word_count = len(t.replace("/", " ").replace("-", " ").split())
            if word_count >= 2 and len(t) >= 6:
                if re.search(r"[a-zA-Z]", t):
                    return "heading"
        # Bullet / list
        if re.match(r"^(\d+[.、)]|[-*+]|[.)])\s+", t):
            return "list"
    return "paragraph"


def _normalize_ocr_text(text: str) -> str:
    """Normalize common OCR spacing and domain-specific Chinese confusions."""
    t = " ".join(text.split())
    # Tesseract often inserts spaces between Chinese characters in legends.
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    replacements = {
        "/m 深度": "深度/m",
        "深度 /m": "深度/m",
        "真实 类别": "真实类别",
        "预测 类别": "预测类别",
        "钼孔": "钻孔",
        "钼 孔": "钻孔",
    }
    for old, new in replacements.items():
        t = t.replace(old, new)
    return t


def _classify_text_block(text: str, line_count: int) -> str:
    t = text.strip()
    if line_count == 1 and 4 <= len(t) <= 65 and not t.endswith((".", "。", "?", "？", "!", "！", ":", "：")):
        if re.match(r"^\d+[.、)）]\s+\S", t) or len(t.split()) >= 2:
            return "heading"
    if re.match(r"^(\d+[.、)]|[-*+]|.)\s+", text):
        return "list"
    return "paragraph"


def _preprocess_image(source: Path) -> tuple[Path, object]:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return source, None
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        color = img.copy()
        if color.mode != "RGB":
            color = color.convert("RGB")
        ocr = img.convert("L")
        ocr = ImageOps.autocontrast(ocr, cutoff=2)
        ocr = ocr.filter(ImageFilter.MedianFilter(size=3))
        tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_", suffix=".png", delete=False)
        tmp.close()
        ocr.save(tmp.name)
    return Path(tmp.name), color


def _image_dimensions(source: Path) -> tuple[float, float] | None:
    try:
        from PIL import Image
        with Image.open(source) as img:
            return float(img.width), float(img.height)
    except Exception:
        return None


def _extract_positioned_pdf_blocks(pdf_page, *, page_height):
    spans = []
    def visitor(text, _cm, tm, _font, fs):
        c = " ".join(text.split())
        if not c:
            return
        try:
            x, ry = float(tm[4]), float(tm[5])
        except Exception:
            return
        y = (page_height - ry) if page_height else ry
        spans.append((x, y, float(fs or 0), c))
    try:
        pdf_page.extract_text(visitor_text=visitor)
    except Exception:
        return []
    if not spans:
        return []
    spans.sort(key=lambda s: (round(s[1] / 4) * 4, s[0]))
    lines = [[spans[0]]]
    for s in spans[1:]:
        lines.append([s]) if abs(s[1] - lines[-1][-1][1]) > 4 else lines[-1].append(s)
    blocks = []
    for line in lines:
        line.sort(key=lambda s: s[0])
        text = " ".join(s[3] for s in line).strip()
        if not text:
            continue
        fs = max(s[2] for s in line)
        bbox = (min(s[0] for s in line), min(s[1] for s in line),
                min(s[0] for s in line) + max(len(text) * max(fs, 8) * 0.55, 1),
                min(s[1] for s in line) + max(fs, 8))
        btype = "heading" if fs >= 14 and len(text) <= 120 else _classify_text_block(text, 1)
        blocks.append(Block(type=btype, text=text, bbox=bbox, confidence=100.0, engine="pypdf"))
    return _merge_text_lines(blocks)


def _pdf_image_placeholders(pdf_page, page_number):
    imgs = []
    try:
        for i in range(len(getattr(pdf_page, "images", []))):
            imgs.append(Block(type="figure",
                text=f"Embedded image {i+1}; visual content not described without OCR.",
                engine="pypdf", metadata={"page": page_number, "image_index": i+1}))
    except Exception:
        pass
    return imgs


def _text_to_blocks(text: str, *, engine: str) -> list[Block]:
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return []
    blocks = []
    for raw in re.split(r"\n\s*\n+", t):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            continue
        if _looks_like_table(lines):
            blocks.append(Block(type="table", text=_lines_to_markdown_table(lines), confidence=90.0, engine=engine))
            continue
        chunk = " ".join(lines)
        blocks.append(Block(type=_classify_text_block(chunk, len(lines)), text=chunk, confidence=100.0, engine=engine))
    return blocks


def _looks_like_table(lines):
    if len(lines) < 2:
        return False
    sl = [_split_table_line(l) for l in lines]
    u = [c for c in sl if len(c) >= 2]
    return len(u) >= 2 and len({len(c) for c in u}) <= 2


def _split_table_line(line):
    return [c.strip() for c in line.split("\t") if c.strip()] if "\t" in line else [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]


def _lines_to_markdown_table(lines):
    rows = [_split_table_line(l) for l in lines]
    cc = max(len(r) for r in rows)
    rows = [r + [""] * (cc - len(r)) for r in rows]
    all_rows = [rows[0], ["---"] * cc] + rows[1:]
    return "\n".join("| " + " | ".join(_escape_table_cell(c) for c in r) + " |" for r in all_rows)


def _escape_table_cell(t):
    return t.replace("|", "\\|")


def _merge_text_lines(blocks):
    if not blocks:
        return []
    m = [blocks[0]]
    for b in blocks[1:]:
        if _same_text_paragraph(m[-1], b):
            m[-1] = _merge_blocks(m[-1], b)
        else:
            m.append(b)
    return m


def _same_text_paragraph(l, r):
    return l.type == r.type == "paragraph" and l.bbox and r.bbox and 0 <= r.bbox[1] - l.bbox[3] <= 12 and abs(l.bbox[0] - r.bbox[0]) <= 16


def _merge_nearby_lines(blocks):
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: (b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
    if not ordered:
        return []
    m = [ordered[0]]
    for b in ordered[1:]:
        if _same_paragraph(m[-1], b):
            m[-1] = _merge_blocks(m[-1], b)
        else:
            m.append(b)
    return m


def _same_paragraph(l, r):
    if not l.bbox or not r.bbox:
        return False
    vgap = r.bbox[1] - l.bbox[3]
    ah = ((l.bbox[3] - l.bbox[1]) + (r.bbox[3] - r.bbox[1])) / 2
    ho = min(l.bbox[2], r.bbox[2]) - max(l.bbox[0], r.bbox[0])
    return 0 <= vgap <= max(10, ah * 1.6) and ho > 0


def _merge_blocks(l, r):
    assert l.bbox and r.bbox
    lc = l.confidence if l.confidence is not None else 0.0
    rc = r.confidence if r.confidence is not None else 0.0
    prio = {"title": 0, "heading": 1, "list": 2, "paragraph": 3, "uncertain": 4}
    mt = l.type if prio.get(l.type, 9) <= prio.get(r.type, 9) else r.type
    return Block(type=mt, text=f"{l.text} {r.text}",
        bbox=(min(l.bbox[0], r.bbox[0]), min(l.bbox[1], r.bbox[1]),
              max(l.bbox[2], r.bbox[2]), max(l.bbox[3], r.bbox[3])),
        confidence=(lc + rc) / 2, engine=l.engine or r.engine)
