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
from .table_extraction import extract_tables_from_image
from .formula_ocr import (
    recognize_formula,
    is_formula_ocr_available,
    reconstruct_formula_text,
    looks_like_latex,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

TESSERACT_CANDIDATES = [
    # Project-local Tesseract (portable, self-contained)
    Path(__file__).resolve().parent.parent / "tools" / "Tesseract-OCR" / "tesseract.exe",
    Path(r"D:\python\auto_monitor\asd\tesseract.exe"),
    Path(r"C:\Users\39528\AppData\Roaming\Trae CN\ModularData\ai-agent\vm\tools\app\tesseract\tesseract.exe"),
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]

# ---------------------------------------------------------------------------
#  Language detection & CJK helpers
# ---------------------------------------------------------------------------

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


def _detect_tofu(text: str) -> bool:
    """Detect tofu characters indicating glyph rendering failure.

    When pypdf extracts CJK text without the proper font mappings,
    glyphs are rendered as the Unicode replacement character (U+FFFD)
    or a white square (U+25A1).  This function checks for those markers.
    """
    if not text:
        return False
    tofu_chars = {'\u25a1', '\ufffd'}
    return any(c in text for c in tofu_chars)


def _resolve_auto_language(source: Path) -> dict:
    """Detect document language for ``--lang auto`` mode.

    Returns a dict with keys ``lang``, ``min_confidence``, and ``psm``.
    """
    suffix = source.suffix.lower()

    # PDF: try native text extraction on first page
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(source))
            if reader.pages:
                text = reader.pages[0].extract_text() or ""
                if _is_cjk(text):
                    return {"lang": "chi_sim", "min_confidence": 35.0, "psm": 6}
        except Exception:
            pass

    # Image / failed PDF: Tesseract OSD script detection
    tesseract = _find_tesseract()
    if tesseract:
        try:
            env = _tesseract_env(tesseract)
            cmd = [str(tesseract), str(source), "stdout", "--psm", "0"]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False,
                               env=env, timeout=15)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    m = re.search(r"Script(?: name)?:\s*(\S+)", line, re.IGNORECASE)
                    if m:
                        script = m.group(1).lower()
                        if script in ("han", "hans", "hant", "cjk", "chinese",
                                       "japanese", "korean"):
                            return {"lang": "chi_sim", "min_confidence": 35.0, "psm": 6}
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Fallback: combined language (safe for mixed docs)
    return {"lang": "chi_sim+eng", "min_confidence": 40.0, "psm": 3}


def extract_document(
    source: Path,
    *,
    lang: str = "eng",
    force_ocr: bool = False,
    min_confidence: float = 45.0,
    handwriting: bool = False,
    deskew: bool = True,
    chart_colors: list[str] | None = None,
    formula_ocr: bool = False,
) -> Document:
    """Main entry point: dispatch to PDF or image extractor."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    # --lang auto: detect language and adjust thresholds
    if lang == "auto":
        lang_info = _resolve_auto_language(source)
        lang = lang_info["lang"]
        # Only auto-adjust min_confidence when user hasn't explicitly set it
        if min_confidence == 45.0:
            min_confidence = lang_info["min_confidence"]

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source, force_ocr=force_ocr, lang=lang, min_confidence=min_confidence,
                           handwriting=handwriting, deskew=deskew, chart_colors=chart_colors,
                           formula_ocr=formula_ocr)
    if suffix in IMAGE_EXTENSIONS:
        return extract_image(source, lang=lang, min_confidence=min_confidence,
                             handwriting=handwriting, deskew=deskew, chart_colors=chart_colors,
                             formula_ocr=formula_ocr)
    raise ValueError(f"Unsupported input type: {source.suffix}")


def extract_pdf(
    source: Path, *, force_ocr: bool = False, lang: str = "eng", min_confidence: float = 45.0,
    handwriting: bool = False,
    deskew: bool = True,
    chart_colors: list[str] | None = None,
    formula_ocr: bool = False,
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
            ocr_path = temp_path
            if deskew:
                ocr_path = _deskew_image(ocr_path)
                ocr_path = _correct_orientation(ocr_path)
            blocks = _ocr_image(ocr_path, lang=lang, min_confidence=min_confidence,
                                handwriting=handwriting, color_image=pil_image)
            # Clean up deskew temp
            if ocr_path != temp_path:
                ocr_path.unlink(missing_ok=True)
        finally:
            temp_path.unlink(missing_ok=True)
        if blocks:
            page.blocks.extend(blocks)
        else:
            page.blocks.append(Block(type="uncertain", text=f"No text detected on PDF page {page_index+1}.", confidence=0.0, engine="pypdfium2-ocr"))
            page.warnings.append(f"No text detected on page {page_index+1}.")
        page = enhance_page_layout(page, color_image=pil_image)
        # Phase 3: Chart analysis on the rendered page (B&W classification
        # charts and coloured plots both benefit; best-effort).
        _attach_chart_analysis(page, pil_image, chart_colors)
        # Phase 3.2: Rule-based LaTeX reconstruction for detected formula
        # blocks (always-on; re-OCR of the full formula line + rule-based
        # reconstruction so scanned formulas render as $$...$$).
        _reconstruct_formula_blocks(page, pil_image)
        # Formula OCR (best-effort, requires --formula-ocr flag)
        if formula_ocr and pil_image is not None:
            _process_formula_blocks(page, pil_image)
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
    import logging
    _log = logging.getLogger(__name__)
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
            # Check for CJK tofu (glyph rendering failure) before accepting native text
            if _detect_tofu(text):
                _log.warning(
                    "CJK glyph rendering issue detected on page %d, "
                    "falling back to OCR for the entire document.", index
                )
                return None
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


def extract_image(source: Path, *, lang: str = "eng", min_confidence: float = 45.0,
                   handwriting: bool = False, deskew: bool = True,
                   chart_colors: list[str] | None = None,
                   formula_ocr: bool = False) -> Document:
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

    ocr_path, color_image = _preprocess_image(source, handwriting=handwriting, deskew=deskew)
    try:
        blocks = _ocr_image(ocr_path, lang=lang, min_confidence=min_confidence,
                            handwriting=handwriting, color_image=color_image)
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

    # Phase 2: Layout enhancement (header/footer, title reclassification, formula detection)
    if page.blocks:
        page = enhance_page_layout(page, color_image=color_image)

    # Phase 2.5: Table structure recovery (line-based cell grid detection)
    if color_image is not None and page.blocks:
        try:
            table_blocks = extract_tables_from_image(
                color_image, page.blocks,
                page_width=page.width, page_height=page.height,
            )
            if table_blocks:
                # Replace any existing table-type blocks with structured ones
                page.blocks = [b for b in page.blocks if b.type != "table"]
                page.blocks.extend(table_blocks)
        except Exception:
            pass  # table extraction is best-effort

    # Phase 3: Chart analysis (only if we have a color image)
    _attach_chart_analysis(page, color_image, chart_colors)

    # Phase 3.2: Rule-based LaTeX reconstruction (always-on)
    _reconstruct_formula_blocks(page, color_image)

    # Phase 3.5: Formula OCR (best-effort, requires --formula-ocr flag)
    if formula_ocr and color_image is not None and page.blocks:
        _process_formula_blocks(page, color_image)

    document.pages.append(page)
    return document


# ── Phase 3 helper: chart analysis ──────────────────────────────

_CHART_DATA_HEADER = " 标签 深度/m 真实类别 预测类别"


def _attach_chart_analysis(
    page: Page, color_image, chart_colors: list[str] | None,
) -> None:
    """Analyse *color_image* for chart content and attach ``chart_data`` to
    the page's first figure block (creating one if none exists yet).

    Best-effort: failures never block the rest of the pipeline.  This runs
    for both image inputs and OCR-rendered PDF pages.
    """
    if color_image is None or not page.blocks:
        return
    try:
        chart_result = analyze_chart(
            color_image, page.blocks,
            page_width=page.width, page_height=page.height,
            chart_colors=chart_colors,
        )
    except Exception:
        # Chart analysis is best-effort; failures do not block OCR output
        return

    chart_data = chart_result.get("chart_data")
    if not chart_data:
        return

    chart_figure = next((b for b in page.blocks if b.type == "figure"), None)
    # The "标签" header describes the interval/point depth tables; only
    # attach it when structured series data is present.  Coarse
    # "chart_detected" fallback entries carry axis labels instead.
    has_structured = any(
        item.get("type") in ("interval", "point") for item in chart_data
    )
    if chart_figure is None:
        caption = "Chart data extracted from image."
        if has_structured:
            caption += _CHART_DATA_HEADER
        chart_figure = Block(
            type="figure",
            text=caption,
            bbox=(0.0, 0.0, float(page.width or 0), float(page.height or 0)),
            engine="chart-analyzer",
        )
        page.blocks.append(chart_figure)
    elif has_structured and "标签" not in chart_figure.text:
        chart_figure.text = (
            chart_figure.text.strip() + _CHART_DATA_HEADER
        ).strip()
    chart_figure.metadata["chart_data"] = chart_data


def _ocr_image(image_path: Path, *, lang: str, min_confidence: float,
               handwriting: bool = False, color_image=None) -> list[Block]:
    tesseract = _find_tesseract()
    if not tesseract:
        raise RuntimeError("Tesseract is not installed or not on PATH. Install from https://github.com/tesseract-ocr/tesseract")

    if handwriting:
        blocks = _ocr_handwriting(image_path, tesseract=tesseract, lang=lang,
                                  min_confidence=min_confidence)
        # Also try EasyOCR for handwriting (optional dependency)
        easyocr_blocks = _ocr_easyocr(image_path, lang=lang)
        if easyocr_blocks:
            blocks = _merge_ocr_results(blocks, easyocr_blocks)
    else:
        psm = _select_psm(image_path, lang_hint=lang)
        blocks = _do_ocr_tesseract(image_path, tesseract=tesseract, lang=lang,
                                   psm=psm, min_confidence=min_confidence)

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


def _select_psm(image_path: Path, lang_hint: str = "eng") -> int:
    """Select optimal Tesseract PSM mode.

    For CJK-dominant documents, PSM 6 (uniform block of text) tends to
    produce better results than PSM 3 (fully automatic).  For Latin text,
    the aspect-ratio-based heuristic (PSM 4 for wide/narrow images,
    PSM 3 otherwise) works well.
    """
    # CJK-aware: prefer PSM 6 for Chinese/Japanese/Korean text
    if lang_hint in ("chi_sim", "chi_tra", "chi_sim+eng", "jpn", "jpn_vert", "kor"):
        return 6

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


# Unit tokens commonly found in engineering/measurement body text.
# Used to keep measurement lines ("qc = 0.9 MPa, fs = 72 kPa") out of
# heading/formula classification — they are body text, always.
_UNIT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(MPa|kPa|GPa|Pa|kN|kN/m2|kN/m²|mm|cm|km|m/s|km/h|bars?|psi|°C|°F|‰|%|m|kg|g|t)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)


def _looks_like_measurement_or_equation(text: str) -> bool:
    """Return True when *text* is body text (measurement or equation line).

    OCR body lines routinely contain equation signs ("=") or numbers
    combined with unit tokens ("0.9 MPa", "72 kPa", "2 m").  Such lines
    are never headings, so they are short-circuited to ``paragraph`` before
    any heading evidence is considered.
    """
    t = text.strip()
    if not t:
        return False
    # Equation signs are never part of a heading
    if re.search(r"[=≈]", t):
        return True
    # Digits combined with a unit token → measurement line
    if re.search(r"\d", t) and _UNIT_TOKEN_RE.search(t):
        return True
    return False


# Standard single-phrase section headings that appear as standalone blocks.
# Exact-match whitelist: a body line that is EXACTLY one of these phrases
# (with a capitalized first letter) is almost certainly a heading, not
# OCR-fragmented prose, so it is safe to promote without false positives.
_KNOWN_HEADING_PHRASES = frozenset({
    "abstract", "introduction", "background", "literature review", "method",
    "methodology", "results", "discussion", "conclusions", "conclusion",
    "summary", "references", "notation", "nomenclature", "acknowledgements",
    "acknowledgment", "appendix", "case history", "case histories",
    "list of symbols", "field data", "testing",
})


def _is_title_case_phrase(text: str, min_words: int = 2, max_words: int = 6) -> bool:
    """Return True when *text* is a short title-case phrase.

    A strong heading signal for OCR output: 2-6 alphabetic words where
    EVERY word starts with an uppercase letter ("Soil Behaviour Type",
    "SOIL BEHAVIOUR TYPE").  Words containing digits or symbols break
    the pattern and are rejected conservatively.
    """
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    for w in words:
        if not re.fullmatch(r"[A-Za-z]+", w):
            return False
        if not w[0].isupper():
            return False
    return True


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

        # Body-text guard: measurements and equations are never headings.
        if _looks_like_measurement_or_equation(t):
            return "paragraph"

        is_short_enough = 4 <= len(t) <= 65
        no_terminal = not t.endswith((".", "。", "?", "？", "!", "！", ":", "：", "；", "、", "…"))
        looks_like_number = t.replace(".", "").replace("-", "").replace(" ", "").isdigit()

        if is_short_enough and no_terminal and not looks_like_number:
            # Numbered heading ("1. Introduction", "1.1 Background", "2.1.3 Xxx")
            if (
                re.match(r"^\d+[.、)）]\s+\S", t)          # "1. Introduction", "1) Intro"
                or re.match(r"^\d+(?:\.\d+)+[.、)）]?\s+\S", t)  # "1.1 Background", "2.1.3 Xxx"
                or re.match(r"^[A-Z][.、)）]\s+", t)          # "A. Appendix"
            ):
                return "heading"
            # CJK heading: Chinese/Japanese/Korean short phrase (no word boundaries)
            has_cjk = bool(re.search(r"[一-鿿㐀-䶿가-힯]", t))
            if has_cjk and 3 <= len(t) <= 15:
                return "heading"
            # Title-case phrase: every word starts uppercase ("Soil Behaviour Type")
            if _is_title_case_phrase(t):
                return "heading"
            # Known single-phrase section heading ("Introduction", "Case history", "Summary")
            if t.lower() in _KNOWN_HEADING_PHRASES and t[0].isupper():
                return "heading"
        # Bullet / list
        if re.match(r"^(\d+[.、)]|[-*+]|[.)])\s+", t):
            return "list"
    return "paragraph"


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR text with comprehensive CJK handling.

    Steps:
      1. Collapse whitespace.
      2. Convert full-width alphanumerics to half-width.
      3. Remove spaces inserted by OCR between CJK characters.
      4. Correct common Tesseract CJK confusions.
      5. Apply domain-specific replacements.
      6. Add correct spacing at Latin-CJK boundaries.
    """
    t = " ".join(text.split())

    # --- Full-width to half-width: digits ---
    _FW_DIGITS = (
        "０１２３４５６７８９"
    )
    _HW_DIGITS = (
        "0123456789"
    )
    t = t.translate(str.maketrans(_FW_DIGITS, _HW_DIGITS))

    # --- Full-width to half-width: uppercase letters ---
    _FW_UPPER = (
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    )
    _HW_UPPER = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    t = t.translate(str.maketrans(_FW_UPPER, _HW_UPPER))

    # --- Full-width to half-width: lowercase letters ---
    _FW_LOWER = (
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    )
    _HW_LOWER = (
        "abcdefghijklmnopqrstuvwxyz"
    )
    t = t.translate(str.maketrans(_FW_LOWER, _HW_LOWER))

    # --- Remove inter-CJK spaces (OCR artifact) ---
    t = re.sub(
        r"(?<=[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])"
        r"\s+"
        r"(?=[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])",
        "",
        t,
    )

    # --- Common Tesseract CJK character confusions ---
    # Direction: Tesseract's wrong output → correct intended character
    _CJK_CONFUSIONS = {
        "已": "己",   # 已经 → correct is 自己
        "未": "末",   # 未来 → correct is 末尾
        "日": "曰",   # 日期 → correct is 子曰
        "人": "入",   # 人民 → correct is 入口
        "土": "士",   # 土地 → correct is 士兵
        "干": "千",   # 干部 → correct is 一千
        "天": "夫",   # 天空 → correct is 丈夫
        "王": "玉",   # 王子 → correct is 玉米
        "午": "牛",   # 中午 → correct is 牛奶
        "右": "石",   # 左右 → correct is 石头
        "刀": "力",   # 刀子 → correct is 力量
        "几": "九",   # 几个 → correct is 九个
        "凤": "风",   # 凤凰 → correct is 大风
        "帅": "师",   # 元帅 → correct is 老师
        "币": "巾",   # 货币 → correct is 毛巾
    }
    for wrong, right in _CJK_CONFUSIONS.items():
        t = t.replace(wrong, right)

    # Multi-character CJK confusions
    _MULTI_CJK_CONFUSIONS = {
        "钼孔": "钻孔",
        "钼 孔": "钻孔",
    }
    for wrong, right in _MULTI_CJK_CONFUSIONS.items():
        t = t.replace(wrong, right)

    # --- Domain-specific replacements (preserved from original) ---
    _DOMAIN_REPLACEMENTS = {
        "/m 深度": "深度/m",
        "深度 /m": "深度/m",
        "真实 类别": "真实类别",
        "预测 类别": "预测类别",
    }
    for old, new in _DOMAIN_REPLACEMENTS.items():
        t = t.replace(old, new)

    # --- Mixed-script spacing: standard CJK-Latin boundary rule ---
    # Latin/ASCII word followed by CJK character: add space
    t = re.sub(
        r"([a-zA-Z0-9])([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])",
        r"\1 \2",
        t,
    )
    # CJK character followed by Latin/ASCII word: add space
    t = re.sub(
        r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])([a-zA-Z0-9])",
        r"\1 \2",
        t,
    )

    return t


def _classify_text_block(text: str, line_count: int) -> str:
    t = text.strip()
    if line_count == 1 and 4 <= len(t) <= 65 and not t.endswith((".", "。", "?", "？", "!", "！", ":", "：")):
        if re.match(r"^\d+[.、)）]\s+\S", t) or len(t.split()) >= 2:
            return "heading"
    if re.match(r"^(\d+[.、)]|[-*+]|.)\s+", text):
        return "list"
    return "paragraph"


def _preprocess_image(source: Path, *, handwriting: bool = False, deskew: bool = False) -> tuple[Path, object]:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        return source, None

    # Apply deskew + orientation correction first if enabled
    working_path = source
    deskew_temps: list[Path] = []
    if deskew:
        deskewed = _deskew_image(working_path)
        if deskewed != working_path:
            deskew_temps.append(deskewed)
        working_path = deskewed
        oriented = _correct_orientation(working_path)
        if oriented != working_path:
            deskew_temps.append(oriented)
        working_path = oriented

    with Image.open(working_path) as img:
        img = ImageOps.exif_transpose(img)
        color = img.copy()
        if color.mode != "RGB":
            color = color.convert("RGB")

        if handwriting:
            ocr = _preprocess_handwriting(img)
        else:
            ocr = img.convert("L")
            ocr = ImageOps.autocontrast(ocr, cutoff=2)
            ocr = ocr.filter(ImageFilter.MedianFilter(size=3))

        tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_", suffix=".png", delete=False)
        tmp.close()
        ocr.save(tmp.name)

    # Clean up deskew/orientation temp files
    for tp in deskew_temps:
        try:
            tp.unlink(missing_ok=True)
        except Exception:
            pass

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
    if not (l.type == r.type == "paragraph" and l.bbox and r.bbox):
        return False
    vgap = r.bbox[1] - l.bbox[3]
    if not (0 <= vgap <= 12 and abs(l.bbox[0] - r.bbox[0]) <= 16):
        return False
    # CJK punctuation-based splitting when gap is large for CJK text
    if _is_cjk(l.text) and _CJK_SENTENCE_END_RE.search(l.text.strip()):
        if vgap > 8:
            return False
    return True


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


_CJK_SENTENCE_END_RE = re.compile(r"[。！？]$")


def _same_paragraph(l, r):
    if not l.bbox or not r.bbox:
        return False
    vgap = r.bbox[1] - l.bbox[3]
    ah = ((l.bbox[3] - l.bbox[1]) + (r.bbox[3] - r.bbox[1])) / 2
    ho = min(l.bbox[2], r.bbox[2]) - max(l.bbox[0], r.bbox[0])
    if not (0 <= vgap <= max(10, ah * 1.6) and ho > 0):
        return False

    # CJK punctuation-based splitting: when a CJK block ends with sentence-ending
    # punctuation and the vertical gap is large relative to line height, treat
    # them as separate paragraphs to avoid merging an entire page into one.
    if _is_cjk(l.text) and _CJK_SENTENCE_END_RE.search(l.text.strip()):
        if vgap > max(15, ah * 1.8):
            return False

    return True


def _merge_blocks(l, r):
    assert l.bbox and r.bbox
    lc = l.confidence if l.confidence is not None else 0.0
    rc = r.confidence if r.confidence is not None else 0.0
    prio = {"title": 0, "heading": 1, "list": 2, "paragraph": 3, "uncertain": 4}
    mt = l.type if prio.get(l.type, 9) <= prio.get(r.type, 9) else r.type

    # CJK-aware joining: no space between pure-CJK blocks
    l_cjk = _is_cjk(l.text)
    r_cjk = _is_cjk(r.text)
    if l_cjk and r_cjk:
        separator = ""          # both CJK → no space (Chinese doesn't use word spacing)
    elif l_cjk or r_cjk:
        separator = " "         # mixed script → preserve space boundary
    else:
        separator = " "         # Latin-only → space join (word separation)

    return Block(type=mt, text=f"{l.text}{separator}{r.text}",
        bbox=(min(l.bbox[0], r.bbox[0]), min(l.bbox[1], r.bbox[1]),
              max(l.bbox[2], r.bbox[2]), max(l.bbox[3], r.bbox[3])),
        confidence=(lc + rc) / 2, engine=l.engine or r.engine)


# ═══════════════════════════════════════════════════════════════════════
#  Rule-based formula LaTeX reconstruction (always-on)
# ═══════════════════════════════════════════════════════════════════════


def _reconstruct_formula_blocks(page: Page, color_image) -> None:
    """Reconstruct LaTeX for every ``type="formula"`` block.

    Scanned formulas are usually OCR'd into tiny fragments ("By =",
    "X 100 =") that span only part of the real formula line.  For each
    formula block this step:

      1. expands the crop to the full text line (union with neighbouring
         blocks on the same line, then out to the nearest column gaps),
      2. re-OCRs the line region with Tesseract (upscaled),
      3. runs the rule-based LaTeX reconstruction
         (:func:`doc_textify.formula_ocr.reconstruct_formula_text`),
      4. keeps the result only when it contains recognizable LaTeX
         (backslash macros, subscripts, …).

    Blocks that already carry ``formula_latex`` metadata (e.g. set by the
    pix2tex path) are left untouched.  Failures are silently skipped — the
    raw OCR fragment text remains as a fallback.
    """
    import PIL.Image

    if not page.blocks or color_image is None:
        return
    tesseract = _find_tesseract()
    if tesseract is None:
        # No OCR backend: at least run the rule-based pass over whatever
        # text the formula blocks already carry.
        for block in page.blocks:
            if block.type != "formula":
                continue
            _apply_formula_reconstruction(block)
        return

    try:
        import numpy as np
        gray = np.array(color_image.convert("L")) if color_image.mode != "L" else np.array(color_image)
    except Exception:
        gray = None
    img_w, img_h = color_image.size

    for block in page.blocks:
        if block.type != "formula" or not block.bbox:
            continue
        if block.metadata.get("formula_latex"):
            continue

        x0, y0, x1, y1 = block.bbox
        bh = max(y1 - y0, 1)
        bw = max(x1 - x0, 1)
        # Skip absurdly large "formula" regions (misclassified captions)
        if bh > img_h * 0.5 or bw > img_w * 0.9:
            _apply_formula_reconstruction(block)
            continue

        try:
            region = _expand_formula_region(
                gray, page.blocks, int(x0), int(y0), int(x1), int(y1), img_w, img_h,
            )
            crop = color_image.crop((region[0], region[1], region[2], region[3]))
            raw = _ocr_formula_line(crop, tesseract)
            if raw:
                latex = reconstruct_formula_text(raw)
                if looks_like_latex(latex):
                    original_latex = reconstruct_formula_text(block.text)
                    if not looks_like_latex(original_latex) or _math_marker_count(latex) >= _math_marker_count(original_latex):
                        block.text = latex
                        block.metadata["formula_latex"] = True
                        continue
        except Exception:
            pass  # best-effort — fall through to text-only reconstruction

        _apply_formula_reconstruction(block)


def _apply_formula_reconstruction(block: Block) -> None:
    """Run the rule-based reconstruction over the block's existing text."""
    try:
        latex = reconstruct_formula_text(block.text)
        if looks_like_latex(latex):
            block.text = latex
            block.metadata["formula_latex"] = True
    except Exception:
        pass


def _expand_formula_region(
    gray, blocks: list[Block], x0: int, y0: int, x1: int, y1: int,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a formula bbox to the full text line it belongs to.

    Step 1: union with any block whose vertical range overlaps the band
    ``[y0 - pad, y1 + pad]`` by at least 25% of the formula height.
    Step 2: expand horizontally out to the nearest column gaps (blank
    vertical gutters) within the band.
    """
    pad = max(10, int(0.75 * (y1 - y0)))
    band_y0, band_y1 = y0 - pad, y1 + pad
    fh = max(y1 - y0, 1)

    ux0, ux1 = x0, x1
    uy0, uy1 = y0, y1
    for b in blocks:
        if b is None or not b.bbox:
            continue
        if b.type in ("figure", "table", "header", "footer"):
            continue
        bx0, by0, bx1, by1 = b.bbox
        if bx1 <= ux0 - img_w or bx0 >= ux1 + img_w:
            continue
        lo = max(by0, band_y0)
        hi = min(by1, band_y1)
        overlap = hi - lo
        if overlap >= fh * 0.25:
            ux0 = min(ux0, int(bx0))
            ux1 = max(ux1, int(bx1))
            uy0 = min(uy0, int(by0))
            uy1 = max(uy1, int(by1))

    # Expand x to nearest blank gutters inside the band (column gaps).
    if gray is not None:
        gy0 = max(0, uy0 - 4)
        gy1 = min(img_h, uy1 + 4)
        if gy1 > gy0:
            band = gray[gy0:gy1, :]
            col_dark = (band < 128).sum(axis=0)
            blank = col_dark < max(2, (gy1 - gy0) * 0.05)

            def _nearest_gap(from_x: int, direction: int) -> int:
                # direction: -1 scan left, +1 scan right
                run_start = None
                x = from_x
                while 0 <= x < img_w:
                    if blank[x]:
                        if run_start is None:
                            run_start = x
                    else:
                        if run_start is not None and abs(from_x - run_start) >= 8:
                            return run_start if direction < 0 else run_start
                        run_start = None
                    x += direction
                return run_start if run_start is not None else (0 if direction < 0 else img_w)

            left_gap = _nearest_gap(ux0, -1)
            right_gap = _nearest_gap(ux1, +1)
            if left_gap is not None and left_gap < ux0:
                ux0 = max(0, left_gap)
            if right_gap is not None and right_gap > ux1:
                ux1 = min(img_w, right_gap)

    # Cap the span so we never swallow the whole page.
    if ux1 - ux0 > img_w * 0.8:
        cx = (x0 + x1) / 2
        half = img_w * 0.4
        ux0, ux1 = max(0, int(cx - half)), min(img_w, int(cx + half))
    return ux0, uy0, ux1, uy1


def _ocr_formula_line(crop, tesseract: Path) -> str | None:
    """OCR an (already expanded) formula line region with Tesseract.

    Upscales 3× (minimum 300 px wide) and prefers ``--psm 6``; falls back
    to ``--psm 7`` when the first pass returns nothing.
    """
    import PIL.Image

    if crop.width < 4 or crop.height < 4:
        return None
    w, h = crop.size
    scale = max(3, int(300 / max(w, 1)), int(90 / max(h, 1)))
    if scale > 1:
        crop = crop.resize((w * scale, h * scale), PIL.Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_formula_line_", suffix=".png", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        crop.save(tmp_path)
        env = _tesseract_env(tesseract)
        for psm in ("6", "7"):
            r = subprocess.run(
                [str(tesseract), str(tmp_path), "stdout", "-l", "eng", "--psm", psm],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=env, timeout=25,
            )
            if r.returncode == 0:
                text = r.stdout.strip()
                if text:
                    return " ".join(text.split())
    except Exception:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)
    return None


def _math_marker_count(text: str) -> int:
    """Count LaTeX math markers in *text* (for choosing the better of two
    reconstruction candidates)."""
    return (
        len(re.findall(r"\\[A-Za-z]+", text or ""))
        + len(re.findall(r"_(?=[A-Za-z0-9{])", text or ""))
    )


# ═══════════════════════════════════════════════════════════════════════
#  Formula OCR processing
# ═══════════════════════════════════════════════════════════════════════


def _process_formula_blocks(page: Page, color_image) -> None:
    """Run formula OCR on every ``type="formula"`` block in *page*.

    Crops each formula region from *color_image*, calls
    :func:`~.formula_ocr.recognize_formula`, and updates the block's
    ``text`` field with the recognised LaTeX (or raw text fallback).

    Failures are silently skipped — formula OCR is best-effort.
    """
    import PIL.Image

    for block in page.blocks:
        if block.type != "formula" or not block.bbox:
            continue
        if block.metadata.get("formula_latex"):
            continue

        try:
            x0, y0, x1, y1 = block.bbox
            crop = color_image.crop((int(x0), int(y0), int(x1), int(y1)))
            if crop.width < 4 or crop.height < 4:
                continue
            result = recognize_formula(crop)
            if result:
                block.text = result
                block.metadata["formula_ocr"] = True
        except Exception:
            pass  # best-effort — formula text from OCR remains as-is


# ═══════════════════════════════════════════════════════════════════════
#  Deskew / orientation correction
# ═══════════════════════════════════════════════════════════════════════


def _deskew_image(image_path: Path) -> Path:
    """Detect and correct skew/rotation of a tilted document image.

    Uses Tesseract OSD first, falls back to projection profile analysis.
    Returns the (possibly rotated) image path. If no significant skew is
    detected (< 0.3°), the original path is returned unchanged.
    """
    angle = _detect_skew_tesseract(image_path)
    if angle is None:
        angle = _detect_skew_projection(image_path)

    if angle is None or abs(angle) < 0.3:
        return image_path

    return _rotate_image(image_path, -angle)


def _detect_skew_tesseract(image_path: Path) -> float | None:
    """Detect fine-grained skew angle via Tesseract OSD (--psm 0).

    Returns the 'Orientation in degrees' value, or None on failure.
    """
    tesseract = _find_tesseract()
    if not tesseract:
        return None
    env = _tesseract_env(tesseract)
    cmd = [str(tesseract), str(image_path), "stdout", "--psm", "0"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env, timeout=30)
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            m = re.search(r"Orientation in degrees:\s*(-?[\d.]+)", line)
            if m:
                return float(m.group(1))
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _detect_skew_projection(image_path: Path) -> float | None:
    """Detect skew by maximising horizontal projection variance.

    Tries angles from -5° to +5° in 0.5° steps. For well-aligned text the
    horizontal projection exhibits sharp peaks at text lines, yielding high
    variance. Rotation smears the projection and lowers variance.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_variance = 0.0
    h, w = binary.shape[:2]
    center = (w / 2, h / 2)

    for angle in np.arange(-5.0, 5.1, 0.5):
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, rot_mat, (w, h), borderValue=0)
        projection = np.sum(rotated, axis=1).astype(np.float64)
        variance = float(np.var(projection))
        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    return float(best_angle)


def _rotate_image(image_path: Path, angle: float) -> Path:
    """Rotate image by *angle* degrees, expand canvas, save to temp file."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return image_path
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(img, rot_mat, (new_w, new_h), borderValue=(255, 255, 255))
        tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_deskew_", suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, rotated)
        return Path(tmp.name)
    except ImportError:
        from PIL import Image
        with Image.open(image_path) as img:
            rotated = img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)
            tmp = tempfile.NamedTemporaryFile(prefix="doc_textify_deskew_", suffix=".png", delete=False)
            tmp.close()
            rotated.save(tmp.name)
        return Path(tmp.name)


def _correct_orientation(image_path: Path) -> Path:
    """Detect 90°/180°/270° rotation via Tesseract OSD and correct it.

    Common when photos are taken in portrait mode. Unlike _deskew_image
    which handles sub-degree tilts, this handles cardinal rotations.
    """
    tesseract = _find_tesseract()
    if not tesseract:
        return image_path
    env = _tesseract_env(tesseract)
    cmd = [str(tesseract), str(image_path), "stdout", "--psm", "0"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env, timeout=30)
        if r.returncode != 0:
            return image_path
        for line in r.stdout.splitlines():
            m = re.search(r"Rotate:\s*(\d+)", line)
            if m:
                rotate_angle = int(m.group(1))
                if rotate_angle in (90, 180, 270):
                    return _rotate_image(image_path, -rotate_angle)
        return image_path
    except (subprocess.TimeoutExpired, OSError):
        return image_path


# ═══════════════════════════════════════════════════════════════════════
#  Handwriting-specific preprocessing & OCR
# ═══════════════════════════════════════════════════════════════════════


def _preprocess_handwriting(pil_img) -> "Image.Image":
    """Handwriting-optimized preprocessing pipeline.

    Uses CLAHE + adaptive Gaussian thresholding + denoising when OpenCV
    is available. Falls back to an equalize/median/autocontrast chain
    with Pillow when OpenCV is not installed.
    """
    try:
        import cv2
        import numpy as np

        gray = np.array(pil_img.convert("L"))

        # CLAHE — Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoising for speckle noise common in handwriting photos
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Adaptive Gaussian thresholding — handles uneven lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # block size
            2,   # constant subtracted
        )

        from PIL import Image
        return Image.fromarray(binary)
    except ImportError:
        # Pillow-only fallback
        from PIL import Image, ImageFilter, ImageOps

        ocr = pil_img.convert("L")
        # Approximate CLAHE with equalize
        ocr = ImageOps.equalize(ocr)
        # Median filter for speckle noise
        ocr = ocr.filter(ImageFilter.MedianFilter(size=3))
        # Aggressive contrast as adaptive-threshold stand-in
        ocr = ImageOps.autocontrast(ocr, cutoff=5)
        return ocr


def _do_ocr_tesseract(image_path: Path, *, tesseract: Path, lang: str,
                       psm: int, min_confidence: float) -> list[Block]:
    """Run a single Tesseract OCR pass with the given PSM mode."""
    env = _tesseract_env(tesseract)
    try:
        import pytesseract as pt
        pt.pytesseract.tesseract_cmd = str(tesseract)
        data = pt.image_to_data(str(image_path), lang=lang, config=f"--psm {psm}",
                                output_type=pt.Output.DICT)
        blocks = _tsv_dict_to_blocks(data, min_confidence=min_confidence)
    except (ImportError, Exception):
        blocks = _ocr_tesseract_tsv_fallback(
            image_path, tesseract=tesseract, lang=lang, psm=psm,
            min_confidence=min_confidence, env=env)
    return blocks


def _ocr_handwriting(image_path: Path, *, tesseract: Path, lang: str,
                      min_confidence: float) -> list[Block]:
    """Try multiple PSM modes optimised for handwriting and pick the best result.

    PSM 7  — Treat the image as a single text line.
    PSM 8  — Treat the image as a single word.
    PSM 13 — Raw line. Treat the image as a single text line, bypassing
             hacks that are Tesseract-specific.

    The mode producing the highest average confidence is selected.
    """
    psm_modes = [7, 8, 13]
    best_blocks: list[Block] = []
    best_confidence = 0.0

    for psm in psm_modes:
        try:
            blocks = _do_ocr_tesseract(image_path, tesseract=tesseract, lang=lang,
                                        psm=psm, min_confidence=min_confidence)
        except Exception:
            continue
        if not blocks:
            continue
        avg_conf = sum(b.confidence or 0 for b in blocks) / len(blocks)
        if avg_conf > best_confidence:
            best_confidence = avg_conf
            best_blocks = blocks

    return best_blocks


def _ocr_easyocr(image_path: Path, *, lang: str = "en") -> list[Block]:
    """Use EasyOCR as an alternative handwriting engine.

    EasyOCR often outperforms Tesseract on Chinese handwriting.
    This is an OPTIONAL dependency — returns [] if not installed.
    """
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return []

    # Tesseract lang → EasyOCR lang mapping
    lang_map: dict[str, list[str]] = {
        "eng": ["en"],
        "chi_sim": ["ch_sim"],
        "chi_sim+eng": ["ch_sim", "en"],
        "chi_tra": ["ch_tra"],
        "jpn": ["ja"],
        "kor": ["ko"],
    }
    easy_langs = lang_map.get(lang, ["en"])

    try:
        reader = easyocr.Reader(easy_langs, gpu=False)  # type: ignore[attr-defined]
        results = reader.readtext(str(image_path))  # type: ignore[attr-defined]
    except Exception:
        return []

    blocks: list[Block] = []
    for bbox, text, conf in results:
        if not text or not text.strip():
            continue
        if conf < 0.3:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append(Block(
            type="paragraph",
            text=text.strip(),
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            confidence=conf * 100.0,
            engine="easyocr",
        ))

    return blocks


def _merge_ocr_results(
    tesseract_blocks: list[Block], easyocr_blocks: list[Block],
) -> list[Block]:
    """Merge Tesseract and EasyOCR results, deduplicating overlaps.

    When two blocks overlap (IoU > 0.5 against the smaller box), keep the
    one with higher confidence.
    """
    if not easyocr_blocks:
        return tesseract_blocks
    if not tesseract_blocks:
        return easyocr_blocks

    merged = list(tesseract_blocks)

    for eb in easyocr_blocks:
        if not eb.bbox:
            merged.append(eb)
            continue

        best_iou = 0.0
        best_idx = -1
        for i, tb in enumerate(merged):
            if not tb.bbox:
                continue
            x1 = max(eb.bbox[0], tb.bbox[0])
            y1 = max(eb.bbox[1], tb.bbox[1])
            x2 = min(eb.bbox[2], tb.bbox[2])
            y2 = min(eb.bbox[3], tb.bbox[3])
            if x1 >= x2 or y1 >= y2:
                continue
            inter_area = (x2 - x1) * (y2 - y1)
            eb_area = (eb.bbox[2] - eb.bbox[0]) * (eb.bbox[3] - eb.bbox[1])
            tb_area = (tb.bbox[2] - tb.bbox[0]) * (tb.bbox[3] - tb.bbox[1])
            iou = inter_area / min(eb_area, tb_area) if min(eb_area, tb_area) > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou > 0.5:
            # Overlap: keep higher-confidence block
            if (eb.confidence or 0) > (merged[best_idx].confidence or 0):
                merged[best_idx] = eb
        else:
            merged.append(eb)

    return merged
