# doc-textify

`doc-textify` is a no-vision-LLM document compiler: it turns PDFs, screenshots, and photographed document images into Markdown/TXT plus JSON metadata for downstream text-only AI systems.

The project intentionally does not call GPT/Gemini/Claude/Qwen-VL or other visual large language models. It uses deterministic extraction where possible and optional dedicated OCR engines where installed.

## What works in this MVP

- Native/digital PDFs: extracts embedded text page by page through `pypdf` when available.
- Images and screenshots: preprocesses through Pillow when available, then uses the local `tesseract` executable when installed.
- Scanned PDFs: represented as explicit page placeholders unless a PDF renderer/OCR backend is added.
- Outputs:
  - `.md` with page markers, headings, paragraphs, tables/placeholders, figures/placeholders, and uncertainty markers.
  - `.txt` simplified from the same document model.
  - `.json` sidecar with pages, blocks, bounding boxes, confidence, engine, and warnings.

## Quick start

```powershell
python -m pip install -e .[pdf,image,dev]
python -m doc_textify input.pdf --out outputs --format both
python -m doc_textify screenshot.png --out outputs --format md --lang chi_sim+eng
```

For OCR on images, install Tesseract separately and make sure `tesseract.exe` is on `PATH`.

```powershell
doc-textify photo.jpg --out outputs --format both --lang chi_sim+eng
```

## Design boundary

This tool is meant to solve the “text-only AI reads images/PDFs” problem by adding a preprocessing layer:

```text
PDF/Image -> doc-textify -> Markdown/TXT + JSON -> text-only AI
```

That is not a paradox because the target AI never sees pixels. A dedicated OCR/document parsing tool converts visual structure into text first.

## Output contract

Markdown uses a conservative, AI-readable contract:

```markdown
# Page 1

## Detected or inferred title

Paragraph text in reading order.

| A | B |
| --- | --- |
| 1 | 2 |

[Figure: page=1, bbox=10,20,100,80, caption=not detected]

[uncertain: low-confidence text]
```

The JSON sidecar preserves extra structure that plain Markdown cannot carry well:

- page dimensions
- block type
- bounding box
- confidence
- extraction engine
- source warnings

## Current limitations

- Complex layout recovery is heuristic in this MVP.
- Scanned PDF OCR needs a page renderer such as PyMuPDF, Poppler, or another backend added later.
- Image content is not described without an OCR result or nearby caption text.
- Handwriting, formulas, and complex tables require specialized OCR/layout engines.

## Recommended next stage

Add optional adapters for Docling, MinerU, Marker, PaddleOCR/PP-Structure, or Azure/Google/AWS Document AI, then normalize all outputs into the same internal block model used here.
