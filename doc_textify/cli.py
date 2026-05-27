from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractors import extract_document
from .renderers import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-textify",
        description="Convert PDFs/images into Markdown/TXT plus JSON without vision LLMs.",
    )
    parser.add_argument("input", type=Path, help="PDF or image file to convert.")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Directory for generated files.")
    parser.add_argument(
        "--format",
        choices=["md", "txt", "both"],
        default="both",
        help="Primary text output format. JSON sidecar is always written.",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language code, for example eng or chi_sim+eng.")
    parser.add_argument("--force-ocr", action="store_true", help="Skip native PDF text extraction and use OCR path.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=45.0,
        help="Minimum OCR word confidence accepted from Tesseract TSV output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        document = extract_document(
            args.input,
            lang=args.lang,
            force_ocr=args.force_ocr,
            min_confidence=args.min_confidence,
        )
        written = write_outputs(document, args.out, output_format=args.format)
    except Exception as exc:  # noqa: BLE001 - CLI should print a concise error
        print(f"doc-textify: error: {exc}", file=sys.stderr)
        return 1

    for label, path in written.items():
        print(f"{label}: {path}")
    for warning in document.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
