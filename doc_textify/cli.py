from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

from .extractors import extract_document, IMAGE_EXTENSIONS
from .renderers import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-textify",
        description="Convert PDFs/images into Markdown/TXT plus JSON without vision LLMs.",
    )
    parser.add_argument(
        "input", nargs="+", type=Path,
        help="PDF or image file(s) to convert. Supports glob patterns and directories.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs"),
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--format",
        choices=["md", "txt", "llm", "llm-v1", "deepseek", "both", "all"],
        default="both",
        help=(
            "Primary text output format. JSON sidecar is always written. "
            "llm: compact v2 AI-readable text. "
            "llm-v1: legacy v1 format. "
            "deepseek: token-optimized format for DeepSeek models."
        ),
    )
    parser.add_argument(
        "--lang", default="eng",
        help="Tesseract language code, for example eng or chi_sim+eng.",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Skip native PDF text extraction and use OCR path.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=45.0,
        help="Minimum OCR word confidence accepted from Tesseract TSV output.",
    )
    parser.add_argument(
        "--chart-colors",
        default="auto",
        help=(
            "Chart colours to extract. Comma-separated list, e.g. red,blue,green. "
            "Use 'auto' for automatic colour detection (default). "
            "Available: red, red_line, blue, blue_dark, green, green_dark, "
            "cyan, yellow, orange, magenta, purple, black, white."
        ),
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively scan input directories.",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=0,
        help="Number of parallel workers (0 = auto = CPU count).",
    )
    parser.add_argument(
        "--no-batch", action="store_true",
        help="Process files sequentially even when multiple inputs are given.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files whose output already exists in the output directory. "
             "Resume interrupted batch jobs.",
    )
    parser.add_argument(
        "--per-file-dir", action="store_true",
        help="Create a separate output subdirectory for each input file "
             "(named after the input file's stem).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed without actually processing them.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Process at most N files in this run. "
             "Combine with --resume for incremental chunked processing.",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Append a custom suffix to output filenames "
             "(e.g. '_v2' produces 'input_v2.md', 'input_v2.json').",
    )
    parser.add_argument(
        "--rag-ready", action="store_true",
        help=(
            "Enable RAG-optimized output with semantic chunk markers "
            "(<!-- CHUNK -->), context windows around data blocks, "
            "page metadata, and consistent title hierarchy."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000,
        metavar="N",
        help="Maximum characters per semantic chunk for --rag-ready (default: 2000).",
    )
    parser.add_argument(
        "--handwriting", action="store_true",
        help="Enable handwriting-optimised OCR "
             "(adaptive CLAHE preprocessing + multi-PSM 7/8/13 + EasyOCR fallback).",
    )
    parser.add_argument(
        "--deskew", action="store_true", default=True,
        help="Enable automatic deskew / rotation correction for tilted photos (default).",
    )
    parser.add_argument(
        "--no-deskew", action="store_false", dest="deskew",
        help="Disable automatic deskew.",
    )
    parser.add_argument(
        "--formula-ocr", action="store_true",
        help="Enable formula OCR (LaTeX recognition via pix2tex or Tesseract eq). "
             "Disabled by default; may require model download on first use.",
    )
    return parser


# ── CLI entry point ─────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve colour parameter
    chart_colors = None if args.chart_colors == "auto" else [
        c.strip() for c in args.chart_colors.split(",") if c.strip()
    ]

    # Collect all input files (expand globs and directories)
    files = _collect_input_files(args.input, recursive=args.recursive)

    if not files:
        print("doc-textify: error: no supported files found in input.", file=sys.stderr)
        return 1

    # ── --resume (apply before dry-run so they compose) ──
    if args.resume:
        skipped, files = _apply_resume_filter(
            files, args.out, args.output_suffix, args.per_file_dir,
        )
        if skipped > 0:
            print(
                f"doc-textify: skipping {skipped} already-processed file(s).",
                file=sys.stderr,
            )
        if not files:
            print(
                "doc-textify: all files already processed -- nothing to do.",
                file=sys.stderr,
            )
            return 0

    # ── --batch-size ──
    if args.batch_size > 0 and len(files) > args.batch_size:
        print(
            f"doc-textify: limiting to {args.batch_size} of {len(files)} file(s).",
            file=sys.stderr,
        )
        files = files[: args.batch_size]

    # ── --dry-run (after resume + batch so it reflects the final file set) ──
    if args.dry_run:
        _dry_run_report(files)
        return 0

    # Single file: direct processing (original path, no batch overhead)
    if len(files) == 1:
        return _process_one(
            files[0], args.out, args.format, args.lang,
            args.force_ocr, args.min_confidence, chart_colors,
            handwriting=args.handwriting, deskew=args.deskew,
            formula_ocr=args.formula_ocr,
            per_file_dir=args.per_file_dir, output_suffix=args.output_suffix,
            rag_ready=args.rag_ready, chunk_size=args.chunk_size,
        )

    # Multi-file: batch mode
    print(f"doc-textify: {len(files)} file(s) to process.", file=sys.stderr)

    if args.no_batch:
        return _process_sequential(
            files, args.out, args.format, args.lang,
            args.force_ocr, args.min_confidence, chart_colors,
            handwriting=args.handwriting, deskew=args.deskew,
            formula_ocr=args.formula_ocr,
            per_file_dir=args.per_file_dir, output_suffix=args.output_suffix,
            rag_ready=args.rag_ready, chunk_size=args.chunk_size,
        )
    else:
        workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        return _process_batch(
            files, args.out, args.format, args.lang,
            args.force_ocr, args.min_confidence, chart_colors,
            workers,
            handwriting=args.handwriting, deskew=args.deskew,
            formula_ocr=args.formula_ocr,
            per_file_dir=args.per_file_dir, output_suffix=args.output_suffix,
            rag_ready=args.rag_ready, chunk_size=args.chunk_size,
        )


# ── File collection ────────────────────────────────────────────────

def _collect_input_files(inputs: list[Path], *, recursive: bool) -> list[Path]:
    """Expand globs, directories and input lists into a flat deduped file list."""
    files: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        p = raw.expanduser()
        if "*" in str(p) or "?" in str(p):
            # Glob pattern
            expanded = sorted(p.parent.glob(p.name))
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            expanded = sorted(p.glob(pattern))
        elif p.exists():
            expanded = [p.resolve()]
        else:
            print(f"doc-textify: warning: input not found: {p}", file=sys.stderr)
            continue

        for f in expanded:
            f = f.resolve()
            if f.is_file() and f not in seen:
                suffix = f.suffix.lower()
                if suffix == ".pdf" or suffix in IMAGE_EXTENSIONS:
                    files.append(f)
                    seen.add(f)

    return files


# ── Dry-run / resume helpers ───────────────────────────────────────

def _dry_run_report(files: list[Path]) -> None:
    """Print the files that would be processed and exit."""
    print(f"doc-textify: {len(files)} file(s) would be processed:\n")
    for f in files:
        print(f"  {f}")
    print()


def _apply_resume_filter(
    files: list[Path],
    output_dir: Path,
    suffix: str,
    per_file_dir: bool,
) -> tuple[int, list[Path]]:
    """Filter out files whose output already exists.

    Returns (skipped_count, remaining_files).
    """
    remaining: list[Path] = []
    skipped = 0
    for f in files:
        if _is_file_done(f, output_dir, suffix, per_file_dir):
            skipped += 1
        else:
            remaining.append(f)
    return skipped, remaining


def _is_file_done(
    source: Path, output_dir: Path, suffix: str, per_file_dir: bool,
) -> bool:
    """Check if output files already exist for the given source.

    Checks for the JSON sidecar which is always written regardless of format.
    """
    stem = source.stem + suffix
    if per_file_dir:
        check_dir = output_dir / source.stem
    else:
        check_dir = output_dir
    return (check_dir / f"{stem}.json").exists()


# ── Output helpers ─────────────────────────────────────────────────

def _resolve_output_dir(
    output_dir: Path, source: Path, per_file_dir: bool,
) -> Path:
    """Determine the actual output directory for a given source file."""
    if per_file_dir:
        return output_dir / source.stem
    return output_dir


def _rename_with_suffix(written: dict[str, Path], suffix: str) -> dict[str, Path]:
    """Rename written output files to include the suffix. Returns updated dict."""
    renamed: dict[str, Path] = {}
    for label, path in written.items():
        new_path = path.with_stem(path.stem + suffix)
        path.rename(new_path)
        renamed[label] = new_path
    return renamed


def _rename_outputs(output_dir: Path, stem: str, suffix: str) -> None:
    """Rename all output files in *output_dir* that start with *stem* to include *suffix*."""
    for old_path in sorted(output_dir.glob(f"{stem}.*")):
        new_name = old_path.name.replace(stem, stem + suffix, 1)
        old_path.rename(old_path.with_name(new_name))


def _format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds <= 0 or seconds != seconds:  # NaN guard
        return "--:--"
    s = round(seconds)
    hours, rem = divmod(s, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ── Processing modes ───────────────────────────────────────────────

def _process_one(
    source: Path, out: Path, fmt: str, lang: str,
    force_ocr: bool, min_confidence: float, chart_colors: list[str] | None,
    handwriting: bool = False, deskew: bool = True,
    formula_ocr: bool = False,
    per_file_dir: bool = False, output_suffix: str = "",
    rag_ready: bool = False, chunk_size: int = 2000,
) -> int:
    """Process a single file (original behaviour)."""
    try:
        document = extract_document(
            source,
            lang=lang,
            force_ocr=force_ocr,
            min_confidence=min_confidence,
            handwriting=handwriting,
            deskew=deskew,
            chart_colors=chart_colors,
            formula_ocr=formula_ocr,
        )
        actual_out = _resolve_output_dir(out, source, per_file_dir)
        written = write_outputs(
            document, actual_out, output_format=fmt,
            rag_ready=rag_ready, chunk_size=chunk_size,
        )
        if output_suffix:
            written = _rename_with_suffix(written, output_suffix)
    except Exception as exc:
        print(f"doc-textify: error: {exc}", file=sys.stderr)
        return 1

    for label, path in written.items():
        print(f"{label}: {path}")
    for warning in document.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _process_sequential(
    files: list[Path], out: Path, fmt: str, lang: str,
    force_ocr: bool, min_confidence: float, chart_colors: list[str] | None,
    handwriting: bool = False, deskew: bool = True,
    formula_ocr: bool = False,
    per_file_dir: bool = False, output_suffix: str = "",
    rag_ready: bool = False, chunk_size: int = 2000,
) -> int:
    """Process files one by one with progress, file durations, and ETA."""
    ok = 0
    fail = 0
    t0 = time.monotonic()
    total = len(files)

    for i, f in enumerate(files, 1):
        # ── ETA ──
        completed = ok + fail
        if completed > 0:
            elapsed = time.monotonic() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0.0
        else:
            eta = 0.0

        print(
            f"[{i}/{total}] {f.name}  (ETA {_format_duration(eta)})",
            file=sys.stderr, end=" ",
        )

        file_t0 = time.monotonic()
        try:
            document = extract_document(
                f,
                lang=lang,
                force_ocr=force_ocr,
                min_confidence=min_confidence,
                handwriting=handwriting,
                deskew=deskew,
                chart_colors=chart_colors,
                formula_ocr=formula_ocr,
            )
            actual_out = _resolve_output_dir(out, f, per_file_dir)
            write_outputs(
                document, actual_out, output_format=fmt,
                rag_ready=rag_ready, chunk_size=chunk_size,
            )
            if output_suffix:
                _rename_outputs(actual_out, document.source.stem, output_suffix)
            dt = time.monotonic() - file_t0
            print(f"\u2713 ({dt:.1f}s)", file=sys.stderr)
            ok += 1
        except Exception as exc:
            print(f"\u2717 ({exc})", file=sys.stderr)
            fail += 1

    elapsed = time.monotonic() - t0
    _print_summary(ok, fail, elapsed)
    return 0 if fail == 0 else 1


def _process_batch(
    files: list[Path], out: Path, fmt: str, lang: str,
    force_ocr: bool, min_confidence: float, chart_colors: list[str] | None,
    workers: int,
    handwriting: bool = False, deskew: bool = True,
    formula_ocr: bool = False,
    per_file_dir: bool = False, output_suffix: str = "",
    rag_ready: bool = False, chunk_size: int = 2000,
) -> int:
    """Process files in parallel using multiprocessing.

    Shows per-file completion (name + status) and ETA.
    """
    import multiprocessing as mp

    total = len(files)
    t0 = time.monotonic()

    # Build task list
    tasks = [
        (
            f,
            _resolve_output_dir(out, f, per_file_dir),
            fmt, lang, force_ocr, min_confidence, chart_colors,
            handwriting, deskew, formula_ocr, output_suffix,
            rag_ready, chunk_size,
        )
        for f in files
    ]

    results: list[tuple[str, int, list[str]]] = []
    with mp.Pool(processes=min(workers, total)) as pool:
        for res in pool.imap_unordered(_batch_worker, tasks):
            results.append(res)
            ok_count = sum(1 for r in results if r[1] == 0)
            fail_count = sum(1 for r in results if r[1] != 0)
            completed = ok_count + fail_count

            # ── ETA ──
            elapsed = time.monotonic() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0.0

            name = res[0]
            status = "\u2713" if res[1] == 0 else "\u2717"

            # Overwrite the line each time (carriage-return + padding)
            line = (
                f"[{completed}/{total}] {status} {name}  "
                f"(ETA {_format_duration(eta)})"
            )
            # Pad to clear previous longer line
            print(f"\r{line:<100s}", end="", file=sys.stderr)

    print(file=sys.stderr)  # newline after progress

    # Print per-file warnings
    for name, rc, warnings in sorted(results, key=lambda r: r[0]):
        for w in warnings:
            print(f"warning [{name}]: {w}", file=sys.stderr)

    ok = sum(1 for _, rc, _ in results if rc == 0)
    fail = sum(1 for _, rc, _ in results if rc != 0)
    elapsed = time.monotonic() - t0
    _print_summary(ok, fail, elapsed)
    return 0 if fail == 0 else 1


def _batch_worker(task: tuple) -> tuple[str, int, list[str]]:
    """Worker function for multiprocessing pool. Returns (name, rc, warnings)."""
    (
        source, out, fmt, lang, force_ocr, min_confidence, chart_colors,
        handwriting, deskew, formula_ocr, output_suffix,
        rag_ready, chunk_size,
    ) = task
    try:
        document = extract_document(
            source,
            lang=lang,
            force_ocr=force_ocr,
            min_confidence=min_confidence,
            handwriting=handwriting,
            deskew=deskew,
            chart_colors=chart_colors,
            formula_ocr=formula_ocr,
        )
        write_outputs(
            document, out, output_format=fmt,
            rag_ready=rag_ready, chunk_size=chunk_size,
        )
        if output_suffix:
            _rename_outputs(out, document.source.stem, output_suffix)
        return (source.name, 0, [w for w in document.warnings])
    except Exception as exc:
        return (source.name, 1, [str(exc)])


def _print_summary(ok: int, fail: int, elapsed: float) -> None:
    rate = (ok + fail) / elapsed if elapsed > 0 else 0
    print(
        f"\ndoc-textify: {ok} succeeded, {fail} failed "
        f"({elapsed:.1f}s, {rate:.1f} files/s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
