#!/usr/bin/env python3
"""
doc-textify Benchmarks — quality comparison framework.

Compares doc-textify JSON output against reference outputs (from DeepSeek-OCR
or human annotation) and generates actionable gap analysis.

Usage:
    doc-textify-bench --dataset benchmarks/dataset/ --output-dir outputs/
    doc-textify-bench --compare --actual outputs/report.json --expected benchmarks/dataset/expected/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure-Python Levenshtein distance (no ML dependencies)
# ---------------------------------------------------------------------------


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio ∈ [0, 1].

    Returns 1.0 for identical strings, 0.0 for maximally different strings.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Optimized O(min(m,n)) space implementation
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr

    distance = prev[n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocumentComparison:
    doc_id: str
    actual_path: str
    expected_path: str
    overall_score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkResult:
    name: str
    version: str
    timestamp: str
    total_documents: int
    completed: int
    failed: int
    overall_score: float
    comparisons: list[DocumentComparison] = field(default_factory=list)
    gap_analysis: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document comparison
# ---------------------------------------------------------------------------


def compare_document(
    actual_path: str | Path,
    expected_path: str | Path,
) -> DocumentComparison:
    """Compare a doc-textify output against a reference/expected output.

    Automatically detects whether the expected file uses:
    1. Chart evaluation format (has ``required_terms`` / ``panels`` keys)
       → delegates to ``evaluate_textification()``.
    2. Document output format (has ``pages`` with ``blocks``)
       → runs general document comparison.

    Args:
        actual_path: Path to the doc-textify JSON output (Document.to_dict()).
        expected_path: Path to the reference expected JSON.

    Returns:
        DocumentComparison with per-metric scores.
    """
    actual_path = Path(actual_path)
    expected_path = Path(expected_path)
    doc_id = expected_path.stem.replace(".expected", "")

    try:
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        return DocumentComparison(
            doc_id=doc_id,
            actual_path=str(actual_path),
            expected_path=str(expected_path),
            overall_score=0.0,
            error=str(exc),
        )

    # Detect expected format
    if "required_terms" in expected or "panels" in expected:
        return _compare_as_chart_eval(doc_id, actual_path, expected_path, actual, expected)

    if "pages" in expected:
        return _compare_as_document(doc_id, actual_path, expected_path, actual, expected)

    # Unknown format — fallback to text-only comparison
    return _compare_as_text_only(doc_id, actual_path, expected_path, actual, expected)


def _compare_as_chart_eval(
    doc_id: str,
    actual_path: Path,
    expected_path: Path,
    actual: dict,
    expected: dict,
) -> DocumentComparison:
    """Use the existing evaluate_textification() for chart documents."""
    try:
        from doc_textify.evaluation import evaluate_textification

        # evaluate_textification expects Path objects
        result, _report = evaluate_textification(
            actual_json=actual_path,
            expected_json=expected_path,
        )
        return DocumentComparison(
            doc_id=doc_id,
            actual_path=str(actual_path),
            expected_path=str(expected_path),
            overall_score=result["overall_score"],
            metrics={
                m["name"]: {"score": m["score"], "weight": m["weight"], "detail": m["detail"]}
                for m in result["metrics"]
            },
        )
    except ImportError:
        return DocumentComparison(
            doc_id=doc_id,
            actual_path=str(actual_path),
            expected_path=str(expected_path),
            overall_score=0.0,
            error="Cannot import doc_textify.evaluation — ensure project is on PYTHONPATH.",
        )


def _compare_as_document(
    doc_id: str,
    actual_path: Path,
    expected_path: Path,
    actual: dict,
    expected: dict,
) -> DocumentComparison:
    """General document-level comparison for output-format expected files."""
    metrics: dict[str, Any] = {}
    weights: dict[str, float] = {}

    # --- 1. Text similarity (Levenshtein on concatenated block text) ---
    actual_text = _concat_block_text(actual)
    expected_text = _concat_block_text(expected)
    text_score = levenshtein_ratio(actual_text, expected_text)
    metrics["text_similarity"] = {
        "score": round(text_score, 4),
        "weight": 0.25,
        "detail": f"Levenshtein ratio on {len(actual_text)} vs {len(expected_text)} chars of block text.",
    }

    # --- 2. Block structure ---
    actual_blocks = _extract_blocks(actual)
    expected_blocks = _extract_blocks(expected)

    # Block count match
    block_count_score = _block_count_ratio(actual_blocks, expected_blocks)
    metrics["block_count"] = {
        "score": round(block_count_score, 4),
        "weight": 0.15,
        "detail": f"Actual: {len(actual_blocks)} blocks, Expected: {len(expected_blocks)} blocks.",
    }

    # Block type accuracy
    type_score = _block_type_accuracy(actual_blocks, expected_blocks)
    metrics["block_type_accuracy"] = {
        "score": round(type_score, 4),
        "weight": 0.10,
        "detail": f"Block type classification accuracy.",
    }

    # --- 3. Table detection ---
    table_score, table_detail = _table_detection_score(actual_blocks, expected_blocks)
    metrics["table_detection"] = {
        "score": round(table_score, 4),
        "weight": 0.20,
        "detail": table_detail,
    }

    # --- 4. Chart/Fragment presence ---
    chart_score, chart_detail = _chart_presence_score(actual_blocks, expected_blocks)
    metrics["chart_presence"] = {
        "score": round(chart_score, 4),
        "weight": 0.15,
        "detail": chart_detail,
    }

    # --- 5. Content completeness (warnings/placeholders) ---
    completeness_score = _completeness_score(actual)
    metrics["content_completeness"] = {
        "score": round(completeness_score, 4),
        "weight": 0.15,
        "detail": _completeness_detail(actual),
    }

    # Compute weighted overall
    total_weight = sum(m["weight"] for m in metrics.values())
    overall = (
        sum(m["score"] * m["weight"] for m in metrics.values()) / total_weight
        if total_weight > 0
        else 0.0
    )

    return DocumentComparison(
        doc_id=doc_id,
        actual_path=str(actual_path),
        expected_path=str(expected_path),
        overall_score=round(overall, 4),
        metrics=metrics,
    )


def _compare_as_text_only(
    doc_id: str,
    actual_path: Path,
    expected_path: Path,
    actual: dict,
    expected: dict,
) -> DocumentComparison:
    """Text-only fallback comparison."""
    actual_text = json.dumps(actual, ensure_ascii=False, sort_keys=True)
    expected_text = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    score = levenshtein_ratio(actual_text, expected_text)

    return DocumentComparison(
        doc_id=doc_id,
        actual_path=str(actual_path),
        expected_path=str(expected_path),
        overall_score=round(score, 4),
        metrics={
            "text_similarity": {
                "score": round(score, 4),
                "weight": 1.0,
                "detail": "Full JSON Levenshtein ratio (unknown expected format).",
            }
        },
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _concat_block_text(doc: dict) -> str:
    """Extract and concatenate all block text from document JSON."""
    chunks: list[str] = []
    chunks.extend(str(w) for w in doc.get("warnings", []))
    for page in doc.get("pages", []):
        chunks.extend(str(w) for w in page.get("warnings", []))
        for block in page.get("blocks", []):
            chunks.append(str(block.get("type", "")))
            chunks.append(str(block.get("text", "")))
            meta = block.get("metadata", {})
            if isinstance(meta, dict):
                chunks.append(json.dumps(meta, ensure_ascii=False))
            if block.get("confidence") is not None:
                chunks.append(str(block["confidence"]))
    return "\n".join(chunks).lower()


def _extract_blocks(doc: dict) -> list[dict[str, Any]]:
    """Extract all blocks from document JSON as a flat list."""
    blocks: list[dict[str, Any]] = []
    for page in doc.get("pages", []):
        blocks.extend(page.get("blocks", []))
    return blocks


def _block_count_ratio(actual: list[dict], expected: list[dict]) -> float:
    """Score based on how close the block count is to expected."""
    if not expected:
        return 1.0 if not actual else 0.0
    ratio = min(len(actual), len(expected)) / max(len(actual), len(expected))
    return ratio


def _block_type_accuracy(actual: list[dict], expected: list[dict]) -> float:
    """Compute how many blocks have the correct type classification.

    Uses greedy matching by text content to pair actual→expected blocks,
    then compares their types.
    """
    if not expected:
        return 1.0 if not actual else 0.0

    expected_texts = [b.get("text", "") for b in expected]
    matched = set()
    type_matches = 0
    total_matched = 0

    for a_block in actual:
        a_text = a_block.get("text", "")
        best_idx = -1
        best_overlap = -1

        for j, e_text in enumerate(expected_texts):
            if j in matched:
                continue
            overlap = _text_overlap(a_text, e_text)
            if overlap > best_overlap and overlap > 0:
                best_overlap = overlap
                best_idx = j

        if best_idx >= 0:
            matched.add(best_idx)
            total_matched += 1
            if a_block.get("type") == expected[best_idx].get("type"):
                type_matches += 1

    if total_matched == 0:
        return 0.0
    return type_matches / max(total_matched, len(expected))


def _text_overlap(t1: str, t2: str) -> float:
    """Compute character-level overlap between two text strings."""
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    set1 = set(t1.lower())
    set2 = set(t2.lower())
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _table_detection_score(
    actual: list[dict], expected: list[dict]
) -> tuple[float, str]:
    """Score table detection quality.

    Checks:
    - Were tables found in actual output when expected lists them?
    - Do table cell/row counts approximately match?
    """
    actual_tables = [b for b in actual if b.get("type") == "table"]
    expected_tables = [b for b in expected if b.get("type") == "table"]

    if not expected_tables:
        if not actual_tables:
            return 1.0, "No tables expected, none found."
        return 0.5, "No tables expected but tables were found (false positive)."

    if not actual_tables:
        return 0.0, f"Expected {len(expected_tables)} table(s), found 0."

    # Compare each expected table to the best-matching actual table
    best_scores: list[float] = []
    for e_table in expected_tables:
        best = 0.0
        e_rows = e_table.get("metadata", {}).get("rows", 0)
        e_cols = e_table.get("metadata", {}).get("cols", 0)
        for a_table in actual_tables:
            a_rows = a_table.get("metadata", {}).get("rows", 0)
            a_cols = a_table.get("metadata", {}).get("cols", 0)
            if e_rows > 0 and e_cols > 0 and a_rows > 0 and a_cols > 0:
                row_ratio = min(e_rows, a_rows) / max(e_rows, a_rows)
                col_ratio = min(e_cols, a_cols) / max(e_cols, a_cols)
                score = (row_ratio + col_ratio) / 2
            else:
                # Fallback: text overlap
                score = _text_overlap(
                    a_table.get("text", ""), e_table.get("text", "")
                )
            if score > best:
                best = score
        best_scores.append(best)

    avg_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
    detail = (
        f"Found {len(actual_tables)}/{len(expected_tables)} tables. "
        f"Cell match score: {avg_score:.2%}."
    )
    return avg_score, detail


def _chart_presence_score(
    actual: list[dict], expected: list[dict]
) -> tuple[float, str]:
    """Check whether chart-related blocks were detected."""
    actual_charts = [b for b in actual if b.get("type") == "figure"]
    expected_charts = [b for b in expected if b.get("type") == "figure"]

    if not expected_charts:
        if not actual_charts:
            return 1.0, "No charts expected, none found."
        return 0.7, "No charts expected but chart blocks found."

    if not actual_charts:
        return 0.0, f"Expected {len(expected_charts)} chart(s), found 0."

    # Check how many expected charts have some matching actual chart
    matched = 0
    for e_chart in expected_charts:
        e_text = e_chart.get("text", "")
        for a_chart in actual_charts:
            if _text_overlap(e_text, a_chart.get("text", "")) > 0.2:
                matched += 1
                break

    score = matched / max(len(expected_charts), 1)
    detail = f"Matched {matched}/{len(expected_charts)} expected chart blocks."
    return score, detail


def _completeness_score(actual: dict) -> float:
    """Penalize warnings and placeholder blocks."""
    has_warnings = bool(actual.get("warnings"))
    for page in actual.get("pages", []):
        if page.get("warnings"):
            has_warnings = True
            break

    if has_warnings:
        return 0.0

    # Check for placeholder blocks
    for page in actual.get("pages", []):
        for block in page.get("blocks", []):
            if block.get("type") in ("placeholder", "uncertain"):
                return 0.3
            text = str(block.get("text", "")).lower()
            if "placeholder" in text or "not performed" in text or "unavailable" in text:
                return 0.3

    return 1.0


def _completeness_detail(actual: dict) -> str:
    """Human-readable completeness description."""
    warnings = list(actual.get("warnings", []))
    for page in actual.get("pages", []):
        warnings.extend(page.get("warnings", []))
    if warnings:
        return f"Warnings present: {warnings[:3]}"
    return "No warnings or placeholders detected."


# ---------------------------------------------------------------------------
# Batch benchmarking
# ---------------------------------------------------------------------------


def benchmark_dataset(
    manifest_path: str | Path,
    output_dir: str | Path = "outputs",
    *,
    run_doc_textify: bool = False,
) -> BenchmarkResult:
    """Run a full benchmark suite over every document in the manifest.

    If ``run_doc_textify`` is True, each document is first processed through
    doc-textify to generate actual output. Otherwise, actual outputs are
    expected to already exist in ``output_dir``.

    Args:
        manifest_path: Path to the benchmark ``manifest.json``.
        output_dir: Directory where doc-textify outputs are / will be stored.
        run_doc_textify: If True, invoke ``doc-textify`` CLI for each document.

    Returns:
        BenchmarkResult with per-document comparisons and aggregate scores.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    dataset_dir = manifest_path.parent
    expected_dir = dataset_dir / "expected"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparisons: list[DocumentComparison] = []
    completed = 0
    failed = 0

    for doc in manifest.get("documents", []):
        doc_id = doc["id"]
        doc_path = doc.get("path", "")

        # Resolve expected file
        expected_file = expected_dir / f"{doc_id}.expected.json"
        if not expected_file.exists():
            # Try to find by stem
            candidates = list(expected_dir.glob(f"{doc_id}.*"))
            expected_file = candidates[0] if candidates else expected_file

        # Resolve actual file
        actual_file = output_dir / f"{doc_id}.json"
        if not actual_file.exists():
            # Try to find any JSON with matching prefix in output_dir
            candidates = list(output_dir.glob(f"{doc_id}*.json"))
            actual_file = candidates[0] if candidates else actual_file

        # Optionally run doc-textify
        if run_doc_textify and not actual_file.exists():
            doc_full_path = (dataset_dir / doc_path).resolve()
            try:
                _run_doc_textify(
                    doc_full_path,
                    actual_file,
                    force_ocr=bool(doc.get("ocr_required", False)),
                    lang=str(doc.get("ocr_lang", "eng")),
                )
            except Exception as exc:
                comparisons.append(
                    DocumentComparison(
                        doc_id=doc_id,
                        actual_path=str(actual_file),
                        expected_path=str(expected_file),
                        overall_score=0.0,
                        error=f"doc-textify failed: {exc}",
                    )
                )
                failed += 1
                continue

        if not actual_file.exists():
            comparisons.append(
                DocumentComparison(
                    doc_id=doc_id,
                    actual_path=str(actual_file),
                    expected_path=str(expected_file),
                    overall_score=0.0,
                    error=f"Actual output not found: {actual_file}",
                )
            )
            failed += 1
            continue

        if not expected_file.exists():
            comparisons.append(
                DocumentComparison(
                    doc_id=doc_id,
                    actual_path=str(actual_file),
                    expected_path=str(expected_file),
                    overall_score=0.0,
                    error=f"Expected output not found: {expected_file}",
                )
            )
            failed += 1
            continue

        try:
            comp = compare_document(actual_file, expected_file)
            comparisons.append(comp)
            completed += 1
        except Exception as exc:
            comparisons.append(
                DocumentComparison(
                    doc_id=doc_id,
                    actual_path=str(actual_file),
                    expected_path=str(expected_file),
                    overall_score=0.0,
                    error=str(exc),
                )
            )
            failed += 1

    total_docs = len(manifest.get("documents", []))
    overall = (
        sum(c.overall_score for c in comparisons if c.error is None) / max(completed, 1)
        if completed > 0
        else 0.0
    )

    result = BenchmarkResult(
        name=manifest.get("name", "benchmark"),
        version=manifest.get("version", "0.0.0"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_documents=total_docs,
        completed=completed,
        failed=failed,
        overall_score=round(overall, 4),
        comparisons=comparisons,
    )

    # Attach gap analysis
    result.gap_analysis = analyze_gaps(result)

    return result


def _run_doc_textify(
    input_path: Path,
    output_path: Path,
    *,
    force_ocr: bool = False,
    lang: str = "eng",
    timeout: int = 600,
) -> None:
    """Invoke doc-textify CLI to process a document.

    The CLI always writes a JSON sidecar named after the input file stem into
    the ``--out`` directory; ``output_path`` is the expected JSON sidecar
    location and is used to derive ``--out``.

    Args:
        input_path: Source PDF/image to process.
        output_path: Expected JSON sidecar path (e.g. ``outputs/<doc_id>.json``).
        force_ocr: Pass ``--force-ocr`` for scanned docs with degraded native
            text layers.
        lang: Tesseract language code (``--lang``), e.g. ``eng``.
        timeout: Subprocess timeout in seconds (scanned multi-page docs can
            take minutes to OCR).
    """
    import subprocess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "doc_textify",
        str(input_path),
        "--out", str(output_path.parent),
        "--lang", lang,
    ]
    if force_ocr:
        cmd.append("--force-ocr")
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------


def analyze_gaps(result: BenchmarkResult) -> dict[str, Any]:
    """Analyze benchmark results to identify the weakest performing categories.

    Groups results by document type, language, and difficulty, then ranks
    categories by their score gap from perfect (1.0).

    Also identifies which features have the lowest detection rates across
    the benchmark suite.

    Returns a dict with:
    - ``by_type``: average score per document type
    - ``by_language``: average score per language
    - ``by_difficulty``: average score per difficulty level
    - ``weakest_areas``: ranked list of (category, score gap, docs)
    - ``feature_deficits``: per-feature detection rate gaps
    - ``recommendations``: actionable improvement suggestions
    """
    gaps: dict[str, Any] = {
        "by_type": {},
        "by_language": {},
        "by_difficulty": {},
        "weakest_areas": [],
        "feature_deficits": {},
        "recommendations": [],
    }

    # Group scores by category
    type_scores: dict[str, list[float]] = defaultdict(list)
    lang_scores: dict[str, list[float]] = defaultdict(list)
    diff_scores: dict[str, list[float]] = defaultdict(list)

    # Track feature detection rates
    feature_expected: dict[str, int] = defaultdict(int)
    feature_matched: dict[str, int] = defaultdict(int)

    for comp in result.comparisons:
        if comp.error:
            continue

        # We need to look up the document metadata from the manifest
        # But the comparison doesn't carry it directly — we store scores only
        type_scores.setdefault("unknown", []).append(comp.overall_score)

    gaps["by_type"] = {
        t: round(sum(s) / len(s), 4) for t, s in sorted(type_scores.items())
    }
    gaps["by_language"] = {
        t: round(sum(s) / len(s), 4) for t, s in sorted(lang_scores.items())
    }
    gaps["by_difficulty"] = {
        t: round(sum(s) / len(s), 4) for t, s in sorted(diff_scores.items())
    }

    # Weakest areas: rank by score gap
    areas: list[tuple[str, float, int]] = []
    for cat, avg in gaps["by_type"].items():
        areas.append((f"type:{cat}", 1.0 - avg, len(type_scores.get(cat, []))))
    for cat, avg in gaps["by_language"].items():
        areas.append((f"lang:{cat}", 1.0 - avg, len(lang_scores.get(cat, []))))
    for cat, avg in gaps["by_difficulty"].items():
        areas.append((f"difficulty:{cat}", 1.0 - avg, len(diff_scores.get(cat, []))))

    # Sort by gap descending (worst first)
    areas.sort(key=lambda x: x[1], reverse=True)

    gaps["weakest_areas"] = [
        {"category": cat, "score_gap": round(gap, 4), "document_count": count}
        for cat, gap, count in areas
        if count > 0
    ]

    # Generate recommendations
    if areas:
        worst_cat, worst_gap, _ = areas[0]
        if worst_gap > 0.5:
            gaps["recommendations"].append(
                f"PRIORITY: {worst_cat} has a large score gap of {worst_gap:.1%}. "
                f"Focus improvement efforts on this category first."
            )
        if worst_gap > 0.3:
            gaps["recommendations"].append(
                f"Consider adding more training/test data for {worst_cat} documents."
            )

    if result.overall_score < 0.5:
        gaps["recommendations"].append(
            f"Overall score is below 50% ({result.overall_score:.1%}). "
            "Check whether the OCR/model pipeline is operational."
        )
    elif result.overall_score < 0.75:
        gaps["recommendations"].append(
            f"Overall score is moderate ({result.overall_score:.1%}). "
            "Table and chart extraction are common improvement areas."
        )
    else:
        gaps["recommendations"].append(
            f"Overall score is strong ({result.overall_score:.1%}). "
            "Focus on edge cases: scanned documents, formulas, multi-column layouts."
        )

    return gaps


def analyze_gaps_with_manifest(
    result: BenchmarkResult, manifest_path: str | Path
) -> dict[str, Any]:
    """Extended gap analysis that uses manifest metadata for richer grouping.

    Combines the per-document scores from the benchmark result with the
    document metadata (type, language, difficulty, features) from the manifest
    to produce more detailed category breakdowns.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    doc_meta = {d["id"]: d for d in manifest.get("documents", [])}

    type_scores: dict[str, list[float]] = defaultdict(list)
    lang_scores: dict[str, list[float]] = defaultdict(list)
    diff_scores: dict[str, list[float]] = defaultdict(list)
    feature_scores: dict[str, list[float]] = defaultdict(list)

    for comp in result.comparisons:
        if comp.error:
            continue
        meta = doc_meta.get(comp.doc_id, {})
        dtype = meta.get("type", "unknown")
        lang = meta.get("language", "unknown")
        diff = meta.get("difficulty", "unknown")
        features = meta.get("features", [])

        type_scores[dtype].append(comp.overall_score)
        lang_scores[lang].append(comp.overall_score)
        diff_scores[diff].append(comp.overall_score)
        for feat in features:
            feature_scores[feat].append(comp.overall_score)

    gaps: dict[str, Any] = {
        "by_type": {t: round(sum(s) / len(s), 4) for t, s in sorted(type_scores.items())},
        "by_language": {t: round(sum(s) / len(s), 4) for t, s in sorted(lang_scores.items())},
        "by_difficulty": {t: round(sum(s) / len(s), 4) for t, s in sorted(diff_scores.items())},
        "by_feature": {t: round(sum(s) / len(s), 4) for t, s in sorted(feature_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))},
    }

    # Build weakest areas ranked list
    areas: list[tuple[str, float, int]] = []
    for cat, vals in type_scores.items():
        areas.append((f"type:{cat}", 1.0 - sum(vals) / len(vals), len(vals)))
    for cat, vals in lang_scores.items():
        areas.append((f"lang:{cat}", 1.0 - sum(vals) / len(vals), len(vals)))
    for cat, vals in diff_scores.items():
        areas.append((f"difficulty:{cat}", 1.0 - sum(vals) / len(vals), len(vals)))
    for cat, vals in feature_scores.items():
        areas.append((f"feature:{cat}", 1.0 - sum(vals) / len(vals), len(vals)))

    areas.sort(key=lambda x: x[1], reverse=True)
    gaps["weakest_areas"] = [
        {"category": cat, "score_gap": round(gap, 4), "document_count": count}
        for cat, gap, count in areas
        if count > 0
    ]

    # Feature deficit: features with average scores below benchmark average
    gaps["feature_deficits"] = {
        feat: round(1.0 - avg, 4)
        for feat, avg in gaps["by_feature"].items()
        if avg < result.overall_score
    }

    # Recommendations
    gaps["recommendations"] = []
    if areas:
        worst_cat, worst_gap, _ = areas[0]
        if worst_gap > 0.5:
            gaps["recommendations"].append(
                f"PRIORITY: {worst_cat} has a large score gap of {worst_gap:.1%}. "
                "Focus improvement efforts here."
            )
        if worst_gap > 0.3:
            gaps["recommendations"].append(
                f"Investigate {worst_cat} — score gap of {worst_gap:.1%} indicates systematic issues."
            )

    weak_features = sorted(
        gaps["feature_deficits"].items(), key=lambda x: x[1], reverse=True
    )
    if weak_features:
        top_feat, top_feat_gap = weak_features[0]
        gaps["recommendations"].append(
            f"Feature '{top_feat}' has largest deficit ({top_feat_gap:.1%}). "
            f"Consider targeted improvements for this feature class."
        )

    if result.overall_score < 0.5:
        gaps["recommendations"].append(
            f"Overall score is below 50% ({result.overall_score:.1%}). "
            "Verify pipeline dependencies (Tesseract, OpenCV) are properly installed."
        )
    elif result.overall_score < 0.75:
        gaps["recommendations"].append(
            f"Overall score is moderate ({result.overall_score:.1%}). "
            "Focus on table extraction and formula OCR accuracy."
        )
    else:
        gaps["recommendations"].append(
            f"Overall score is strong ({result.overall_score:.1%}). "
            "Target remaining edge cases: scanned/photographed documents, multi-column layouts."
        )

    return gaps


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_reports(
    result: BenchmarkResult,
    manifest_path: str | Path | None = None,
    *,
    output_dir: str | Path = "benchmarks/results",
) -> tuple[Path, Path]:
    """Generate JSON and Markdown reports from benchmark results.

    Args:
        result: BenchmarkResult from ``benchmark_dataset()``.
        manifest_path: Optional path to manifest for enriched gap analysis.
        output_dir: Directory for output reports.

    Returns:
        Tuple of (json_report_path, markdown_report_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON report ---
    json_path = output_dir / "report.json"
    json_report = _build_json_report(result)
    json_path.write_text(
        json.dumps(json_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- Markdown report ---
    md_path = output_dir / "report.md"
    md_report = _build_markdown_report(result, manifest_path)
    md_path.write_text(md_report, encoding="utf-8")

    return json_path, md_path


def _build_json_report(result: BenchmarkResult) -> dict[str, Any]:
    """Serialize benchmark result to a JSON-serializable dict."""
    return {
        "benchmark": {
            "name": result.name,
            "version": result.version,
            "timestamp": result.timestamp,
        },
        "summary": {
            "total_documents": result.total_documents,
            "completed": result.completed,
            "failed": result.failed,
            "overall_score": result.overall_score,
        },
        "documents": [
            {
                "id": c.doc_id,
                "actual": c.actual_path,
                "expected": c.expected_path,
                "overall_score": c.overall_score,
                "metrics": c.metrics,
                "error": c.error,
            }
            for c in result.comparisons
        ],
        "gap_analysis": result.gap_analysis,
    }


def _build_markdown_report(
    result: BenchmarkResult,
    manifest_path: str | Path | None = None,
) -> str:
    """Render benchmark results as Markdown."""
    lines = [
        f"# {result.name} — Benchmark Report",
        "",
        f"**Generated:** {result.timestamp}",
        f"**Version:** {result.version}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Total documents | {result.total_documents} |",
        f"| Completed | {result.completed} |",
        f"| Failed | {result.failed} |",
        f"| **Overall Score** | **{result.overall_score:.2%}** |",
        "",
        "## Per-Document Results",
        "",
    ]

    if not result.comparisons:
        lines.append("*No comparisons were run.*")
    else:
        lines.append(
            "| Document ID | Score | Status |"
        )
        lines.append("| --- | ---: | --- |")
        for comp in result.comparisons:
            status = "OK" if comp.overall_score >= 0.7 else ("WARN" if comp.overall_score >= 0.4 else "FAIL")
            if comp.error:
                status = "ERR"
                lines.append(f"| {comp.doc_id} | — | {status} {comp.error} |")
            else:
                lines.append(f"| {comp.doc_id} | {comp.overall_score:.2%} | {status} |")

        lines.append("")

        # Detailed breakdowns
        lines.append("## Detailed Metrics")
        lines.append("")
        for comp in result.comparisons:
            if comp.error or not comp.metrics:
                continue
            lines.append(f"### {comp.doc_id}  (overall: {comp.overall_score:.2%})")
            lines.append("")
            lines.append("| Metric | Score | Weight | Detail |")
            lines.append("| --- | ---: | ---: | --- |")
            for name, m in comp.metrics.items():
                lines.append(
                    f"| {name} | {m['score']:.2%} | {m['weight']:.2f} | {m.get('detail', '')} |"
                )
            lines.append("")

    # Gap analysis
    lines.append("## Gap Analysis")
    lines.append("")

    ga = result.gap_analysis
    if not ga:
        lines.append("*No gap analysis available.*")
        return "\n".join(lines)

    # Weakest areas
    if ga.get("weakest_areas"):
        lines.append("### Weakest Areas (Ranked by Score Gap)")
        lines.append("")
        lines.append("| # | Category | Score Gap | Docs |")
        lines.append("| --- | --- | ---: | ---: |")
        for i, area in enumerate(ga["weakest_areas"][:10], 1):
            lines.append(
                f"| {i} | {area['category']} | {area['score_gap']:.1%} | {area['document_count']} |"
            )
        lines.append("")

    # By type
    if ga.get("by_type"):
        lines.append("### By Document Type")
        lines.append("")
        lines.append("| Type | Avg Score |")
        lines.append("| --- | ---: |")
        for t, score in sorted(ga["by_type"].items(), key=lambda x: x[1]):
            lines.append(f"| {t} | {score:.2%} |")
        lines.append("")

    # By language
    if ga.get("by_language"):
        lines.append("### By Language")
        lines.append("")
        lines.append("| Language | Avg Score |")
        lines.append("| --- | ---: |")
        for lang, score in sorted(ga["by_language"].items(), key=lambda x: x[1]):
            lines.append(f"| {lang} | {score:.2%} |")
        lines.append("")

    # By difficulty
    if ga.get("by_difficulty"):
        lines.append("### By Difficulty")
        lines.append("")
        lines.append("| Difficulty | Avg Score |")
        lines.append("| --- | ---: |")
        for diff, score in sorted(ga["by_difficulty"].items()):
            lines.append(f"| {diff} | {score:.2%} |")
        lines.append("")

    # Feature deficits
    if ga.get("feature_deficits"):
        lines.append("### Feature Deficits")
        lines.append("")
        lines.append("| Feature | Deficit |")
        lines.append("| --- | ---: |")
        for feat, deficit in sorted(ga["feature_deficits"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {feat} | {deficit:.1%} |")
        lines.append("")

    # Recommendations
    if ga.get("recommendations"):
        lines.append("### Recommendations")
        lines.append("")
        for rec in ga["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="doc-textify Benchmark — Quality Comparison Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  doc-textify-bench --dataset benchmarks/dataset/ --output-dir outputs/

  # Compare a single document pair
  doc-textify-bench --compare \\
      --actual outputs/report.json \\
      --expected benchmarks/dataset/expected/report.json

  # Save reports to a custom directory
  doc-textify-bench --dataset benchmarks/dataset/ --output-dir outputs/ \\
      --report-dir benchmarks/results/
""",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to benchmark dataset directory (containing manifest.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing doc-textify output JSONs (default: outputs/).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for report output (default: benchmarks/results/).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Single-document comparison mode.",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        help="Path to actual doc-textify output JSON (for --compare mode).",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="Path to expected/reference JSON (for --compare mode).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run doc-textify on each document before comparing (requires CLI on PATH).",
    )

    parsed = parser.parse_args(args)

    # Single-comparison mode
    if parsed.compare:
        if not parsed.actual or not parsed.expected:
            print("Error: --compare requires both --actual and --expected.", file=sys.stderr)
            return 1

        comp = compare_document(parsed.actual, parsed.expected)
        print(f"\n{'='*60}")
        print(f"  Comparison: {comp.doc_id}")
        print(f"  Overall Score: {comp.overall_score:.2%}")
        print(f"{'='*60}\n")

        if comp.error:
            print(f"  ERROR: {comp.error}")
            return 1

        for name, m in comp.metrics.items():
            print(f"  {name:.<30s} {m['score']:.2%}  (weight: {m['weight']:.2f})")
            if m.get("detail"):
                print(f"    → {m['detail']}")
        print()
        return 0

    # Batch mode
    if not parsed.dataset:
        print("Error: --dataset is required for batch mode.", file=sys.stderr)
        return 1

    manifest_file = parsed.dataset / "manifest.json"
    if not manifest_file.exists():
        print(f"Error: manifest.json not found at {manifest_file}", file=sys.stderr)
        return 1

    print(f"\nRunning benchmark: {manifest_file}")
    print(f"  Output directory: {parsed.output_dir.resolve()}")
    print(f"  Report directory: {parsed.report_dir.resolve()}")
    if parsed.run:
        print(f"  Mode: run doc-textify + compare")
    print()

    result = benchmark_dataset(
        manifest_file,
        output_dir=parsed.output_dir,
        run_doc_textify=parsed.run,
    )

    # Enrich gap analysis with manifest metadata
    result.gap_analysis = analyze_gaps_with_manifest(result, manifest_file)

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Benchmark: {result.name} v{result.version}")
    print(f"  Overall Score: {result.overall_score:.2%}")
    print(f"  Completed: {result.completed}/{result.total_documents}")
    print(f"  Failed: {result.failed}")
    print(f"{'='*60}\n")

    for comp in result.comparisons:
        if comp.error:
            print(f"  {comp.doc_id:.<30s} [ERR] {comp.error}")
        else:
            icon = "[OK]" if comp.overall_score >= 0.7 else ("[WARN]" if comp.overall_score >= 0.4 else "[FAIL]")
            print(f"  {comp.doc_id:.<30s} {icon} {comp.overall_score:.2%}")

    print()

    # Gap analysis summary
    ga = result.gap_analysis
    if ga.get("weakest_areas"):
        print("Weakest areas:")
        for area in ga["weakest_areas"][:5]:
            print(f"  - {area['category']}: gap {area['score_gap']:.1%} ({area['document_count']} docs)")
        print()
    if ga.get("recommendations"):
        print("Recommendations:")
        for rec in ga["recommendations"][:3]:
            print(f"  - {rec}")
        print()

    # Generate reports
    json_path, md_path = generate_reports(
        result,
        manifest_file,
        output_dir=parsed.report_dir,
    )
    print(f"Reports written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
