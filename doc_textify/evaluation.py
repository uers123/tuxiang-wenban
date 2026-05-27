from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Metric:
    name: str
    score: float
    weight: float
    detail: str


def evaluate_textification(actual_json: Path, expected_json: Path) -> tuple[dict[str, Any], str]:
    actual = json.loads(actual_json.read_text(encoding="utf-8"))
    expected = json.loads(expected_json.read_text(encoding="utf-8"))

    text = _actual_text(actual)
    metrics = [
        _image_presence_metric(actual),
        _required_terms_metric(text, expected),
        _panel_layout_metric(text, expected),
        _chart_data_metric(text, actual, expected),
        _uncertainty_metric(actual),
    ]
    total_weight = sum(metric.weight for metric in metrics)
    overall = sum(metric.score * metric.weight for metric in metrics) / total_weight if total_weight else 0.0

    result = {
        "actual": str(actual_json),
        "expected": str(expected_json),
        "overall_score": round(overall, 4),
        "metrics": [
            {
                "name": metric.name,
                "score": round(metric.score, 4),
                "weight": metric.weight,
                "detail": metric.detail,
            }
            for metric in metrics
        ],
    }
    return result, render_report(result)


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Textification Evaluation Report",
        "",
        f"- actual: `{result['actual']}`",
        f"- expected: `{result['expected']}`",
        f"- overall_score: **{result['overall_score']:.2%}**",
        "",
        "| Metric | Score | Weight | Detail |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric in result["metrics"]:
        lines.append(
            f"| {metric['name']} | {metric['score']:.2%} | {metric['weight']} | {metric['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A useful image-to-text result should score well on semantic text, layout, and visual data.",
            "- A result that only says `[Figure]` should score low even if the file was processed successfully.",
            "- This metric accepts explicit uncertainty fields such as `depth_tolerance`; the goal is faithful textification, not fake precision.",
            "",
        ]
    )
    return "\n".join(lines)


def _actual_text(actual: dict[str, Any]) -> str:
    chunks: list[str] = []
    chunks.extend(str(item) for item in actual.get("warnings", []))
    for page in actual.get("pages", []):
        chunks.extend(str(item) for item in page.get("warnings", []))
        for block in page.get("blocks", []):
            chunks.append(str(block.get("type", "")))
            chunks.append(str(block.get("text", "")))
            chunks.append(json.dumps(block.get("metadata", {}), ensure_ascii=False))
    return "\n".join(chunks).lower()


def _image_presence_metric(actual: dict[str, Any]) -> Metric:
    pages = actual.get("pages", [])
    has_dimensions = any(page.get("width") and page.get("height") for page in pages)
    has_blocks = any(page.get("blocks") for page in pages)
    score = 1.0 if has_dimensions and has_blocks else 0.0
    return Metric(
        name="image_presence",
        score=score,
        weight=0.10,
        detail="Checks whether the input image/page was represented with dimensions and at least one block.",
    )


def _required_terms_metric(text: str, expected: dict[str, Any]) -> Metric:
    terms = [str(term).lower() for term in expected.get("required_terms", [])]
    if not terms:
        return Metric("required_terms", 1.0, 0.20, "No required terms were provided.")
    found = [term for term in terms if term in text]
    score = len(found) / len(terms)
    detail = f"Found {len(found)}/{len(terms)} required terms."
    return Metric("required_terms", score, 0.20, detail)


def _panel_layout_metric(text: str, expected: dict[str, Any]) -> Metric:
    panels = expected.get("panels", [])
    if not panels:
        return Metric("panel_layout", 1.0, 0.20, "No expected panels were provided.")

    expected_items: list[str] = []
    for panel in panels:
        expected_items.extend(
            [
                panel.get("caption", ""),
                panel.get("x_axis", {}).get("label", ""),
                panel.get("y_axis", {}).get("label", ""),
            ]
        )
    expected_items = [item.lower() for item in expected_items if item]
    found = [item for item in expected_items if item in text]
    score = len(found) / len(expected_items) if expected_items else 0.0
    detail = f"Matched {len(found)}/{len(expected_items)} panel, caption, and axis labels."
    return Metric("panel_layout", score, 0.20, detail)


def _chart_data_metric(text: str, actual: dict[str, Any], expected: dict[str, Any]) -> Metric:
    expected_intervals = []
    expected_points = []
    for panel in expected.get("panels", []):
        for interval in panel.get("predicted_intervals", []):
            expected_intervals.append((panel.get("id", ""), interval))
        for point in panel.get("predicted_points", []):
            expected_points.append((panel.get("id", ""), point))

    if not expected_intervals and not expected_points:
        return Metric("chart_data", 1.0, 0.40, "No chart data expectations were provided.")

    actual_chart_objects = _actual_chart_objects(actual)
    matched_intervals = sum(
        1 for panel_id, interval in expected_intervals if _interval_found(text, actual_chart_objects, panel_id, interval)
    )
    matched_points = sum(
        1 for panel_id, point in expected_points if _point_found(text, actual_chart_objects, panel_id, point)
    )
    total = len(expected_intervals) + len(expected_points)
    score = (matched_intervals + matched_points) / total
    detail = (
        f"Matched {matched_intervals}/{len(expected_intervals)} intervals and "
        f"{matched_points}/{len(expected_points)} points."
    )
    return Metric("chart_data", score, 0.40, detail)


def _uncertainty_metric(actual: dict[str, Any]) -> Metric:
    text = _actual_text(actual)
    warnings = actual.get("warnings", [])
    has_placeholder = "placeholder" in text or "not performed" in text or "unavailable" in text
    if warnings or has_placeholder:
        return Metric(
            "usable_confidence",
            0.0,
            0.10,
            "Warnings/placeholders indicate the output is not yet usable for reconstruction.",
        )
    return Metric("usable_confidence", 1.0, 0.10, "No top-level warnings or placeholders detected.")


def _actual_chart_objects(actual: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for page in actual.get("pages", []):
        for block in page.get("blocks", []):
            metadata = block.get("metadata", {})
            if isinstance(metadata, dict):
                chart_data = metadata.get("chart_data")
                if isinstance(chart_data, list):
                    objects.extend(item for item in chart_data if isinstance(item, dict))
                elif isinstance(chart_data, dict):
                    objects.append(chart_data)
    return objects


def _interval_found(
    text: str,
    chart_objects: list[dict[str, Any]],
    panel_id: str,
    interval: dict[str, Any],
) -> bool:
    klass = str(interval.get("class", ""))
    start = interval.get("start_depth")
    end = interval.get("end_depth")
    for item in chart_objects:
        if item.get("type") != "interval":
            continue
        if panel_id and item.get("panel_id") != panel_id:
            continue
        tolerance = _item_tolerance(item)
        if (
            _class_matches(item, klass)
            and _close(item.get("start_depth"), start, tolerance)
            and _close(item.get("end_depth"), end, tolerance)
        ):
            return True

    class_patterns = [f"class {klass}", f"类别 {klass}", f"标签 {klass}"]
    depth_patterns = [_number_pattern(start), _number_pattern(end)]
    return any(pattern in text for pattern in class_patterns) and all(
        re.search(pattern, text) for pattern in depth_patterns if pattern
    )


def _point_found(text: str, chart_objects: list[dict[str, Any]], panel_id: str, point: dict[str, Any]) -> bool:
    klass = str(point.get("class", ""))
    depth = point.get("depth")
    for item in chart_objects:
        if item.get("type") != "point":
            continue
        if panel_id and item.get("panel_id") != panel_id:
            continue
        if _class_matches(item, klass) and _close(item.get("depth"), depth, _item_tolerance(item)):
            return True
    return f"class {klass}" in text and bool(re.search(_number_pattern(depth), text))


def _close(left: Any, right: Any, tolerance: float = 0.15) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _item_tolerance(item: dict[str, Any]) -> float:
    try:
        return max(0.15, float(item.get("depth_tolerance", 0.15)))
    except (TypeError, ValueError):
        return 0.15


def _class_matches(item: dict[str, Any], klass: str) -> bool:
    if str(item.get("class")) == klass:
        return True
    candidates = item.get("class_candidates", [])
    if isinstance(candidates, list):
        return klass in {str(candidate) for candidate in candidates}
    return False


def _number_pattern(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric.is_integer():
        return rf"\b{int(numeric)}(?:\.0+)?\b"
    return rf"\b{numeric:g}\b"
