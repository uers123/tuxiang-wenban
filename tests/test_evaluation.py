import json
from pathlib import Path

from doc_textify.evaluation import evaluate_textification


def test_placeholder_output_scores_low(tmp_path: Path) -> None:
    actual = {
        "warnings": ["Image OCR backend unavailable."],
        "pages": [
            {
                "number": 1,
                "width": 1280,
                "height": 1971,
                "blocks": [
                    {
                        "type": "figure",
                        "text": "Image OCR was not performed because Tesseract is not installed or not on PATH.",
                        "metadata": {},
                    }
                ],
            }
        ],
    }
    expected = {
        "required_terms": ["标签", "深度/m"],
        "panels": [
            {
                "id": "a",
                "caption": "(a) 钻孔 ZK-4",
                "x_axis": {"label": "标签"},
                "y_axis": {"label": "深度/m"},
                "predicted_intervals": [{"class": 0, "start_depth": 0, "end_depth": 5.5}],
            }
        ],
    }
    actual_path = tmp_path / "actual.json"
    expected_path = tmp_path / "expected.json"
    actual_path.write_text(json.dumps(actual, ensure_ascii=False), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    result, report = evaluate_textification(actual_path, expected_path)

    assert result["overall_score"] < 0.2
    assert "overall_score" in report


def test_structured_chart_output_scores_high(tmp_path: Path) -> None:
    actual = {
        "warnings": [],
        "pages": [
            {
                "number": 1,
                "width": 1280,
                "height": 1971,
                "blocks": [
                    {
                        "type": "figure",
                        "text": "标签 深度/m 真实类别 预测类别 (a) 钻孔 ZK-4",
                        "metadata": {
                            "chart_data": [
                                {
                                    "type": "interval",
                                    "panel_id": "a",
                                    "class": 0,
                                    "start_depth": 0,
                                    "end_depth": 5.5,
                                }
                            ]
                        },
                    }
                ],
            }
        ],
    }
    expected = {
        "required_terms": ["标签", "深度/m", "真实类别", "预测类别"],
        "panels": [
            {
                "id": "a",
                "caption": "(a) 钻孔 ZK-4",
                "x_axis": {"label": "标签"},
                "y_axis": {"label": "深度/m"},
                "predicted_intervals": [{"class": 0, "start_depth": 0, "end_depth": 5.5}],
            }
        ],
    }
    actual_path = tmp_path / "actual.json"
    expected_path = tmp_path / "expected.json"
    actual_path.write_text(json.dumps(actual, ensure_ascii=False), encoding="utf-8")
    expected_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    result, _report = evaluate_textification(actual_path, expected_path)

    assert result["overall_score"] > 0.85
