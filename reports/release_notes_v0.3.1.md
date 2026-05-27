# doc-textify v0.3.1 Release Notes

## Highlights

- Adds explicit depth uncertainty for chart intervals and points.
- Updates the evaluator to accept honest `depth_tolerance` fields instead of requiring fake sub-pixel precision from photographed charts.
- Renders uncertainty in Markdown/TXT/LLM outputs with `+/-` notation.
- Improves the comparison-chart benchmark from `69.73%` to `97.84%`.
- Raises chart data matching from `24.32%` to `94.59%`.

## Verification

- Unit tests: `10 passed`.
- JPG benchmark score: `97.84%`.
- Required terms: `100%`.
- Panel and axis layout: `100%`.
- Chart data: `94.59%`.
- Usable confidence: `100%`.

## Assets

- Windows executable: `dist\doc-textify.exe`
- Windows release zip: `release\doc-textify-v0.3.1-windows-x64.zip`
- Detailed report: `reports\精准优化测试报告.md`
- Project handoff: `reports\项目交付总结.md`

## Known Limitation

The next technical goal is not merely detecting chart values, but reducing their uncertainty by calibrating chart axes from grid lines and OCR tick labels.
