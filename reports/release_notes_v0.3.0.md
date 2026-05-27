# doc-textify v0.3.0 Release Notes

## Highlights

- Adds compact `.llm.txt` output for low-token text-only LLM ingestion.
- Adds `--format llm` and `--format all`.
- Adds layout analysis and chart extraction modules.
- Improves Chinese OCR normalization for labels such as `标签`, `深度/m`, `真实类别`, `预测类别`, and `钻孔`.
- Adds chart `interval` and `point` data into JSON metadata.
- Updates README, environment notes, reports, and publishing script.

## Verification

- Unit tests: `10 passed`.
- JPG benchmark score: `97.84%`.
- Required terms: `100%`.
- Panel and axis layout: `100%`.
- Usable confidence: `100%`.

## Assets

- Windows executable: `dist\doc-textify.exe`
- Windows release zip: `release\doc-textify-v0.3.0-windows-x64.zip`
- Detailed report: `reports\精准优化测试报告.md`
- Project handoff: `reports\项目交付总结.md`

## Known Limitation

The main remaining technical bottleneck is reducing numeric uncertainty. The pipeline now exports explicit depth tolerances so text-only LLMs receive measured chart values with error bars instead of fake precision.
