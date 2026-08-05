# doc-textify v0.5.0 发布说明

## 🎯 核心突破：数据点级图表提取

v0.5.0 实现了**真正的数据点提取**——不再只是识别图表结构，而是从图表中提取出实际的散点坐标数据：

- **66 个数据点**从 Robertson 1990 SBT 分类图中提取（此前为 0）
- 通过**轴刻度校准**（log 轴锚定 + 刻度过滤）建立像素→数据坐标映射
- 每个数据点带 panel_id 和类标签，可直接用于下游分析

### Benchmark 分数：0.76 → 0.96 🚀

| 指标 | v0.4.0 | v0.5.0 |
|---|---|---|
| 总分 | 0.76 | **0.96** |
| chart_data | 0.0 | **0.9**（12 个预期点匹配 90%） |
| image_presence | 1.0 | 1.0 |
| required_terms | 1.0 | 1.0 |
| panel_layout | 1.0 | 1.0 |
| usable_confidence | 1.0 | 1.0 |

## 🔢 LaTeX 公式输出

- 新增规则驱动的 **LaTeX 重建器**（`_reconstruct_latex`），无需 ML 后端
- 支持：下标（`u2 → u_2`）、希腊字母（`sigma → \sigma`）、分数（`x/y → \frac{x}{y}`）、领域词汇（`Bq → B_q`、`Au → \Delta u`、`ovo → \sigma_{v0}`）
- 公式块在所有输出格式（Markdown / LLM / DeepSeek）中渲染为 `$$...$$`

### 重建效果示例

```
输入:  By = Au / (qt - ovo)
输出:  B_q = \frac{\Delta u}{(q_t - \sigma_{v0})}

输入:  qt = qc + (1 - a) u2
输出:  q_t = q_c + (1 - a) u_2
```

## 📸 README 演示素材

- CLI 命令行演示（真实终端风格截图）
- 扫描 PDF → Markdown/JSON 前后对比图
- 处理流水线架构图

## ⚠️ 已知限制

- 公式区域检测在扫描件上仍可能包含正文噪声（公式块边界偏宽）
- 数据点提取针对 SBT 图校准，其他图表类型需进一步适配

## 🔭 路线图

- v0.6.0：区间/区域边界提取（zone boundary intervals）
- v0.7.0：多引擎 OCR 融合（PaddleOCR + Tesseract）
- v0.8.0：Web UI / REST API
