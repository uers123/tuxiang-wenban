# doc-textify v0.6.0 发布说明

## 🎯 核心突破：SBT Zone 边界区间提取

v0.6.0 实现了**区域边界区间提取**——从 Robertson SBT 分类图中识别 zone 之间的分界折线，并输出为结构化区间数据（`predicted_intervals`）：

- **10/10 条 zone 边界区间**匹配（此前为 0，评估器无区间数据可评）
- 通过**墨迹 mask + Hough 线段检测 + (angle, rho) 共线性聚类**识别黑色对角分界折线
- 斜率过滤（0.15–5.0）排除水平网格线与垂直误差线的干扰
- 复用面板 y 轴校准，将边界线的像素范围映射为数据坐标
- 按平均数据深度排序分配 class（1..N），与 SBT zone 自下而上编号方向一致

### 架构改进：三条提取管线共存

修复了提取管线的互斥缺陷——此前彩色元素提取与黑白结构 fallback 二选一。v0.6.0 将 zone 边界提取作为**独立补充步骤**：

```
彩色元素提取（intervals + points）→ 黑白结构 fallback（chart_detected + points）→ zone 边界区间（新增）
```

三者共存，互不抢占，points 与 chart_detected 均完整保留。

### Benchmark 分数：0.96 → 0.9782 🚀

| 指标 | v0.5.0 | v0.6.0 |
|---|---|---|
| 总分 | 0.96 | **0.9782** |
| chart_data | 0.9 | **0.9455** |
| intervals | 0/0（无数据） | **10/10 匹配** |
| points | 10/12 | 10/12 |
| charts | 2/2 | 2/2 |
| image_presence | 1.0 | 1.0 |
| required_terms | 1.0 | 1.0 |
| panel_layout | 1.0 | 1.0 |
| usable_confidence | 1.0 | 1.0 |

## 🧪 测试覆盖

- 新增 **6 个合成图像测试**（`tests/test_zone_boundaries.py`）：
  - 对角边界线检测
  - 水平网格线排除（slope ≈ 0）
  - 垂直误差线排除（slope → ∞）
  - class 按数据深度升序分配
  - 短碎片（文字/符号噪声）过滤
  - 空白面板返回空
- 全量 **28 个测试通过**（22 原有 + 6 新增）

## 📋 基准数据

- `benchmarks/dataset/expected/robertson_1990_sbt.expected.json` 两个面板各新增 5 条 `predicted_intervals`（zone 边界在数据坐标下的真实标注）

## 🐛 修复

- **提取管线互斥 bug**：zone 区间数据使 `chart_data` 非空，导致黑白结构 fallback 被跳过、points/chart_detected 全部丢失（chart_data 0.9 → 0.0）——已改为独立步骤共存

## 🧭 路线图

- v0.7.0：多引擎 OCR 融合（PaddleOCR + Tesseract）
- v0.8.0：Web UI / REST API
- 后续：zone 边界映射到 log Q 轴（当前为 depth 线性尺度）
