# doc-textify

## 项目介绍

`doc-textify` 是一个面向文本大模型的图片/PDF 文本化预处理器。它的目标是把 PDF、扫描件、截图、拍照文档和部分图表转换成大模型容易读取的 Markdown、TXT、低 Token 文本协议和 JSON 结构化数据。

这个项目的核心思路是：先用 OCR、PDF 解析、版面分析和图表解析把视觉文件编译成文本，再交给没有视觉能力或不希望消耗视觉 Token 的大模型读取。

```text
PDF / Image -> doc-textify -> Markdown / TXT / LLM Text / JSON -> Text-only LLM
```

它不是通用视觉大模型的完整替代品，也不声称在所有图片理解任务上都更强。它更适合处理文档、表格、图表、截图、扫描件等“视觉中包含结构化文字事实”的场景。对自然照片、复杂场景、情绪、意图、物体关系等开放式视觉理解，视觉大模型仍然更合适。

## 项目说明

### 支持的输入

- 数字 PDF：优先抽取原生文字层。
- 扫描 PDF：将页面渲染为图片后进入 OCR 管线。
- 图片文件：支持常见图片格式。
- 截图：适合界面文字、表格、标签、按钮等内容提取。
- 拍照文档：适合文档页面、图表页面和带文字的纸质资料。

### 输出格式

- Markdown：保留页面、标题、段落、图表和结构化表格。
- TXT：输出更朴素的文本结果。
- LLM 文本：面向大模型输入的紧凑文本协议，减少无效 Token。
- JSON：保存页面、文本块、坐标、置信度、图表数据和元信息。

### 核心能力

- OCR 识别：支持中文、英文和中英混排。
- PDF 解析：优先使用原生文字层，必要时走 OCR。
- 版面分析：识别标题、段落、列表、页眉页脚和基础阅读顺序。
- 图表文本化：提取面板、轴标签、区间、散点和结构化 `chart_data`。
- 误差表达：图表读数可输出 `+/-` 误差范围，避免伪装成不可靠的精确值。
- 低 Token 表达：把图片中的有效事实压缩为大模型更容易处理的文本。

### 项目边界

`doc-textify` 不调用 GPT、Gemini、Claude、Qwen-VL 等视觉大模型作为核心识别管线。它允许使用专用 OCR、PDF 渲染和传统图像处理能力，因为这些工具承担的是“视觉文件到文本事实”的前置编译工作。

## 项目与传统项目对比

| 维度 | 传统 OCR | 直接使用视觉大模型 | doc-textify |
| --- | --- | --- | --- |
| 主要目标 | 把图片中的文字识别出来 | 直接理解图片并回答问题 | 把图片/PDF 编译成大模型可读文本 |
| 输出形式 | 普通文本或带文字层 PDF | 自然语言回答 | Markdown、TXT、LLM 文本、JSON |
| Token 成本 | 后续仍需整理文本 | 图像 Token 成本较高 | 只把提取后的有效事实交给大模型 |
| 结构化能力 | 通常较弱 | 依赖模型临场理解 | 明确输出页面、块、坐标、置信度和图表数据 |
| 可评测性 | 可评 OCR 字符准确率 | 开放式回答较难稳定评测 | 可用 expected JSON 做结构化评分 |
| 隐私与部署 | 可本地运行 | 通常依赖云端模型 | 可本地处理，再决定是否发送文本 |
| 图表/表格 | 容易丢结构 | 能描述但不一定稳定 | 面向结构化数据提取设计 |
| 适用场景 | 简单扫描件、票据、纯文字图片 | 自然图片、开放式视觉问答 | 文档、论文、截图、表格、工程图表 |
| 主要局限 | 版面和语义理解弱 | 成本高、结果不易复查 | 不擅长开放式自然场景理解 |

## 使用方法

### 安装

安装 Python 依赖：

```powershell
python -m pip install -e .[all]
```

如果需要处理图片或扫描 PDF，请安装 Tesseract OCR，并安装需要的语言包。中文和英文混排通常使用：

```powershell
chi_sim+eng
```

### 基础命令

```powershell
doc-textify input.pdf --out outputs --format all
```

`--format all` 会同时输出 Markdown、TXT、LLM 文本和 JSON。

### 处理 PDF

处理数字 PDF：

```powershell
doc-textify input.pdf --out outputs --format all
```

强制 PDF 走 OCR 路径：

```powershell
doc-textify input.pdf --out outputs --format all --force-ocr --lang chi_sim+eng
```

### 处理图片

处理普通图片：

```powershell
doc-textify image.jpg --out outputs --format all --lang chi_sim+eng
```

降低 OCR 置信度阈值以保留更多疑似文字：

```powershell
doc-textify image.jpg --out outputs --format all --lang chi_sim+eng --min-confidence 20
```

### 输出 LLM 低 Token 文本

只生成面向大模型输入的紧凑文本和 JSON：

```powershell
doc-textify image.jpg --out outputs --format llm --lang chi_sim+eng
```

LLM 文本示例：

```text
DOC_TEXTIFY_LLM_PROTOCOL v1
source: image.jpg
pages: 1

[page 1 size=1280x1971]
chart_data:
  panel a:
    intervals: class 0 depth 0.0-5.4 +/- 0.7; class 1 depth 5.0-7.2 +/- 0.7
    points: class 1 depth 4.0 +/- 0.56; class 2 depth 0.5 +/- 0.56
figure_note: Chart data extracted from image. 标签 深度/m 真实类别 预测类别
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--out` | 指定输出目录 |
| `--format md` | 输出 Markdown 和 JSON |
| `--format txt` | 输出 TXT 和 JSON |
| `--format llm` | 输出 LLM 文本和 JSON |
| `--format both` | 输出 Markdown、TXT 和 JSON |
| `--format all` | 输出 Markdown、TXT、LLM 文本和 JSON |
| `--lang` | 指定 OCR 语言，例如 `eng` 或 `chi_sim+eng` |
| `--force-ocr` | PDF 强制使用 OCR 路径 |
| `--min-confidence` | 设置 OCR 词语置信度过滤阈值 |

### 评测示例

项目支持使用人工标注的 expected JSON 对输出结果进行结构化评分：

```powershell
doc-textify image.jpg --out outputs --format all --lang chi_sim+eng
doc-textify-eval --actual actual.json --expected expected.json --out report.md
```

评测重点包括：

- 图像或页面是否被正确表示。
- 关键术语是否被提取。
- 面板、标题、坐标轴等布局信息是否存在。
- 图表区间和散点是否转为结构化数据。
- 输出是否包含明确警告、占位符或不确定性说明。

## 当前能力与限制

当前项目在文档和工程图表类样例上已经可以输出可评测的结构化结果，并能用较少文本表达图片中的关键信息。它适合作为文本大模型的视觉前处理层，尤其适合需要批量处理、成本敏感、结果可复查的场景。

仍需继续优化的方向：

- 更精确的图表坐标校准。
- 更强的复杂表格结构恢复。
- 更稳定的公式、手写体和低质量拍照识别。
- 更多类型 benchmark 和下游问答评测。
- 更完整的批量处理和工程化接口。

## License / 贡献

当前仓库尚未声明许可证。使用、分发或二次开发前，请先确认项目维护者后续发布的许可说明。

欢迎围绕以下方向贡献：

- 新增 OCR 或文档解析后端。
- 增加更多 benchmark 样例。
- 改进图表、表格、公式解析。
- 优化 LLM 文本协议。
- 完善测试、文档和发行流程。
