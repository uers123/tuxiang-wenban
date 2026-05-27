# doc-textify

`doc-textify` 是一个不依赖视觉大模型的图片/PDF 文本化工具。它的目标是把 PDF、截图、扫描件、拍照文档和部分图表转换成文本大模型可以直接读取的 Markdown/TXT/JSON/LLM 协议文本。

项目定位不是“让文生图模型复原原图”，而是做一个低成本的视觉预处理器：

```text
PDF / Image -> doc-textify -> compact text + JSON -> text-only LLM
```

这样，后面的模型不需要图像识别能力，也不需要消耗高价视觉 token，就能获得页面文字、阅读顺序、版面块、置信度、图表结构和部分数值信息。

## 当前能力

- 数字 PDF：优先抽取原生文字层，保留页面、段落、标题、列表和图片占位。
- 扫描 PDF：可通过 `pypdfium2` 渲染页面后进入 OCR 管线。
- 图片/截图/拍照文档：使用 Pillow 预处理，再调用本地 Tesseract OCR。
- 中文/英文混排：支持 `chi_sim+eng`，并做中文 OCR 空格、轴标签、常见误识别的后处理。
- 版面恢复：包含标题、页眉页脚、列表、阅读顺序和基础多栏排序。
- 图表文本化：对红色竖线区间、散点、面板、坐标轴标签做结构化提取，输出 `chart_data`。
- 输出格式：Markdown、TXT、紧凑 LLM 文本、JSON sidecar。

## 不使用什么

核心管线不调用 GPT、Gemini、Claude、Qwen-VL 等视觉大模型。项目允许使用专用 OCR、PDF 渲染和传统图像处理库，因为它们属于前置文档编译器，而不是让目标大模型自己看图。

## 安装

```powershell
python -m pip install -e .[all]
```

图片 OCR 需要安装 Tesseract 和对应语言包。Windows 常见语言参数：

```powershell
doc-textify input.jpg --out outputs --format all --lang chi_sim+eng
```

如果只处理数字 PDF，可以不安装 Tesseract：

```powershell
doc-textify input.pdf --out outputs --format all
```

## 命令行

```powershell
doc-textify INPUT --out outputs --format both --lang chi_sim+eng
```

参数：

- `--format md`：输出 Markdown 和 JSON。
- `--format txt`：输出 TXT 和 JSON。
- `--format llm`：输出低 Token 的 `.llm.txt` 和 JSON。
- `--format both`：输出 Markdown、TXT 和 JSON。
- `--format all`：输出 Markdown、TXT、LLM 文本和 JSON。
- `--force-ocr`：PDF 强制走渲染 + OCR 路径。
- `--min-confidence`：过滤低置信度 OCR 词。

## LLM 协议输出

`--format llm` 会生成面向文本大模型的紧凑文本，例如：

```text
DOC_TEXTIFY_LLM_PROTOCOL v1
source: 对比图.jpg
pages: 1

[page 1 size=1280x1971]
chart_data:
  panel a:
    intervals: class 0 depth 0.0-5.4; class 1 depth 5.0-7.2
    points: class 1 depth 4.0; class 2 depth 0.5
figure_note: Chart data extracted from image. 标签 深度/m 真实类别 预测类别
```

这个输出比完整 JSON 更省 token，比普通 OCR 文本更有结构，适合直接作为文本大模型的输入上下文。

## 评测

项目包含一个人工标注的图表样例评测：

```powershell
doc-textify "G:\aaaaaaaaaaaaaaaaa\dili\dili\对比图.jpg" --out test_outputs\jpg_exe_final --format all --lang chi_sim+eng --min-confidence 20

python scripts\evaluate_textification.py `
  --actual test_outputs\jpg_exe_final\对比图.json `
  --expected benchmarks\duibitu.expected.json `
  --out test_outputs\jpg_exe_final\评分报告.md
```

当前可执行版在该样例上的结果：

- 综合得分：69.73%
- 关键术语：100%
- 面板与轴标签：100%
- 可用性：100%
- 图表数值数据：24.32%

这说明项目已经能把图片中的文字和图表结构转换成文本，但图表数值反算仍是下一阶段的主要瓶颈。

## 测试

```powershell
python -m pytest -q --basetemp .pytest_tmp\pytest
```

## 发行文件

本地 Windows 可执行文件：

```text
dist\doc-textify.exe
```

本地发行压缩包：

```text
release\doc-textify-v0.3.0-windows-x64.zip
```

## 报告

详细测试和优化报告位于：

```text
reports\精准优化测试报告.md
```

## 下一步

最值得继续做的是图表坐标校准器：

1. 检测每个图表面板的水平网格线。
2. OCR 识别深度刻度。
3. 建立像素 y 到实际深度的映射。
4. 重新计算红色区间和散点深度。
5. 剔除线段端点、印刷噪声和图例干扰。

完成这一层后，项目会更接近“文本大模型的眼睛”：不是描述图片的大概含义，而是用更低 token 成本提供可验证、可引用、可推理的视觉文本事实。
