# doc-textify 与视觉大模型的图像理解和 Token 评估

## 结论

`doc-textify` 不应被宣传为“在所有图像上比视觉大模型更好”。更准确的定位是：

- 对文档、截图、表格、论文页面、扫描件、坐标图、工程图表这类“视觉中的文字和结构事实”，`doc-textify` 可以比通用视觉大模型更可控、更可评测。
- 对自然照片、复杂场景、物体关系、隐含语义、情绪、空间常识、开放式描述，视觉大模型通常更强。
- 对低 Token 成本，`doc-textify` 的优势明确：它把图像先编译成紧凑文本，只把有效事实交给文本大模型，而不是把整张图作为图像 token 输入。

所以它更像“文档/图表视觉编译器”，不是通用视觉大模型替代品。

## 图像识别大模型如何理解图片

现代视觉大模型通常不是把整张图作为一块无限清晰的对象来理解。工程上大致是：

1. 图像被缩放、裁剪，或切成 patch/tile。
2. 每个 patch/tile 被视觉编码器变成向量表示。
3. 这些视觉向量被投影到语言模型可处理的 token/embedding 空间。
4. 语言模型在文本 token 与图像 token 的共同上下文里推理并生成答案。

这意味着视觉模型看到的是“被压缩和重采样后的视觉 token”。如果图片里有小字、细网格、旋转文字、低对比度数据点，模型可能会误读。OpenAI 官方文档也明确列出视觉模型在小字、旋转、精确空间定位、计数等方面存在限制。

## 图像 Token 怎么算

不同厂商算法不同，但核心都是：图片越大、细节越高、tile/patch 越多，token 越多。

### OpenAI

OpenAI 文档说明：图像输入像文本一样按 token 计量，但不同模型转换方式不同。部分模型使用 32px x 32px patch；部分模型使用 tile。tile 规则里，high detail 会先缩放到 2048x2048 内，再让短边变为 768px，然后按 512px 方块计数，再加 base token。

以 GPT-4o/4.1/4.5 系列为例，官方表格给出 base=85、tile=170。GPT-5 系列给出 base=70、tile=140。

对样例图 `对比图.jpg`：

- 原始尺寸：1280 x 1971
- high detail 缩放后约：768 x 1183
- 512 tile 数：ceil(768/512) * ceil(1183/512) = 2 * 3 = 6
- GPT-4o/4.1/4.5 估算：85 + 6 * 170 = 1105 image tokens
- GPT-5 估算：70 + 6 * 140 = 910 image tokens

low detail 通常更便宜，但模型只收到低分辨率图像，不适合小字、坐标轴、细网格和数据点。

### Claude

Anthropic 文档给出近似公式：

~~~text
image tokens ~= width * height / 750
~~~

对 1280 x 1971 的样例图，原始近似值为：

~~~text
1280 * 1971 / 750 ~= 3364 tokens
~~~

实际还会受模型上限和缩放策略影响。Anthropic 文档列出普通模型 native 图像 token 上限约 1568，高分辨率 Opus 可到约 4784。

### Gemini

Google Gemini 文档说明：如果图片两个维度都不超过 384px，计 258 tokens；更大的图片会切成 768 x 768 tile，每个 tile 258 tokens。

对 1280 x 1971 的样例图，按 768 tile 粗算：

~~~text
ceil(1280/768) * ceil(1971/768) = 2 * 3 = 6 tiles
6 * 258 = 1548 tokens
~~~

## doc-textify 的 Token 对比

样例输出：

- 输入图像：`G:\aaaaaaaaaaaaaaaaa\dili\dili\对比图.jpg`
- 图像尺寸：1280 x 1971
- `doc-textify` 输出：`test_outputs\v031_exe_final\对比图.llm.txt`
- `.llm.txt` 大小：1551 bytes
- 字符数：1457
- 空白分词粗略词数：262

该 `.llm.txt` 已包含：

- 页面尺寸
- 面板 a/b
- 深度区间
- 预测点
- `+/-` 误差范围
- 关键中文标签：标签、深度/m、真实类别、预测类别、钻孔 ZK-4、ZK-10

严格说，文本 token 数要由目标模型 tokenizer 决定；但从体积看，它已经是约 1.5KB 的结构化事实文本。相比让视觉大模型直接读取整张高分辨率图片，它更接近“只把有效事实送给大模型”。

## 是否比视觉大模型更好

### 更好的地方

`doc-textify` 在以下场景可能比通用视觉大模型更好：

1. 需要可复查的 OCR 文本。
2. 需要坐标、bbox、置信度、误差范围。
3. 需要稳定输出 JSON/Markdown/LLM 协议。
4. 需要批量处理大量 PDF/图片，成本敏感。
5. 下游模型没有视觉能力，只能读文本。
6. 评测目标是明确字段、图表数值、表格内容，而不是开放式描述。

### 不如视觉大模型的地方

`doc-textify` 在以下场景通常不如视觉大模型：

1. 自然场景理解，例如“这张照片发生了什么”。
2. 物体关系、情绪、意图、隐喻、风格判断。
3. 需要识别非文字视觉内容，但没有规则或专用解析器。
4. 图像内容高度开放，无法预先定义输出协议。
5. 需要从图片中推断常识，而不仅是提取事实。

## 正确评估方法

不要用“让文生图复原原图”作为主要指标。那会混入生成模型自己的能力，无法判断文本化是否真的好。

建议评估链路：

~~~text
图片/PDF -> doc-textify -> .llm.txt / JSON -> 文本大模型问答 -> 与人工答案比较
~~~

核心指标：

- OCR 字符准确率
- 阅读顺序准确率
- 表格结构准确率
- 图表数据匹配率
- 下游文本大模型问答正确率
- 平均 token 成本
- 平均处理耗时

## 当前项目样例结果

`对比图.jpg` benchmark：

- 综合得分：97.84%
- 关键术语：100%
- 面板/轴标签：100%
- chart_data：94.59%
- 可用性：100%

这说明在该类工程图表/文档图像上，项目已经可以作为“文本大模型的视觉预处理器”。但这个结论不能外推到所有图像类型。

## 来源

- OpenAI Images and Vision 文档：https://developers.openai.com/api/docs/guides/images-vision
- Anthropic Claude Vision 文档：https://platform.claude.com/docs/en/build-with-claude/vision
- Google Gemini Image Understanding 文档：https://ai.google.dev/gemini-api/docs/image-understanding
