# LLM 30 天学习实录：从 PyTorch 到 RAG

一个 30 天的动手计划：从零手写训练循环、注意力机制、mini GPT，到 BERT/LoRA 微调、LLM API、Function Calling，最终落到一个完整的 RAG 问答系统。

不调包抄作业，每个环节都自己写一遍——目的是搞清楚"每一层到底在做什么"，而不是"能跑起来就行"。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | PyTorch 基础 → 手写训练循环 → MNIST → 注意力机制 → mini GPT | ✅ 完成 |
| 第 2 周 | Tokenizer → HuggingFace → BERT 微调 → LLM API → Function Calling → LoRA | ✅ 完成 |
| 第 3 周 | RAG 项目：切分、向量检索、混合检索 + 重排、生成、评估 | 🚧 进行中 |
| 第 4 周 | 简历、面试题、投递 | ⬜ 未开始 |

## 亮点结果

| 项目 | 结果 | 说明 |
|---|---|---|
| MNIST 手写数字分类 | 测试准确率 **97%+** | 3 层 MLP，5 epoch，CPU 训练 |
| BERT 情感分类微调 | 测试准确率 **≈88%** | `distilbert-base-uncased`，3000 条训练样本，2 epoch；微调前基线约 75% |
| LoRA 微调 Qwen2.5-0.5B | 只训练 **0.2%** 参数 | r=8 / alpha=16 / target=q,k,v,o_proj；adapter **15 MB**，完整模型约 1 GB |
| mini GPT（从零）| 5 万参数，loss **4.2 → 1.0** | 字符级 tokenizer + 单头因果自注意力，CPU 约 2 分钟训练 |
| Function Calling | 模型自主连续调用 2 次工具 | 比较两个城市天气，`while` 循环驱动多步工具调用 |

> 数字均为本仓库代码的实际运行结果，可在对应脚本的输出中复现。

## 目录说明

```
llm-30days/
├── week1_basics/                   PyTorch 基础与 Transformer 原理
│   ├── day1_tensors.py             张量创建/索引/广播/形状/运算（50 个练习）
│   ├── day2_autograd_manual.py     autograd 求导 + 手写梯度下降（不用 nn.Module）
│   ├── day2_full_reference.py      带逐行注释的完整版：手写版 + nn.Module 版 + 学习率实验
│   ├── day3_mnist_mlp.py           MNIST 分类：Dataset/DataLoader/MLP/CrossEntropyLoss
│   ├── day4_attention.py           Q/K/V 手推，逐步打印 shape，含因果掩码
│   └── day5_mini_gpt.py            字符级 mini GPT：嵌入 + 因果自注意力 + 训练 + 生成
│
├── week2_engineering/              工程化：用现成模型、调 API、微调
│   ├── day8_tokenizer.py           手写 BPE + HuggingFace tokenizer 对比
│   ├── day9_huggingface.py         pipeline 的 5 步拆解、datasets 加载、批量推理
│   ├── day10_finetune_bert.py      微调 BERT 做情感分类（含学习率/数据量/冻结实验）
│   ├── day11_llm_api.py            原始 HTTP → SDK → 流式 → JSON 结构化输出
│   ├── day12_function_calling.py   few-shot / CoT / 工具调用闭环 / 20 行小 Agent
│   └── day13_lora_qwen.py          LoRA 微调 Qwen2.5-0.5B，含 labels 掩码与前后对比
│
├── docs/
│   ├── 踩坑记录.md                  实际踩过的 8 个坑：现象 → 原因 → 解决
│   ├── day3_必背清单.md             20 条值得背的结论（不是 API 手册）
│   └── 数据预处理对照表.md          图像/文本分类/文本生成三种预处理路径对照
│
└── results/                        运行产物：样本可视化、loss 曲线、生成样例
```

**脚本原名对照**（之前在 `api_use/` 下的文件名）：

| 现在 | 原来 |
|---|---|
| `week1_basics/day1_tensors.py` | `torch_learn.py` |
| `week1_basics/day2_autograd_manual.py` | `autogradlearning.py` |
| `week1_basics/day3_mnist_mlp.py` | `day3.py` |
| `week1_basics/day4_attention.py` | `day4/test.py` |
| `week2_engineering/day8_tokenizer.py` | `day8/day8.py` |
| （其余 `dayN/dayN.py` 同名） | |

## 环境

- Python 3.11+（实测 3.13.6）
- PyTorch（CPU 即可跑完前两周；N 卡可选）
- 依赖见各脚本头部的说明。主要包：

```bash
pip install torch torchvision transformers datasets peft accelerate \
            openai python-dotenv matplotlib
```

- API key 放在项目根目录 `.env`：

```
DEEPSEEK_API_KEY=sk-xxxx
```

`.env` 已被 `.gitignore` 排除，不会进入版本库。

## 怎么跑

每个脚本顶部都写了"这是第几天、在学什么、预期看到什么输出"，可以直接单独运行：

```bash
python week1_basics/day5_mini_gpt.py
python week2_engineering/day13_lora_qwen.py
```

- `day1` / `day4`：不需要下载数据，纯本地计算，秒级
- `day3` / `day10`：首次会下载 MNIST / BERT 权重
- `day5`：CPU 约 1-3 分钟
- `day13`：首次下载 Qwen2.5-0.5B（约 1 GB），CPU 训练 10-25 分钟

## 踩坑记录

见 [docs/踩坑记录.md](docs/踩坑记录.md)。这些不是从文档里抄的，是实际跑错后记下来的，包含：数据集 ID 改版报错、过拟合导致风格固化、梯度未清零、双重 softmax、labels 掩码写反等。

## 后续计划

第 3 周做一个完整的 RAG 问答系统：文档切分 → 向量检索 → BM25 混合 + reranker 重排 → 带引用生成 → RAGAS 评估 + bad case 迭代 → Gradio 界面。代码将放在独立仓库，完成后这里补链接。

## 说明

仓库里的代码是学习过程的产物，优先保证"能看懂每一步在做什么"，因此注释密度远高于生产代码，部分脚本保留了实验用的多份配置和打印语句。
