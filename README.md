# Transformer LM from Scratch

本项目是基于 Stanford CS336 Assignment 1 的完整实现。
该项目的核心目标是不调用 `torch.nn` 的高层封装（如 `nn.Transformer`、`nn.Linear`、`nn.LayerNorm` 等），仅使用 PyTorch 的基础 Tensor 算子从零构建一个功能完备的 Transformer 语言模型。

## 项目结构 (Project Structure)

项目采用模块化设计，将核心算法解耦在 `basics/` 文件夹中，方便独立开发与测试：

```text
.
├── model.py                # 项目主程序：整合所有组件进行模型训练与推理生成
├── basics/                  # 核心组件库
│   ├── tokenizer.py        # BPE 分词器实现（含词表训练、编解码逻辑）
│   ├── utils.py            # Transformer 架构（Attention, RMSNorm, RoPE, SwiGLU等），以及checkpoint
│   ├── optimizer.py        # AdamW 优化器、训练循环、学习率调度
│   ├── loss.py             # Cross-Entropy 损失函数实现
│   └── model.py            # Transformer类的封装
├── tests/                  # 单元测试文件夹，验证各模块的数值精度
├── runs/                   # 实验产出：存放 Checkpoints、日志、Loss 曲线及生成结果
├── data/                   # 语料目录：存放 TinyStories 和 OpenWebText 数据集
└── README.md

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### 注意：和原项目要求不同，未使用uv进行测试，如果需要与行./tests下的测试模块：
```sh
请使用 python pytest
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

