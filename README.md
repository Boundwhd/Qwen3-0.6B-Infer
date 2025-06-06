# 🚀 Qwen3 大模型推理框架

> **从零实现 Qwen3 模型架构 | 官方权重支持 | 本地推理引擎**

## 📖 项目介绍

本项目是从零实现的 Qwen3 模型推理框架，支持加载官方预训练权重进行本地部署推理。系统实现了 Qwen3 (非 MOE 版本) 的完整模型架构，包含高效自回归生成能力，并通过两个核心脚本提供开箱即用的功能：

1. **`script_api.py`** - 实时交互式聊天系统
2. **`script_test.py`** - 模型性能基准测试工具

![](https://img.shields.io/badge/PyTorch-2.5+-orange)
![](https://img.shields.io/badge/Python-3.9+-blue)
![](https://img.shields.io/badge/License-Apache2.0-green)

## 🌟 核心特性
### 🧠 极简实现
- **仅不到700行代码** - 世界最简单的生产级大模型实现加应用
- 无复杂封装，100%透明实现大模型算法与推理全流程
- 精简架构设计，特别适合教学和学习使用

### 🚫 摆脱Transformers依赖
- **纯PyTorch原生实现** - 零第三方大模型库依赖
- 彻底摆脱transformers/huggingface生态
- 手把手实现KV缓存、PD分离、自回归生成等核心技术与算法

### 🌐 跨平台部署
- 支持NVIDIA GPU加速
- 完整CPU推理支持
- Apple Silicon芯片MPS加速
- 单行代码切换设备

## 🚀 快速开始

1. 加载官方权重并导出到.pth文件
```bash
cd src
python load.py --model_name Qwen/Qwen3-0.6B --output_file ../qwen3_0.6B_weights.pth
```

2. 运行交互式聊天机器人
```bash
python script_api.py --max_length 512 --device "cpu" 
```

3. 运行性能测试脚本
```bash
python script_test.py --prompt_len 128 --output_len 128 --device "cpu"
```

## 🖥️效果展示
1. 交互对话演示
```bash
Bound@MacBook-Pro src % python script_api.py --max_length 512 --device "cpu"
Start loading model weight......
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:00<00:00]
✅ Model loaded successfully.
💡 Deep thinking mode: Disabled
🔁 Enter your prompts below. Type 'exit' to quit.

User: Hi

Assistant: Hello! How can I assist you today?

User: Tell me something about a large language model

Assistant: A large language model (LLM) is a type of artificial intelligence model that can understand and generate human language. These models are trained on vast amounts of text data to learn patterns and understand context. They can perform a wide range of tasks, from writing text to answering questions, translating between languages, and even creating creative content. LLMs are used in various applications, including language translation, customer service, content creation, and more.
```

2. 性能测试结果
```bash
Bound@MacBook-Pro src % python script_test.py -o 32 -p 32 -w 1 -t 2 -d "cpu"
Start loading model weight......
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:00<00:00]
✅ Model loaded successfully.
Device: cpu
Data Type: torch.bfloat16
Input Length: 32 tokens
Output Length: 32 tokens
Number of Tests: 2
Warmup Rounds: 1
Complete test 1/2
Complete test 2/2

================================================================================
                         Model Performance Test Results
================================================================================
Device:                       cpu
Average Prefill Time:         368.21 ms
--------------------------------------------------------------------------------
Total Average Decode Time:    1113.65 ms
Average Decode Time per Token:34.80 ms
Decode Throughput:            28.73 tokens/s
--------------------------------------------------------------------------------
```
