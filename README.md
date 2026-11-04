# 🧩 Machine Learning from Scratch

从基础机器学习到 Transformer 与大语言模型，手写代码理解模型背后的数学、架构和训练过程。

这个仓库记录我的学习与实现过程。目标不只是让代码运行起来，更是弄清楚：数据如何流经模型、参数如何更新，以及每个模块为什么这样设计。

---

## 🎯 学习目标
- 补齐机器学习、深度学习和 LLM 所需的基础知识
- 从数学公式出发，用代码实现核心算法和模型
- 理解 Transformer，并实现一个可训练、可文本生成的小型 GPT
- 逐步学习语言模型的数据处理、预训练、微调、评估与推理
- 在计算资源允许的范围内，从零训练自己的语言模型

## ✅ 实现原则
- **核心模块自己写**：学习 Attention 时，不直接调用框架内置 Attention / Transformer 高层接口
- **基础工具合理用**：依赖 NumPy、PyTorch 的张量运算与自动求导，聚焦模型原理
- **紧盯张量形状**：理清每一步输入、输出、各维度的含义
- **用实验验证理解**：检查梯度、参数更新、损失变化，验证因果掩码防止未来信息泄露
- **先最小实现，再迭代扩展**：单头注意力 → 完整模块，单步参数更新 → 完整训练循环

## 📌 当前进度：Transformer
目前正在实现单头 Self-Attention
- [x] 推导 Q、K、V 与缩放点积注意力
- [x] 实现单头 Self-Attention 前向计算
- [x] 支持 batch 输入，封装为 `nn.Module`
- [x] 理解损失函数与反向传播
- [x] 手动实现梯度下降，验证参数变化与损失下降
- [x] 完成 causal mask，验证无法读取未来信息
- [ ] 实现 Multi-Head Attention
- [ ] 实现前馈网络、残差连接与 LayerNorm
- [ ] 组合完整 Transformer Block

## 🗺️ 学习路线
> 仅为学习计划，不代表仓库已全部实现

### 1. 机器学习基础
- 线性回归、逻辑回归
- 损失函数、梯度下降、正则化
- 数据集划分、过拟合、模型评估
- 决策树、K-Means、PCA 等经典算法

### 2. 深度学习基础
- 多层感知机、激活函数
- 反向传播与自动求导
- 参数初始化、优化器
- 训练循环、模型保存
- 归一化、Dropout、训练稳定性

### 3. Transformer
- Token Embedding、位置编码
- Self-Attention、Multi-Head Attention
- Causal Mask、Padding Mask
- 前馈网络、残差连接、LayerNorm
- Encoder / Decoder、Cross-Attention
- 原版 Transformer 和 GPT 的架构差异

### 4. 从零训练小型语言模型
- 字符级建模、分词、词表构建
- Next-token prediction 输入与标签构造
- 小型 GPT 实现
- 交叉熵损失训练
- 训练/验证集划分，追踪损失与生成样例
- 自回归生成，temperature、top-k / top-p 采样
- Checkpoint 保存与加载

先跑通完整链路，再根据数据、算力扩展模型。

### 5. LLM 进阶知识
- **架构**：RMSNorm、RoPE、SwiGLU、GQA
- **数据**：清洗、去重、分词、数据质量评估
- **训练**：学习率调度、混合精度、梯度累积、分布式基础
- **微调**：SFT、LoRA、偏好优化基础
- **推理**：KV Cache、量化、推理加速
- **评估**：困惑度、任务表现、生成质量

## 📓 实验记录
每个主题尽量包含：
1. 原理与关键公式
2. 核心代码 + 张量形状说明
3. 最小可运行示例、正确性校验
4. 实验配置、结果、局限
5. 踩坑记录、修复思路、学习笔记

保留调试时的原始疑问，例如：**为什么参数没有更新？mask 为什么要加在 softmax 之前？**，把调试过程也作为学习材料。

## 🛠️ 工具栈
Python | NumPy | PyTorch
依赖随模块单独补充。

## ℹ️ 关于仓库
个人持续更新的学习仓库。优先保证可读性与原理清晰，性能优化在正确性验证之后再迭代。

---

# 🧩 Machine Learning from Scratch
From fundamental machine learning to Transformer and large language models. Handwritten code to understand the math, architecture and training pipeline behind models.

This repo records my learning & implementation journey. The goal is not merely making code run, but understanding how data flows through models, how parameters are updated, and why each component is designed this way.

## 🎯 Goals
- Build foundations for ML, DL and LLM
- Implement core algorithms from mathematical formulas
- Master Transformer and build a small trainable GPT with text generation
- Learn data processing, pretraining, fine-tuning, evaluation and inference for language models
- Train a small language model from scratch within computing limits

## ✅ Implementation Principles
- **Handwrite core modules**: No direct high-level Attention / Transformer APIs when learning attention
- **Use basic utilities wisely**: Leverage NumPy and PyTorch tensor & autograd, focus on model logic
- **Track tensor shapes**: Clarify dimensions of inputs and outputs
- **Validate with experiments**: Inspect gradients, parameter updates, loss and causal mask
- **Minimal first, then expand**: Single-head attention → full block, single update → complete training loop

## 📌 Current Progress: Transformer
Working on single-head self-attention
- [x] Derive Q, K, V and scaled dot-product attention
- [x] Forward pass for single-head self-attention
- [x] Batch support, wrapped as `nn.Module`
- [x] Loss function and backpropagation
- [x] Manual gradient descent, verify parameter & loss change
- [x] Causal mask implementation
- [ ] Multi-Head Attention
- [ ] Feed-forward network, residual connection, LayerNorm
- [ ] Full Transformer Block


## 🗺️ Roadmap
> Learning plan, not all implemented yet
### 1. Classical ML
Linear & Logistic Regression | Loss & Gradient Descent | Regularization | Train/Test Split | Overfitting | Decision Tree, K-Means, PCA

### 2. Deep Learning Basics
MLP | Activation | Backprop & Autograd | Initialization | Optimizer | Training Loop | Normalization & Dropout

### 3. Transformer
Embedding & Positional Encoding | Self / Multi-Head Attention | Causal / Padding Mask | FFN, Residual, LayerNorm | Encoder / Decoder & Cross-Attention | GPT vs vanilla Transformer

### 4. Small Language Model from Scratch
Character-level modeling | Tokenization & Vocab | Next-token prediction | Small GPT | CrossEntropy Loss | Train / Val split | Autoregressive Sampling (temperature, top-k, top-p) | Checkpoint

### 5. Advanced LLM
RMSNorm, RoPE, SwiGLU, GQA | Data cleaning & tokenization | LR scheduler, mixed precision, gradient accumulation | SFT, LoRA | KV Cache, quantization | Perplexity & generation evaluation

## 📓 Experiment Log
Each module includes: theory, formula, annotated code, tensor shape notes, minimal demo, results, bugs and takeaways.

## 🛠️ Stack
Python | NumPy | PyTorch

## ℹ️ About
Personal learning repo. Readability and correctness come first; performance optimizations are added after verification.

---
### GitHub 仓库简介（直接复制）
