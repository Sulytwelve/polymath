# 4096 Vocab × 600 Steps — 基线报告

## 1. 训练曲线
| Step | Train Loss | Val Loss | Gap | Throughput (tok/s) | rho(A) |
|------|-----------|---------|-----|---------------------|--------|
| 200  | 2.40      | 2.60    | 0.19| 109                 | —      |
| 400  | 1.64      | 2.05    | 0.41| 63 ← 暴跌           | —      |
| 600  | 1.25      | 2.00    | 0.64| 62                  | —      |

*注：600 步时的具体评估（Eval）Train Loss 为 1.3559，Val Loss 刷新最低记录为 1.9958，此处表格取平时滑动平均 Train Loss `1.2494` 简写为 1.25，Gap 取 `2.00 - 1.36 = 0.64`。*

## 2. 吞吐量问题
- Step 200 EVAL 后吞吐量从 109 → 63 tok/s（-42.8%）
- 根因：`generate()` 阶段产生了显存碎片且未释放，导致后续的前向/反向传播中分配小块显存时频繁 fallback，造成巨大延迟。
- 解决：将在新版代码中通过 `torch.cuda.empty_cache()` 和 `torch.cuda.synchronize()` 解决。

## 3. 样本生成（4096 词表）
> def add is round add add function con corre add add add evaluate add add add dec add add add add accept add add add pro add add factorial add add f push add add datetime add add reverse add accept add add add accept add add add encrypt add add add factorial add 
> add add add numberlines add
- 问题：无换行、无缩进、无限堆砌关键字，词不达意，像结巴一样重复 `add`。
- 根因：4096 词表过小，对于 Python 代码中高频使用的缩进（如 `    `）、换行符（`\n`）以及内置函数缺乏专门的独立 Token，导致模型缺少 EOS 学习信号和结构化学习能力。

## 4. 结论
此为**未优化基线**。目前验证了模型确实能够稳定下降并进行拟合（Val Loss 从最初的 >5 降至 1.9958）。但模型被词表大小和训练策略束缚了手脚。

优化项（**8192 词表 + 静态 causal_mask 缓冲区 + empty_cache 防止显存碎片掉速 + dropout 0.1 防止过拟合 + 学习率降低至 0.0003 稳定后期微调**）将在接下来马上开启的 1600 步 Run 中得到全面验证。
