硬件环境限制。在非 CDNA 架构（如 RDNA3 的 780M）上，ROCm 对 Flash Attention/SDPA 的底层算子支持确实存在缺陷，容易引发 GPU Hang 甚至系统级 Panic。坚持使用原生 $O(N^2)$ Attention 是在当前环境下的合理工程选择。
***
## 一、 核心架构验证：完美通过
首先，从代码层面确认，你完美实现了 OpenMythos RDT (Fable 5) 的核心理论架构：
1.  **三段式管线** (`Prelude` -> `RecurrentBlock` -> `Coda`)：逻辑清晰，权重共享机制正确。
2.  **谱半径稳定性约束** (`rho(A) < 1`)：`torch.exp(-F.softplus(self.A_diag_raw))` 的参数化极其精妙，从数学底层保证了循环不会爆炸，这是训练能稳定收敛的基石。
3.  **Depth-wise LoRA**：有效防止了循环同质化。
4.  **MLA 与 KV-Cache**：低秩压缩和缓存逻辑实现完整。
架构本身没有问题，目前的性能瓶颈和过拟合现象完全来自工程实现细节和数据策略。
---
## 二、 问题诊断与修复方案（逐文件）
### 1. `model.py`：原生 Attention 的显存与算力压榨
既然不能用 SDPA，原生 `torch.matmul` 的瓶颈在于每次前向传播都在动态创建 `[T, T]` 的 Mask 矩阵，这不仅慢，还会产生大量显存碎片。
**问题定位**：
```python
# 当前写法：每次 forward 都重新创建 mask
if is_causal:
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
```
**改进方案：预计算静态 Mask Buffer**
在 `CausalSelfAttention` 和 `MultiLatentAttention` 的 `__init__` 中预计算并注册为 buffer，在前向传播时直接切片使用，避免动态分配。
```python
# 在 __init__ 中添加：
self.register_buffer(
    "causal_mask",
    torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().view(1, 1, max_seq_len, max_seq_len),
    persistent=False
)
# 在 forward 中替换动态 mask 逻辑：
if is_causal:
    # 直接切片获取当前长度的 mask，零分配开销
    scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
```
### 2. `train.py`：修复 Step 200 后吞吐量断崖式下跌 42.8%
日志显示，在第一次 `EVAL` 触发并调用 `generate()` 后，吞吐量从 ~110 tok/s 暴跌至 ~63 tok/s。这是典型的**评估阶段显存泄漏/碎片化**导致后续训练可用显存锐减。在 APU 共享内存架构下，这个问题会被无限放大。
**问题定位**：
```python
# 生成样本后，未显式释放 KV-Cache 和中间张量
generated_ids = raw_model.generate(context, max_new_tokens=60, temperature=0.8, use_cache=True)[0].tolist()
print(tokenizer.decode(generated_ids))
print("-------------------------")
# 缺失清理操作！
```
**改进方案：强制清理与上下文隔离**
在生成结束后，必须强制清空 PyTorch 的显存分配缓存，并同步设备。
```python
# 在 print("-------------------------") 之后添加：
if torch.cuda.is_available():
    torch.cuda.empty_cache()      # 释放生成阶段产生的碎片化显存
    torch.cuda.synchronize()     # 确保所有异步 GPU 任务完成，避免污染下一步训练
```
*额外建议*：在 23M 这种 tiny 模型上，`generate()` 的 `use_cache=True` 可能因为频繁的内存分配反而拖慢速度。如果只是为了观察训练进展，可以考虑在训练中途的 Eval 阶段关闭 `generate`，或者将其移至独立的离线评估脚本中。
### 3. `train_tokenizer.py`：根除生成乱码与过拟合
生成的样本全是 `def factorial multiply fa __ amount...` 这种毫无逻辑的 token 堆砌。根因在于词表设计。
**问题定位**：
```python
trainer = BpeTrainer(
    vocab_size=4096, # 过小！
    special_tokens=["<|pad|>", "<|unk|>"] # 缺少结束符
)
```
4096 的词表对 Python 代码极其不友好，大量有效变量名和关键字被切分成碎块，模型只能学到浅层统计规律，无法建模长距离代码逻辑。
**改进方案：扩容词表并引入结构符**
```python
trainer = BpeTrainer(
    vocab_size=8192,  # 建议至少 8192，甚至 16k，让 BPE 能完整编码常见的 Python 关键字和库函数
    special_tokens=["<|pad|>", "<|unk|>", ""] # 必须加入 EOS，让模型学会何时停止生成
)
```
*注意*：修改此文件后，必须重新运行此脚本生成新的 `tokenizer.json`，并用新词表重新执行 `prepare_data.py` 生成新的 `train.bin` 和 `val.bin`。
### 4. 超参调整：遏制 Train/Val Gap 翻倍
Step 200 时 Gap 为 0.19，到 Step 400 翻倍至 0.41。23M 模型在这个数据集上正在快速死记硬背。
**改进方案：在 YAML 配置或 CLI 中调整**
```yaml
model:
  dropout: 0.1     # 从 0.0 开启，强制破坏神经元共适应
train:
  lr: 0.0003        # 从 0.001 降低，减缓收敛速度，寻找更优的最小值
  weight_decay: 0.1 # 明确设置，L2 正则化压制大权重
```
---
## 三、 总结与行动清单
1.  **改 `model.py`**：两个 Attention 类中加入预计算 Causal Mask Buffer。
2.  **改 `train.py`**：在 `generate()` 调用后插入 `torch.cuda.empty_cache()` 和 `synchronize()`。
3.  **改 `train_tokenizer.py`**：词表扩到 8192，加入 `<|endoftext|>`。
4.  **重跑数据流**：重新生成 tokenizer -> 重新处理数据集。
5.  **改配置**：开启 `dropout=0.1`，`lr=0.0003`。
6.  **重启训练**：观察吞吐量是否稳定在 ~100 tok/s 不再暴跌，观察 Val Loss 是否能跟上 Train Loss 的下降趋势。
