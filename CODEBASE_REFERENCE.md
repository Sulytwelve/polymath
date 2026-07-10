# llm_learn: Codebase Reference & Developer Guide

This document provides a comprehensive guide to the `llm_learn` codebase for developers, researchers, and code-reviewing agents. It outlines the project's design philosophy, architectural implementations of modern Transformer components (RoPE, GQA, QK-Norm, KV-Cache) and four residual routing variants (`standard`, `attn_res`, `delta_attn_res`, `mhc`), directory map, data processing pipeline, and usage commands.

---

## 1. Project Overview & Architectural Innovations

`llm_learn` is a production-grade, modular Autoregressive Transformer (GPT-style language model) framework engineered to train small-to-medium language models (**up to 2B parameters**). The framework combines state-of-the-art LLM building blocks with four distinct inter-layer connection and signal routing paradigms:

### Core Modern Building Blocks (All Modes)
1. **Rotary Position Embedding (RoPE):**
   * Replaces absolute learned position embeddings (`pos_emb`) with relative position rotations applied directly to query and key vectors (`Q` and `K`).
   * Guarantees long-context extrapolation capability and relative distance invariance across sequence lengths up to `max_seq_len`.
2. **Grouped Query Attention (GQA):**
   * Decouples `n_heads` (query heads) from `n_kv_heads` (key/value heads).
   * When `n_kv_heads < n_heads`, key and value tensors are shared across groups of query heads (`n_rep = n_heads // n_kv_heads`). This dramatically reduces parameter counts in linear KV projections and decreases inference-time KV-Cache memory consumption by $2\times$ to $8\times$.
3. **KV-Cache Incremental Decoding:**
   * Supports $O(1)$ per-token attention computation during autoregressive generation (`model.generate(..., use_cache=True)`).
4. **QK-Norm & SwiGLU FFN:**
   * **QK-Norm:** Applies RMSNorm independently to `Q` and `K` prior to computing attention dot products, eliminating attention logit explosions in ultra-deep models.
   * **SwiGLU FFN:** Replaces standard ReLU/GELU MLPs with gated `SiLU` projections: $\text{FFN}(x) = (w_1(x) \odot \text{SiLU}(w_2(x))) w_3$.

---

### Four Residual Routing Paradigms

#### Mode 1: Standard PreNorm Residuals (`standard`)
* **Source:** Traditional Transformer architecture ("Attention is All You Need").
* **Formula:** 
  $$x_{l+1} = x_l + \text{SubLayer}(\text{RMSNorm}(x_l))$$
* **Characteristics:** Simple addition, but hidden states grow with depth ($O(\sqrt{L})$), leading to "residual dilution" in deep networks.

#### Mode 2: Attention Residuals (`attn_res`)
* **Source:** Moonshot AI / Kimi Team (arXiv:2603.15031).
* **Formula:**
  $$u_l = \text{Softmax}\left(\frac{w_l^T \phi(X_{<l})}{\sqrt{d}}\right) \cdot X_{<l}$$
  $$x_l = u_l + \text{SubLayer}(\text{RMSNorm}(u_l))$$
* **Characteristics:** Replaces simple additive connections with depth-wise content-based attention over all prior cumulative hidden states $X_{<l}$. Uses a zero-initialized learned `pseudo_query` $w_l$ and parameter-free normalization $\phi$ (`RMSNormNoWeight`) to stabilize initialization.

#### Mode 3: Delta Attention Residuals (`delta_attn_res`)
* **Source:** May 2026 improvement (arXiv:2605.18855).
* **Formula:** Same attention routing as AttnRes, but the history stack $V_{<l}$ contains **layer deltas** (the raw outputs of preceding sublayers $\Delta h_i = h_i - h_{i-1}$) instead of cumulative states.
* **Characteristics:** Solves the "routing collapse" problem of AttnRes where cumulative states become highly redundant. Encourages high-contrast sparse attention routing.

#### Mode 4: Manifold-Constrained Hyper-Connections (`mhc`)
* **Source:** DeepSeek-AI (arXiv:2512.24880).
* **Formula:** Splits representation into $N$ parallel streams ($S_l$).
  $$u_l = \text{Softmax}(H_l^{pre}) \cdot x_l \quad \text{(Aggregate streams)}$$
  $$y_l = \text{SubLayer}(\text{RMSNorm}(u_l))$$
  $$\Delta x_l = \text{Softmax}(H_l^{post}) \cdot y_l \quad \text{(Distribute back)}$$
  $$x_{l+1} = H_l^{res\_mixed} \cdot x_l + \Delta x_l \quad \text{(Mix streams)}$$
  $$H_l^{res\_mixed} = (1 - \alpha_l) I + \alpha_l S_l$$
* **Characteristics:** $S_l$ is projected onto the **Birkhoff Polytope** (doubly stochastic matrices) using the iterative **Sinkhorn-Knopp** algorithm. Because doubly stochastic matrices are non-expansive (spectral radius $\le 1$), this guarantees training stability across ultra-deep architectures without residual explosions.

---

## 2. Directory & File Map

```
LLM-learn/
├── pyproject.toml              # Dependencies (torch, numpy, tqdm, tiktoken, pyyaml)
├── .gitignore                  # Git exclusions (*.pt, *.bin, data/, checkpoints/)
├── configs/                    # Preset YAML configuration recipes
│   ├── tiny.yaml               # (~15M params) Debug / fast CPU verification
│   ├── small.yaml              # (~125M params) GPT-3 Small class
│   ├── medium.yaml             # (~350M params) GPT-3 Medium class
│   └── large.yaml              # (~1.5B params) Production <2B model
└── llm_learn/
    ├── __init__.py
    ├── config.py               # Structured dataclasses (ModelConfig, TrainConfig, DataConfig)
    ├── tokenizer.py            # Unified Tokenizer API (CharTokenizer vs Tiktoken BPE)
    ├── prepare_data.py         # mmap binary dataset serializer (train.bin, val.bin)
    ├── dataset.py              # CharDataset & MmapDataset + PyTorch DataLoader integration
    ├── model.py                # TransformerLM, RoPE, GQA, QK-Norm, KV-Cache, 4 routing modes
    ├── train.py                # Distributed training loop (AMP, DDP, GradAccum, GradClip, Resume)
    └── sample.py               # Autoregressive sampling script with fast KV-Cache support
```

---

## 3. Training & Data Pipeline

### Step 1: Data Preprocessing (`prepare_data.py`)
Instead of loading entire text files into RAM during training, `prepare_data.py` tokenizes large text corpora using `tiktoken` (`cl100k_base` or `o200k_base`) and writes `numpy.memmap` binary files (`train.bin`, `val.bin`) to disk using `uint16` or `uint32` storage:
```bash
# Tokenize a custom text dataset into data/train.bin and data/val.bin
PYTHONPATH=llm_learn uv run python llm_learn/prepare_data.py --input_file corpus.txt --output_dir data --tokenizer tiktoken --encoding_name cl100k_base
```

### Step 2: Training (`train.py`)
`train.py` supports structured YAML configs (`configs/*.yaml`) alongside CLI argument overrides. It natively features:
* **Gradient Accumulation (`gradient_accumulation_steps`) & Gradient Clipping (`grad_clip`)**
* **Automatic Mixed Precision (`use_amp`):** BF16 / FP16 `autocast` with `GradScaler`
* **Distributed Data Parallel (`DDP`):** Multi-GPU training via `torchrun`
* **Checkpoint Resumption (`--resume`):** Restores model weights, optimizer states, schedule step, and random number generator state.

```bash
# Train using a preset YAML configuration on single GPU or CPU
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/small.yaml

# Override model routing mode and training iterations from CLI
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/small.yaml --mode delta_attn_res --iters 5000

# Multi-GPU Distributed Training (e.g., 4 GPUs)
PYTHONPATH=llm_learn torchrun --nproc_per_node=4 llm_learn/train.py --config configs/medium.yaml
```

### Step 3: Autoregressive Sampling (`sample.py`)
`sample.py` reads a saved checkpoint, automatically rebuilds the exact architecture configuration and tokenizer, and generates text using $O(1)$ step KV-Cache decoding:
```bash
PYTHONPATH=llm_learn uv run python llm_learn/sample.py --checkpoint checkpoints/model_standard.pt --prompt "Once upon a time" --num_tokens 300 --temp 0.8 --top_k 50
```

---

## 4. Key Configuration Parameters (`config.py`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model.d_model` | `int` | `768` | Hidden embedding dimension |
| `model.n_heads` | `int` | `12` | Number of query attention heads |
| `model.n_kv_heads` | `int` | `4` | Number of key/value heads for GQA |
| `model.max_seq_len` | `int` | `1024` | Maximum context window size |
| `model.rope_theta` | `float` | `10000.0` | Rotary frequency base parameter (`10000.0` to `500000.0`) |
| `model.mode` | `str` | `"standard"` | Routing mode: `standard`, `attn_res`, `delta_attn_res`, `mhc` |
| `model.qk_norm` | `bool` | `True` | Apply RMSNorm to Q and K before dot product |
| `train.batch_size` | `int` | `32` | Batch size per GPU / worker |
| `train.gradient_accumulation_steps` | `int` | `4` | Number of micro-steps before optimizer update |
| `train.use_amp` | `bool` | `True` | Enable BF16/FP16 automatic mixed precision |
