# polymath: Codebase Reference & Developer Guide

This document provides a comprehensive guide to the `polymath` codebase for developers, researchers, and code-reviewing agents. It outlines the project's design philosophy, architectural implementations of modern Transformer components (RoPE, GQA/MLA, QK-Norm, KV-Cache) and five residual routing and structural variants (`polymath`, `standard`, `attn_res`, `delta_attn_res`, `openmythos_rdt`), directory map, data processing pipeline, and usage commands.

---

## 1. Project Overview & Architectural Innovations

`polymath` is a production-grade, modular Autoregressive Transformer (GPT-style language model) framework engineered to train small-to-medium language models (**up to 2B parameters**). The framework combines state-of-the-art LLM building blocks with five distinct inter-layer connection and signal routing paradigms:

### Core Modern Building Blocks (All Modes)
1. **Rotary Position Embedding (RoPE):**
   * Replaces absolute learned position embeddings (`pos_emb`) with relative position rotations applied directly to query and key vectors (`Q` and `K`).
   * Guarantees long-context extrapolation capability and relative distance invariance across sequence lengths up to `max_seq_len`.
2. **Grouped Query Attention (GQA) & Multi-Latent Attention (MLA):**
   * **GQA (`attn_type: gqa`):** Decouples `n_heads` (query heads) from `n_kv_heads` (key/value heads). When `n_kv_heads < n_heads`, key and value tensors are shared across groups of query heads (`n_rep = n_heads // n_kv_heads`).
   * **MLA (`attn_type: mla`):** DeepSeek-V2 joint low-rank KV compression ($c_{KV} = W_{DKV}(x) \in \mathbb{R}^{kv\_lora\_rank}$) with decoupled RoPE queries/keys ($q_{pe}, k_{pe} \in \mathbb{R}^{qk\_rope\_head\_dim}$). Reduces KV-Cache bandwidth by $>80\%$.
3. **KV-Cache Incremental Decoding & Adaptive Depth:**
   * Supports $O(1)$ per-token attention computation during autoregressive generation (`model.generate(..., use_cache=True)`).
   * Supports `--effort` (`low`, `medium`, `high`, `xhigh`) dynamic reasoning loop control for `openmythos_rdt` mode.
4. **QK-Norm, SwiGLU & Sparse MoE FFN:**
   * **QK-Norm:** Applies RMSNorm independently to `Q` and `K` prior to computing attention dot products, eliminating attention logit explosions in ultra-deep models.
   * **SwiGLU FFN (`ffn_type: swiglu`):** Replaces standard ReLU/GELU MLPs with gated `SiLU` projections: $\text{FFN}(x) = (w_1(x) \odot \text{SiLU}(w_2(x))) w_3$.


---

### Five Residual Routing & Structural Paradigms

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

#### Mode 4: PolyMath (`polymath`)
* **Source:** Unified Architecture merging Anthropic reverse-engineered Mythos / Fable 5 Recurrent-Depth Loop, Kimi AttnRes, and DeepSeek MLA.
* **Formula:** Combines Loop-Aware AttnRes historical stream mixing (`sources`) across a three-stage `Prelude -> RecurrentBlock -> Coda` pipeline:
  $$u_t = \text{AttnRes}(h_0, h_1, \dots, h_t)$$
  $$\Delta h_t = \text{TransformerBlock}(u_t) + \text{DepthWiseLoRA}(t, u_t)$$
  $$h_{t+1} = A \cdot u_t + B \cdot e + \Delta h_t$$
* **Characteristics:** Eliminates redundant mHC streams while providing maximum expressivity and adaptive inference compute depth (`--effort`).

#### Mode 5: OpenMythos Recurrent-Depth Transformer (`openmythos_rdt`)
* **Source:** Anthropic reverse-engineered Mythos / Fable 5 (Mythos + Safe restrictions for user version) & Academic Literature (Saunshi et al., 2025; arXiv:2604.07822).
* **Formula:** Organizes the architecture into a three-stage pipeline (`Prelude -> RecurrentBlock -> Coda`). The recurrent block shares a single Transformer layer weight and loops iteratively $T$ times (`max_loop_iters`):
  $$e = \text{Prelude}(h_0), \quad h_{(0)} = e$$
  $$\Delta h_t = \text{TransformerBlock}(h_t, \text{past\_kv}) + \text{DepthWiseLoRA}(t, h_t)$$
  $$h_{t+1} = A \cdot h_t + B \cdot e + \Delta h_t$$
  $$\text{Out} = \text{Coda}(h_{(T)})$$
* **Characteristics & Critical Features:**
  * **Multi-Latent Attention (MLA):** When `attn_type="mla"`, compresses Key/Value representations into a low-rank latent vector $c_{KV} = W_{DKV}(x) \in \mathbb{R}^{kv\_lora\_rank}$ alongside decoupled RoPE query/key branches ($q_{pe}, k_{pe} \in \mathbb{R}^{qk\_rope\_head\_dim}$). This cuts KV-cache bandwidth requirements by $>80\%$.

  * **Depth-Wise LoRA Perturbation (`DepthWiseLoRA`):** Injects step-conditioned low-rank matrices across loop iterations $t \in [0, T-1]$, preventing loop homogenization while maintaining weight-sharing efficiency.
  * **Spectral Radius Stability ($\rho(A) < 1$):** Parameterizes the recurrent contraction matrix $A$ as $A = \text{diag}(\exp(-\text{softplus}(A\_raw)))$. This strictly bounds $\rho(A) \in (0, 1)$, ensuring the LTI recurrence is a contraction mapping that never suffers from hidden-state explosion across arbitrary loop depths.
  * **Adaptive Computation Depth (`--effort`):** Allows dynamic adjustment of loop iterations during generation (`sample.py --effort low|medium|high|xhigh`), giving the model adaptive thinking depth without retraining.

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
│   ├── large.yaml              # (~1.5B params) Production <2B model
│   ├── polymath_tiny.yaml      # (~20M params) PolyMath RDT + MLA (CPU fast check)
│   └── polymath_small.yaml     # (~200M params) PolyMath production preset
└── polymath/
    ├── __init__.py
    ├── config.py               # Structured dataclasses (ModelConfig, TrainConfig, DataConfig)
    ├── tokenizer.py            # Unified Tokenizer API (CharTokenizer vs Tiktoken BPE)
    ├── prepare_data.py         # mmap binary dataset serializer (train.bin, val.bin)
    ├── dataset.py              # CharDataset & MmapDataset + PyTorch DataLoader integration
    ├── model.py                # TransformerLM, RoPE, GQA/MLA, 5 routing modes
    ├── train.py                # Distributed training loop (AMP, DDP, GradAccum, GradClip, Resume)
    └── sample.py               # Autoregressive sampling script (`--effort` adaptive depth)
```

---

## 3. Training & Data Pipeline

### Step 1: Data Preprocessing (`prepare_data.py`)
Instead of loading entire text files into RAM during training, `prepare_data.py` tokenizes large text corpora using `tiktoken` (`cl100k_base` or `o200k_base`) and writes `numpy.memmap` binary files (`train.bin`, `val.bin`) to disk using `uint16` or `uint32` storage:
```bash
# Tokenize a custom text dataset into data/train.bin and data/val.bin
PYTHONPATH=polymath uv run python polymath/prepare_data.py --input_file corpus.txt --output_dir data --tokenizer tiktoken --encoding_name cl100k_base
```

### Step 2: Training (`train.py`)
`train.py` supports structured YAML configs (`configs/*.yaml`) alongside CLI argument overrides. It natively features:
* **Gradient Accumulation (`gradient_accumulation_steps`) & Gradient Clipping (`grad_clip`)**
* **Automatic Mixed Precision (`use_amp`):** BF16 / FP16 `autocast` with `GradScaler`
* **Distributed Data Parallel (`DDP`):** Multi-GPU training via `torchrun`
* **Checkpoint Resumption (`--resume`):** Restores model weights, optimizer states, schedule step, and random number generator state.
* **Spectral Radius Monitoring (`rho(A)`):** Automatically monitors internal LTI contraction mapping stability when in `polymath` / `openmythos_rdt` mode.

```bash
# Train using a preset YAML configuration on single GPU or CPU
PYTHONPATH=polymath uv run python polymath/train.py --config configs/small.yaml

# Train PolyMath Unified Architecture with Multi-Latent Attention
PYTHONPATH=polymath uv run python polymath/train.py --config configs/polymath_tiny.yaml

# Override model routing mode and training iterations from CLI
PYTHONPATH=polymath uv run python polymath/train.py --config configs/small.yaml --mode polymath --iters 5000

# Multi-GPU Distributed Training (e.g., 4 GPUs)
PYTHONPATH=polymath torchrun --nproc_per_node=4 polymath/train.py --config configs/medium.yaml
```

### Step 3: Autoregressive Sampling (`sample.py`)
`sample.py` reads a saved checkpoint, automatically rebuilds the exact architecture configuration and tokenizer, and generates text using $O(1)$ step KV-Cache decoding and `--effort` adaptive computation depth:
```bash
PYTHONPATH=polymath uv run python polymath/sample.py --checkpoint checkpoints/model_polymath.pt --prompt "Once upon a time" --num_tokens 300 --temp 0.8 --effort high
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
| `model.mode` | `str` | `"standard"` | Routing mode: `polymath`, `standard`, `attn_res`, `delta_attn_res`, `openmythos_rdt` |
| `model.attn_type` | `str` | `"gqa"` | Attention type: `gqa` (Grouped Query) or `mla` (Multi-Latent Attention) |
| `model.kv_lora_rank` | `int` | `64` | Latent KV compression rank when `attn_type="mla"` |

| `model.max_loop_iters` | `int` | `6` | Default loop count $T$ for `polymath` / `openmythos_rdt` mode |
| `model.qk_norm` | `bool` | `True` | Apply RMSNorm to Q and K before dot product |
| `train.batch_size` | `int` | `32` | Batch size per GPU / worker |
| `train.gradient_accumulation_steps` | `int` | `4` | Number of micro-steps before optimizer update |
| `train.use_amp` | `bool` | `True` | Enable BF16/FP16 automatic mixed precision |
