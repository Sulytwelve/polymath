# LLM-learn

**Production-Grade Modular Transformer Framework for Training ≤2B Parameter Language Models**

A from-scratch PyTorch implementation of autoregressive decoder-only Transformers, combining state-of-the-art building blocks (RoPE, GQA/MLA, QK-Norm, SwiGLU, Sparse MoE, KV-Cache) with **five distinct inter-layer routing paradigms** — from classic PreNorm residuals to OpenMythos Recurrent-Depth Transformers.

Built for personal research, experimentation, and training small-to-medium language models on consumer hardware.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
  - [Core Building Blocks](#core-building-blocks)
  - [Five Routing Modes](#five-routing-modes)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Environment Setup](#1-environment-setup)
  - [Tiny Verification](#2-tiny-verification-cpu)
  - [Data Preparation](#3-data-preparation)
  - [Training](#4-training)
  - [Checkpoint Resume](#5-checkpoint-resume)
  - [Text Generation](#6-text-generation)
- [Configuration Presets](#configuration-presets)
- [Scaling to ~2B Parameters](#scaling-to-2b-parameters)
- [Code Review & Merge Status](#code-review--merge-status)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Documentation](#documentation)
- [License](#license)

---

## Architecture Overview

### Core Building Blocks

All five routing modes share the same modern Transformer primitives:

| Component | Description |
|---|---|
| **RoPE** | Rotary Position Embedding — relative positional encoding via complex-valued rotation of Q/K vectors. Supports long-context extrapolation with configurable `rope_theta` (10K–500K). |
| **GQA** | Grouped Query Attention — decouples query heads (`n_heads`) from KV heads (`n_kv_heads`). Reduces KV-cache memory by `n_heads/n_kv_heads` ratio. |
| **MLA** | Multi-Latent Attention (DeepSeek-V2) — joint low-rank KV compression ($c_{KV} \in \mathbb{R}^{\text{kv\_lora\_rank}}$) with decoupled RoPE query/key branches. Cuts KV-cache bandwidth by >80%. |
| **QK-Norm** | Pre-attention RMSNorm on Q and K to prevent attention logit explosions in deep networks. |
| **SwiGLU FFN** | Gated SiLU feed-forward: $\text{FFN}(x) = (W_1 x \odot \text{SiLU}(W_2 x)) W_3$ |
| **MoE FFN** | Sparse Mixture-of-Experts: 1 always-active shared expert + $N$ routed experts, top-$k$ gating with load-balancing auxiliary loss. |
| **KV-Cache** | $O(1)$ per-token incremental decoding for autoregressive generation. |

### Five Routing Modes

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Mode 1: standard          Classic PreNorm additive residuals               │
│  Mode 2: attn_res          Kimi/Moonshot AI Attention Residuals             │
│  Mode 3: delta_attn_res    Delta (sparse) Attention Residuals               │
│  Mode 4: polymath          PolyMath Unified Architecture (RDT+AttnRes+MLA)  │
│  Mode 5: openmythos_rdt    Recurrent-Depth Transformer (Prelude→Loop→Coda) │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Mode 1 — `standard` (PreNorm Residuals)
$$x_{l+1} = x_l + \text{SubLayer}(\text{RMSNorm}(x_l))$$
Traditional Transformer architecture. Simple but susceptible to residual dilution at depth.

#### Mode 2 — `attn_res` (Attention Residuals)
*Source: Moonshot AI / Kimi (arXiv:2603.15031)*

Replaces additive connections with depth-wise content-based attention over all prior cumulative hidden states. Uses zero-initialized pseudo-query and parameter-free RMSNorm.

#### Mode 3 — `delta_attn_res` (Delta Attention Residuals)
*Source: arXiv:2605.18855*

Same routing mechanism as `attn_res`, but the history stack stores **layer deltas** (raw sublayer outputs) instead of cumulative states. Solves the routing collapse problem.

#### Mode 4 — `polymath` (PolyMath Unified Architecture)
*Source: Unified architecture merging Anthropic reverse-engineered Mythos / Fable 5 Recurrent-Depth Loop, Kimi AttnRes, and DeepSeek MLA/MoE*

Unifies Loop-Aware Attention Residuals across a `Prelude -> RecurrentBlock -> Coda` pipeline with Multi-Latent Attention (MLA) and Sparse Mixture of Experts (MoE). Maximizes expressivity while keeping active parameters minimal ($O(T)$ dynamic depth via `--effort`).

#### Mode 5 — `openmythos_rdt` (Recurrent-Depth Transformer)
*Source: Anthropic reverse-engineered Mythos / Fable 5 (Mythos + Safe restrictions for user version) + academic literature*

Three-stage pipeline with weight-shared recurrent iteration:

```
Input → [Prelude Layers] → [Recurrent Block × T iterations] → [Coda Layers] → Output
```

Key features:
- **Contraction mapping stability:** $A = \text{diag}(\exp(-\text{softplus}(A_{\text{raw}})))$ guarantees $\rho(A) \in (0, 1)$
- **DepthWiseLoRA:** Step-conditioned low-rank perturbation prevents loop homogenization
- **Adaptive computation depth:** `--effort low|medium|high|xhigh` adjusts loop iterations at inference
- **Compatible with MLA + MoE** for maximum efficiency

---

## Project Structure

```
LLM-learn/
├── pyproject.toml                 # Dependencies (torch, numpy, tqdm, tiktoken, pyyaml)
├── configs/                       # YAML configuration presets
│   ├── tiny.yaml                  # ~15M params — CPU verification
│   ├── small.yaml                 # ~125M params — GPT-3 Small class
│   ├── medium.yaml                # ~350M params — GPT-3 Medium class
│   ├── large.yaml                 # ~1.5B params — Production <2B
│   ├── polymath_tiny.yaml         # ~20M params — PolyMath RDT+AttnRes+MLA+MoE (CPU check)
│   ├── polymath_small.yaml        # ~200M params — PolyMath production preset
│   └── polymath_2b.yaml           # ~2.0B params — PolyMath 2B scale preset
├── polymath/
│   ├── __init__.py
│   ├── config.py                  # Structured dataclasses + YAML/CLI config loader
│   ├── tokenizer.py               # CharTokenizer & tiktoken BPE wrapper
│   ├── prepare_data.py            # Corpus → memmap binary serializer
│   ├── dataset.py                 # CharDataset & MmapDataset + DataLoader factory
│   ├── model.py                   # All architecture: RoPE, GQA/MLA, MoE, 5 modes (820 lines)
│   ├── train.py                   # Training loop: AMP, DDP, GradAccum, Resume (322 lines)
│   └── sample.py                  # Autoregressive generation with KV-Cache & --effort
└── docs/
    ├── CODEBASE_REFERENCE.md      # Full technical reference & developer guide
    └── Improve_doc.md             # Anthropic Mythos reverse engineering & mathematical proofs
```

**Total codebase: ~1,700 lines of Python** — compact and auditable.

---

## Quick Start

### 1. Environment Setup

Requires Python ≥3.13 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

> **Note:** The default `pyproject.toml` installs CPU-only PyTorch. For GPU training (required for ≥350M models), modify the PyTorch index URL to your CUDA version. See [Scaling to ~2B](#scaling-to-2b-parameters).

### 2. Tiny Verification (CPU)

Run a quick sanity check with character-level tokenization:

```bash
# Standard Transformer (~15M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/tiny.yaml --iters 100 --eval_interval 20

# PolyMath Unified Architecture (~20M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/polymath_tiny.yaml
```

### 3. Data Preparation

Tokenize raw text corpus into memory-mapped binary files:

```bash
PYTHONPATH=polymath uv run python polymath/prepare_data.py \
  --input_file path/to/your_corpus.txt \
  --output_dir data \
  --tokenizer tiktoken \
  --encoding_name cl100k_base
```

This produces `data/train.bin` and `data/val.bin` (90/10 split by default) using `uint16` or `uint32` storage depending on vocabulary size.

### 4. Training

```bash
# Single GPU — GPT-3 Small class (~125M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/small.yaml

# Override routing mode from CLI
PYTHONPATH=polymath uv run python polymath/train.py --config configs/small.yaml --mode polymath

# PolyMath RDT + MLA + MoE production preset (~200M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/polymath_small.yaml

# Multi-GPU DDP (4× GPUs)
PYTHONPATH=polymath torchrun --nproc_per_node=4 polymath/train.py --config configs/large.yaml
```

Training features:
- **Cosine annealing** LR schedule with linear warmup
- **AMP** (BF16/FP16 autocast + GradScaler)
- **Gradient accumulation** and **gradient clipping**
- **MoE auxiliary loss** for load balancing (when `ffn_type: moe`)
- **Spectral radius $\rho(A)$ monitoring** (when `mode: polymath` or `openmythos_rdt`)

### 5. Checkpoint Resume

```bash
PYTHONPATH=polymath uv run python polymath/train.py \
  --config configs/small.yaml \
  --resume checkpoints/model_standard.pt
```

Restores model weights, optimizer state, LR schedule step, and RNG states for exact reproducibility.

### 6. Text Generation

```bash
PYTHONPATH=polymath uv run python polymath/sample.py \
  --checkpoint checkpoints/model_polymath.pt \
  --prompt "Once upon a time" \
  --num_tokens 300 \
  --temp 0.8 \
  --top_k 50 \
  --effort high
```

The `--effort` flag controls adaptive computation depth for `polymath`/`openmythos_rdt` models:

| Effort | Loop Iterations | Use Case |
|---|---|---|
| `low` | 1 | Fast drafting |
| `medium` | T/2 | Balanced |
| `high` | T (default) | Full reasoning |
| `xhigh` | 2T | Extended thinking |

---

## Configuration Presets

| Config | Params | d_model | Layers | Heads | KV Heads | Mode | FFN | Notes |
|---|---|---|---|---|---|---|---|---|
| `tiny.yaml` | ~15M | 288 | 6 | 6 | 6 | standard | SwiGLU | CPU verification, char tokenizer |
| `small.yaml` | ~125M | 768 | 12 | 12 | 4 | standard | SwiGLU | GPT-3 Small class |
| `medium.yaml` | ~350M | 1024 | 24 | 16 | 4 | standard | SwiGLU | GPT-3 Medium class |
| `large.yaml` | ~1.5B | 2048 | 24 | 32 | 8 | standard | SwiGLU | Production <2B |
| `polymath_tiny.yaml` | ~20M | 288 | 4 | 6 | 6 | polymath | MoE(4e/2k) | MLA + MoE, CPU check |
| `polymath_small.yaml` | ~200M | 768 | 6 | 12 | 4 | polymath | MoE(8e/2k) | MLA + MoE, production RDT |
| `polymath_2b.yaml` | ~2.0B | 2048 | 8 | 32 | 8 | polymath | MoE(8e/2k) | PolyMath 2B scale preset |

All configs support CLI overrides: `--mode`, `--batch_size`, `--lr`, `--iters`, etc.

---

## Scaling to ~2B Parameters

The framework is designed for ≤2B models. The existing `large.yaml` targets ~1.5B. To reach ~2B:

### Option A: Dense Standard Transformer (~2.0B)

```yaml
model:
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8
  d_ff: 5632
  n_layers: 32        # ← increase from 24 to 32
  max_seq_len: 4096
  rope_theta: 500000.0
  mode: standard       # or delta_attn_res, attn_res
  qk_norm: true
  dropout: 0.05
```

### Option B: PolyMath RDT + AttnRes + MLA + MoE (~2B total, ~800M active per token)

Weight-sharing in the recurrent block combined with Loop-Aware AttnRes and MoE provides massive model capacity while keeping active computation minimal ($O(T)$ adaptive depth via `--effort`):

```yaml
model:
  d_model: 2048
  n_heads: 32
  n_kv_heads: 8
  d_ff: 5632
  max_seq_len: 4096
  rope_theta: 500000.0
  mode: polymath
  attn_res_mode: full
  attn_type: mla
  kv_lora_rank: 256
  qk_rope_head_dim: 64
  ffn_type: moe
  n_experts: 8
  n_experts_per_tok: 2
  prelude_layers: 3
  coda_layers: 3
  max_loop_iters: 10
  uniform_loop_sampling: true
  lora_rank: 64
  qk_norm: true
  dropout: 0.05
```

### GPU Requirements for ~2B

| Config | VRAM (Training) | Recommended GPU |
|---|---|---|
| Dense 2B, bs=4, AMP | ~40 GB | A100 80GB / 2×A6000 |
| RDT+MoE 2B, bs=8, AMP | ~24 GB | A6000 / RTX 4090 |

> **Important:** Switch `pyproject.toml` PyTorch index from CPU to CUDA:
> ```toml
> [[tool.uv.index]]
> name = "pytorch-cu124"
> url = "https://download.pytorch.org/whl/cu124"
> explicit = true
> ```

---

## Code Review & Merge Status

### Branch Status

**OpenMythos RDT is already on `main`** — there is no separate feature branch. The implementation was merged in commit `3e76565` alongside the full framework upgrade in `8233fea`. The working tree is clean with no pending changes.

### Review Summary (2026-07-10)

| Component | File | Lines | Quality | Verdict |
|---|---|---|---|---|
| Model Architecture | `model.py` | 820 | ⭐⭐⭐⭐⭐ | Clean, math-correct, well-documented |
| Config System | `config.py` | 135 | ⭐⭐⭐⭐ | Functional, needs validation |
| Training Loop | `train.py` | 322 | ⭐⭐⭐⭐ | AMP+DDP+GradAccum all correct |
| Sampling/Generation | `sample.py` | 76 | ⭐⭐⭐⭐⭐ | KV-cache + effort working |
| Dataset Pipeline | `dataset.py` | 134 | ⭐⭐⭐⭐⭐ | memmap efficient |
| Tokenizer | `tokenizer.py` | 118 | ⭐⭐⭐⭐⭐ | Clean abstraction |
| Data Preparation | `prepare_data.py` | 90 | ⭐⭐⭐⭐⭐ | Correct, handles edge cases |

### Key Findings

**✅ Production Ready:**
- All five routing modes correctly implement their documented formulas
- RoPE, GQA, MLA, MoE, QK-Norm, SwiGLU, KV-Cache all functional
- Spectral radius guarantee ($\rho(A) < 1$) correctly enforced via `diag(exp(-softplus(·)))`
- DepthWiseLoRA properly prevents loop homogenization
- Training loop handles AMP, DDP, gradient accumulation, checkpoint resume correctly
- MoE load-balancing auxiliary loss implemented
- `--effort` adaptive depth works for inference-time compute scaling

**⚠️ Improvement Opportunities (Non-blocking):**
1. **No Flash Attention** — standard attention is O(n²) memory; consider `torch.nn.functional.scaled_dot_product_attention` for 4K+ sequences
2. **Sequential MoE routing** — experts are iterated in a Python loop; batched expert dispatch would improve throughput
3. **No gradient checkpointing** — needed for fitting 2B dense models in 24GB VRAM
4. **No experiment tracking** — stdout only; no TensorBoard/WandB integration
5. **Config validation** — invalid parameter combinations (e.g., `mode=standard` + `attn_type=mla`) aren't caught at config time
6. **CPU-only PyTorch in pyproject.toml** — needs manual switch for GPU training

---

## Known Limitations & Future Work

- [ ] Flash Attention / `scaled_dot_product_attention` integration
- [ ] Gradient checkpointing for large model memory efficiency
- [ ] Batched MoE expert dispatch (replace sequential loop)
- [ ] TensorBoard / WandB experiment tracking
- [ ] Config validation layer (catch invalid mode + attn_type combinations)
- [ ] Dedicated `openmythos_2b.yaml` config preset
- [ ] Multi-node distributed training (currently single-node DDP only)
- [ ] GGUF / ONNX export for local inference

---

## Documentation

Detailed technical documentation is in the [`docs/`](docs/) directory:

- **[CODEBASE_REFERENCE.md](docs/CODEBASE_REFERENCE.md)** — Full architectural reference, mathematical formulas for all five routing modes, directory map, parameter tables, and usage commands.
- **[Improve_doc.md](docs/Improve_doc.md)** — OpenMythos / Fable 5 reverse engineering notes, academic literature survey, mathematical proofs for the Prelude→Recurrent→Coda pipeline, MLA, MoE, DepthWiseLoRA, and spectral radius stability guarantees.

---

## License

This project is for personal research and educational use.
