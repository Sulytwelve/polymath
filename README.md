# PolyMath

**Unified ≤2B Architecture merging Anthropic reverse-engineered Mythos/Fable 5 Recurrent-Depth Loop, Kimi AttnRes, and DeepSeek MLA/MoE.**

Built for personal research, experimentation, and training small-to-medium language models on consumer hardware.

---

## Architecture Overview

PolyMath unifies the most efficient architectural innovations into a single, highly expressive model:

- **Recurrent-Depth Loop (Anthropic Mythos/Fable 5):** Three-stage pipeline (`Prelude -> RecurrentBlock -> Coda`). The recurrent block shares weights and loops iteratively, guaranteeing contraction mapping stability ($\rho(A) \in (0, 1)$) and enabling adaptive computation depth (`--effort`).
- **Attention Residuals (Kimi/Moonshot AI):** Replaces additive connections with depth-wise content-based attention over historical layer states.
- **Multi-Latent Attention (DeepSeek):** Joint low-rank KV compression ($c_{KV} \in \mathbb{R}^{\text{kv\_lora\_rank}}$) with decoupled RoPE query/key branches. Cuts KV-cache bandwidth by >80%.
- **Sparse Mixture-of-Experts (MoE):** 1 always-active shared expert + $N$ routed experts, top-$k$ gating with load-balancing auxiliary loss.
- **RoPE & QK-Norm:** Rotary embeddings for long-context extrapolation and pre-attention RMSNorm to prevent logit explosions.

---

## Project Structure

```
polymath/
├── configs/
│   ├── polymath_tiny.yaml      # (~20M params) PolyMath CPU fast check
│   ├── polymath_small.yaml     # (~200M params) PolyMath production preset
│   └── polymath_2b.yaml        # (~2.0B params) PolyMath 2B scale preset
└── polymath/
    ├── __init__.py
    ├── config.py               # Structured dataclasses
    ├── tokenizer.py            # Unified Tokenizer API (CharTokenizer vs Tiktoken)
    ├── prepare_data.py         # mmap binary dataset serializer (train.bin, val.bin)
    ├── dataset.py              # PyTorch DataLoader integration
    ├── model.py                # Core PolyMath architecture
    ├── train.py                # Distributed training loop (AMP, DDP, GradAccum)
    └── sample.py               # Autoregressive sampling (`--effort` adaptive depth)
```

---

## Quick Start

### 1. Environment Setup

```bash
uv sync
```

### 2. Data Preparation

Tokenize raw text corpus into memory-mapped binary files:

```bash
PYTHONPATH=polymath uv run python polymath/prepare_data.py \
  --input_file path/to/your_corpus.txt \
  --output_dir data \
  --tokenizer tiktoken \
  --encoding_name cl100k_base
```

### 3. Training

```bash
# Fast verification (~20M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/polymath_tiny.yaml

# Production preset (~200M params)
PYTHONPATH=polymath uv run python polymath/train.py --config configs/polymath_small.yaml

# Multi-GPU Distributed Training (e.g., 4 GPUs)
PYTHONPATH=polymath torchrun --nproc_per_node=4 polymath/train.py --config configs/polymath_small.yaml
```

Training features:
- **AMP** (BF16/FP16 autocast + GradScaler)
- **Gradient accumulation** and **gradient clipping**
- **MoE auxiliary loss** for load balancing
- **Spectral radius $\rho(A)$ monitoring**

### 4. Text Generation

```bash
PYTHONPATH=polymath uv run python polymath/sample.py \
  --checkpoint checkpoints/model_polymath.pt \
  --prompt "Once upon a time" \
  --num_tokens 300 \
  --temp 0.8 \
  --top_k 50 \
  --effort high
```

The `--effort` flag controls adaptive computation depth (loop iterations):
| Effort | Loop Iterations | Use Case |
|---|---|---|
| `low` | 1 | Fast drafting |
| `medium` | 3 | Default balance |
| `high` | 6 | Complex reasoning |
| `xhigh` | 12 | Maximum compute |

---

## Scaling to ~2B Parameters

Weight-sharing in the recurrent block combined with Loop-Aware AttnRes and MoE provides massive model capacity while keeping active computation minimal. The `configs/polymath_2b.yaml` scales to ~2.0B parameters.

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
```

## Documentation
- Detailed internal developer guide: [CODEBASE_REFERENCE.md](docs/CODEBASE_REFERENCE.md)
- Mathematical proofs and reverse-engineering notes: [Improve_doc.md](docs/Improve_doc.md)
