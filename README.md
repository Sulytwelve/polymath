# LLM-learn: Production-Grade Small LLM Training Framework

`LLM-learn` is a modular, engineering-focused PyTorch repository designed for training small-to-medium Autoregressive Decoder-only Transformers (**up to 2B parameters**).

It serves both as a pedagogical sandbox and an experimental framework for comparing **five advanced architectural routing variants** (`standard`, `attn_res`, `delta_attn_res`, `mhc`, `openmythos_rdt`) alongside modern LLM components like **RoPE**, **GQA/MLA**, **QK-Norm**, **MoE FFN**, and **KV-Cache**.

---

## Key Highlights

- **Modern Model Architecture (`llm_learn/model.py`)**:
  - **Rotary Position Embedding (RoPE):** Relative positional encoding supporting long-context extrapolation.
  - **Grouped Query & Multi-Latent Attention (GQA / MLA):** Configurable K/V heads (`n_kv_heads`) or low-rank joint KV compression (`kv_lora_rank`) to minimize memory bandwidth and KV-Cache footprint.
  - **QK-Norm, SwiGLU & Sparse MoE FFN:** Pre-attention Q/K RMSNorm, SwiGLU non-linearities, and routed top-k Mixture-of-Experts (`MoEFFN`).
  - **KV-Cache Incremental Decoding:** Fast $O(1)$ per-token generation in `sample.py` (`--effort` adaptive depth).

- **Five Inter-Layer Routing & Structural Variants**:
  1. `standard`: Traditional PreNorm additive connections.
  2. `attn_res`: Kimi/Moonshot AI Attention Residuals (arXiv:2603.15031).
  3. `delta_attn_res`: May 2026 sparse delta attention routing (arXiv:2605.18855).
  4. `mhc`: DeepSeek Manifold-Constrained Hyper-Connections with Sinkhorn-Knopp projection (arXiv:2512.24880).
  5. `openmythos_rdt`: OpenMythos / Fable 5 Recurrent-Depth Transformer (`Prelude->Recurrent->Coda`) with Depth-wise LoRA and guaranteed contraction mapping spectral radius ($\rho(A) < 1$).

- **Production Engineering (`llm_learn/train.py` & `llm_learn/prepare_data.py`)**:
  - **YAML Configuration System (`configs/*.yaml`)**: Clean, reproducible experiments (`tiny`, `small`, `medium`, `large`).
  - **BPE Tokenization (`tiktoken`)**: Seamless integration with OpenAI's `cl100k_base` and `o200k_base` vocabularies alongside legacy character-level tokenization (`char`).
  - **`numpy.memmap` Binary Dataset**: Zero-overhead random window slicing across GB-scale training corpora.
  - **Distributed & Mixed Precision Training**: Supports **AMP** (`autocast` + `GradScaler`), **Gradient Accumulation**, **Gradient Clipping**, and multi-GPU **Distributed Data Parallel (DDP)** via `torchrun`.
  - **Full Checkpoint Resumption**: Seamless `--resume` support preserving RNG state, optimizer state, and step progression.

---

## Quick Start

### 1. Environment Setup
Install dependencies and sync virtual environment using [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 2. Fast CPU / Tiny Verification (`tiny.yaml`)
Run a quick test training loop using character-level tokenization and tiny model configuration:
```bash
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/tiny.yaml --iters 100 --eval_interval 20
```

### 3. Preprocessing Custom Text Corpus (`prepare_data.py`)
Tokenize raw text using `tiktoken` BPE and serialize into binary `memmap` files (`train.bin`, `val.bin`):
```bash
PYTHONPATH=llm_learn uv run python llm_learn/prepare_data.py \
  --input_file path/to/your_corpus.txt \
  --output_dir data \
  --tokenizer tiktoken \
  --encoding_name cl100k_base
```

### 4. Training (`train.py`)
Train a model using preset configurations or custom CLI overrides:
```bash
# Single GPU / CPU training with small configuration (~125M params)
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/small.yaml

# Override model routing mode and training parameters from CLI
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/small.yaml --mode delta_attn_res --batch_size 16

# Multi-GPU Distributed Training (e.g., 4 GPUs) via torchrun
PYTHONPATH=llm_learn torchrun --nproc_per_node=4 llm_learn/train.py --config configs/medium.yaml
```

### 5. Resuming from Checkpoint
Continue training seamlessly from a saved `.pt` checkpoint:
```bash
PYTHONPATH=llm_learn uv run python llm_learn/train.py --config configs/small.yaml --resume checkpoints/model_standard.pt
```

### 6. Autoregressive Generation (`sample.py`)
Generate text using fast KV-Cache incremental decoding:
```bash
PYTHONPATH=llm_learn uv run python llm_learn/sample.py \
  --checkpoint checkpoints/model_standard.pt \
  --prompt "First Citizen:\n" \
  --num_tokens 200 \
  --temp 0.8 \
  --top_k 50
```

---

## Documentation
All detailed technical documentation, mathematical derivations, reverse engineering notes, and review guidelines are archived in the [`docs/`](docs/) directory:
- [`docs/CODEBASE_REFERENCE.md`](file:///home/suly/Develop/Sources/LLM-learn/docs/CODEBASE_REFERENCE.md): Full Codebase Reference & Developer Guide (Architectural Innovations, 5 Routing Modes, Directory Map, and Parameters).
- [`docs/Improve_doc.md`](file:///home/suly/Develop/Sources/LLM-learn/docs/Improve_doc.md): OpenMythos / Fable 5 Reverse Engineering & Academic Proofs (`Prelude->Recurrent->Coda`, MLA, MoE, Depth-wise LoRA, and Spectral Radius stability).
