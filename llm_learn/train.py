import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from model import TransformerLM
from dataset import get_dataset, CharDataset, MmapDataset
from tokenizer import get_tokenizer

DEFAULT_CORPUS = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved, resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""

def setup_ddp():
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        assert torch.cuda.is_available() or torch.backends.mps.is_available() or True
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device

def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()

def get_lr(step: int, config: Config) -> float:
    warmup_iters = config.train.warmup_iters
    max_iters = config.train.max_iters
    lr = config.train.lr
    min_lr = config.train.min_lr

    if step < warmup_iters:
        return lr * (step + 1) / (warmup_iters + 1)
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

@torch.no_grad()
def estimate_loss(model: nn.Module, train_dataset, val_dataset, config: Config, device: str) -> dict:
    out = {}
    model.eval()
    for split, dataset in [("train", train_dataset), ("val", val_dataset)]:
        if dataset is None:
            continue
        losses = torch.zeros(config.train.eval_iters)
        for k in range(config.train.eval_iters):
            x, y = dataset.get_batch(config.train.batch_size, device=device)
            with torch.amp.autocast(device_type=device.split(":")[0], enabled=config.train.use_amp and device != "cpu"):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Production-grade Training Loop for <=2B Autoregressive Transformer Models.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/tiny.yaml).")
    parser.add_argument("--mode", type=str, default=None, choices=["standard", "attn_res", "delta_attn_res", "mhc", "openmythos_rdt"], help="Override model routing mode.")
    parser.add_argument("--attn_type", type=str, default=None, choices=["gqa", "mla"], help="Override attention mechanism (GQA vs Multi-Latent Attention).")
    parser.add_argument("--ffn_type", type=str, default=None, choices=["swiglu", "moe"], help="Override feed-forward block type (SwiGLU vs Sparse MoE).")
    parser.add_argument("--max_loop_iters", type=int, default=None, help="Override max loop iterations for openmythos_rdt.")
    parser.add_argument("--n_experts", type=int, default=None, help="Override routed experts count when ffn_type is moe.")
    parser.add_argument("--iters", type=int, default=None, help="Override training iterations.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--block_size", type=int, default=None, help="Override sequence max_seq_len.")
    parser.add_argument("--d_model", type=int, default=None, help="Override embedding and hidden dimension.")
    parser.add_argument("--n_heads", type=int, default=None, help="Override attention heads.")
    parser.add_argument("--n_kv_heads", type=int, default=None, help="Override GQA key/value heads.")
    parser.add_argument("--n_layers", type=int, default=None, help="Override number of layers.")
    parser.add_argument("--tokenizer", type=str, default=None, choices=["char", "tiktoken"], help="Override tokenizer type.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume training from.")
    parser.add_argument("--eval_interval", type=int, default=None, help="Interval for evaluation.")
    args = parser.parse_args()

    is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()
    if master_process:
        print(f"Executing on device: {device} (DDP world size: {ddp_world_size})")

    # Initialize configuration
    if args.config and os.path.exists(args.config):
        if master_process:
            print(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        if master_process:
            print("No YAML config provided or found, initializing default Config.")
        config = Config()

    config.update_from_args(args)
    if args.resume:
        config.train.resume_checkpoint = args.resume

    # Setup Dataset and Tokenizer
    if config.data.dataset_type == "char_memory":
        text = DEFAULT_CORPUS
        if config.data.input_file and os.path.exists(config.data.input_file):
            with open(config.data.input_file, "r", encoding="utf-8") as f:
                text = f.read()
        train_dataset = get_dataset("char_memory", block_size=config.model.max_seq_len, text=text)
        val_dataset = train_dataset
        config.model.vocab_size = train_dataset.get_vocab_size()
        if master_process:
            print(f"Loaded char_memory dataset (vocab_size: {config.model.vocab_size})")
    else:
        assert config.data.mmap_data_dir is not None, "mmap dataset requires `data.mmap_data_dir` to be set."
        train_dataset = get_dataset("mmap", block_size=config.model.max_seq_len, split="train", data_dir=config.data.mmap_data_dir)
        try:
            val_dataset = get_dataset("mmap", block_size=config.model.max_seq_len, split="val", data_dir=config.data.mmap_data_dir)
        except Exception:
            val_dataset = train_dataset
        config.model.vocab_size = train_dataset.get_vocab_size()
        if master_process:
            print(f"Loaded mmap dataset from {config.data.mmap_data_dir} (vocab_size: {config.model.vocab_size})")

    # Initialize Model
    if master_process:
        print(f"Initializing TransformerLM in '{config.model.mode}' mode with d_model={config.model.d_model}, n_layers={config.model.n_layers}...")
    model = TransformerLM(config.model, vocab_size=config.model.vocab_size).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f"Total trainable parameters: {param_count:,}")

    # Optimizer with weight decay grouping
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "pseudo_query" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.train.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.train.lr, betas=(0.9, 0.95))

    start_step = 1
    # Checkpoint Resume
    if config.train.resume_checkpoint and os.path.exists(config.train.resume_checkpoint):
        if master_process:
            print(f"Resuming checkpoint from: {config.train.resume_checkpoint}")
        checkpoint = torch.load(config.train.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if master_process:
            print(f"Resumed successfully at step {start_step}")

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank] if device != "cpu" else None)

    raw_model = model.module if is_ddp else model

    # Setup Mixed Precision Scaler
    use_amp = config.train.use_amp and device != "cpu"
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # Training Loop
    raw_model.train()
    if master_process:
        print("Starting training loop...")

    t0 = time.time()
    for step in range(start_step, config.train.max_iters + 1):
        # Set learning rate for this step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.train.gradient_accumulation_steps):
            if is_ddp:
                model.require_backward_grad_sync = (micro_step == config.train.gradient_accumulation_steps - 1)
            
            xb, yb = train_dataset.get_batch(config.train.batch_size, device=device)
            
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                loss = loss / config.train.gradient_accumulation_steps

            accum_loss += loss.item()
            scaler.scale(loss).backward()

        if config.train.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if master_process and (step % config.train.log_interval == 0 or step == start_step):
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_processed = config.train.batch_size * config.train.gradient_accumulation_steps * config.model.max_seq_len * ddp_world_size
            tokens_per_sec = tokens_processed / max(dt, 1e-5) if step > start_step else 0.0
            rho_str = ""
            if config.model.mode.lower() == "openmythos_rdt":
                rho = raw_model.get_spectral_radius()
                if rho is not None:
                    rho_str = f" | rho(A): {rho:.4f}"
            print(f"Step {step:5d}/{config.train.max_iters} | LR: {lr:.6f} | Loss: {accum_loss:.4f}{rho_str} | Throughput: {tokens_per_sec:,.0f} tokens/s")

        # Evaluation & Checkpoint Saving
        if step % config.train.eval_interval == 0 or step == config.train.max_iters:
            losses = estimate_loss(raw_model, train_dataset, val_dataset, config, device)
            if master_process:
                print(f"--> [EVAL step {step}] Train Loss: {losses.get('train', 0):.4f} | Val Loss: {losses.get('val', 0):.4f}")

                # Generate sample text
                print("--- Sample Generation ---")
                tokenizer = get_tokenizer(config.tokenizer.tokenizer_type, config.tokenizer.tiktoken_encoding, text=getattr(train_dataset, "text", None))
                prompt_ids = tokenizer.encode("\nFirst Citizen:\n") if config.tokenizer.tokenizer_type == "char" else tokenizer.encode("First Citizen:\n")
                if not prompt_ids:
                    prompt_ids = [0]
                context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                generated_ids = raw_model.generate(context, max_new_tokens=60, temperature=0.8, use_cache=True)[0].tolist()
                print(tokenizer.decode(generated_ids))
                print("-------------------------")

                # Save Checkpoint
                os.makedirs(config.train.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(config.train.checkpoint_dir, f"model_{config.model.mode}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "vocab_size": config.model.vocab_size,
                    "rng_state": torch.get_rng_state()
                }, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

    if master_process:
        print("Training successfully completed!")
    cleanup_ddp(is_ddp)

if __name__ == "__main__":
    main()
