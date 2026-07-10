import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 8          # GQA KV heads (if < n_heads, Grouped Query Attention is used)
    d_ff: Optional[int] = None   # If None, defaults to SwiGLU multiple of d_model
    n_layers: int = 6
    vocab_size: int = 100277     # Default for tiktoken cl100k_base (or set dynamically by CharDataset)
    max_seq_len: int = 512       # Maximum sequence length / context window
    rope_theta: float = 10000.0  # Rotary position embedding base frequency
    mode: str = "polymath"       # "polymath" (Unified RDT+AttnRes+MLA+MoE), "standard", "attn_res", "delta_attn_res", "openmythos_rdt"
    attn_res_mode: str = "full"  # "full" or "block" for attn_res / polymath
    block_size_layers: int = 2   # Number of layers per block when attn_res_mode == "block"
    dropout: float = 0.0
    qk_norm: bool = True         # Apply RMSNorm to Q and K before attention dot product

    # Options for PolyMath / Anthropic reverse-engineered Mythos (Fable 5) architecture
    attn_type: str = "gqa"       # "gqa" or "mla" (Multi-Latent Attention)
    kv_lora_rank: int = 64       # Latent KV compression rank when attn_type == "mla"
    qk_rope_head_dim: int = 32   # Decoupled RoPE head dim when attn_type == "mla"

    prelude_layers: int = 1      # Number of prelude transformer blocks when mode in ("polymath", "openmythos_rdt")
    coda_layers: int = 1         # Number of coda transformer blocks when mode in ("polymath", "openmythos_rdt")
    max_loop_iters: int = 6      # Max loop iterations T for recurrent block when mode in ("polymath", "openmythos_rdt")
    uniform_loop_sampling: bool = True  # Uniformly sample T in [1, max_loop_iters] during training
    lora_rank: int = 16          # Depth-wise LoRA rank injected across loop iterations

    def __post_init__(self):
        if self.d_ff is None:
            # Standard SwiGLU hidden dimension calculation ~ (8/3) * d_model rounded up to multiple of 64
            hidden_dim = int(2 * 4 * self.d_model / 3)
            self.d_ff = 64 * ((hidden_dim + 63) // 64)
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        valid_modes = {"polymath", "standard", "attn_res", "delta_attn_res", "openmythos_rdt"}
        assert self.mode.lower() in valid_modes, f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}"
        assert self.attn_type.lower() in {"gqa", "mla"}, f"Invalid attn_type '{self.attn_type}'"

@dataclass
class TrainConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_iters: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True          # Automatic Mixed Precision
    eval_interval: int = 100
    eval_iters: int = 20          # Number of batches to evaluate for validation loss
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    resume_checkpoint: Optional[str] = None
    activation_checkpointing: bool = False  # Trade computation for memory at <=2B scale
    compile_model: bool = False             # Use torch.compile for optimized execution

@dataclass
class TokenizerConfig:
    tokenizer_type: str = "char"  # "char" or "tiktoken"
    tiktoken_encoding: str = "cl100k_base"  # "cl100k_base" or "o200k_base"

@dataclass
class DataConfig:
    dataset_type: str = "char_memory"  # "char_memory" (in-memory CharDataset) or "mmap" (numpy binary files)
    input_file: Optional[str] = None   # Raw text file path (for char_memory or on-the-fly preparation)
    mmap_data_dir: Optional[str] = None # Directory containing train.bin and val.bin
    num_workers: int = 0

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        model_dict = raw_dict.get("model", {})
        train_dict = raw_dict.get("train", {})
        tokenizer_dict = raw_dict.get("tokenizer", {})
        data_dict = raw_dict.get("data", {})

        return cls(
            model=ModelConfig(**model_dict),
            train=TrainConfig(**train_dict),
            tokenizer=TokenizerConfig(**tokenizer_dict),
            data=DataConfig(**data_dict)
        )

    def to_yaml(self, yaml_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def update_from_args(self, args: Any):
        """
        Update configuration using non-None attributes from argparse namespace or dictionary.
        """
        args_dict = vars(args) if hasattr(args, "__dict__") else args
        if not isinstance(args_dict, dict):
            return

        # Map common CLI arguments to config sections
        model_keys = set(ModelConfig().__dict__.keys())
        train_keys = set(TrainConfig().__dict__.keys())
        tokenizer_keys = set(TokenizerConfig().__dict__.keys())
        data_keys = set(DataConfig().__dict__.keys())

        for k, v in args_dict.items():
            if v is None:
                continue
            if k == "iters":  # CLI shortcut for max_iters
                self.train.max_iters = v
            elif k == "block_size":  # CLI shortcut for max_seq_len
                self.model.max_seq_len = v
            elif k in model_keys:
                setattr(self.model, k, v)
            elif k in train_keys:
                setattr(self.train, k, v)
            elif k in tokenizer_keys:
                setattr(self.tokenizer, k, v)
            elif k in data_keys:
                setattr(self.data, k, v)
            elif k == "tokenizer":
                self.tokenizer.tokenizer_type = v
