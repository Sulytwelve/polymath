import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Optional, Tuple, List, Union, Dict, Any
try:
    from polymath.config import ModelConfig
except ImportError:
    from config import ModelConfig

# ==========================================
# 1. Normalization layers
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization with a learnable scaling parameter.
    Uses FP32 inside variance calculation to avoid overflow in mixed precision (AMP).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to float32 for stable variance calculation
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.to(torch.float32)).to(x.dtype)


class RMSNormNoWeight(nn.Module):
    """
    Parameter-free Root Mean Square Normalization.
    Used for normalizing keys in Attention Residuals (AttnRes) to prevent
    large-magnitude representations from dominating attention logits.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(variance + self.eps)).to(x.dtype)


# ==========================================
# 2. Rotary Position Embedding (RoPE)
# ==========================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative position awareness and long-context extrapolation.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, dtype=torch.float32):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate freqs along head_dim to match half rotation format
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, dtype=torch.float32)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_key_value_length: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[2]
    cos = cos[past_key_value_length : past_key_value_length + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[past_key_value_length : past_key_value_length + seq_len].unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads to match query heads for Grouped Query Attention (GQA)."""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


# ==========================================
# 3. Attention Residuals (Kimi style)
# ==========================================

class AttnResOperator(nn.Module):
    """
    Attention Residuals Operator (AttnRes) from Moonshot AI.
    Replaces standard additive residual connections with a learned, content-aware
    softmax attention mechanism over preceding layer/sub-layer outputs.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        # sources shape: [N_src, B, T, d_model]
        K = self.key_norm(sources)
        logits = torch.einsum('d, n b t d -> n b t', self.pseudo_query, K)
        weights = torch.softmax(logits, dim=0)
        out = torch.einsum('n b t, n b t d -> b t d', weights, sources)
        return out




# ==========================================
# 5. Attention Modules: GQA & Multi-Latent Attention (MLA)
# ==========================================

class CausalSelfAttention(nn.Module):
    """
    Grouped Query Causal Self-Attention with RoPE, QK-Norm, and KV-Cache.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048,
                 rope_theta: float = 10000.0, dropout: float = 0.0, qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len, base=rope_theta)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-computed static causal mask buffer: slice at forward time, zero allocation overhead
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().view(1, 1, max_seq_len, max_seq_len),
            persistent=False
        )

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        past_kv_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        total_len = past_kv_len + T

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        v = v.transpose(1, 2)  # [B, n_kv_heads, T, head_dim]

        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len=total_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, past_key_value_length=past_kv_len)

        # Update KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # Repeat K/V heads if GQA (n_kv_heads < n_heads)
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Compute Attention using SDPA if available (PyTorch >= 2.0)
        is_causal = (past_key_value is None and T > 1)
        if False: # Forced fallback
            pass
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == False, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            context = torch.matmul(weights, v_rep)

        context = context.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(context))
        return out, present_key_value


class MultiLatentAttention(nn.Module):
    """
    DeepSeek-V2 style Multi-Latent Attention (MLA).
    Uses low-rank joint KV compression (`kv_lora_rank`) and decoupled RoPE (`qk_rope_head_dim`)
    to drastically compress KV-cache memory bandwidth during autoregressive loops and generation.
    """
    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int = 64, qk_rope_head_dim: int = 32,
                 max_seq_len: int = 2048, rope_theta: float = 10000.0, dropout: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # Down-projection for KV latent compression
        self.dkv_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.dkv_norm = RMSNorm(kv_lora_rank) if qk_norm else nn.Identity()

        # Up-projection from compressed latent vector c_KV to content Key & Value
        self.uk_proj = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.uv_proj = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)

        # Content Query projection
        self.q_content_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # Decoupled RoPE Query & Key projections (positional branch)
        self.q_pe_proj = nn.Linear(d_model, n_heads * qk_rope_head_dim, bias=False)
        self.k_pe_proj = nn.Linear(d_model, qk_rope_head_dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.qk_norm = qk_norm

        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(qk_rope_head_dim, max_position_embeddings=max_seq_len, base=rope_theta)
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-computed static causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().view(1, 1, max_seq_len, max_seq_len),
            persistent=False
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        past_kv_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        total_len = past_kv_len + T

        # 1. Content query
        q_content = self.q_content_proj(x).view(B, T, self.n_heads, self.head_dim)
        if self.qk_norm:
            q_content = self.q_norm(q_content)

        # 2. Down-project input to latent c_KV, then up-project to K_content and V_content
        c_kv = self.dkv_norm(self.dkv_proj(x))  # [B, T, kv_lora_rank]
        k_content = self.uk_proj(c_kv).view(B, T, self.n_heads, self.head_dim)
        v_content = self.uv_proj(c_kv).view(B, T, self.n_heads, self.head_dim)
        if self.qk_norm:
            k_content = self.k_norm(k_content)

        # 3. Decoupled RoPE branch
        q_pe = self.q_pe_proj(x).view(B, T, self.n_heads, self.qk_rope_head_dim)
        k_pe = self.k_pe_proj(x).view(B, T, 1, self.qk_rope_head_dim)

        q_pe = q_pe.transpose(1, 2)      # [B, n_heads, T, qk_rope_head_dim]
        k_pe = k_pe.transpose(1, 2)      # [B, 1, T, qk_rope_head_dim]
        cos, sin = self.rotary_emb(k_pe, seq_len=total_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, past_key_value_length=past_kv_len)

        # Expand k_pe across all n_heads
        k_pe = k_pe.expand(B, self.n_heads, T, self.qk_rope_head_dim)

        # Transpose content terms
        q_content = q_content.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k_content = k_content.transpose(1, 2)  # [B, n_heads, T, head_dim]
        v = v_content.transpose(1, 2)          # [B, n_heads, T, head_dim]

        # Concatenate content and RoPE dimensions for Q and K
        q = torch.cat([q_content, q_pe], dim=-1)  # [B, n_heads, T, head_dim + qk_rope_head_dim]
        k = torch.cat([k_content, k_pe], dim=-1)  # [B, n_heads, T, head_dim + qk_rope_head_dim]

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v) if use_cache else None

        is_causal = (past_key_value is None and T > 1)
        full_head_dim = self.head_dim + self.qk_rope_head_dim
        if False: # Forced fallback
            pass
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(full_head_dim)
            if is_causal:
                scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == False, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            context = torch.matmul(weights, v)

        context = context.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(context))
        return out, present_key_value


# ==========================================
# 6. Feed-Forward Networks: SwiGLU & Sparse MoE
# ==========================================

class FeedForward(nn.Module):
    """
    Standard SwiGLU Feed-Forward Network.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))




# ==========================================
# 7. Depth-wise LoRA & Recurrent Block (OpenMythos RDT / Fable 5)
# ==========================================

class DepthWiseLoRA(nn.Module):
    """
    Loop-conditioned Depth-wise LoRA perturbation.
    Injected into the weight-shared Recurrent Block at step t so each loop iteration
    operates with a slightly distinct transformation profile (preventing loop homogenization).
    """
    def __init__(self, d_model: int, lora_rank: int = 16, max_loops: int = 32):
        super().__init__()
        self.lora_rank = lora_rank
        self.max_loops = max_loops
        self.lora_down = nn.Parameter(torch.randn(max_loops, d_model, lora_rank) * 0.02)
        self.lora_up = nn.Parameter(torch.zeros(max_loops, lora_rank, d_model))

    def forward(self, loop_idx: int, x: torch.Tensor) -> torch.Tensor:
        idx = min(loop_idx, self.max_loops - 1)
        down = torch.matmul(x, self.lora_down[idx])
        up = torch.matmul(down, self.lora_up[idx])
        return up


class RecurrentBlock(nn.Module):
    """
    Recurrent-Depth Transformer (RDT) Block for PolyMath / Anthropic reverse-engineered Mythos (Fable 5) architecture.
    Applies a weight-shared Transformer block iteratively T times with:
    1. Continuous input injection `e` at every step.
    2. Depth-wise LoRA perturbation per loop iteration.
    3. Spectral Radius stability constraint (`rho(A) < 1`) via exponential negative softplus diagonal.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block = TransformerDecoderBlock(config, skip_residual=True)
        self.max_loop_iters = config.max_loop_iters
        self.d_model = config.d_model
        self.use_activation_checkpointing = False

        # Parameterize stable diagonal contraction matrix A: diag(exp(-softplus(A_raw)))
        self.A_diag_raw = nn.Parameter(torch.ones(config.d_model) * 1.0)
        # Input injection matrix B
        self.B_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Loop-conditioned LoRA
        self.lora_per_loop = DepthWiseLoRA(config.d_model, lora_rank=config.lora_rank, max_loops=max(32, config.max_loop_iters))

    def get_A_diag(self) -> torch.Tensor:
        """Returns the strictly bounded diagonal matrix entries in (0, 1) guaranteeing contraction."""
        return torch.exp(-F.softplus(self.A_diag_raw))

    def get_spectral_radius(self) -> float:
        """Computes current spectral radius rho(A). Guaranteed < 1.0."""
        return self.get_A_diag().max().item()

    def forward(self, h: torch.Tensor, e: torch.Tensor, n_loops: int,
                sources: Optional[List[torch.Tensor]] = None,
                past_key_value: Optional[Union[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        present_kvs = [] if use_cache else None
        A = self.get_A_diag().view(1, 1, -1)
        B_e = self.B_proj(e)

        for t in range(n_loops):
            past_kv = None
            if past_key_value is not None:
                if isinstance(past_key_value, dict):
                    past_kv = past_key_value.get(f"recurrent_{t}")
                elif t < len(past_key_value):
                    past_kv = past_key_value[t]

            if self.use_activation_checkpointing and self.training and not use_cache:
                delta, sources, present_kv = torch_checkpoint(
                    self.block, h, sources, past_kv, use_cache, use_reentrant=False
                )
            else:
                delta, sources, present_kv = self.block(h, sources=sources, past_key_value=past_kv, use_cache=use_cache)

            if use_cache:
                present_kvs.append(present_kv)

            # Add loop-conditioned LoRA perturbation
            delta = delta + self.lora_per_loop(t, h)
            # Stable LTI recurrence update: h_{t+1} = A * h_t + B * e + delta
            h = (h * A) + B_e + delta
            if sources is not None:
                sources.append(h)

        return h, sources, present_kvs


# ==========================================
# 8. Modular Transformer Decoder Block
# ==========================================

class TransformerDecoderBlock(nn.Module):
    """
    Modular Transformer Block supporting standard PreNorm residuals, AttnRes, and PolyMath / OpenMythos RDT.
    Dynamically routes between GQA / MLA attention and SwiGLU / MoE FFN.
    When skip_residual=True (used inside RecurrentBlock), returns pure delta without internal residuals.
    """
    def __init__(self, config: ModelConfig, skip_residual: bool = False):
        super().__init__()
        self.mode = config.mode.lower()
        self.skip_residual = skip_residual

        d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model

        # Attention selection
        if getattr(config, "attn_type", "gqa").lower() == "mla":
            self.attn = MultiLatentAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                qk_norm=config.qk_norm
            )
        else:
            self.attn = CausalSelfAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                qk_norm=config.qk_norm
            )

        # FFN
        self.ffn = FeedForward(config.d_model, d_ff, dropout=config.dropout)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        if self.mode in ['attn_res', 'delta_attn_res', 'polymath']:
            self.attn_res1 = AttnResOperator(config.d_model)
            self.attn_res2 = AttnResOperator(config.d_model)

    def forward(self, state: torch.Tensor, sources: Optional[List[torch.Tensor]] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        if self.mode in ['standard', 'openmythos_rdt']:
            attn_out, present_kv = self.attn(self.norm1(state), past_key_value=past_key_value, use_cache=use_cache)
            if self.skip_residual:
                # Pure delta mode for RecurrentBlock: no internal residual connections
                ffn_out = self.ffn(self.norm2(attn_out))
                return attn_out + ffn_out, sources, present_kv
            else:
                x = state + attn_out
                out = x + self.ffn(self.norm2(x))
                return out, sources, present_kv

        elif self.mode == 'polymath':
            if self.skip_residual:
                # ★ RecurrentBlock 内部：禁用 AttnRes，保护谱半径约束 rho(A) < 1
                # AttnRes 的无衰减全历史聚合会绕过 A 矩阵的衰减约束，导致 Loss 震荡与模式坍缩
                attn_out, present_kv = self.attn(self.norm1(state), past_key_value=past_key_value, use_cache=use_cache)
                ffn_out = self.ffn(self.norm2(attn_out))
                return attn_out + ffn_out, sources, present_kv
            elif sources is not None and len(sources) > 0:
                # Prelude/Coda blocks: AttnRes 全局特征融合
                stacked_sources1 = torch.stack(sources, dim=0)
                u1 = self.attn_res1(stacked_sources1)
                attn_out, present_kv = self.attn(self.norm1(u1), past_key_value=past_key_value, use_cache=use_cache)
                h1 = u1 + attn_out
                sources.append(h1)
                stacked_sources2 = torch.stack(sources, dim=0)
                u2 = self.attn_res2(stacked_sources2)
                ffn_out = self.ffn(self.norm2(u2))
                out = u2 + ffn_out
                sources.append(out)
                return out, sources, present_kv
            else:
                # Fallback when sources is not initialized
                attn_out, present_kv = self.attn(self.norm1(state), past_key_value=past_key_value, use_cache=use_cache)
                if self.skip_residual:
                    ffn_out = self.ffn(self.norm2(attn_out))
                    return attn_out + ffn_out, None, present_kv
                else:
                    x = state + attn_out
                    out = x + self.ffn(self.norm2(x))
                    return out, None, present_kv

        elif self.mode == 'attn_res':
            assert sources is not None, "In attn_res mode, sources list must be provided."
            stacked_sources1 = torch.stack(sources, dim=0)
            u1 = self.attn_res1(stacked_sources1)
            y1, present_kv = self.attn(self.norm1(u1), past_key_value=past_key_value, use_cache=use_cache)
            h1 = u1 + y1
            sources.append(h1)

            stacked_sources2 = torch.stack(sources, dim=0)
            u2 = self.attn_res2(stacked_sources2)
            y2 = self.ffn(self.norm2(u2))
            h2 = u2 + y2
            sources.append(h2)
            return h2, sources, present_kv

        elif self.mode == 'delta_attn_res':
            assert sources is not None, "In delta_attn_res mode, sources list must be provided."
            stacked_sources1 = torch.stack(sources, dim=0)
            u1 = self.attn_res1(stacked_sources1)
            y1, present_kv = self.attn(self.norm1(u1), past_key_value=past_key_value, use_cache=use_cache)
            sources.append(y1)

            stacked_sources2 = torch.stack(sources, dim=0)
            u2 = self.attn_res2(stacked_sources2)
            y2 = self.ffn(self.norm2(u2))
            sources.append(y2)

            state_out = torch.sum(stacked_sources2, dim=0) + y2
            return state_out, sources, present_kv

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ==========================================
# 9. Complete Language Model
# ==========================================

class TransformerLM(nn.Module):
    """
    Complete Decoder-only Autoregressive Language Model supporting <= 2B parameter scales.
    Incorporates RoPE, GQA / MLA, QK-Norm, KV-Cache, MoEFFN, and five routing modes:
    `standard`, `attn_res`, `delta_attn_res`, `polymath` (Unified RDT+AttnRes+MLA+MoE), and `openmythos_rdt`.
    """
    def __init__(self, config: Union[ModelConfig, dict], vocab_size: Optional[int] = None):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig(**config)
        
        self.config = config
        self.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.mode = config.mode.lower()
        self.attn_res_mode = config.attn_res_mode.lower()
        self.block_size_layers = config.block_size_layers

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        if self.mode in ['openmythos_rdt', 'polymath']:
            # Three-stage RDT/PolyMath: Prelude -> RecurrentBlock -> Coda
            self.prelude = nn.ModuleList([
                TransformerDecoderBlock(config) for _ in range(config.prelude_layers)
            ])
            self.recurrent = RecurrentBlock(config)
            self.coda = nn.ModuleList([
                TransformerDecoderBlock(config) for _ in range(config.coda_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerDecoderBlock(config) for _ in range(config.n_layers)
            ])

        self.norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        self.use_activation_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_spectral_radius(self) -> Optional[float]:
        """Returns the current recurrent block spectral radius if operating in openmythos_rdt/polymath mode."""
        if self.mode in ['openmythos_rdt', 'polymath'] and hasattr(self, 'recurrent'):
            return self.recurrent.get_spectral_radius()
        return None

    def forward(self, idx: torch.Tensor, past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
                use_cache: bool = False, n_loops: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[List[Any], Dict[str, Any]]]]:
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.emb_dropout(x)

        present_key_values = [] if use_cache else None

        if self.mode in ['openmythos_rdt', 'polymath']:
            state = x
            past_kv_dict = past_key_values if isinstance(past_key_values, dict) else None
            present_kv_dict = {} if use_cache else None

            sources = self._cached_sources if (self.mode == 'polymath' and past_key_values is not None and hasattr(self, "_cached_sources") and self._cached_sources is not None) else ([state] if self.mode == 'polymath' else None)

            # 1. Prelude blocks
            for i, block in enumerate(self.prelude):
                cache_key = f"prelude_{i}"
                past_kv = past_kv_dict.get(cache_key) if past_kv_dict is not None else (past_key_values[i] if past_key_values is not None and isinstance(past_key_values, list) and i < len(past_key_values) else None)
                if self.use_activation_checkpointing and self.training and not use_cache:
                    state, sources, present_kv = torch_checkpoint(block, state, sources, past_kv, use_cache, use_reentrant=False)
                else:
                    state, sources, present_kv = block(state, sources=sources, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_kv_dict[cache_key] = present_kv
            
            # Input injection vector e is captured after prelude
            e = state

            # 2. Determine loop count (adaptive depth or uniform sampling during training)
            if n_loops is None:
                if self.training and getattr(self.config, "uniform_loop_sampling", True):
                    n_loops = torch.randint(1, self.config.max_loop_iters + 1, (1,)).item()
                else:
                    n_loops = self.config.max_loop_iters

            recurrent_past_kv = None
            if past_kv_dict is not None:
                recurrent_past_kv = {f"recurrent_{t}": past_kv_dict.get(f"recurrent_{t}") for t in range(n_loops)}
            elif past_key_values is not None and isinstance(past_key_values, list):
                prelude_len = len(self.prelude)
                recurrent_past_kv = past_key_values[prelude_len : prelude_len + n_loops]

            state, sources, rec_present_kvs = self.recurrent(state, e, n_loops=n_loops, sources=sources, past_key_value=recurrent_past_kv, use_cache=use_cache)
            if use_cache and rec_present_kvs is not None:
                for t, kv in enumerate(rec_present_kvs):
                    present_kv_dict[f"recurrent_{t}"] = kv

            # 3. Coda blocks
            for i, block in enumerate(self.coda):
                cache_key = f"coda_{i}"
                if past_kv_dict is not None:
                    past_kv = past_kv_dict.get(cache_key)
                elif past_key_values is not None and isinstance(past_key_values, list):
                    idx_coda = len(self.prelude) + (len(rec_present_kvs) if rec_present_kvs else n_loops) + i
                    past_kv = past_key_values[idx_coda] if idx_coda < len(past_key_values) else None
                else:
                    past_kv = None

                if self.use_activation_checkpointing and self.training and not use_cache:
                    state, sources, present_kv = torch_checkpoint(block, state, sources, past_kv, use_cache, use_reentrant=False)
                else:
                    state, sources, present_kv = block(state, sources=sources, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_kv_dict[cache_key] = present_kv

            if use_cache and self.mode == 'polymath' and sources is not None:
                self._cached_sources = sources

            out = self.norm(state)
            if use_cache:
                present_key_values = present_kv_dict

        elif self.mode == 'standard':
            state = x
            for i, block in enumerate(self.blocks):
                past_kv = past_key_values[i] if past_key_values is not None else None
                if self.use_activation_checkpointing and self.training and not use_cache:
                    state, _, present_kv = torch_checkpoint(block, state, None, past_kv, use_cache, use_reentrant=False)
                else:
                    state, _, present_kv = block(state, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_key_values.append(present_kv)
            out = self.norm(state)

        elif self.mode in ['attn_res', 'delta_attn_res']:
            if past_key_values is not None and len(past_key_values) > 0 and hasattr(self, "_cached_sources") and self._cached_sources is not None:
                sources = self._cached_sources
            else:
                sources = [x]
            state = x

            for i, block in enumerate(self.blocks):
                past_kv = past_key_values[i] if past_key_values is not None else None
                if self.use_activation_checkpointing and self.training and not use_cache:
                    state, sources, present_kv = torch_checkpoint(block, state, sources, past_kv, use_cache, use_reentrant=False)
                else:
                    state, sources, present_kv = block(state, sources=sources, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_key_values.append(present_kv)

                if self.attn_res_mode == 'block' and (i + 1) % self.block_size_layers == 0:
                    sources = [sources[0]] + [sources[k] for k in range(1, len(sources)) if k % (2 * self.block_size_layers) == 0]
            
            if use_cache:
                self._cached_sources = sources
            out = self.norm(state)



        logits = self.lm_head(out)
        if use_cache:
            return logits, present_key_values
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
                 use_cache: bool = True, effort: str = "high") -> torch.Tensor:
        """
        Autoregressive text generation using fast KV-cache decoding ($O(T)$ step cost).
        Supports Anthropic reverse-engineered Mythos (Fable 5) adaptive computation depth via the `effort` parameter when in `polymath`/`openmythos_rdt` mode.
        """
        self.eval()
        past_key_values = None
        if hasattr(self, "_cached_sources"):
            self._cached_sources = None

        # Map effort to recurrent loop depth T
        effort_map = {"low": 2, "medium": 4, "high": self.config.max_loop_iters, "xhigh": max(16, self.config.max_loop_iters * 2)}
        n_loops = effort_map.get(effort.lower(), self.config.max_loop_iters) if self.mode in ['openmythos_rdt', 'polymath'] else None

        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -self.max_seq_len:]

            if use_cache:
                logits, past_key_values = self(idx_cond, past_key_values=past_key_values, use_cache=True, n_loops=n_loops)
            else:
                logits = self(idx_cond, use_cache=False, n_loops=n_loops)

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
