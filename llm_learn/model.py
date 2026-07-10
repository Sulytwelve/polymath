import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
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
# 4. Manifold-Constrained Hyper-Connections (DeepSeek style)
# ==========================================

def sinkhorn_knopp_projection(logits: torch.Tensor, max_iter: int = 20, eps: float = 1e-12) -> torch.Tensor:
    """
    Applies the Sinkhorn-Knopp algorithm to project an unconstrained matrix
    onto the Birkhoff polytope (manifold of doubly stochastic matrices).
    """
    M = torch.exp(logits)
    for _ in range(max_iter):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


# ==========================================
# 5. Standard Sub-layers (GQA Causal Attention & SwiGLU FFN)
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
        if hasattr(F, "scaled_dot_product_attention"):
            context = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Fallback for older PyTorch versions
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            context = torch.matmul(weights, v_rep)

        context = context.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(context))
        return out, present_key_value


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
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
# 6. Modular Transformer Decoder Block
# ==========================================

class TransformerDecoderBlock(nn.Module):
    """
    Modular Transformer Block supporting standard PreNorm residual connections,
    Attention Residuals (AttnRes), and Manifold-Constrained Hyper-Connections (mHC).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mode = config.mode.lower()
        self.n_streams = config.n_streams
        self.sinkhorn_iters = config.sinkhorn_iters

        d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
            qk_norm=config.qk_norm
        )
        self.ffn = FeedForward(config.d_model, d_ff, dropout=config.dropout)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        if self.mode in ['attn_res', 'delta_attn_res']:
            self.attn_res1 = AttnResOperator(config.d_model)
            self.attn_res2 = AttnResOperator(config.d_model)
        elif self.mode == 'mhc':
            self.H_res1_logits = nn.Parameter(torch.randn(self.n_streams, self.n_streams))
            self.H_pre1_logits = nn.Parameter(torch.randn(self.n_streams))
            self.H_post1_logits = nn.Parameter(torch.randn(self.n_streams))
            self.alpha1_logits = nn.Parameter(torch.zeros(1))

            self.H_res2_logits = nn.Parameter(torch.randn(self.n_streams, self.n_streams))
            self.H_pre2_logits = nn.Parameter(torch.randn(self.n_streams))
            self.H_post2_logits = nn.Parameter(torch.randn(self.n_streams))
            self.alpha2_logits = nn.Parameter(torch.zeros(1))

    def forward(self, state: torch.Tensor, sources: Optional[List[torch.Tensor]] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        if self.mode == 'standard':
            attn_out, present_kv = self.attn(self.norm1(state), past_key_value=past_key_value, use_cache=use_cache)
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

        elif self.mode == 'mhc':
            streams = state
            # Attention sub-layer
            H_pre1 = torch.softmax(self.H_pre1_logits, dim=0)
            u1 = torch.einsum('s, s b t d -> b t d', H_pre1, streams)
            y1, present_kv = self.attn(self.norm1(u1), past_key_value=past_key_value, use_cache=use_cache)

            H_post1 = torch.softmax(self.H_post1_logits, dim=0)
            delta1 = torch.einsum('s, b t d -> s b t d', H_post1, y1)

            H_res1 = sinkhorn_knopp_projection(self.H_res1_logits, max_iter=self.sinkhorn_iters)
            alpha1 = torch.sigmoid(self.alpha1_logits)
            I = torch.eye(self.n_streams, device=streams.device)
            H_res1_mixed = (1.0 - alpha1) * I + alpha1 * H_res1
            streams = torch.einsum('i j, j b t d -> i b t d', H_res1_mixed, streams) + delta1

            # FFN sub-layer
            H_pre2 = torch.softmax(self.H_pre2_logits, dim=0)
            u2 = torch.einsum('s, s b t d -> b t d', H_pre2, streams)
            y2 = self.ffn(self.norm2(u2))

            H_post2 = torch.softmax(self.H_post2_logits, dim=0)
            delta2 = torch.einsum('s, b t d -> s b t d', H_post2, y2)

            H_res2 = sinkhorn_knopp_projection(self.H_res2_logits, max_iter=self.sinkhorn_iters)
            alpha2 = torch.sigmoid(self.alpha2_logits)
            H_res2_mixed = (1.0 - alpha2) * I + alpha2 * H_res2
            streams = torch.einsum('i j, j b t d -> i b t d', H_res2_mixed, streams) + delta2

            return streams, None, present_kv

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ==========================================
# 7. Complete Language Model
# ==========================================

class TransformerLM(nn.Module):
    """
    Complete Decoder-only Autoregressive Language Model supporting <= 2B parameter scales.
    Incorporates RoPE, GQA, QK-Norm, KV-Cache, and four routing variants.
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
        self.n_streams = config.n_streams
        self.attn_res_mode = config.attn_res_mode.lower()
        self.block_size_layers = config.block_size_layers

        # Token embedding (no absolute position embedding since RoPE is used)
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.n_layers)
        ])

        # Final Normalization and Output Projection
        self.norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        if self.mode == 'mhc':
            self.final_mix_logits = nn.Parameter(torch.randn(self.n_streams))

        # Apply custom initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.emb_dropout(x)

        present_key_values = [] if use_cache else None

        if self.mode == 'standard':
            state = x
            for i, block in enumerate(self.blocks):
                past_kv = past_key_values[i] if past_key_values is not None else None
                state, _, present_kv = block(state, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_key_values.append(present_kv)
            out = self.norm(state)

        elif self.mode in ['attn_res', 'delta_attn_res']:
            if past_key_values is not None and len(past_key_values) > 0 and hasattr(self, "_cached_sources") and self._cached_sources is not None:
                # During KV-cache incremental step in attn_res, we reuse prior cached sources
                sources = self._cached_sources
            else:
                sources = [x]
            state = x

            for i, block in enumerate(self.blocks):
                past_kv = past_key_values[i] if past_key_values is not None else None
                state, sources, present_kv = block(state, sources=sources, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_key_values.append(present_kv)

                if self.attn_res_mode == 'block' and (i + 1) % self.block_size_layers == 0:
                    sources = [sources[0]] + [sources[k] for k in range(1, len(sources)) if k % (2 * self.block_size_layers) == 0]
            
            if use_cache:
                self._cached_sources = sources
            out = self.norm(state)

        elif self.mode == 'mhc':
            streams = x.unsqueeze(0).repeat(self.n_streams, 1, 1, 1)
            for i, block in enumerate(self.blocks):
                past_kv = past_key_values[i] if past_key_values is not None else None
                streams, _, present_kv = block(streams, past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    present_key_values.append(present_kv)

            final_mix = torch.softmax(self.final_mix_logits, dim=0)
            state = torch.einsum('s, s b t d -> b t d', final_mix, streams)
            out = self.norm(state)

        logits = self.lm_head(out)
        if use_cache:
            return logits, present_key_values
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
                 use_cache: bool = True) -> torch.Tensor:
        """
        Autoregressive text generation using fast KV-cache decoding ($O(T)$ step cost).
        """
        self.eval()
        past_key_values = None
        if hasattr(self, "_cached_sources"):
            self._cached_sources = None

        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -self.max_seq_len:]

            if use_cache:
                logits, past_key_values = self(idx_cond, past_key_values=past_key_values, use_cache=True)
            else:
                logits = self(idx_cond, use_cache=False)

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
