"""
Transformer encoder with Rotary Position Embeddings, SwiGLU FFN, and Flash Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class RotaryPositionalEmbedding(nn.Module):
    """RoPE with optional linear scaling for sequences longer than max_tokens."""

    def __init__(self, dim: int, max_tokens: int = 512, base: int = 10000,
                 scaling_type: str | None = "linear"):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_tokens = max_tokens
        self.scaling_type = scaling_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        if seq_len > self.max_tokens and self.scaling_type == "linear":
            positions = positions * (self.max_tokens / seq_len)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)[None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor):
    cos, sin = pos_emb.cos(), pos_emb.sin()
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)

class AttentionBlock(nn.Module):
    """
    Pre-norm transformer block.

    Attention uses ``scaled_dot_product_attention`` which dispatches to the
    Flash Attention kernel when no explicit mask is passed.  Padding tokens
    are zeroed in key/value so they contribute negligible attention weight.
    """

    def __init__(self, embed_dim: int, nhead: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        ff_dim = int(embed_dim * ff_mult * 2 / 3)
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                rope_emb: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        if mask is not None:
            mask = mask.to(torch.bool)

        # --- Self-attention ---
        h = self.norm1(x)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        q = self.q_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(-1)
            k = k.masked_fill(~m, 0.0)
            v = v.masked_fill(~m, 0.0)

        if rope_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_emb.unsqueeze(1))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.o_proj(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + self.dropout(out)

        # --- SwiGLU FFN ---
        h = self.norm2(x)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        ff = self.w3(F.silu(self.w1(h)) * self.w2(h))
        if mask is not None:
            ff = ff.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + self.dropout(ff)
        return x


class Encoder(nn.Module):
    """Stacked transformer encoder with RoPE and optional gradient checkpointing."""

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 256,
                 num_layers: int = 4, nhead: int = 8, ff_mult: int = 4,
                 dropout: float = 0.1, max_tokens: int = 1024,
                 padding_idx: int = 0, rope_scaling_type: str | None = "linear"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(
            embed_dim // nhead, max_tokens, scaling_type=rope_scaling_type,
        )
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, nhead, ff_mult, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.emb_dropout(self.token_emb(tokens))
        padding_mask = mask.to(torch.bool) if mask is not None else None
        rope_emb = self.rope(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(layer, x, padding_mask, rope_emb, use_reentrant=False)
            else:
                x = layer(x, padding_mask, rope_emb)

        x = self.final_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return x
