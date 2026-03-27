"""
Attention-based sequence aggregation: token embeddings [B, L, D] → family embedding [B, D].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    """
    Learnable query attends over token embeddings to produce a single
    sequence-level representation.  Uses ``scaled_dot_product_attention``
    for Flash Attention dispatch on supported hardware.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.align = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.align.weight)

    def forward(self, token_embs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = token_embs.shape
        H, D = self.num_heads, self.head_dim

        query = self.pool_query.expand(B, -1, -1)
        q = self.q_proj(query).view(B, 1, H, D).transpose(1, 2)
        k = self.k_proj(token_embs).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(token_embs).view(B, L, H, D).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            bool_mask = mask.to(torch.bool)
            attn_mask = torch.zeros(B, 1, 1, L, device=token_embs.device, dtype=token_embs.dtype)
            attn_mask = attn_mask.masked_fill(~bool_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            m = bool_mask.unsqueeze(1).unsqueeze(-1)
            k = k.masked_fill(~m, 0.0)
            v = v.masked_fill(~m, 0.0)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        pooled = self.norm(self.o_proj(out)).squeeze(1)
        return self.align(pooled)
