import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """RoPE with optional linear scaling for token sequences > max_tokens."""

    def __init__(
        self,
        dim: int,
        max_tokens: int = 512,
        base: int = 10000,
        scaling_type: str | None = "linear",
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_tokens = max_tokens
        self.scaling_type = scaling_type

    def _scaled_positions(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        if seq_len <= self.max_tokens or not self.scaling_type:
            return positions

        if self.scaling_type == "linear":
            scale = self.max_tokens / seq_len
            return positions * scale

        raise ValueError(f"Unknown RoPE scaling_type: {self.scaling_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = self._scaled_positions(seq_len, x.device, self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb[None, :, :]

class ScaledPositionalEmbedding(nn.Module):
    """Learned embeddings with ALiBi-style extrapolation beyond max index."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        extend_strategy: str = "alibi",
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.extend_strategy = extend_strategy
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx

        max_index = num_embeddings - 1
        if padding_idx is not None:
            content_max_index = max(0, min(max_index - 1, padding_idx - 1))
        else:
            content_max_index = max_index
        self.register_buffer("max_index", torch.tensor(max_index), persistent=False)
        self.register_buffer("content_max_index", torch.tensor(content_max_index), persistent=False)

        if extend_strategy == "alibi":
            slopes = torch.logspace(0, -3, steps=embedding_dim, base=2.0)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.register_buffer("alibi_slopes", None, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.long()
        if self.padding_idx is not None:
            pad_mask = positions == self.padding_idx
            clamped = positions.clamp(min=0, max=int(self.content_max_index.item()))
            clamped = torch.where(pad_mask, torch.full_like(clamped, self.padding_idx), clamped)
        else:
            pad_mask = torch.zeros_like(positions, dtype=torch.bool)
            clamped = positions.clamp(min=0, max=int(self.max_index.item()))
        embeddings = self.embedding(clamped)

        if self.extend_strategy == "alibi":
            excess = (positions - self.content_max_index).clamp_min(0)
            if self.padding_idx is not None:
                excess = torch.where(pad_mask, torch.zeros_like(excess), excess)
            embeddings = embeddings + excess.unsqueeze(-1) * self.alibi_slopes

        if self.padding_idx is not None:
            embeddings = embeddings.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return embeddings

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor):
    cos = pos_emb.cos()
    sin = pos_emb.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

