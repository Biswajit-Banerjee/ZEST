"""
Task-specific heads for pretraining and fine-tuning.

- ProfileExpansionHead: expands token embeddings to residue-level HMM emission predictions
- SimilarityHead: pairwise similarity scoring from projected embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from zest.model.encoder import ScaledPositionalEmbedding


class ProfileExpansionHead(nn.Module):
    """
    Expand BPE token embeddings to per-residue logits over 20 amino acids.

    Each BPE token covers 1-N residues. This head repeats the token embedding
    for each covered position, adds learned local and global positional
    embeddings, then projects to a 20-dim emission distribution.
    """

    LOCAL_PAD_IDX = 31

    def __init__(self, embed_dim: int, max_residues: int, vocab_size: int = 20,
                 extend_strategy: str = "alibi"):
        super().__init__()
        self.max_residues = max_residues
        self.local_pos_emb = ScaledPositionalEmbedding(
            num_embeddings=self.LOCAL_PAD_IDX + 1,
            embedding_dim=embed_dim,
            extend_strategy=extend_strategy,
            padding_idx=self.LOCAL_PAD_IDX,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def _expand_tokens(self, z: torch.Tensor, token_lengths: torch.Tensor,
                       target_len: int):
        """Vectorized token → residue expansion via searchsorted."""
        B, T, D = z.shape
        device = z.device
        if not torch.is_tensor(token_lengths):
            token_lengths = torch.tensor(token_lengths, device=device, dtype=torch.long)
        token_lengths = token_lengths.clamp(min=0)

        cumsum = torch.cumsum(token_lengths, dim=1)
        total = cumsum[:, -1]
        positions = torch.arange(target_len, device=device).unsqueeze(0).expand(B, -1)

        tok_idx = torch.searchsorted(cumsum, positions + 1).clamp(0, T - 1)
        z_exp = torch.gather(z, 1, tok_idx.unsqueeze(-1).expand(-1, -1, D))

        shifted = F.pad(cumsum[:, :-1], (1, 0), value=0)
        starts = torch.gather(shifted, 1, tok_idx)
        local_idx = positions - starts

        pad_mask = positions >= total.unsqueeze(1)
        z_exp = z_exp.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        local_idx = torch.where(pad_mask, torch.full_like(local_idx, self.LOCAL_PAD_IDX),
                                local_idx.clamp(min=0))
        return z_exp, local_idx

    def get_residue_embeddings(self, z, token_lengths, global_indices, global_pos_emb):
        target_len = global_indices.shape[1]
        z_exp, local_idx = self._expand_tokens(z, token_lengths, target_len)
        pos_global = global_pos_emb(global_indices.clamp(min=0)).to(z_exp.dtype)
        pos_local = self.local_pos_emb(local_idx).to(z_exp.dtype)
        return z_exp + pos_global + pos_local

    def forward(self, z, token_lengths, global_indices, global_pos_emb):
        z_final = self.get_residue_embeddings(z, token_lengths, global_indices, global_pos_emb)
        return self.output_proj(z_final)


class SimilarityHead(nn.Module):
    """
    Pairwise similarity head producing a probability regression logit
    and a binary homologue classification logit from symmetric interaction
    features: [mean(u,v), |u-v|, u*v].
    """

    PROB_LO = 0.24
    PROB_HI = 1.00
    PROB_RANGE = PROB_HI - PROB_LO

    def __init__(self, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        d = proj_dim * 3
        self.prob_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.homo_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        feat = torch.cat([(u + v) / 2, torch.abs(u - v), u * v], dim=-1)
        return self.prob_head(feat), self.homo_head(feat)

    @staticmethod
    def decode_probability(logit: torch.Tensor) -> torch.Tensor:
        return SimilarityHead.PROB_LO + SimilarityHead.PROB_RANGE * torch.sigmoid(logit)
