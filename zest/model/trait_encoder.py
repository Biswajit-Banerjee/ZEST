"""
TraitEncoder: Layered Extraction of Meaningful Order in Nature.

Two-stage encoder model:
  1. Masked Profile Modeling (MPM) — learns domain-specific conservation patterns
  2. Hierarchical contrastive fine-tuning — learns structural similarity
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zest.model.encoder import Encoder, ScaledPositionalEmbedding
from zest.model.pooling import AttentionAggregator
from zest.model.heads import ProfileExpansionHead, SimilarityHead

_log = logging.getLogger(__name__)


class TraitEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        num_layers: int = 8,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 1024,
        proj_dim: int = 128,
        pool_type: str = "attention",
        padding_idx: int = 0,
        rope_scaling_type: str | None = "linear",
        global_extend_strategy: str = "alibi",
        num_proj_layers: int = 2,
        proj_ff_mult: int = 2,
        # Legacy / ignored kwargs for checkpoint compat
        cath_counts=None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        if max_seq_len is not None and max_tokens == 1024:
            max_tokens = max_seq_len

        self.core = Encoder(
            vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers,
            nhead=nhead, ff_mult=ff_mult, dropout=dropout, max_tokens=max_tokens,
            padding_idx=padding_idx, rope_scaling_type=rope_scaling_type,
        )
        self.max_tokens = max_tokens
        self.max_residues = max_tokens * 4

        # Positional embeddings for residue-level heads
        self.global_pos_emb = ScaledPositionalEmbedding(
            self.max_residues, embed_dim,
            extend_strategy=global_extend_strategy, padding_idx=None,
        )

        # Pretraining heads
        self.profile_head = ProfileExpansionHead(
            embed_dim, self.max_residues, vocab_size=20,
            extend_strategy=global_extend_strategy,
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 20),
        )
        self.token_mlm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim),
        )

        # Similarity components
        self.proj_dim = proj_dim or embed_dim
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.aggregator = AttentionAggregator(embed_dim, dropout=dropout)

        # Projector: bottleneck MLP
        self.num_proj_layers = num_proj_layers
        self.proj_ff_mult = proj_ff_mult
        layers = []
        hidden = embed_dim * proj_ff_mult
        for _ in range(num_proj_layers):
            layers.extend([
                nn.Linear(embed_dim, hidden), nn.BatchNorm1d(hidden),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, embed_dim), nn.BatchNorm1d(embed_dim),
                nn.GELU(), nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(embed_dim, self.proj_dim))
        self.projector = nn.Sequential(*layers)

        self.sim_head = SimilarityHead(self.proj_dim, dropout=dropout)

        self.config = {
            "vocab_size": vocab_size, "embed_dim": embed_dim,
            "num_layers": num_layers, "nhead": nhead, "ff_mult": ff_mult,
            "dropout": dropout, "max_tokens": max_tokens,
            "max_seq_len": max_tokens,
            "proj_dim": self.proj_dim, "pool_type": pool_type,
            "padding_idx": padding_idx,
            "rope_scaling_type": rope_scaling_type,
            "global_extend_strategy": global_extend_strategy,
            "num_proj_layers": num_proj_layers, "proj_ff_mult": proj_ff_mult,
        }

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def _encode_tokens(self, tokens, mask=None):
        if self._core_is_frozen():
            with torch.no_grad():
                return self.core(tokens, mask).detach()
        return self.core(tokens, mask)

    def _core_is_frozen(self) -> bool:
        return not next(self.core.parameters()).requires_grad

    def project_family(self, tokens, mask=None, normalize=True):
        """Encode tokens → aggregated → projected embedding [B, proj_dim]."""
        token_embs = self._encode_tokens(tokens, mask)
        family = self.aggregator(token_embs, mask)
        proj = self.projector(family)
        if normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        return proj

    def forward_mpm(self, tokens, mask, global_indices, token_lengths):
        """MPM forward: profile logits + projected embedding."""
        token_embs = self.core(tokens, mask)
        logits = self.profile_head(token_embs, token_lengths, global_indices,
                                   self.global_pos_emb)
        family = self.aggregator(token_embs, mask)
        proj = F.normalize(self.projector(family), p=2, dim=-1)
        return logits, proj

    def forward_mpm_mlm(self, tokens, mask, global_indices, token_lengths):
        """Combined forward: profile + residue MLM + token MLM in one pass."""
        token_embs = self.core(tokens, mask)
        z = self.profile_head.get_residue_embeddings(
            token_embs, token_lengths, global_indices, self.global_pos_emb,
        )
        profile_logits = self.profile_head.output_proj(z)
        mlm_logits = self.mlm_head(z)
        h = self.token_mlm_proj(token_embs)
        token_logits = F.linear(h, self.core.token_emb.weight)
        family = self.aggregator(token_embs, mask)
        proj = F.normalize(self.projector(family), p=2, dim=-1)
        return profile_logits, mlm_logits, token_logits, proj

    def forward_token_mlm(self, tokens, mask):
        """Token-level MLM with tied input embedding weights."""
        token_embs = self.core(tokens, mask)
        h = self.token_mlm_proj(token_embs)
        return F.linear(h, self.core.token_emb.weight)

    def compute_similarity(self, u_tok, u_mask, v_tok, v_mask):
        u = self.project_family(u_tok, u_mask)
        v = self.project_family(v_tok, v_mask)
        cos = (u * v).sum(dim=-1)
        prob_logit, homo_logit = self.sim_head(u, v)
        return prob_logit, homo_logit, cos, u, v

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_core(self):
        for p in self.core.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_for_similarity(self):
        """Freeze everything except aggregator, projector, similarity head."""
        self.freeze_core()
        for module in [self.mlm_head, self.profile_head, self.global_pos_emb,
                       self.token_mlm_proj]:
            for p in module.parameters():
                p.requires_grad = False
        self.temperature.requires_grad = False

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, path, extra=None):
        state = {"model_state_dict": self.state_dict(), "config": self.config}
        if extra:
            state.update(extra)
        torch.save(state, path)

    @classmethod
    def load_pretrained(cls, path, device="cpu", **config_overrides):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = dict(ckpt.get("config", {}))
        config.pop("num_superfamilies", None)
        config.pop("cath_counts", None)
        sd = ckpt.get("model_state_dict", {})

        # Strip legacy keys
        for prefix in ("superfamily_head.", "cath_head."):
            for k in [k for k in sd if k.startswith(prefix)]:
                sd.pop(k)

        config.update(config_overrides)

        # If projector config was overridden, drop old projector weights
        if any(k in config_overrides for k in ("num_proj_layers", "proj_dim", "proj_ff_mult")):
            stripped = [k for k in sd if any(k.startswith(p) for p in ("projector.", "sim_head."))]
            for k in stripped:
                sd.pop(k)
            _log.info("Projector overridden — stripped %d old weights", len(stripped))

        model = cls(**config)

        # Handle size mismatches gracefully
        model_sd = model.state_dict()
        for key in list(sd.keys()):
            if key in model_sd and sd[key].shape != model_sd[key].shape:
                c, m = sd[key], model_sd[key]
                if c.ndim != m.ndim:
                    sd.pop(key)
                    continue
                _log.info("Size mismatch %s: %s vs %s, copying overlap",
                          key, list(c.shape), list(m.shape))
                slices = tuple(slice(0, min(a, b)) for a, b in zip(c.shape, m.shape))
                model_sd[key][slices] = c[slices]
                sd[key] = model_sd[key]

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _log.info("New parameters (random init): %s",
                       list({k.split(".")[0] for k in missing}))
        if unexpected:
            _log.warning("Ignored unexpected keys: %s", unexpected)
        return model, ckpt


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

def create_trait_encoder(preset: str, vocab_size: int = 32000, max_tokens: int = 1024) -> TraitEncoder:
    """Create a TraitEncoder model from a named preset."""
    presets = {
        "base": dict(embed_dim=384, num_layers=8, nhead=6, proj_dim=128, dropout=0.1),
        "efficient-deep": dict(embed_dim=512, num_layers=16, nhead=8, proj_dim=128,
                               dropout=0.08, num_proj_layers=3),
        "deep-wide": dict(embed_dim=768, num_layers=24, nhead=12, proj_dim=384, dropout=0.05),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    return TraitEncoder(vocab_size=vocab_size, max_tokens=max_tokens, **presets[preset])
