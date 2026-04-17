"""
ThemeEncoder.
Two-stage encoder model:
  1. Masked Language Modeling (MLM)
  2. Hierarchical contrastive fine-tuning
"""
import logging
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from zest.model.encoder import Encoder
from zest.model.pooling import AttentionAggregator
from zest.model.heads import ExpansionHead


_log = logging.getLogger(__name__)

class ThemeEncoder(nn.Module):

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
        padding_idx: int = 0,
        rope_scaling_type: str | None = "linear",
        num_proj_layers: int = 2,
        proj_ff_mult: int = 2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim or embed_dim
        self.num_proj_layers = num_proj_layers
        self.proj_ff_mult = proj_ff_mult

        
        self.core = Encoder(
            vocab_size=self.vocab_size, embed_dim=self.embed_dim, num_layers=num_layers,
            nhead=nhead, ff_mult=ff_mult, dropout=dropout, max_tokens=self.max_tokens,
            padding_idx=padding_idx, rope_scaling_type=rope_scaling_type,
        )

        # Similarity components
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())
        self.aggregator = AttentionAggregator(self.embed_dim, dropout=dropout)
        self.profile_expansion_head = ExpansionHead(self.embed_dim, max_residues=3*self.max_tokens)
        self.global_pos_emb = nn.Embedding(3*self.max_tokens, self.embed_dim)

        # Projector: bottleneck MLP
        hidden = self.embed_dim * self.proj_ff_mult
        self.projector = self.bottleneck_mlp(self.embed_dim, hidden, dropout, self.num_proj_layers, output_dim=self.proj_dim) 
        
        self.config = {
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
            "num_layers": num_layers, "nhead": nhead, "ff_mult": ff_mult,
            "dropout": dropout, "max_tokens": self.max_tokens,
            "proj_dim": self.proj_dim,
            "padding_idx": padding_idx,
            "rope_scaling_type": rope_scaling_type,
            "num_proj_layers": num_proj_layers, "proj_ff_mult": proj_ff_mult,
        }

    def bottleneck_mlp(self, input_dim, hidden, dropout, num_proj_layers, output_dim=None):
        layers = []
        
        # Linear Norm GELU
        lng_layer = lambda input_dim, hidden, dropout: [
            nn.Linear(input_dim, hidden), 
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        
        for _ in range(num_proj_layers):
            layers.extend(
                lng_layer(input_dim, hidden, dropout) 
                + lng_layer(hidden, input_dim, dropout)
            )
        
        if output_dim is not None:
            layers.append(nn.Linear(input_dim, output_dim))
        
        return nn.Sequential(*layers)
        
    @property
    def temperature(self) -> torch.Tensor:
        """Learned temperature, clamped to [0.01, 1.0] for stability."""
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_tokens(self, tokens, mask=None):
        if self._core_is_frozen():
            with torch.no_grad():
                return self.core(tokens, mask).detach()
        return self.core(tokens, mask)

    def _core_is_frozen(self) -> bool:
        return not next(self.core.parameters()).requires_grad

    def project_family(self, tokens, mask=None, normalize=True, temperature="learned"):
        """Encode tokens → aggregated → projected embedding [B, proj_dim].

        Args:
            temperature: Controls similarity scaling applied to the projection.
                ``"learned"`` (default) — divide by the model's learned temperature.
                ``None`` or ``False`` — no temperature scaling.
                ``float`` — divide by the given manual value.
        """
        token_embs = self.encode_tokens(tokens, mask)
        family = self.aggregator(token_embs, mask)
        proj = self.projector(family)
        if normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        proj = self._apply_temperature(proj, temperature)
        return proj

    def _apply_temperature(self, proj, temperature):
        """Scale projection by 1/temperature when requested."""
        if temperature is None or temperature is False:
            return proj
        if temperature == "learned":
            return proj / self.temperature
        return proj / float(temperature)

    def similarity(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine similarity [B_a, B_b]. Inputs should be L2-normalised."""
        return (emb_a @ emb_b.T) / self.temperature

    def forward(self, tokens, mask=None):
        """Default forward: token-level MLM with tied input embedding weights."""
        token_embs = self.encode_tokens(tokens, mask)
        return F.linear(token_embs, self.core.token_emb.weight)

    def forward_residue_mlm(self, tokens, mask, token_lengths, global_indices):
        """Residue-level MLM: encode → expand to residues → predict AAs [B, L, 20]."""
        z = self.encode_tokens(tokens, mask)
        return self.profile_expansion_head(z, token_lengths, global_indices, self.global_pos_emb)

    def freeze_core(self):
        for p in self.core.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


    _CKPT_VERSION = 1.0

    def save_pretrained(self, path, extra=None):
        sd = self.state_dict()
        state = {
            "ckpt_version": self._CKPT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "model_state_dict": sd,
            "shape_manifest": {k: tuple(v.shape) for k, v in sd.items()},
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    @classmethod
    def load_pretrained(cls, path, device="cpu", **config_overrides):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = dict(ckpt.get("config", {}))
        config.update(config_overrides)

        model = cls(**config)
        model.load_state_dict(ckpt["model_state_dict"])

        _log.info("Loaded checkpoint v%.1f from %s",
                  ckpt.get("ckpt_version", 1.0), ckpt.get("saved_at", "unknown"))
        return model, ckpt


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

def create_theme_encoder(preset: str, vocab_size: int = 32000, max_tokens: int = 1024) -> ThemeEncoder:
    """Create a ThemeEncoder model from a named preset."""
    presets = {
        # ~33.7M params (core=30.0M, projector=1.2M, aggregator=0.7M, profile_head=0.6M, global_pos=1.2M)
        "test": dict(embed_dim=384, num_layers=10, nhead=6,  proj_dim=64,  dropout=0.10, num_proj_layers=2, ff_mult=4, proj_ff_mult=2),
        # ~77.1M params (core=66.7M, projector=6.4M, aggregator=1.3M, profile_head=1.1M, global_pos=1.6M)
        "base": dict(embed_dim=512, num_layers=16, nhead=8,  proj_dim=128, dropout=0.08, num_proj_layers=3, ff_mult=4, proj_ff_mult=4),
        # ~221.5M params (core=194.5M, projector=19.2M, aggregator=3.0M, profile_head=2.4M, global_pos=2.4M)
        "deep": dict(embed_dim=768, num_layers=24, nhead=12, proj_dim=384, dropout=0.04, num_proj_layers=4, ff_mult=4, proj_ff_mult=4),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    return ThemeEncoder(vocab_size=vocab_size, max_tokens=max_tokens, **presets[preset])


