"""Hugging Face PretrainedConfig for LEMON-ZEST."""
from __future__ import annotations

from transformers import PretrainedConfig


class LemonConfig(PretrainedConfig):
    """Configuration class for LEMON (`ThemeEncoder` with `RecycledResidueDecoder`)."""

    model_type = "lemon"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        ff_mult: int = 4,
        hidden_dropout_prob: float = 0.04,
        embedding_dropout_prob: float = 0.1,
        token_head_dropout_prob: float = 0.1,
        residue_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        proj_dim: int = 384,
        num_proj_layers: int = 4,
        proj_ff_mult: int = 4,
        aggregator_num_heads: int = 4,
        rope_scaling_type: str | None = "linear",
        rope_theta: float = 10000.0,
        residue_hidden_dim: int = 384,
        residue_num_heads: int = 6,
        recycled_num_cycles: int = 7,
        recycled_token_window: int = 2,
        recycled_residue_window: int = 16,
        residue_vocab_size: int = 20,
        zero_kv_pad_mask: bool = True,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        cls_token_id: int = 2,
        eos_token_id: int = 3,
        mask_token_id: int = 4,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.token_head_dropout_prob = token_head_dropout_prob
        self.residue_dropout_prob = residue_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.proj_dim = proj_dim
        self.num_proj_layers = num_proj_layers
        self.proj_ff_mult = proj_ff_mult
        self.aggregator_num_heads = aggregator_num_heads
        self.rope_scaling_type = rope_scaling_type
        self.rope_theta = rope_theta
        self.residue_hidden_dim = residue_hidden_dim
        self.residue_num_heads = residue_num_heads
        self.recycled_num_cycles = recycled_num_cycles
        self.recycled_token_window = recycled_token_window
        self.recycled_residue_window = recycled_residue_window
        self.residue_vocab_size = residue_vocab_size
        self.zero_kv_pad_mask = zero_kv_pad_mask
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.mask_token_id = mask_token_id
        self.layer_norm_eps = layer_norm_eps

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_zest_config(cls, cfg: dict, num_cycles: int | None = None, **extra) -> "LemonConfig":
        """Build a ``LemonConfig`` from a ``ThemeEncoder.config`` dict."""
        cycles = num_cycles if num_cycles is not None else cfg.get("recycled_num_cycles", 3)
        return cls(
            vocab_size=cfg.get("vocab_size", 32000),
            hidden_size=cfg.get("embed_dim", 768),
            num_hidden_layers=cfg.get("num_layers", 24),
            num_attention_heads=cfg.get("nhead", 12),
            ff_mult=cfg.get("ff_mult", 4),
            hidden_dropout_prob=cfg.get("dropout", 0.04),
            embedding_dropout_prob=cfg.get("embedding_dropout", 0.1),
            token_head_dropout_prob=cfg.get("token_head_dropout", 0.1),
            residue_dropout_prob=cfg.get("residue_dropout", 0.1),
            max_position_embeddings=cfg.get("max_tokens", 1024),
            proj_dim=cfg.get("proj_dim", 384),
            num_proj_layers=cfg.get("num_proj_layers", 4),
            proj_ff_mult=cfg.get("proj_ff_mult", 4),
            rope_scaling_type=cfg.get("rope_scaling_type", "linear"),
            residue_hidden_dim=cfg.get("residue_hidden_dim", 384),
            residue_num_heads=cfg.get("residue_heads", 6),
            recycled_num_cycles=cycles,
            recycled_token_window=cfg.get("recycled_token_window", 2),
            recycled_residue_window=cfg.get("recycled_residue_window", 16),
            pad_token_id=cfg.get("padding_idx", 0),
            **extra,
        )
