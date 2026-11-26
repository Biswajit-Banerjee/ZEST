"""
ThemeEncoder.
Two-stage protein language model with iterative residue recycling:
  1. Masked Language Modeling (token + residue MLM with RecycledResidueDecoder)
  2. Hierarchical contrastive fine-tuning (CATH remote homology retrieval)
"""
import inspect
import logging
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from zest.model.encoder import Encoder
from zest.model.heads import RecycledResidueDecoder
from zest.model.pooling import AttentionAggregator


_log = logging.getLogger(__name__)

#: Every residue-head parameter lives under this prefix.
HEAD_PREFIX = "profile_expansion_head."

#: Dropped config keys from earlier head variants that may still appear in saved
#: checkpoint configs (e.g., hp2 / cycle_enc_v2 checkpoints). Ignored on load.
_IGNORED_LEGACY_CONFIG_KEYS = frozenset({
    "sequence_head",
    "residue_blocks",
    "residue_window",
})


class ThemeEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        num_layers: int = 8,
        nhead: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        token_head_dropout: float = 0.1,
        residue_dropout: float = 0.1,
        max_tokens: int = 1024,
        proj_dim: int = 128,
        padding_idx: int = 0,
        rope_scaling_type: str | None = "linear",
        num_proj_layers: int = 2,
        proj_ff_mult: int = 2,
        residue_hidden_dim: int = 384,
        residue_heads: int = 6,
        recycled_num_cycles: int = 3,
        recycled_token_window: int = 2,
        recycled_residue_window: int = 16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim or embed_dim
        self.num_proj_layers = num_proj_layers
        self.proj_ff_mult = proj_ff_mult
        self.sequence_head = "recycled_decoder"

        self.core = Encoder(
            vocab_size=self.vocab_size, embed_dim=self.embed_dim, num_layers=num_layers,
            nhead=nhead, ff_mult=ff_mult, dropout=dropout, max_tokens=self.max_tokens,
            embedding_dropout=embedding_dropout,
            padding_idx=padding_idx, rope_scaling_type=rope_scaling_type,
        )

        # Similarity components
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())
        self.aggregator = AttentionAggregator(self.embed_dim, dropout=dropout)

        # Residue decoder (RecycledResidueDecoder)
        self.profile_expansion_head = RecycledResidueDecoder(
            self.embed_dim, max_residues=3 * self.max_tokens,
            hidden_dim=residue_hidden_dim, num_heads=residue_heads,
            num_cycles=recycled_num_cycles,
            token_window=recycled_token_window,
            residue_window=recycled_residue_window,
            dropout=residue_dropout,
        )
        self.global_pos_emb = nn.Embedding(3 * self.max_tokens, self.embed_dim)
        self.token_head_dropout = nn.Dropout(token_head_dropout)

        # Projector: bottleneck MLP
        hidden = self.embed_dim * self.proj_ff_mult
        self.projector = self.bottleneck_mlp(
            self.embed_dim, hidden, dropout, self.num_proj_layers, output_dim=self.proj_dim
        )

        self.config = {
            "vocab_size": self.vocab_size, "embed_dim": self.embed_dim,
            "num_layers": num_layers, "nhead": nhead, "ff_mult": ff_mult,
            "dropout": dropout, "embedding_dropout": embedding_dropout,
            "token_head_dropout": token_head_dropout,
            "residue_dropout": residue_dropout, "max_tokens": self.max_tokens,
            "proj_dim": self.proj_dim,
            "padding_idx": padding_idx,
            "rope_scaling_type": rope_scaling_type,
            "num_proj_layers": num_proj_layers, "proj_ff_mult": proj_ff_mult,
            "sequence_head": self.sequence_head,
            "residue_hidden_dim": residue_hidden_dim,
            "residue_heads": residue_heads,
            "recycled_num_cycles": recycled_num_cycles,
            "recycled_token_window": recycled_token_window,
            "recycled_residue_window": recycled_residue_window,
        }

    def bottleneck_mlp(self, input_dim, hidden, dropout, num_proj_layers, output_dim=None):
        layers = []

        # Linear Norm GELU
        lng_layer = lambda input_dim, hidden, dropout: [
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
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
        return not any(parameter.requires_grad for parameter in self.core.parameters())

    def project_family(self, tokens, mask=None, normalize=True, temperature="learned"):
        """Encode tokens -> aggregated -> projected embedding [B, proj_dim].

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

    def forward(self, tokens, mask=None, token_lengths=None, global_indices=None,
                pos_tokens=None, pos_mask=None,
                pos_token_lengths=None, pos_global_indices=None,
                return_residue_features=False):
        """Run token MLM, optionally together with residue MLM.

        When ``pos_tokens`` is provided, runs in **joint contrastive + residue
        MLM** mode designed for DDP-wrapped training:

        Returns ``(anc_proj, pos_proj, anc_residue_logits, pos_residue_logits)``
        where projections are L2-normalised (without temperature scaling) and
        residue logits are ``None`` when ``token_lengths`` / ``global_indices``
        are not supplied.

        When ``return_residue_features=True``, returns an extra tensor of refined
        residue features (before the aa_logits projection) for token-level
        reconstruction objectives.
        """
        if pos_tokens is not None:
            anc_embs = self.encode_tokens(tokens, mask)
            pos_embs = self.encode_tokens(pos_tokens, pos_mask)

            anc_proj = F.normalize(
                self.projector(self.aggregator(anc_embs, mask)), p=2, dim=-1)
            pos_proj = F.normalize(
                self.projector(self.aggregator(pos_embs, pos_mask)), p=2, dim=-1)

            anc_res = pos_res = None
            if token_lengths is not None and global_indices is not None:
                gi_a = global_indices.clamp(0, self.max_residues - 1)
                gi_p = pos_global_indices.clamp(0, self.max_residues - 1)
                anc_res = self.profile_expansion_head(
                    anc_embs, token_lengths, gi_a, self.global_pos_emb)
                pos_res = self.profile_expansion_head(
                    pos_embs, pos_token_lengths, gi_p, self.global_pos_emb)

            return anc_proj, pos_proj, anc_res, pos_res

        token_embs = self.encode_tokens(tokens, mask)
        token_logits = F.linear(self.token_head_dropout(token_embs), self.core.token_emb.weight)
        if token_lengths is None or global_indices is None:
            return token_logits
        global_indices = global_indices.clamp(0, self.max_residues - 1)
        if return_residue_features:
            residue_logits, z_res = self.profile_expansion_head(
                token_embs, token_lengths, global_indices, self.global_pos_emb,
                return_features=True)
            return token_logits, residue_logits, z_res
        residue_logits = self.profile_expansion_head(
            token_embs, token_lengths, global_indices, self.global_pos_emb)
        return token_logits, residue_logits

    @property
    def max_residues(self) -> int:
        """Number of residue positions the global position embedding covers."""
        return self.global_pos_emb.num_embeddings

    def forward_residue_mlm(self, tokens, mask, token_lengths, global_indices,
                            return_features=False):
        """Residue-level MLM: encode -> expand to residues -> predict AAs [B, L, 20]."""
        z = self.encode_tokens(tokens, mask)
        global_indices = global_indices.clamp(0, self.max_residues - 1)
        return self.profile_expansion_head(
            z, token_lengths, global_indices, self.global_pos_emb,
            return_features=return_features)

    def forward_all_cycles_residue(self, tokens, mask, token_lengths,
                                   global_indices, num_cycles=None,
                                   noise_prob=0.0,
                                   stop_grad_before_last_n=0,
                                   forced_error_rate=0.0, aa_labels=None,
                                   commit_threshold=0.0):
        """Per-cycle residue logits for deep supervision / diagnostics."""
        z = self.encode_tokens(tokens, mask)
        global_indices = global_indices.clamp(0, self.max_residues - 1)
        return self.profile_expansion_head.forward_all_cycles(
            z, token_lengths, global_indices, self.global_pos_emb,
            num_cycles=num_cycles, noise_prob=noise_prob,
            stop_grad_before_last_n=stop_grad_before_last_n,
            forced_error_rate=forced_error_rate, aa_labels=aa_labels,
            commit_threshold=commit_threshold)

    def freeze_core(self):
        for p in self.core.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    _CKPT_VERSION = 2.0

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
    def load_pretrained(cls, path, device="cpu", reinit_head: bool = False,
                        **config_overrides):
        """Load a RecycledResidueDecoder ThemeEncoder checkpoint with strict guardrails.

        Args:
            path: Checkpoint file path.
            device: Map location for ``torch.load``.
            reinit_head: If True, drop saved ``profile_expansion_head.*`` weights
                and freshly initialise ``RecycledResidueDecoder`` while requiring
                the encoder, aggregator, and projector to load intact.
            **config_overrides: Optional constructor overrides (``None`` values
                are ignored).

        Raises:
            ValueError: If the checkpoint declares a non-recycling legacy head
                and ``reinit_head`` is False.
            RuntimeError: If any non-head parameters are missing, or if
                unexpected head parameters are present without ``reinit_head=True``.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        ckpt_config = dict(ckpt.get("config", {}))

        ckpt_head = ckpt_config.get("sequence_head", "recycled_decoder")
        if ckpt_head != "recycled_decoder" and not reinit_head:
            raise ValueError(
                f"Checkpoint {path} uses legacy sequence_head='{ckpt_head}'. "
                f"Only 'recycled_decoder' is supported; pass reinit_head=True to "
                f"load the encoder/aggregator/projector with a fresh RecycledResidueDecoder."
            )

        config = dict(ckpt_config)
        config.update({k: v for k, v in config_overrides.items() if v is not None})

        # Filter dropped legacy config keys (e.g. residue_blocks, residue_window)
        valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown_keys = set(config) - valid_params - _IGNORED_LEGACY_CONFIG_KEYS
        if unknown_keys:
            _log.warning("Ignoring unknown checkpoint config keys: %s", sorted(unknown_keys))
        init_kwargs = {k: v for k, v in config.items() if k in valid_params}

        model = cls(**init_kwargs)
        state = dict(ckpt["model_state_dict"])

        # Reconcile cycle_film size (e.g. 7-cycle checkpoints vs default 3 cycles)
        cycle_film_key = HEAD_PREFIX + "cycle_film.weight"
        if cycle_film_key in state and not reinit_head:
            ckpt_n = state[cycle_film_key].shape[0]
            head = model.profile_expansion_head
            if head.cycle_film.num_embeddings != ckpt_n:
                dim2 = state[cycle_film_key].shape[1]
                head.cycle_film = nn.Embedding(ckpt_n, dim2)
                _log.info("Resized cycle_film to %d cycles to match checkpoint.", ckpt_n)
            if head.num_cycles != ckpt_n:
                head.num_cycles = ckpt_n
                _log.info("Updated head.num_cycles to %d to match checkpoint.", ckpt_n)

        if reinit_head:
            for key in [k for k in state if k.startswith(HEAD_PREFIX)]:
                del state[key]

        missing, unexpected = model.load_state_dict(state, strict=False)

        missing_outside = [k for k in missing if not k.startswith(HEAD_PREFIX)]
        if missing_outside:
            raise RuntimeError(
                f"Checkpoint {path} is missing {len(missing_outside)} parameters outside "
                f"the residue head. Refusing to load, because these would stay randomly "
                f"initialised: {sorted(missing_outside)[:20]}"
                + (" ..." if len(missing_outside) > 20 else "")
            )

        unexpected_outside = [k for k in unexpected if not k.startswith(HEAD_PREFIX)]
        if unexpected_outside:
            _log.warning("Ignoring %d unused checkpoint tensors: %s",
                         len(unexpected_outside), sorted(unexpected_outside)[:10])

        head_unexpected = [k for k in unexpected if k.startswith(HEAD_PREFIX)]
        head_missing = [k for k in missing if k.startswith(HEAD_PREFIX)]
        if head_unexpected and not reinit_head:
            raise RuntimeError(
                f"Residue-head keys in {path} do not match RecycledResidueDecoder. "
                f"Pass reinit_head=True if you intend to reinitialise the head. "
                f"Differing keys: {sorted(head_unexpected)[:10]}"
            )
        if head_missing and not reinit_head:
            _log.info(
                "Head parameter(s) not found in checkpoint (using default init): %s",
                sorted(head_missing)[:10],
            )

        if reinit_head:
            fresh = sum(p.numel() for n, p in model.named_parameters()
                        if n.startswith(HEAD_PREFIX))
            _log.warning(
                "RecycledResidueDecoder freshly initialised: %d parameters (%.2fM). "
                "Encoder/aggregator/projector loaded intact.",
                fresh, fresh / 1e6,
            )

        _log.info("Loaded checkpoint v%.1f from %s",
                  ckpt.get("ckpt_version", 1.0), ckpt.get("saved_at", "unknown"))
        return model, ckpt


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

def create_theme_encoder(preset: str, vocab_size: int = 32000, max_tokens: int = 1024,
                         **overrides) -> ThemeEncoder:
    """Create a ThemeEncoder model from a named preset."""
    presets = {
        # ~33.7M params
        "test": dict(embed_dim=384, num_layers=10, nhead=6,  proj_dim=64,  dropout=0.10, num_proj_layers=2, ff_mult=4, proj_ff_mult=2),
        # ~77.1M params
        "base": dict(embed_dim=512, num_layers=16, nhead=8,  proj_dim=128, dropout=0.08, num_proj_layers=3, ff_mult=4, proj_ff_mult=4),
        # ~203.7M-221.5M params
        "deep": dict(embed_dim=768, num_layers=24, nhead=12, proj_dim=384, dropout=0.04, num_proj_layers=4, ff_mult=4, proj_ff_mult=4),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    config = dict(presets[preset])
    config.update({k: v for k, v in overrides.items() if v is not None})
    return ThemeEncoder(vocab_size=vocab_size, max_tokens=max_tokens, **config)
