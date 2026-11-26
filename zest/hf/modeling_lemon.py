"""Self-contained Hugging Face Transformers modeling code for LEMON-ZEST.

Provides:
  - LemonModel: Transformer encoder + AttentionAggregator + contrastive projector
    (works with ``pipeline("feature-extraction")`` and sentence embeddings).
  - LemonForMaskedLM: Token-level tied MLM head + optional RecycledResidueDecoder
    (works with ``pipeline("fill-mask")``).
  - LemonForSequenceClassification: Classification / regression head on pooled embedding.
  - LemonForTokenClassification: Per-token or per-residue (via RecycledResidueDecoder)
    classification head (works with ``pipeline("token-classification")``).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .configuration_lemon import LemonConfig


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = pos_emb.cos()
    sin = pos_emb.sin()
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LemonRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_tokens: int = 1024,
        base: float = 10000.0,
        scaling_type: str | None = "linear",
    ):
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


# ---------------------------------------------------------------------------
# Transformer Encoder & Attention Aggregator
# ---------------------------------------------------------------------------

class LemonAttentionBlock(nn.Module):
    def __init__(self, config: LemonConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.nhead = config.num_attention_heads
        self.head_dim = self.embed_dim // self.nhead
        self.zero_kv_pad_mask = config.zero_kv_pad_mask

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        ff_dim = int(self.embed_dim * config.ff_mult * 2 / 3)
        self.w1 = nn.Linear(self.embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(self.embed_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, self.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        bool_mask = mask.to(torch.bool) if mask is not None else None

        h = self.norm1(x)
        if bool_mask is not None:
            h = h.masked_fill(~bool_mask.unsqueeze(-1), 0.0)

        q = self.q_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        attn_mask = None
        if bool_mask is not None:
            m = bool_mask.unsqueeze(1).unsqueeze(-1)
            k = k.masked_fill(~m, 0.0)
            v = v.masked_fill(~m, 0.0)
            if not self.zero_kv_pad_mask:
                attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~bool_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if rope_emb is not None:
            q, k = _apply_rotary_pos_emb(q, k, rope_emb.unsqueeze(1))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.o_proj(out)
        if bool_mask is not None:
            out = out.masked_fill(~bool_mask.unsqueeze(-1), 0.0)
        x = x + self.dropout(out)

        h = self.norm2(x)
        if bool_mask is not None:
            h = h.masked_fill(~bool_mask.unsqueeze(-1), 0.0)
        ff = self.w3(F.silu(self.w1(h)) * self.w2(h))
        if bool_mask is not None:
            ff = ff.masked_fill(~bool_mask.unsqueeze(-1), 0.0)
        return x + self.dropout(ff)


class LemonEncoder(nn.Module):
    def __init__(self, config: LemonConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.token_emb = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.emb_dropout = nn.Dropout(config.embedding_dropout_prob)
        self.rope = LemonRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_tokens=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_type=config.rope_scaling_type,
        )
        self.layers = nn.ModuleList(
            [LemonAttentionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        x = self.emb_dropout(self.token_emb(input_ids))
        padding_mask = attention_mask.to(torch.bool) if attention_mask is not None else None
        rope_emb = self.rope(x)

        all_hidden = (x,) if output_hidden_states else None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(layer, x, padding_mask, rope_emb, use_reentrant=False)
            else:
                x = layer(x, padding_mask, rope_emb)
            if output_hidden_states:
                all_hidden = all_hidden + (x,)

        x = self.final_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return x, all_hidden


class LemonAttentionAggregator(nn.Module):
    def __init__(self, config: LemonConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.aggregator_num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = config.hidden_dropout_prob

        self.pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.align = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self, token_embs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = token_embs.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(self.pool_query.expand(B, -1, -1)).view(B, 1, H, D).transpose(1, 2)
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
        return self.align(self.norm(self.o_proj(out)).squeeze(1))


def _build_bottleneck_mlp(
    input_dim: int, hidden: int, dropout: float, num_layers: int, output_dim: int
) -> nn.Sequential:
    layers = []
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Dropout(dropout),
        ])
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# RecycledResidueDecoder
# ---------------------------------------------------------------------------

def _gather_window(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch, _, dim = x.shape
    _, length, width = idx.shape
    flat = idx.reshape(batch, length * width, 1).expand(batch, length * width, dim)
    return x.gather(1, flat).reshape(batch, length, width, dim)


class LemonRefinementBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_v = nn.Linear(hidden_dim, hidden_dim)
        self.cross_out = nn.Linear(hidden_dim, hidden_dim)

        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_q = nn.Linear(hidden_dim, hidden_dim)
        self.self_k = nn.Linear(hidden_dim, hidden_dim)
        self.self_v = nn.Linear(hidden_dim, hidden_dim)
        self.self_out = nn.Linear(hidden_dim, hidden_dim)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ff_dim = int(hidden_dim * ff_mult * 2 / 3)
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _windowed_attn(self, q_seq, kv_seq, window_idx, key_valid, q_proj, k_proj, v_proj, out_proj):
        B, L, _ = q_seq.shape
        W = window_idx.shape[-1]
        H, D = self.num_heads, self.head_dim
        qh = q_proj(q_seq).view(B, L, H, D)
        k = _gather_window(k_proj(kv_seq), window_idx).view(B, L, W, H, D)
        v = _gather_window(v_proj(kv_seq), window_idx).view(B, L, W, H, D)
        scores = torch.einsum("blhd,blkhd->blhk", qh, k) * self.scale
        scores = scores.masked_fill(~key_valid.unsqueeze(2), torch.finfo(scores.dtype).min)
        attn = self.dropout(scores.softmax(dim=-1))
        ctx = torch.einsum("blhk,blkhd->blhd", attn, v).reshape(B, L, H * D)
        return self.dropout(out_proj(ctx))

    def forward(self, z_res, kv_tok, tok_win, tok_valid, res_win, res_valid):
        h = self.cross_norm(z_res)
        z_res = z_res + self._windowed_attn(
            h, kv_tok, tok_win, tok_valid, self.cross_q, self.cross_k, self.cross_v, self.cross_out
        )
        h = self.self_norm(z_res)
        z_res = z_res + self._windowed_attn(
            h, h, res_win, res_valid, self.self_q, self.self_k, self.self_v, self.self_out
        )
        h = self.ffn_norm(z_res)
        return z_res + self.dropout(self.w3(F.silu(self.w1(h)) * self.w2(h)))


class LemonRecycledResidueDecoder(nn.Module):
    def __init__(self, config: LemonConfig):
        super().__init__()
        self.PADDING_INDEX = 31
        self.hidden_dim = config.residue_hidden_dim
        self.num_cycles = config.recycled_num_cycles
        self.token_window = config.recycled_token_window
        self.residue_window = config.recycled_residue_window

        self.input_proj = nn.Linear(config.hidden_size, self.hidden_dim)
        self.token_proj = nn.Linear(config.hidden_size, self.hidden_dim)
        self.film = nn.Embedding(self.PADDING_INDEX + 1, 2 * self.hidden_dim)
        self.cycle_film = nn.Embedding(self.num_cycles, 2 * self.hidden_dim)
        self.refinement = LemonRefinementBlock(
            self.hidden_dim, config.residue_num_heads, ff_mult=2, dropout=config.residue_dropout_prob
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.aa_logits_proj = nn.Linear(self.hidden_dim, config.residue_vocab_size)

    def _expand_full(self, z, token_lengths, target_len):
        B, T, D = z.shape
        device = z.device
        token_lengths = torch.clamp(token_lengths.to(device=device, dtype=torch.long), min=0)
        cumsum = torch.cumsum(token_lengths, dim=1)
        total_res = cumsum[:, -1]
        positions = torch.arange(target_len, device=device).unsqueeze(0).expand(B, -1)
        token_indices = torch.searchsorted(cumsum, positions + 1).clamp(0, T - 1)
        z_expanded = torch.gather(z, 1, token_indices.unsqueeze(-1).expand(-1, -1, D))
        cumsum_shifted = F.pad(cumsum[:, :-1], (1, 0), value=0)
        local_indices = positions - torch.gather(cumsum_shifted, 1, token_indices)
        padding_mask = positions >= total_res.unsqueeze(1)
        z_expanded = z_expanded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        local_indices = torch.where(
            padding_mask,
            torch.full_like(local_indices, self.PADDING_INDEX),
            local_indices.clamp(min=0, max=self.PADDING_INDEX - 1),
        )
        return z_expanded, local_indices, token_indices, padding_mask, token_lengths

    def _build_token_windows(self, token_indices, num_tokens, token_lengths, padding_mask):
        device = token_indices.device
        offsets = torch.arange(-self.token_window, self.token_window + 1, device=device)
        raw = token_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        win_idx = raw.clamp(0, num_tokens - 1)
        tok_valid = (token_lengths > 0).unsqueeze(-1).to(torch.uint8)
        key_valid = (raw >= 0) & (raw < num_tokens) & _gather_window(tok_valid, win_idx).squeeze(-1).bool()
        key_valid = key_valid | padding_mask.unsqueeze(-1)
        centre = torch.zeros_like(key_valid)
        centre[..., self.token_window] = True
        return win_idx, key_valid | centre

    def _build_residue_windows(self, target_len, padding_mask, device):
        B = padding_mask.shape[0]
        w = self.residue_window
        offsets = torch.arange(-w, w + 1, device=device)
        raw = torch.arange(target_len, device=device).unsqueeze(-1) + offsets.unsqueeze(0)
        win_idx = raw.clamp(0, target_len - 1).unsqueeze(0).expand(B, -1, -1)
        in_bounds = ((raw >= 0) & (raw < target_len)).unsqueeze(0).expand(B, -1, -1)
        valid = (~padding_mask).unsqueeze(-1).expand(-1, -1, win_idx.shape[-1])
        key_valid = in_bounds & torch.gather(valid, 1, win_idx).bool()
        centre = torch.zeros_like(key_valid)
        centre[..., w] = True
        return win_idx, key_valid | centre

    def forward(
        self,
        z: torch.Tensor,
        token_lengths: torch.Tensor,
        global_indices: torch.Tensor,
        global_pos_emb: nn.Embedding,
        num_cycles: Optional[int] = None,
        return_features: bool = False,
    ):
        target_len = global_indices.shape[1]
        z_exp, local_idx, tok_idx, pad_mask, token_lengths = self._expand_full(z, token_lengths, target_len)
        pos_global = global_pos_emb(global_indices.clamp(0, global_pos_emb.num_embeddings - 1)).to(z_exp.dtype)
        z_res = self.input_proj(z_exp + pos_global)
        gamma, beta = self.film(local_idx).to(z_res.dtype).chunk(2, dim=-1)
        z_res = z_res * (1.0 + gamma) + beta

        kv_tok = self.token_proj(z)
        tok_win, tok_valid = self._build_token_windows(tok_idx, z.shape[1], token_lengths, pad_mask)
        res_win, res_valid = self._build_residue_windows(target_len, pad_mask, z.device)

        cycles = num_cycles if num_cycles is not None else self.num_cycles
        for i in range(cycles):
            cyc_idx = min(i, self.cycle_film.num_embeddings - 1)
            c_gamma, c_beta = self.cycle_film(
                torch.tensor(cyc_idx, device=z_res.device, dtype=torch.long)
            ).chunk(2)
            z_res = z_res * (1.0 + c_gamma) + c_beta
            z_res = self.refinement(z_res, kv_tok, tok_win, tok_valid, res_win, res_valid)

        z_res = self.out_norm(z_res.masked_fill(pad_mask.unsqueeze(-1), 0.0))
        logits = self.aa_logits_proj(z_res)
        if return_features:
            return logits, z_res
        return logits


# ---------------------------------------------------------------------------
# Hugging Face PreTrainedModel Classes
# ---------------------------------------------------------------------------

class LemonPreTrainedModel(PreTrainedModel):
    config_class = LemonConfig
    base_model_prefix = "lemon"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class LemonModel(LemonPreTrainedModel):
    """LEMON backbone with contrastive pooling head and RecycledResidueDecoder."""

    def __init__(self, config: LemonConfig):
        super().__init__(config)
        self.core = LemonEncoder(config)
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())
        self.aggregator = LemonAttentionAggregator(config)
        self.projector = _build_bottleneck_mlp(
            config.hidden_size,
            config.hidden_size * config.proj_ff_mult,
            config.hidden_dropout_prob,
            config.num_proj_layers,
            output_dim=config.proj_dim,
        )
        self.profile_expansion_head = LemonRecycledResidueDecoder(config)
        self.global_pos_emb = nn.Embedding(3 * config.max_position_embeddings, config.hidden_size)
        self.post_init()
        nn.init.zeros_(self.profile_expansion_head.film.weight)
        nn.init.zeros_(self.profile_expansion_head.cycle_film.weight)
        nn.init.eye_(self.aggregator.align.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.core.token_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.core.token_emb = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        token_embs, hidden_states = self.core(
            input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states
        )
        family = self.aggregator(token_embs, mask=attention_mask)
        pooler_output = F.normalize(self.projector(family), p=2, dim=-1)

        if not return_dict:
            out = (token_embs, pooler_output)
            if output_hidden_states:
                out = out + (hidden_states,)
            return out

        return BaseModelOutputWithPooling(
            last_hidden_state=token_embs,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )

    def encode_residues(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_lengths: torch.Tensor,
        global_indices: torch.Tensor,
        num_cycles: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + RecycledResidueDecoder -> ``(aa_logits [B,L,20], residue_features [B,L,H])``."""
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        token_embs, _ = self.core(input_ids, attention_mask=attention_mask)
        return self.profile_expansion_head(
            token_embs,
            token_lengths,
            global_indices,
            self.global_pos_emb,
            num_cycles=num_cycles,
            return_features=True,
        )


class LemonForMaskedLM(LemonPreTrainedModel):
    """LEMON with tied token MLM head (compatible with ``pipeline('fill-mask')``)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LemonConfig):
        super().__init__(config)
        self.lemon = LemonModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.lemon.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.lemon(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        if not return_dict:
            return ((loss,) if loss is not None else ()) + (logits,)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class LemonForSequenceClassification(LemonPreTrainedModel):
    """LEMON with sequence-level classification/regression head over contrastive embeddings."""

    def __init__(self, config: LemonConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lemon = LemonModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.proj_dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.lemon(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(outputs.pooler_output))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.float().squeeze(-1))
            else:
                loss = F.cross_entropy(logits, labels)

        if not return_dict:
            return ((loss,) if loss is not None else ()) + (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class LemonForTokenClassification(LemonPreTrainedModel):
    """LEMON for per-residue (when ``token_lengths`` provided) or per-token classification."""

    def __init__(self, config: LemonConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lemon = LemonModel(config)
        self.dropout = nn.Dropout(config.residue_dropout_prob)
        self.residue_classifier = nn.Linear(config.residue_hidden_dim, config.num_labels)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_lengths: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if token_lengths is not None and global_indices is not None:
            _, res_feats = self.lemon.encode_residues(
                input_ids, attention_mask, token_lengths, global_indices
            )
            logits = self.residue_classifier(self.dropout(res_feats))
            hidden_states = None
        else:
            outputs = self.lemon(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            logits = self.token_classifier(self.dropout(outputs.last_hidden_state))
            hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100)

        if not return_dict:
            return ((loss,) if loss is not None else ()) + (logits,)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
