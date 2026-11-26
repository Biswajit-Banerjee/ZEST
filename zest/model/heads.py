"""Residue decoding head with AlphaFold-style iterative recycling."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_window(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``x`` [B, T, D] at window indices ``idx`` [B, L, K] -> [B, L, K, D]."""
    batch, _, dim = x.shape
    _, length, width = idx.shape
    flat = idx.reshape(batch, length * width, 1).expand(batch, length * width, dim)
    return x.gather(1, flat).reshape(batch, length, width, dim)



# ---------------------------------------------------------------------------
# Recycled Residue Decoder (AlphaFold-style iterative refinement)
# ---------------------------------------------------------------------------

class RefinementBlock(nn.Module):
    """Shared refinement block applied N times with shared weights.

    Each application performs:
      1. Cross-attention: residues query their parent ± W tokens
      2. Local self-attention: residues attend to ± W neighboring residues
      3. SwiGLU FFN: per-residue nonlinear transformation
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cross-attention: residues → tokens
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_v = nn.Linear(hidden_dim, hidden_dim)
        self.cross_out = nn.Linear(hidden_dim, hidden_dim)

        # Local self-attention: residues ↔ residues
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_q = nn.Linear(hidden_dim, hidden_dim)
        self.self_k = nn.Linear(hidden_dim, hidden_dim)
        self.self_v = nn.Linear(hidden_dim, hidden_dim)
        self.self_out = nn.Linear(hidden_dim, hidden_dim)

        # SwiGLU FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ff_dim = int(hidden_dim * ff_mult * 2 / 3)
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _windowed_attn(self, q_seq, kv_seq, window_idx, key_valid,
                       q_proj, k_proj, v_proj, out_proj):
        """Multi-head attention with gathered key windows.

        Args:
            q_seq:      [B, L, D] query sequence
            kv_seq:     [B, S, D] key/value source (S=T for cross, S=L for self)
            window_idx: [B, L, K] indices into kv_seq per query position
            key_valid:  [B, L, K] boolean mask for valid keys
        Returns:
            output:     [B, L, D]
        """
        batch, length, _ = q_seq.shape
        width = window_idx.shape[-1]
        heads, hd = self.num_heads, self.head_dim

        qh = q_proj(q_seq).view(batch, length, heads, hd)

        k = _gather_window(k_proj(kv_seq), window_idx)
        v = _gather_window(v_proj(kv_seq), window_idx)
        k = k.view(batch, length, width, heads, hd)
        v = v.view(batch, length, width, heads, hd)

        scores = torch.einsum("blhd,blkhd->blhk", qh, k) * self.scale
        scores = scores.masked_fill(
            ~key_valid.unsqueeze(2), torch.finfo(scores.dtype).min)
        attn = self.dropout(scores.softmax(dim=-1))

        ctx = torch.einsum("blhk,blkhd->blhd", attn, v)
        ctx = ctx.reshape(batch, length, heads * hd)
        return self.dropout(out_proj(ctx))

    def forward(self, z_res, kv_tok,
                tok_win_idx, tok_key_valid,
                res_win_idx, res_key_valid):
        """One refinement cycle.

        Args:
            z_res:         [B, L, D] current residue representations
            kv_tok:        [B, T, D] token representations (in hidden_dim)
            tok_win_idx:   [B, L, K_tok] token window indices per residue
            tok_key_valid: [B, L, K_tok] validity mask for token windows
            res_win_idx:   [B, L, K_res] residue window indices per residue
            res_key_valid: [B, L, K_res] validity mask for residue windows
        Returns:
            z_res:         [B, L, D] refined residue representations
        """
        # (a) Cross-attention: residues attend to parent ± W tokens
        h = self.cross_norm(z_res)
        z_res = z_res + self._windowed_attn(
            h, kv_tok, tok_win_idx, tok_key_valid,
            self.cross_q, self.cross_k, self.cross_v, self.cross_out,
        )

        # (b) Local self-attention: residues attend to ± W neighbor residues
        h = self.self_norm(z_res)
        z_res = z_res + self._windowed_attn(
            h, h, res_win_idx, res_key_valid,
            self.self_q, self.self_k, self.self_v, self.self_out,
        )

        # (c) SwiGLU FFN
        h = self.ffn_norm(z_res)
        z_res = z_res + self.dropout(self.w3(F.silu(self.w1(h)) * self.w2(h)))

        return z_res


class RecycledResidueDecoder(nn.Module):
    """Residue decoder with AlphaFold-style recycling.

    Maps token-level encoder outputs to residue-level representations and
    amino-acid logits.  Forward signature:
    forward signature: ``(z, token_lengths, global_indices, global_pos_emb)``
    returning ``[B, L, 20]`` amino-acid logits.

    Architecture:
      1. **Expand** token embeddings to residue length and initialise with
         FiLM-conditioned local offset + global position embedding.
      2. **Recycle** a shared ``RefinementBlock`` *N* times.  Each cycle does
         cross-attention back to the token stream (re-grounding) followed by
         local self-attention among residues (context exchange).
      3. **Project** to amino-acid logits.

    The final ``z_residues`` before projection is the per-residue embedding
    for downstream tasks (SS3, contacts, binding sites, etc.).
    """

    def __init__(
        self,
        embed_dim: int,
        max_residues: int,
        vocab_size: int = 20,
        padding_index: int = 31,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_cycles: int = 3,
        token_window: int = 2,
        residue_window: int = 16,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_residues = max_residues
        self.PADDING_INDEX = padding_index
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_cycles = num_cycles
        self.token_window = token_window
        self.residue_window = residue_window

        # Projection from encoder space to decoder space
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.token_proj = nn.Linear(embed_dim, hidden_dim)

        # FiLM: local offset → (gamma, beta) for multiplicative conditioning
        self.film = nn.Embedding(padding_index + 1, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)

        # Cycle-position FiLM: tells the shared decoder *which iteration* it is
        # currently executing.  Analogous to positional encoding on the sequence
        # axis — without this the decoder has no signal distinguishing cycle 0
        # ("rough first draft") from cycle 6 ("final fine-grained refinement"),
        # so the shared-weight block converges to a fixed point after 1-2 passes.
        #
        # Initialised to all-zeros → identity transform (gamma=0 → ×1, beta=0 →
        # +0), so loading from a checkpoint that pre-dates this parameter is
        # safe: the first forward pass is identical to the unmodified model and
        # gradients immediately start shaping the cycle embeddings.
        self.cycle_film = nn.Embedding(num_cycles, 2 * hidden_dim)
        nn.init.zeros_(self.cycle_film.weight)

        # Single refinement block, recycled num_cycles times
        self.refinement = RefinementBlock(
            hidden_dim, num_heads, ff_mult=ff_mult, dropout=dropout)

        # Output
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.aa_logits_proj = nn.Linear(hidden_dim, vocab_size)

    # -- Token-to-residue expansion -----------------------------------------

    def _expand_tokens(self, z, token_lengths, target_len):
        """Expand tokens to residues → ``(z_expanded, local_indices)``.

        Returns ``(z_expanded, local_indices)`` for downstream probes.
        """
        z_exp, local, _, _, _ = self._expand_full(z, token_lengths, target_len)
        return z_exp, local

    def _expand_full(self, z, token_lengths, target_len):
        batch, num_tokens, dim = z.shape
        device = z.device

        if not torch.is_tensor(token_lengths):
            token_lengths = torch.tensor(
                token_lengths, device=device, dtype=torch.long)
        token_lengths = torch.clamp(token_lengths, min=0)

        cumsum = torch.cumsum(token_lengths, dim=1)
        total_residues = cumsum[:, -1]

        positions = torch.arange(
            target_len, device=device).unsqueeze(0).expand(batch, -1)
        token_indices = torch.searchsorted(cumsum, positions + 1)
        token_indices = torch.clamp(token_indices, 0, num_tokens - 1)

        gather_idx = token_indices.unsqueeze(-1).expand(-1, -1, dim)
        z_expanded = torch.gather(z, 1, gather_idx)

        cumsum_shifted = F.pad(cumsum[:, :-1], (1, 0), value=0)
        start_positions = torch.gather(cumsum_shifted, 1, token_indices)
        local_indices = positions - start_positions

        padding_mask = positions >= total_residues.unsqueeze(1)
        z_expanded = z_expanded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        local_indices = torch.where(
            padding_mask,
            torch.full_like(local_indices, self.PADDING_INDEX),
            local_indices.clamp(min=0, max=self.PADDING_INDEX - 1),
        )
        return z_expanded, local_indices, token_indices, padding_mask, token_lengths

    # -- Window construction ------------------------------------------------

    def _build_token_windows(self, token_indices, num_tokens, token_lengths,
                             padding_mask):
        """Build ± token_window indices and validity mask per residue.

        Returns:
            window_idx: [B, L, 2W+1] clamped token indices
            key_valid:  [B, L, 2W+1] boolean — True where the key is real
        """
        device = token_indices.device
        offsets = torch.arange(
            -self.token_window, self.token_window + 1, device=device)
        raw = token_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        window_idx = raw.clamp(0, num_tokens - 1)

        token_valid = token_lengths > 0                            # [B, T]
        key_valid = (raw >= 0) & (raw < num_tokens)
        key_valid = key_valid & _gather_window(
            token_valid.unsqueeze(-1).to(torch.uint8), window_idx,
        ).squeeze(-1).bool()
        # Padded residue rows: keep centre key alive so softmax stays finite
        key_valid = key_valid | padding_mask.unsqueeze(-1).logical_not().logical_not()
        centre = torch.zeros_like(key_valid)
        centre[..., self.token_window] = True
        key_valid = key_valid | centre
        return window_idx, key_valid

    @staticmethod
    def _build_residue_windows(target_len, window, padding_mask, device):
        """Build ± residue_window indices and validity mask.

        Returns:
            window_idx: [B, L, 2W+1] clamped residue indices
            key_valid:  [B, L, 2W+1] boolean
        """
        batch = padding_mask.shape[0]
        offsets = torch.arange(-window, window + 1, device=device)
        pos = torch.arange(target_len, device=device)
        raw = pos.unsqueeze(-1) + offsets.unsqueeze(0)             # [L, 2W+1]
        window_idx = raw.clamp(0, target_len - 1)
        window_idx = window_idx.unsqueeze(0).expand(batch, -1, -1) # [B, L, 2W+1]

        in_bounds = (raw >= 0) & (raw < target_len)                # [L, 2W+1]
        in_bounds = in_bounds.unsqueeze(0).expand(batch, -1, -1)   # [B, L, 2W+1]
        # Keys at padded positions are invalid
        key_not_padded = ~padding_mask                              # [B, L]
        gathered_valid = key_not_padded.unsqueeze(-1).expand_as(window_idx)
        # Gather validity at window positions
        gathered_valid = torch.gather(
            key_not_padded.unsqueeze(-1).expand(-1, -1, window_idx.shape[-1]),
            1,
            window_idx,
        )
        key_valid = in_bounds & gathered_valid.bool()
        # Centre key always valid to prevent NaN in softmax
        centre = torch.zeros_like(key_valid)
        centre[..., window] = True
        key_valid = key_valid | centre
        return window_idx, key_valid

    # -- Core decode --------------------------------------------------------

    def _decode(self, z, token_lengths, global_indices, global_pos_emb,
                commit_threshold=0.0):
        """Full decode pipeline → refined residue features [B, L, hidden_dim].

        Args:
            commit_threshold: if > 0, positions whose predicted AA probability
                exceeds this threshold have their representation replaced with
                the weight-tied AA embedding between cycles (confidence-gated
                identity feedback).  Set to 0 to disable.
        """
        target_len = global_indices.shape[1]
        z_expanded, local_indices, token_indices, padding_mask, token_lengths = \
            self._expand_full(z, token_lengths, target_len)

        global_indices = torch.clamp(global_indices, min=0)
        pos_global = global_pos_emb(global_indices).to(z_expanded.dtype)

        # Initialise residue representations in decoder space
        z_res = self.input_proj(z_expanded + pos_global)
        gamma, beta = self.film(local_indices).to(z_res.dtype).chunk(2, dim=-1)
        z_res = z_res * (1.0 + gamma) + beta

        # Project tokens into decoder space (done once, reused across cycles)
        kv_tok = self.token_proj(z)

        # Build window indices (done once)
        tok_win, tok_valid = self._build_token_windows(
            token_indices, z.shape[1], token_lengths, padding_mask)
        res_win, res_valid = self._build_residue_windows(
            target_len, self.residue_window, padding_mask, z.device)

        # Confidence-gated feedback state
        committed = torch.zeros(z_res.shape[:2], dtype=torch.bool,
                                device=z_res.device) if commit_threshold > 0 else None

        # Recycled refinement
        for i in range(self.num_cycles):
            # ── Cycle-position conditioning ────────────────────────────
            # Apply a learned per-cycle FiLM transform before the shared
            # refinement block so the block knows which iteration it is in.
            # Without this the shared weights see identical inputs in every
            # cycle and converge to a fixed point after 1-2 passes.
            cyc_idx = min(i, self.cycle_film.num_embeddings - 1)
            c_gamma, c_beta = self.cycle_film(
                torch.tensor(cyc_idx, device=z_res.device, dtype=torch.long)
            ).chunk(2)
            z_res = z_res * (1.0 + c_gamma) + c_beta

            z_res = self.refinement(
                z_res, kv_tok,
                tok_win, tok_valid,
                res_win, res_valid,
            )

            # Confidence-gated identity feedback: replace high-confidence
            # positions with their predicted AA embedding so subsequent
            # cycles see differentiated context instead of homogeneous
            # representations.  No new parameters — uses weight-tied
            # aa_logits_proj.weight as an embedding table.
            if commit_threshold > 0 and i < self.num_cycles - 1:
                logits_k = self.aa_logits_proj(self.out_norm(z_res))
                probs_k = logits_k.softmax(-1)
                conf_k, pred_k = probs_k.max(-1)
                newly_confident = ((conf_k > commit_threshold)
                                   & ~committed & ~padding_mask)
                if newly_confident.any():
                    pred_embed = F.embedding(
                        pred_k, self.aa_logits_proj.weight)  # [B,L,D]
                    z_res = torch.where(
                        newly_confident.unsqueeze(-1), pred_embed, z_res)
                    committed = committed | newly_confident

        # Zero out padding positions
        z_res = z_res.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.out_norm(z_res)

    def decode_all_z_res(self, z, token_lengths, global_indices, global_pos_emb,
                         num_cycles=None):
        """Like _decode but returns list of z_res at each cycle (for diagnostics).

        Returns:
            list of num_cycles tensors, each [B, L, hidden_dim], after out_norm.
        """
        num_cycles = num_cycles or self.num_cycles
        target_len = global_indices.shape[1]
        z_expanded, local_indices, token_indices, padding_mask, token_lengths = \
            self._expand_full(z, token_lengths, target_len)

        global_indices = torch.clamp(global_indices, min=0)
        pos_global = global_pos_emb(global_indices).to(z_expanded.dtype)

        z_res = self.input_proj(z_expanded + pos_global)
        gamma, beta = self.film(local_indices).to(z_res.dtype).chunk(2, dim=-1)
        z_res = z_res * (1.0 + gamma) + beta

        kv_tok = self.token_proj(z)
        tok_win, tok_valid = self._build_token_windows(
            token_indices, z.shape[1], token_lengths, padding_mask)
        res_win, res_valid = self._build_residue_windows(
            target_len, self.residue_window, padding_mask, z.device)

        all_z_res = []
        for i in range(num_cycles):
            if hasattr(self, 'cycle_film'):
                cyc_idx = min(i, self.cycle_film.num_embeddings - 1)
                c_gamma, c_beta = self.cycle_film(
                    torch.tensor(cyc_idx, device=z_res.device, dtype=torch.long)
                ).chunk(2)
                z_res = z_res * (1.0 + c_gamma) + c_beta

            z_res = self.refinement(z_res, kv_tok, tok_win, tok_valid,
                                    res_win, res_valid)
            z_normed = self.out_norm(z_res.masked_fill(padding_mask.unsqueeze(-1), 0.0))
            all_z_res.append(z_normed)

        return all_z_res

    # -- Deep-supervision / per-cycle interface -----------------------------

    def forward_all_cycles(self, z, token_lengths, global_indices,
                           global_pos_emb, num_cycles=None, noise_prob=0.0,
                           stop_grad_before_last_n=0,
                           forced_error_rate=0.0, aa_labels=None,
                           commit_threshold=0.0):
        """Decode with per-cycle logits for deep supervision / diagnostics.

        Args:
            forced_error_rate: probability of corrupting each non-padding
                position to a guaranteed-wrong token between cycles (SUNDAE-
                style).  Requires ``aa_labels`` to ensure the injected token
                differs from the ground truth.  Supersedes ``noise_prob`` when
                > 0; both may be 0 to disable inter-cycle perturbation.
            aa_labels: ``[B, L]`` ground-truth amino-acid indices; positions
                that should not be predicted are ``-100``.
            commit_threshold: if > 0, positions whose predicted AA
                probability exceeds this threshold (at eval time only,
                when forced_error_rate is not active) have their
                representation replaced with the weight-tied AA
                embedding.  This provides genuine identity feedback
                between cycles — confident positions anchor their
                neighbors' self-attention context.

        Returns:
            all_logits: list of ``[B, L, vocab_size]`` tensors, one per cycle.
            z_res:      ``[B, L, hidden_dim]`` refined features after the last
                        cycle (normed, padding zeroed).
        """
        num_cycles = num_cycles or self.num_cycles
        target_len = global_indices.shape[1]

        # ── Same setup as _decode ─────────────────────────────────────
        z_expanded, local_indices, token_indices, padding_mask, token_lengths = \
            self._expand_full(z, token_lengths, target_len)

        global_indices = torch.clamp(global_indices, min=0)
        pos_global = global_pos_emb(global_indices).to(z_expanded.dtype)

        z_res = self.input_proj(z_expanded + pos_global)
        gamma, beta = self.film(local_indices).to(z_res.dtype).chunk(2, dim=-1)
        z_res = z_res * (1.0 + gamma) + beta

        kv_tok = self.token_proj(z)

        tok_win, tok_valid = self._build_token_windows(
            token_indices, z.shape[1], token_lengths, padding_mask)
        res_win, res_valid = self._build_residue_windows(
            target_len, self.residue_window, padding_mask, z.device)

        vocab = self.aa_logits_proj.weight.shape[0]  # 20

        # Confidence-gated feedback tracking.  Enabled at eval time
        # (as before) and optionally during training when commit_threshold > 0
        # and SUNDAE corruption is off.  Training-time commitment teaches the
        # model to use cycle-k predictions as anchors for cycle k+1, which is
        # the key mechanism that prevents the recycling plateau.
        use_commit = (commit_threshold > 0 and forced_error_rate == 0)
        committed = (torch.zeros(z_res.shape[:2], dtype=torch.bool,
                                 device=z_res.device)
                     if use_commit else None)

        # ── Recycled refinement with per-cycle logits ─────────────────
        all_logits = []
        for i in range(num_cycles):
            # SUNDAE-style forced-error corruption: replace 15 % of
            # non-padding positions with the representation of a
            # guaranteed-wrong amino acid.  Re-embeds via weight-tied
            # transpose of aa_logits_proj (no new parameters).
            # Fires between cycles (not before cycle 0, not before the
            # last cycle so the final prediction is uncorrupted).
            if (self.training and forced_error_rate > 0
                    and aa_labels is not None and 0 < i < num_cycles - 1):
                B, L = z_res.shape[:2]
                corrupt_mask = (torch.rand(B, L, device=z_res.device)
                                < forced_error_rate)
                corrupt_mask = corrupt_mask & ~padding_mask  # skip padding
                if corrupt_mask.any():
                    aa_clamp = aa_labels[:, :L].clamp(min=0)  # -100 → 0
                    # random token guaranteed different from ground truth
                    rand_offset = torch.randint(
                        1, vocab, (B, L), device=z_res.device)
                    wrong_tok = (aa_clamp + rand_offset) % vocab
                    # weight-tied re-embedding: aa_logits_proj.weight is
                    # [vocab, hidden_dim]; index it as an embedding table
                    wrong_embed = F.embedding(
                        wrong_tok, self.aa_logits_proj.weight)  # [B,L,D]
                    z_res = torch.where(
                        corrupt_mask.unsqueeze(-1), wrong_embed, z_res)

            # Legacy noise injection (zero-out); disabled when
            # forced_error_rate > 0
            elif (self.training and noise_prob > 0
                    and forced_error_rate == 0 and 0 < i < num_cycles - 1):
                noise_mask = (torch.rand(z_res.shape[:2], device=z_res.device)
                              < noise_prob)
                z_res = z_res.masked_fill(noise_mask.unsqueeze(-1), 0.0)

            # ── Cycle-position conditioning ────────────────────────────
            cyc_idx = min(i, self.cycle_film.num_embeddings - 1)
            c_gamma, c_beta = self.cycle_film(
                torch.tensor(cyc_idx, device=z_res.device, dtype=torch.long)
            ).chunk(2)
            z_res = z_res * (1.0 + c_gamma) + c_beta

            z_res = self.refinement(
                z_res, kv_tok,
                tok_win, tok_valid,
                res_win, res_valid,
            )

            # Stop-gradient: detach the carried-forward state so gradients
            # only flow through the last N cycles of the recurrence (AF2
            # style).  Early cycles still get a direct loss signal from
            # their own deep-supervision CE term.
            if (stop_grad_before_last_n > 0
                    and i < num_cycles - stop_grad_before_last_n):
                z_res = z_res.detach()

            logits_k = self.aa_logits_proj(self.out_norm(z_res))
            all_logits.append(logits_k)

            # Confidence-gated identity feedback: replace high-confidence
            # positions with their predicted AA embedding so subsequent
            # cycles see differentiated neighbor context.
            #
            # During training with aa_labels provided, ONLY commit positions
            # that were originally masked (aa_labels == -100).  Non-masked
            # positions already have their true token in kv_tok and committing
            # them trivially leaks the answer to neighbours via self-attention,
            # causing shortcut learning.  At eval time (aa_labels=None) there
            # is no ground truth available so we commit all confident positions.
            if use_commit and i < num_cycles - 1:
                probs_k = logits_k.softmax(-1)
                conf_k, pred_k = probs_k.max(-1)
                eligible = ~committed & ~padding_mask
                if self.training and aa_labels is not None:
                    L = z_res.shape[1]
                    is_masked = (aa_labels[:, :L] == -100)
                    eligible = eligible & is_masked
                newly_confident = (conf_k > commit_threshold) & eligible
                if newly_confident.any():
                    pred_embed = F.embedding(
                        pred_k, self.aa_logits_proj.weight)
                    z_res = torch.where(
                        newly_confident.unsqueeze(-1), pred_embed, z_res)
                    committed = committed | newly_confident

        z_res = z_res.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return all_logits, self.out_norm(z_res)

    # -- Public interface ---------------------------------------------------

    def get_residue_embeddings(self, z, token_lengths, global_indices,
                               global_pos_emb):
        """Residue embeddings in ``embed_dim`` space (for probe compatibility).

        Returns expanded token embeddings + global position in the original
        encoder dimension, bypassing the decoder.  Use ``get_residue_logits``
        for the refined decoder-space features.
        """
        target_len = global_indices.shape[1]
        z_expanded, _, _, _, _ = self._expand_full(z, token_lengths, target_len)
        global_indices = torch.clamp(global_indices, min=0)
        pos_global = global_pos_emb(global_indices).to(z_expanded.dtype)
        return z_expanded + pos_global

    def get_residue_logits(self, z, token_lengths, global_indices,
                           global_pos_emb, commit_threshold=0.0):
        """Refined residue features [B, L, hidden_dim] after all cycles."""
        return self._decode(z, token_lengths, global_indices, global_pos_emb,
                            commit_threshold=commit_threshold)

    def forward(self, z, token_lengths, global_indices, global_pos_emb,
                return_features=False, commit_threshold=0.0):
        z_res = self._decode(z, token_lengths, global_indices, global_pos_emb,
                             commit_threshold=commit_threshold)
        logits = self.aa_logits_proj(z_res)
        if return_features:
            return logits, z_res
        return logits
