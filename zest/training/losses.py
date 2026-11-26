"""
Loss functions and memory bank for contrastive fine-tuning.

Protein domain:
    - ``hierarchical_infonce`` — multi-level InfoNCE at A / T / H CATH levels
    - ``hierarchical_smooth_ap`` — differentiable AP at each hierarchy level
    - ``hierarchical_margin_ranking`` — cosine ordering loss per hierarchy group
    - ``selective_coarse_preservation_loss`` — cross-SF KL divergence for teacher preservation

Code domain:
    - ``code_infonce`` — single-level InfoNCE for algorithm class clone detection
    - ``code_smooth_ap`` — single-level differentiable AP
    - ``_batch_map_at_r`` — MAP@R metric (standard POJ-104 evaluation)

Shared:
    - ``MemoryBank`` — FIFO queue of detached embeddings for extra negatives
    - ``UncertaintyWeightedLoss`` — learned multi-task weighting (Kendall et al.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-task uncertainty weighting
# ---------------------------------------------------------------------------

class UncertaintyWeightedLoss(nn.Module):
    """Learned uncertainty weighting for multi-task losses (Kendall et al.)."""
    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_token, loss_residue, priority_token=0.7, priority_residue=0.3):
        precision_token = torch.exp(-self.log_vars[0])
        loss_1 = precision_token * loss_token + 0.5 * self.log_vars[0]
        precision_residue = torch.exp(-self.log_vars[1])
        loss_2 = precision_residue * loss_residue + 0.5 * self.log_vars[1]
        weighted_loss = priority_token * loss_1 + priority_residue * loss_2

        with torch.no_grad():
            ew_tok = priority_token * precision_token
            ew_res = priority_residue * precision_residue
            t_a = ew_tok * loss_token
            t_b = ew_res * loss_residue
            total = t_a + t_b + 1e-8
        return weighted_loss, {
            "w_tok": ew_tok.item(), "w_res": ew_res.item(),
            "pct_tok": (t_a / total).item(), "pct_res": (t_b / total).item(),
        }


# ---------------------------------------------------------------------------
# Token reconstruction from residue features
# ---------------------------------------------------------------------------

class TokenReconstructionHead(nn.Module):
    """Token-level reconstruction objectives from refined residue embeddings.

    Pools residue embeddings back to token-level via scatter-mean, then
    predicts:
      1. **Token ID** — cross-entropy against the original ZEST token,
         using a learned projection up to ``embed_dim`` tied with the
         encoder's token embedding (weight-tied output).
      2. **Token length** — cross-entropy over discrete length classes.

    Only scored at masked positions (``token_labels != -100``).

    Parameters added: ``up_proj`` (~hidden_dim * embed_dim) and
    ``length_head`` (~hidden_dim * max_token_length).
    """

    def __init__(self, hidden_dim: int, embed_dim: int,
                 max_token_length: int = 32, dropout: float = 0.1):
        super().__init__()
        self.max_token_length = max_token_length
        # Project from decoder space (hidden_dim) to encoder space (embed_dim)
        # for weight-tied token ID prediction
        self.up_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Token length classification
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_token_length),
        )

    @staticmethod
    def pool_residues_to_tokens(z_res, token_lengths):
        """Scatter-mean pool residue features [B, L, D] → [B, T, D].

        Uses the same searchsorted logic as the expansion head to map
        residue positions back to their parent token.
        """
        B, L, D = z_res.shape
        T = token_lengths.shape[1]
        device = z_res.device

        cumsum = torch.cumsum(token_lengths, dim=1)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        token_idx = torch.searchsorted(cumsum, positions + 1).clamp(0, T - 1)

        token_idx_exp = token_idx.unsqueeze(-1).expand(-1, -1, D)
        z_sum = torch.zeros(B, T, D, device=device, dtype=z_res.dtype)
        z_sum.scatter_add_(1, token_idx_exp, z_res)

        lengths = token_lengths.clamp(min=1).unsqueeze(-1).to(z_res.dtype)
        return z_sum / lengths

    def forward(self, z_res, token_lengths, token_labels, embed_weight,
                label_smoothing=0.0):
        """Compute token reconstruction losses at masked positions.

        Args:
            z_res:          [B, L, hidden_dim] refined residue features.
            token_lengths:  [B, T] length of each token.
            token_labels:   [B, T] original token IDs at masked positions,
                            -100 elsewhere.
            embed_weight:   [V, embed_dim] encoder token embedding weight
                            (used as tied output projection).
            label_smoothing: label smoothing for token ID CE.

        Returns:
            token_loss:  CE loss for token ID prediction.
            length_loss: CE loss for token length prediction.
            metrics:     dict with tok_recon_acc, len_acc.
        """
        z_tok = self.pool_residues_to_tokens(z_res, token_lengths)

        # Token ID prediction (weight-tied with encoder embedding)
        z_up = self.up_proj(z_tok)
        token_logits = F.linear(z_up, embed_weight)
        token_loss = F.cross_entropy(
            token_logits.reshape(-1, token_logits.size(-1)),
            token_labels.reshape(-1),
            ignore_index=-100, label_smoothing=label_smoothing)

        # Token length prediction (only at masked positions)
        length_logits = self.length_head(z_tok)
        length_targets = token_lengths.clone()
        masked = token_labels != -100
        length_targets[~masked] = -100
        # Shift to 0-indexed classes: length 1 → class 0, etc.
        valid = length_targets != -100
        length_targets[valid] = (length_targets[valid] - 1).clamp(
            0, self.max_token_length - 1)
        length_loss = F.cross_entropy(
            length_logits.reshape(-1, length_logits.size(-1)),
            length_targets.reshape(-1),
            ignore_index=-100)

        with torch.no_grad():
            t_valid = token_labels != -100
            if t_valid.any():
                tok_recon_acc = (
                    token_logits.argmax(-1)[t_valid]
                    == token_labels[t_valid]
                ).float().mean().item()
            else:
                tok_recon_acc = 0.0

            l_valid = length_targets != -100
            if l_valid.any():
                l_preds = length_logits.argmax(-1) + 1  # shift back
                len_acc = (
                    l_preds[l_valid] == token_lengths[l_valid]
                ).float().mean().item()
            else:
                len_acc = 0.0

        return token_loss, length_loss, {
            "tok_recon_acc": tok_recon_acc,
            "len_acc": len_acc,
        }


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------

class MemoryBank:
    """FIFO queue of detached embeddings + labels for extra InfoNCE negatives."""

    def __init__(self, capacity: int, embed_dim: int, device: torch.device):
        self.capacity = capacity
        self.embeddings = torch.zeros(capacity, embed_dim, device=device)
        self.a_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self.t_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self.h_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self._ptr = 0
        self._full = False

    def enqueue(self, emb, a, t, h):
        n = emb.shape[0]
        if n == 0:
            return
        emb = emb.detach()
        if n >= self.capacity:
            emb, a, t, h = emb[-self.capacity:], a[-self.capacity:], t[-self.capacity:], h[-self.capacity:]
            n = self.capacity
        end = self._ptr + n
        if end <= self.capacity:
            self.embeddings[self._ptr:end] = emb
            self.a_labels[self._ptr:end] = a
            self.t_labels[self._ptr:end] = t
            self.h_labels[self._ptr:end] = h
        else:
            first = self.capacity - self._ptr
            self.embeddings[self._ptr:] = emb[:first]
            self.a_labels[self._ptr:] = a[:first]
            self.t_labels[self._ptr:] = t[:first]
            self.h_labels[self._ptr:] = h[:first]
            rest = n - first
            self.embeddings[:rest] = emb[first:]
            self.a_labels[:rest] = a[first:]
            self.t_labels[:rest] = t[first:]
            self.h_labels[:rest] = h[first:]
        self._ptr = end % self.capacity
        if end >= self.capacity:
            self._full = True

    def get(self):
        size = self.capacity if self._full else self._ptr
        if size == 0:
            return None
        return (self.embeddings[:size], self.a_labels[:size],
                self.t_labels[:size], self.h_labels[:size])

    def sample(self, max_size: int):
        size = self.capacity if self._full else self._ptr
        if size == 0:
            return None
        if max_size <= 0 or max_size >= size:
            return self.get()
        idx = torch.randperm(size, device=self.embeddings.device)[:max_size]
        return (
            self.embeddings.index_select(0, idx),
            self.a_labels.index_select(0, idx),
            self.t_labels.index_select(0, idx),
            self.h_labels.index_select(0, idx),
        )


class MemoryBank4Level(MemoryBank):
    """MemoryBank extended with profile (P) labels for 4-level hierarchy."""

    def __init__(self, capacity: int, embed_dim: int, device: torch.device):
        super().__init__(capacity, embed_dim, device)
        self.p_labels = torch.zeros(capacity, dtype=torch.long, device=device)

    def enqueue(self, emb, a, t, h, p=None):
        if p is None:
            super().enqueue(emb, a, t, h)
            return
        n = emb.shape[0]
        if n == 0:
            return
        emb = emb.detach()
        if n >= self.capacity:
            emb = emb[-self.capacity:]
            a, t, h, p = a[-self.capacity:], t[-self.capacity:], h[-self.capacity:], p[-self.capacity:]
            n = self.capacity
        end = self._ptr + n
        if end <= self.capacity:
            self.embeddings[self._ptr:end] = emb
            self.a_labels[self._ptr:end] = a
            self.t_labels[self._ptr:end] = t
            self.h_labels[self._ptr:end] = h
            self.p_labels[self._ptr:end] = p
        else:
            first = self.capacity - self._ptr
            self.embeddings[self._ptr:] = emb[:first]
            self.a_labels[self._ptr:] = a[:first]
            self.t_labels[self._ptr:] = t[:first]
            self.h_labels[self._ptr:] = h[:first]
            self.p_labels[self._ptr:] = p[:first]
            rest = n - first
            self.embeddings[:rest] = emb[first:]
            self.a_labels[:rest] = a[first:]
            self.t_labels[:rest] = t[first:]
            self.h_labels[:rest] = h[first:]
            self.p_labels[:rest] = p[first:]
        self._ptr = end % self.capacity
        if end >= self.capacity:
            self._full = True

    def get(self):
        size = self.capacity if self._full else self._ptr
        if size == 0:
            return None
        return (self.embeddings[:size], self.a_labels[:size],
                self.t_labels[:size], self.h_labels[:size],
                self.p_labels[:size])

    def sample(self, max_size: int):
        size = self.capacity if self._full else self._ptr
        if size == 0:
            return None
        if max_size <= 0 or max_size >= size:
            return self.get()
        idx = torch.randperm(size, device=self.embeddings.device)[:max_size]
        return (
            self.embeddings.index_select(0, idx),
            self.a_labels.index_select(0, idx),
            self.t_labels.index_select(0, idx),
            self.h_labels.index_select(0, idx),
            self.p_labels.index_select(0, idx),
        )


# ---------------------------------------------------------------------------
# Protein-domain losses (3-level CATH hierarchy)
# ---------------------------------------------------------------------------

def _hierarchical_masks(name, query_a, query_t, query_h,
                        key_a, key_t, key_h, self_mask,
                        benchmark_aligned=False):
    not_self = ~self_mask
    same_a = query_a.unsqueeze(1) == key_a.unsqueeze(0)
    same_t = query_t.unsqueeze(1) == key_t.unsqueeze(0)
    same_h = query_h.unsqueeze(1) == key_h.unsqueeze(0)
    if not benchmark_aligned:
        same = {"a": same_a, "t": same_t, "h": same_h}[name]
        return same & not_self, not_self
    if name == "a":
        return same_a & ~same_t & not_self, ~same_t & not_self
    if name == "t":
        return same_t & ~same_h & not_self, ~same_h & not_self
    return same_h & not_self, not_self


def hierarchical_infonce(embeddings, a_labels, t_labels, h_labels,
                         temp_a=0.10, temp_t=0.07, temp_h=0.05,
                         weight_a=0.15, weight_t=0.35, weight_h=0.50,
                         mem_emb=None, mem_a=None, mem_t=None, mem_h=None,
                         hard_neg_weight=1.0, benchmark_aligned=False):
    """
    Multi-level InfoNCE with optional memory bank and hard negative weighting.

    Hard negatives: entries sharing a coarser label but differing at the current
    level (e.g. same topology, different superfamily) receive upweighted
    contribution to the denominator.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_a = torch.cat([a_labels, mem_a.to(device)])
        key_t = torch.cat([t_labels, mem_t.to(device)])
        key_h = torch.cat([h_labels, mem_h.to(device)])
    else:
        keys, key_a, key_t, key_h = embeddings, a_labels, t_labels, h_labels
    K = keys.shape[0]

    sim = embeddings @ keys.T
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)

    results = {}
    total = zero

    for name, ql, kl, temp, w, coarse_q, coarse_k in [
        ("a", a_labels, key_a, temp_a, weight_a, None, None),
        ("t", t_labels, key_t, temp_t, weight_t, a_labels, key_a),
        ("h", h_labels, key_h, temp_h, weight_h, t_labels, key_t),
    ]:
        pos, valid = _hierarchical_masks(
            name, a_labels, t_labels, h_labels,
            key_a, key_t, key_h, self_mask, benchmark_aligned)
        has_pos = pos.any(dim=1)
        if not has_pos.any():
            results[f"loss_{name}"] = zero
            results[f"acc_{name}"] = zero
            continue

        logits = (sim / temp).masked_fill(~valid, -1e9)
        exp = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        exp = exp.masked_fill(~valid, 0.0)

        if hard_neg_weight > 1.0 and coarse_q is not None:
            hard = ((coarse_q.unsqueeze(1) == coarse_k.unsqueeze(0))
                    & (ql.unsqueeze(1) != kl.unsqueeze(0)) & valid)
            exp = exp * torch.where(hard, hard_neg_weight, 1.0)

        pos_exp = (exp * pos.float()).sum(1)
        loss_i = -torch.log(pos_exp / (exp.sum(1) + 1e-8) + 1e-8)
        level_loss = loss_i[has_pos].mean()
        top1 = sim.masked_fill(~valid, -1e9).argmax(dim=1)
        acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

        results[f"loss_{name}"] = level_loss
        results[f"acc_{name}"] = acc
        total = total + w * level_loss

    results["loss"] = total
    return results


# ---------------------------------------------------------------------------
# Hard-positive weighting: upweight low-probability (remote) same-H pairs
# ---------------------------------------------------------------------------

def hard_positive_infonce(
    embeddings, h_labels, pair_probs,
    temp=0.05, gamma=1.0, prob_floor=0.3,
    mem_emb=None, mem_h=None,
):
    """H-level InfoNCE that upweights hard (low-probability) same-H pairs.

    Uses the SupCon weighted-average formulation: each positive gets its
    own log-ratio term, weighted by ``(1 - probability)^gamma`` for
    probabilities >= ``prob_floor``, and 0 below the floor.

    The floor exists because very low broadcast probabilities can mean either
    "genuinely remote homolog" (what we want to amplify) or "noisy/low-coverage
    broadcast label" (what we do NOT want to amplify).

    Args:
        embeddings: [2B, D] L2-normalised (anchors then positives concatenated).
        h_labels:   [2B] superfamily labels.
        pair_probs: [B] HHblits probability for each (anchor_i, positive_i) pair.
        temp:       temperature for logits.
        gamma:      exponent for hard-positive weighting.
        prob_floor: probabilities below this are dropped (weight 0).
        mem_emb:    optional memory bank embeddings.
        mem_h:      optional memory bank H-labels.

    Returns:
        dict with 'loss', 'acc_h', 'n_pos_per_query'.
    """
    B = pair_probs.shape[0]
    N = embeddings.shape[0]  # = 2B
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_h = torch.cat([h_labels, mem_h.to(device)])
    else:
        keys, key_h = embeddings, h_labels
    K = keys.shape[0]

    sim = embeddings @ keys.T  # [N, K]
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)

    same_h = h_labels.unsqueeze(1) == key_h.unsqueeze(0)  # [N, K]
    pos = same_h & ~self_mask
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return {"loss": zero, "acc_h": zero, "n_pos_per_query": zero}

    # Build weight matrix: default 1.0 for all same-H positives,
    # override paired entries (anc_i <-> pos_i) with hard-positive weight
    w = pos.float()

    # Compute hard-positive weights: high weight for low probability
    hp_weight = torch.where(
        pair_probs >= prob_floor,
        (1.0 - pair_probs).clamp(min=0.0) ** gamma,
        torch.zeros_like(pair_probs),
    )

    if N >= 2 * B and K >= 2 * B:
        idx = torch.arange(B, device=device)
        w[idx, B + idx] = hp_weight       # anc_i -> pos_i
        w[B + idx, idx] = hp_weight       # pos_i -> anc_i

    logits = sim / temp  # [N, K]
    logits_masked = logits.masked_fill(self_mask, -1e9)

    log_sum_exp = torch.logsumexp(logits_masked, dim=1, keepdim=True)
    log_prob = logits_masked - log_sum_exp  # [N, K]

    # Weighted average of per-positive NLLs (SupCon formulation)
    per_pos_nll = -log_prob
    numerator = (w * per_pos_nll).sum(1)
    denominator = w.sum(1).clamp(min=1e-8)
    loss_per_query = numerator / denominator
    loss = loss_per_query[has_pos].mean()

    top1 = logits_masked.argmax(dim=1)
    acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()
    n_pos = pos.float().sum(1)[has_pos].mean()

    return {"loss": loss, "acc_h": acc, "n_pos_per_query": n_pos.detach()}


# ---------------------------------------------------------------------------
# 4-level hierarchy: A / T / H / P (profile)
# ---------------------------------------------------------------------------

def _build_soft_pos_matrix(p_q, p_k, h_q, h_k, profile_sims, not_self):
    """Build a soft positive weight matrix for P-level from cross-profile sims.

    Returns a float tensor [N, K] where entry (i, j) is the normalized
    HHblits probability (0–1) if profiles i and j are in ``profile_sims``
    and share the same H-label, else 0.
    """
    N, K = p_q.shape[0], p_k.shape[0]
    device = p_q.device
    soft = torch.zeros(N, K, device=device)

    same_h = h_q.unsqueeze(1) == h_k.unsqueeze(0)
    diff_p = p_q.unsqueeze(1) != p_k.unsqueeze(0)
    valid_q = p_q >= 0
    valid_k = p_k >= 0
    candidates = (same_h & diff_p & not_self
                  & valid_q.unsqueeze(1) & valid_k.unsqueeze(0))

    if not candidates.any():
        return soft

    qi, ki = candidates.nonzero(as_tuple=True)
    pq_vals = p_q[qi].cpu().tolist()
    pk_vals = p_k[ki].cpu().tolist()

    sims = []
    for pq, pk in zip(pq_vals, pk_vals):
        key = (min(pq, pk), max(pq, pk))
        sims.append(profile_sims.get(key, 0.0))

    soft[qi, ki] = torch.tensor(sims, device=device, dtype=torch.float32)
    return soft


def hierarchical_infonce_4level(
    embeddings, a_labels, t_labels, h_labels, p_labels,
    temp_a=0.10, temp_t=0.07, temp_h=0.05, temp_p=0.03,
    weight_a=0.10, weight_t=0.25, weight_h=0.35, weight_p=0.30,
    mem_emb=None, mem_a=None, mem_t=None, mem_h=None, mem_p=None,
    hard_neg_weight=1.0,
    profile_sims=None,
    soft_pos_scale=0.5,
    p_exclude_same_h=False,
    h_profile_boost=0.0,
):
    """
    4-level InfoNCE: Architecture / Topology / Superfamily / Profile.

    Extends ``hierarchical_infonce`` with a Profile (P) level that
    provides sub-superfamily discrimination using TED-LH S-group data.

    At the P level:
      - **Positives**: same-profile domains (binary positive).
      - **Soft positives**: cross-profile same-H pairs weighted by
        normalized HHblits probability from ``profile_sims``
        (only when ``p_exclude_same_h=False``).
      - **Hard negatives**: same-H, different-profile pairs with no
        HHblits match (treated as dissimilar, upweighted by
        ``hard_neg_weight``; only when ``p_exclude_same_h=False``).

    When ``p_exclude_same_h=True``, same-H different-P pairs are excluded
    from the P-level denominator entirely (neither positive nor negative).
    Only cross-SF pairs serve as P-level negatives.  This prevents the
    P-level loss from pushing apart domains within the same superfamily,
    which would hurt SF-level retrieval metrics.

    When ``h_profile_boost > 0``, same-profile pairs at the H level
    receive an additional positive weight boost, enriching the H-level
    gradient with sub-SF structure without creating adversarial negatives.

    Args:
        embeddings:     [N, D] L2-normalized embeddings.
        a/t/h/p_labels: [N] integer label tensors (-1 = invalid).
        temp_*:         per-level temperatures.
        weight_*:       per-level loss weights.
        mem_*:          optional memory bank tensors.
        hard_neg_weight: upweight factor for hard negatives.
        profile_sims:   dict {(min_p, max_p): float} from prepare_profile_data.
        soft_pos_scale:  scaling for cross-profile soft positives (0–1).
        p_exclude_same_h: if True, exclude same-H different-P from P-level
            denominator so only cross-SF pairs act as negatives.
        h_profile_boost: extra positive weight for same-profile pairs at
            H level (0 = disabled).  E.g. 1.0 means same-profile pairs
            get weight 2.0 while cross-profile same-H pairs get 1.0.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_a = torch.cat([a_labels, mem_a.to(device)])
        key_t = torch.cat([t_labels, mem_t.to(device)])
        key_h = torch.cat([h_labels, mem_h.to(device)])
        key_p = torch.cat([p_labels, mem_p.to(device)])
    else:
        keys = embeddings
        key_a, key_t, key_h, key_p = a_labels, t_labels, h_labels, p_labels
    K = keys.shape[0]

    sim = embeddings @ keys.T
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)
    not_self = ~self_mask

    results = {}
    total = zero

    # --- Levels A, T, H ---
    for name, ql, kl, temp, w, coarse_q, coarse_k in [
        ("a", a_labels, key_a, temp_a, weight_a, None, None),
        ("t", t_labels, key_t, temp_t, weight_t, a_labels, key_a),
        ("h", h_labels, key_h, temp_h, weight_h, t_labels, key_t),
    ]:
        pos, valid = _hierarchical_masks(
            name, a_labels, t_labels, h_labels,
            key_a, key_t, key_h, self_mask, benchmark_aligned=False)
        has_pos = pos.any(dim=1)
        if not has_pos.any():
            results[f"loss_{name}"] = zero
            results[f"acc_{name}"] = zero
            continue

        logits = (sim / temp).masked_fill(~valid, -1e9)
        exp = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        exp = exp.masked_fill(~valid, 0.0)

        if hard_neg_weight > 1.0 and coarse_q is not None:
            hard = ((coarse_q.unsqueeze(1) == coarse_k.unsqueeze(0))
                    & (ql.unsqueeze(1) != kl.unsqueeze(0)) & valid)
            exp = exp * torch.where(hard, hard_neg_weight, 1.0)

        # H-level profile boost: upweight same-profile positives
        if name == "h" and h_profile_boost > 0:
            p_ok_q = (p_labels >= 0).unsqueeze(1)
            p_ok_k = (key_p >= 0).unsqueeze(0)
            same_prof = (p_labels.unsqueeze(1) == key_p.unsqueeze(0)) & p_ok_q & p_ok_k
            # same-profile same-H pairs get weight (1 + boost);
            # cross-profile same-H pairs keep weight 1.0
            pos_weight_h = pos.float() + h_profile_boost * (same_prof & pos).float()
            pos_exp = (exp * pos_weight_h).sum(1)
        else:
            pos_exp = (exp * pos.float()).sum(1)

        loss_i = -torch.log(pos_exp / (exp.sum(1) + 1e-8) + 1e-8)
        level_loss = loss_i[has_pos].mean()
        top1 = sim.masked_fill(~valid, -1e9).argmax(dim=1)
        acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

        results[f"loss_{name}"] = level_loss
        results[f"acc_{name}"] = acc
        total = total + w * level_loss

    # --- Level P: Profile ---
    p_valid_q = p_labels >= 0
    p_valid_k = key_p >= 0
    both_valid = p_valid_q.unsqueeze(1) & p_valid_k.unsqueeze(0) & not_self

    if both_valid.any():
        same_p = (p_labels.unsqueeze(1) == key_p.unsqueeze(0)) & both_valid

        if p_exclude_same_h:
            # Exclude same-H different-P from P-level entirely:
            # only cross-SF pairs act as negatives, same-P as positives.
            # This prevents the P loss from pushing apart domains within
            # the same superfamily.
            same_h_mask = h_labels.unsqueeze(1) == key_h.unsqueeze(0)
            p_level_valid = both_valid & ~(same_h_mask & ~same_p)
            # No soft positives in this mode (cross-profile same-H is excluded)
            pos_weight = same_p.float()
        else:
            p_level_valid = both_valid
            # Soft positives from cross-profile similarity
            if profile_sims is not None and soft_pos_scale > 0:
                soft_pos = _build_soft_pos_matrix(
                    p_labels, key_p, h_labels, key_h, profile_sims, not_self)
                pos_weight = same_p.float() + soft_pos_scale * soft_pos
            else:
                pos_weight = same_p.float()

        has_pos_p = pos_weight.sum(dim=1) > 0
        if has_pos_p.any():
            logits_p = (sim / temp_p).masked_fill(~p_level_valid, -1e9)
            exp_p = torch.exp(logits_p - logits_p.max(dim=1, keepdim=True).values)
            exp_p = exp_p.masked_fill(~p_level_valid, 0.0)

            # Hard negatives only when NOT excluding same-H
            if hard_neg_weight > 1.0 and not p_exclude_same_h:
                same_h_diff_p = (
                    (h_labels.unsqueeze(1) == key_h.unsqueeze(0))
                    & (p_labels.unsqueeze(1) != key_p.unsqueeze(0))
                    & both_valid)
                if profile_sims is not None and soft_pos_scale > 0:
                    has_sim = soft_pos > 0
                    hard_p = same_h_diff_p & ~has_sim
                else:
                    hard_p = same_h_diff_p
                exp_p = exp_p * torch.where(hard_p, hard_neg_weight, 1.0)

            pos_exp_p = (exp_p * pos_weight).sum(1)
            loss_p_i = -torch.log(pos_exp_p / (exp_p.sum(1) + 1e-8) + 1e-8)
            p_loss = loss_p_i[has_pos_p].mean()

            top1_p = sim.masked_fill(~p_level_valid, -1e9).argmax(dim=1)
            acc_p = same_p.gather(1, top1_p.unsqueeze(1)).squeeze(1)[has_pos_p].float().mean()

            results["loss_p"] = p_loss
            results["acc_p"] = acc_p
            total = total + weight_p * p_loss
        else:
            results["loss_p"] = zero
            results["acc_p"] = zero
    else:
        results["loss_p"] = zero
        results["acc_p"] = zero

    results["loss"] = total
    return results


def hierarchical_smooth_ap(embeddings, a_labels, t_labels, h_labels,
                           temp=0.01, weight_a=0.15, weight_t=0.35,
                           weight_h=0.50, max_samples=256,
                           benchmark_aligned=False):
    """
    Differentiable Average Precision (SmoothAP) at each hierarchy level.

    Uses sigmoid approximation of the step function so gradients flow
    through the ranking operation.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if N > max_samples:
        idx = torch.randperm(N, device=device)[:max_samples]
        embeddings = embeddings[idx]
        a_labels, t_labels, h_labels = a_labels[idx], t_labels[idx], h_labels[idx]
        N = max_samples

    sim = embeddings @ embeddings.T
    eye = torch.eye(N, dtype=torch.bool, device=device)
    diff = sim.unsqueeze(1) - sim.unsqueeze(2)
    soft = torch.sigmoid(diff / temp)

    results = {}
    total = zero
    for name, labels, w in [("a", a_labels, weight_a), ("t", t_labels, weight_t),
                             ("h", h_labels, weight_h)]:
        if w == 0:
            results[f"sap_{name}"] = zero
            continue
        pos, valid = _hierarchical_masks(
            name, a_labels, t_labels, h_labels,
            a_labels, t_labels, h_labels, eye, benchmark_aligned)
        has = pos.any(dim=1)
        if not has.any():
            results[f"sap_{name}"] = zero
            continue
        candidate_distinct = ~eye.unsqueeze(0)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2) & candidate_distinct
        level_soft = soft * pair_valid.float()
        d_pos = 1.0 + (level_soft * pos.float().unsqueeze(1)).sum(2)
        d_all = 1.0 + level_soft.sum(2)
        prec = d_pos / (d_all + 1e-8)
        ap = (prec * pos.float()).sum(1) / pos.float().sum(1).clamp(min=1)
        level_loss = 1.0 - ap[has].mean()
        results[f"sap_{name}"] = level_loss
        total = total + w * level_loss

    results["sap_loss"] = total
    return results


def hierarchical_margin_ranking(anc_proj, pos_proj,
                                a_anc, t_anc, h_anc, a_pos, t_pos, h_pos,
                                probabilities, margin=1.0, min_neighbors=3,
                                level_weights=(0.2, 0.3, 0.5)):
    """
    Multi-level margin ranking loss enforcing cosine ordering proportional
    to HHblits probability differences within each hierarchy group.
    """
    pair_cos = (anc_proj * pos_proj).sum(-1)
    device = anc_proj.device
    zero = anc_proj.new_zeros(())

    shared_a = torch.where(a_anc == a_pos, a_anc, torch.tensor(-1, device=device))
    shared_t = torch.where(t_anc == t_pos, t_anc, torch.tensor(-1, device=device))
    shared_h = torch.where(h_anc == h_pos, h_anc, torch.tensor(-1, device=device))

    results = {}
    total = zero
    for name, shared, w in [("A", shared_a, level_weights[0]),
                             ("T", shared_t, level_weights[1]),
                             ("H", shared_h, level_weights[2])]:
        valid = shared != -1
        if not valid.any():
            results[f"rank_loss_{name}"] = zero
            continue

        vl, vc, vp = shared[valid], pair_cos[valid], probabilities[valid]
        sl, perm = torch.sort(vl)
        sc, sp = vc[perm], vp[perm]
        _, counts = torch.unique_consecutive(sl, return_counts=True)
        active = counts >= min_neighbors
        if not active.any():
            results[f"rank_loss_{name}"] = zero
            continue

        gids = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
        amask = active[gids]
        sc, sp = sc[amask], sp[amask]
        gids = (active.long().cumsum(0) - 1)[gids[amask]]
        c = counts[active]
        offsets = torch.cat([c.new_zeros(1), c.cumsum(0)[:-1]])
        pos_in = torch.arange(sc.shape[0], device=device) - offsets[gids]
        mg = int(c.max().item())

        g_cos = pair_cos.new_zeros(c.shape[0], mg)
        g_prob = probabilities.new_zeros(c.shape[0], mg)
        g_valid = torch.zeros(c.shape[0], mg, dtype=torch.bool, device=device)
        g_cos[gids, pos_in] = sc
        g_prob[gids, pos_in] = sp
        g_valid[gids, pos_in] = True

        pdiff = g_prob.unsqueeze(1) - g_prob.unsqueeze(2)
        cdiff = g_cos.unsqueeze(1) - g_cos.unsqueeze(2)
        pmask = g_valid.unsqueeze(1) & g_valid.unsqueeze(2) & (pdiff > 1e-6)
        if not pmask.any():
            results[f"rank_loss_{name}"] = zero
            continue

        ploss = F.relu(margin * pdiff - cdiff) * pmask.float()
        pc = pmask.sum(dim=(1, 2))
        act = pc > 0
        if not act.any():
            results[f"rank_loss_{name}"] = zero
            continue

        gl = ploss.sum(dim=(1, 2)) / pc.clamp(min=1)
        level_loss = gl[act].mean()
        results[f"rank_loss_{name}"] = level_loss
        total = total + w * level_loss

    results["rank_loss"] = total
    return results


# ---------------------------------------------------------------------------
# Selective coarse-preservation KL divergence
# ---------------------------------------------------------------------------

def selective_coarse_preservation_loss(
    student_emb,        # [N, D] L2-normalized, from student projector
    teacher_emb,        # [N, D] L2-normalized, from frozen teacher projector
    h_labels,           # [N] integer superfamily labels
    label_valid,        # [N] bool — True where the H-label is known/trusted
    seq_hashes=None,    # [N] int64 — hash of each domain sequence for dedup
    temperature=0.1,
):
    """KL(teacher || student) restricted to cross-superfamily pairs.

    Preserves the teacher's coarse-level (Architecture/Topology) retrieval
    structure by matching cross-superfamily similarity distributions.
    Within-superfamily pairs are excluded, leaving those relationships
    without direct KD supervision (though shared embeddings still couple
    both sets of relationships).

    Excludes:
      - Self-pairs (diagonal)
      - Same-superfamily pairs
      - Pairs where either endpoint has ``label_valid=False``
      - Pairs where both endpoints share the same sequence hash (dedup)

    Only anchors with at least 2 valid cross-SF candidates contribute.
    With 1 candidate both normalized distributions equal [1.0] and KL is
    identically zero — no preservation signal.

    Returns scalar loss (mean over valid anchors), or zero when no anchor
    qualifies.
    """
    N = student_emb.shape[0]
    device = student_emb.device

    if N < 3:
        return student_emb.new_zeros(())

    # --- Build validity mask [N, N] ---
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_sf = h_labels.unsqueeze(0) == h_labels.unsqueeze(1)

    # Both endpoints must have valid labels
    both_valid = label_valid.unsqueeze(0) & label_valid.unsqueeze(1)

    # Exclude duplicate sequences within the batch
    if seq_hashes is not None:
        same_seq = seq_hashes.unsqueeze(0) == seq_hashes.unsqueeze(1)
    else:
        same_seq = self_mask  # only exclude literal self

    valid = ~self_mask & ~same_sf & both_valid & ~same_seq

    # Require at least 2 valid candidates per anchor
    n_valid = valid.sum(dim=1)
    has_enough = n_valid >= 2
    if not has_enough.any():
        return student_emb.new_zeros(())

    # --- Teacher distribution (float32, no grad) ---
    with torch.no_grad():
        t_sim = (teacher_emb @ teacher_emb.T).float() / temperature
        t_sim.masked_fill_(~valid, float('-inf'))
        t_probs = F.softmax(t_sim, dim=-1)  # zeros where masked

    # --- Student distribution (float32) ---
    s_sim = (student_emb @ student_emb.T).float() / temperature
    s_sim.masked_fill_(~valid, float('-inf'))
    s_log_probs = F.log_softmax(s_sim, dim=-1)

    # --- Per-anchor KL ---
    # F.kl_div with reduction='none' computes p*(log(p) - log_q) per element.
    # Masked positions have t_probs=0, contributing 0*(-inf - ...) which is
    # handled correctly by softmax output (exact zeros) times the log term.
    # We sum over candidates (dim=-1) to get per-anchor KL.
    kl_elements = F.kl_div(s_log_probs, t_probs, reduction='none', log_target=False)
    # Zero out any NaN from 0*inf (defensive — should not occur with softmax zeros)
    kl_elements = torch.where(torch.isfinite(kl_elements), kl_elements, torch.zeros_like(kl_elements))
    kl_per_anchor = kl_elements.sum(dim=-1)

    # Average over anchors with enough candidates
    loss = kl_per_anchor[has_enough].mean()
    return loss


# ---------------------------------------------------------------------------
# Code-domain losses (single-level: algorithm class)
# ---------------------------------------------------------------------------

def code_infonce(embeddings, labels, temp=0.07,
                 mem_emb=None, mem_labels=None, hard_neg_weight=1.0):
    """
    Single-level InfoNCE for code clone detection.

    Positives share the same algorithm class label; all others are negatives.
    Returns loss, top-1 accuracy, and MAP@R.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_labels = torch.cat([labels, mem_labels.to(device)])
    else:
        keys, key_labels = embeddings, labels
    K = keys.shape[0]

    sim = embeddings @ keys.T
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)

    pos = (labels.unsqueeze(1) == key_labels.unsqueeze(0)) & ~self_mask
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return {"loss": zero, "acc": zero, "map_at_r": zero}

    logits = (sim / temp).masked_fill(self_mask, -1e9)
    exp = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    exp = exp.masked_fill(self_mask, 0.0)

    pos_exp = (exp * pos.float()).sum(1)
    loss_i = -torch.log(pos_exp / (exp.sum(1) + 1e-8) + 1e-8)
    loss = loss_i[has_pos].mean()

    # Top-1 clone accuracy
    sim_ns = sim.masked_fill(self_mask, -1e9)
    top1 = sim_ns.argmax(dim=1)
    acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

    # MAP@R — for each query, precision at rank R where R = #positives
    map_at_r = _batch_map_at_r(sim_ns[has_pos], pos[has_pos])

    return {"loss": loss, "acc": acc, "map_at_r": map_at_r}


def _batch_map_at_r(sim, pos_mask):
    """Compute MAP@R over a batch of queries.

    For each query, R = number of positives. We compute precision at each
    rank up to R and average — this is the standard MAP@R metric for
    code clone detection (POJ-104).
    """
    device = sim.device
    N, K = sim.shape
    R = pos_mask.float().sum(dim=1).long()  # (N,)

    # Sort similarities descending
    _, indices = sim.sort(dim=1, descending=True)
    sorted_pos = pos_mask.gather(1, indices).float()  # (N, K)

    # Cumulative hits at each rank
    cumhits = sorted_pos.cumsum(dim=1)  # (N, K)
    ranks = torch.arange(1, K + 1, device=device).unsqueeze(0).float()  # (1, K)
    precision_at_k = cumhits / ranks  # (N, K)

    # Mask: only count precision at positions where the item is a positive AND rank <= R
    r_mask = ranks <= R.unsqueeze(1).float()  # (N, K)
    prec_at_pos = precision_at_k * sorted_pos * r_mask

    ap = prec_at_pos.sum(1) / R.clamp(min=1).float()
    return ap.mean()


def code_smooth_ap(embeddings, labels, temp=0.01, max_samples=256):
    """Single-level differentiable Average Precision for code domain."""
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if N > max_samples:
        idx = torch.randperm(N, device=device)[:max_samples]
        embeddings = embeddings[idx]
        labels = labels[idx]
        N = max_samples

    sim = embeddings @ embeddings.T
    eye = torch.eye(N, dtype=torch.bool, device=device)
    not_self = (~eye).float().unsqueeze(1)
    not_j = (~eye).float().unsqueeze(0)
    diff = sim.unsqueeze(1) - sim.unsqueeze(2)
    soft = torch.sigmoid(diff / temp) * not_self * not_j

    pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
    has = pos.any(dim=1)
    if not has.any():
        return {"sap_loss": zero}

    d_pos = 1.0 + (soft * pos.float().unsqueeze(1)).sum(2)
    d_all = 1.0 + soft.sum(2)
    prec = d_pos / (d_all + 1e-8)
    ap = (prec * pos.float()).sum(1) / pos.float().sum(1).clamp(min=1)
    loss = 1.0 - ap[has].mean()

    return {"sap_loss": loss}
