"""
Loss functions and memory bank for contrastive fine-tuning.

Protein domain:
    - ``hierarchical_infonce`` — multi-level InfoNCE at A / T / H CATH levels
    - ``hierarchical_smooth_ap`` — differentiable AP at each hierarchy level
    - ``hierarchical_margin_ranking`` — cosine ordering loss per hierarchy group

Code domain:
    - ``code_infonce`` — single-level InfoNCE for algorithm class clone detection
    - ``code_smooth_ap`` — single-level differentiable AP
    - ``_batch_map_at_r`` — MAP@R metric (standard POJ-104 evaluation)

Shared:
    - ``MemoryBank`` — FIFO queue of detached embeddings for extra negatives
"""
import torch
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Protein-domain losses (3-level CATH hierarchy)
# ---------------------------------------------------------------------------

def hierarchical_infonce(embeddings, a_labels, t_labels, h_labels,
                         temp_a=0.10, temp_t=0.07, temp_h=0.05,
                         weight_a=0.15, weight_t=0.35, weight_h=0.50,
                         mem_emb=None, mem_a=None, mem_t=None, mem_h=None,
                         hard_neg_weight=1.0):
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
    sim_ns = sim.masked_fill(self_mask, -1e9)
    top1 = sim_ns.argmax(dim=1)

    for name, ql, kl, temp, w, coarse_q, coarse_k in [
        ("a", a_labels, key_a, temp_a, weight_a, None, None),
        ("t", t_labels, key_t, temp_t, weight_t, a_labels, key_a),
        ("h", h_labels, key_h, temp_h, weight_h, t_labels, key_t),
    ]:
        pos = (ql.unsqueeze(1) == kl.unsqueeze(0)) & ~self_mask
        has_pos = pos.any(dim=1)
        if not has_pos.any():
            results[f"loss_{name}"] = zero
            results[f"acc_{name}"] = zero
            continue

        logits = (sim / temp).masked_fill(self_mask, -1e9)
        exp = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        exp = exp.masked_fill(self_mask, 0.0)

        if hard_neg_weight > 1.0 and coarse_q is not None:
            hard = ((coarse_q.unsqueeze(1) == coarse_k.unsqueeze(0))
                    & (ql.unsqueeze(1) != kl.unsqueeze(0)) & ~self_mask)
            exp = exp * torch.where(hard, hard_neg_weight, 1.0)

        pos_exp = (exp * pos.float()).sum(1)
        loss_i = -torch.log(pos_exp / (exp.sum(1) + 1e-8) + 1e-8)
        level_loss = loss_i[has_pos].mean()
        acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

        results[f"loss_{name}"] = level_loss
        results[f"acc_{name}"] = acc
        total = total + w * level_loss

    results["loss"] = total
    return results


def hierarchical_smooth_ap(embeddings, a_labels, t_labels, h_labels,
                           temp=0.01, weight_a=0.15, weight_t=0.35,
                           weight_h=0.50, max_samples=256):
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
    not_self = (~eye).float().unsqueeze(1)
    not_j = (~eye).float().unsqueeze(0)
    diff = sim.unsqueeze(1) - sim.unsqueeze(2)
    soft = torch.sigmoid(diff / temp) * not_self * not_j

    results = {}
    total = zero
    for name, labels, w in [("a", a_labels, weight_a), ("t", t_labels, weight_t),
                             ("h", h_labels, weight_h)]:
        if w == 0:
            results[f"sap_{name}"] = zero
            continue
        pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        has = pos.any(dim=1)
        if not has.any():
            results[f"sap_{name}"] = zero
            continue
        d_pos = 1.0 + (soft * pos.float().unsqueeze(1)).sum(2)
        d_all = 1.0 + soft.sum(2)
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
