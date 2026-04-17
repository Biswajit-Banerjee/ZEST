"""
Encoding helpers and train/eval step functions for contrastive fine-tuning.

Provides domain-dispatched forward passes:
    - ``_compute_losses_protein`` — 3-level CATH hierarchy (A / T / H)
    - ``_compute_losses_code`` — single-level algorithm class (clone detection)
    - ``_compute_losses`` — dispatcher based on ``args.domain``

Plus shared encoding utilities and the ``train_step`` / ``eval_step`` wrappers.
"""
import torch
import torch.nn.functional as F

from zest.training.losses import (
    MemoryBank,
    hierarchical_infonce,
    hierarchical_smooth_ap,
    hierarchical_margin_ranking,
    code_infonce,
    code_smooth_ap,
)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _get_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _encode_batch(model, tokens, mask):
    m = _get_model(model)
    embs = m.encode_tokens(tokens, mask)
    family = m.aggregator(embs, mask)
    proj = F.normalize(m.projector(family), p=2, dim=-1)
    return proj


def _encode_pair_batch(model, anc_tok, anc_mask, pos_tok, pos_mask):
    if anc_tok.shape[1] == pos_tok.shape[1]:
        tokens = torch.cat([anc_tok, pos_tok], dim=0)
        mask = torch.cat([anc_mask, pos_mask], dim=0)
        proj = _encode_batch(model, tokens, mask)
        split = anc_tok.shape[0]
        return proj[:split], proj[split:]
    return _encode_batch(model, anc_tok, anc_mask), _encode_batch(model, pos_tok, pos_mask)


def _encode_batch_with_embs(model, tokens, mask):
    """Encode and return both projections and token embeddings."""
    m = _get_model(model)
    embs = m.encode_tokens(tokens, mask)
    family = m.aggregator(embs, mask)
    proj = F.normalize(m.projector(family), p=2, dim=-1)
    return proj, embs


def _encode_pair_batch_with_embs(model, anc_tok, anc_mask, pos_tok, pos_mask):
    """Encode pair and return projections + token embeddings for both."""
    if anc_tok.shape[1] == pos_tok.shape[1]:
        tokens = torch.cat([anc_tok, pos_tok], dim=0)
        mask = torch.cat([anc_mask, pos_mask], dim=0)
        proj, embs = _encode_batch_with_embs(model, tokens, mask)
        B = anc_tok.shape[0]
        return proj[:B], proj[B:], embs[:B], embs[B:]
    proj_a, emb_a = _encode_batch_with_embs(model, anc_tok, anc_mask)
    proj_p, emb_p = _encode_batch_with_embs(model, pos_tok, pos_mask)
    return proj_a, proj_p, emb_a, emb_p


def _compute_residue_mlm(model, token_embs, batch, device, prefix):
    """Compute residue-level MLM loss from token embeddings."""
    m = _get_model(model)
    token_lengths = batch[f"{prefix}_token_lengths"].to(device, non_blocking=True)
    global_indices = batch[f"{prefix}_global_indices"].to(device, non_blocking=True)
    aa_labels = batch[f"{prefix}_aa_labels"].to(device, non_blocking=True)

    residue_logits = m.profile_expansion_head(
        token_embs, token_lengths, global_indices, m.global_pos_emb)

    logits_flat = residue_logits.reshape(-1, residue_logits.size(-1))
    labels_flat = aa_labels.reshape(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

    valid = labels_flat != -100
    if valid.sum() > 0:
        acc = (logits_flat[valid].argmax(-1) == labels_flat[valid]).float().mean()
    else:
        acc = torch.tensor(0.0, device=device)
    return loss, acc


# ---------------------------------------------------------------------------
# Protein domain forward pass
# ---------------------------------------------------------------------------

def _compute_losses_protein(model, batch, device, amp, args, memory_bank=None):
    """Protein domain forward pass — 3-level CATH hierarchy metrics."""
    anc_tok = batch["anchor_ids"].to(device, non_blocking=True)
    anc_mask = batch["anchor_mask"].to(device, non_blocking=True)
    pos_tok = batch["positive_ids"].to(device, non_blocking=True)
    pos_mask = batch["positive_mask"].to(device, non_blocking=True)
    a_anc = batch["a_labels"].to(device, non_blocking=True)
    t_anc = batch["t_labels"].to(device, non_blocking=True)
    h_anc = batch["h_labels"].to(device, non_blocking=True)
    a_pos = batch["a_pos_labels"].to(device, non_blocking=True)
    t_pos = batch["t_pos_labels"].to(device, non_blocking=True)
    h_pos = batch["h_pos_labels"].to(device, non_blocking=True)

    has_residue = "anchor_token_lengths" in batch
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if has_residue and getattr(args, "residue_mlm_weight", 0) > 0:
            anc_proj, pos_proj, anc_embs, pos_embs = _encode_pair_batch_with_embs(
                model, anc_tok, anc_mask, pos_tok, pos_mask)
        else:
            anc_proj, pos_proj = _encode_pair_batch(model, anc_tok, anc_mask, pos_tok, pos_mask)

        all_emb = torch.cat([anc_proj, pos_proj])
        all_a = torch.cat([a_anc, a_pos])
        all_t = torch.cat([t_anc, t_pos])
        all_h = torch.cat([h_anc, h_pos])

        # Filter to same-SF pairs for InfoNCE
        same_sf = h_anc == h_pos
        mem = memory_bank.sample(args.memory_bank_samples) if memory_bank else None

        # Scale CLI temperatures by learned temperature (preserves per-level ratios)
        m = _get_model(model)
        tau = m.temperature
        init_temp = 0.07  # must match log_temperature init
        scale = tau / init_temp
        temp_a = args.temp_a * scale
        temp_t = args.temp_t * scale
        temp_h = args.temp_h * scale

        if same_sf.sum() > 0:
            sf_anc = anc_proj[same_sf]
            sf_pos = pos_proj[same_sf]
            sf_a_anc, sf_t_anc, sf_h_anc = a_anc[same_sf], t_anc[same_sf], h_anc[same_sf]
            sf_a_pos, sf_t_pos, sf_h_pos = a_pos[same_sf], t_pos[same_sf], h_pos[same_sf]
            nce_emb = torch.cat([sf_anc, sf_pos])
            nce_a = torch.cat([sf_a_anc, sf_a_pos])
            nce_t = torch.cat([sf_t_anc, sf_t_pos])
            nce_h = torch.cat([sf_h_anc, sf_h_pos])

            infonce = hierarchical_infonce(
                nce_emb, nce_a, nce_t, nce_h,
                temp_a=temp_a, temp_t=temp_t, temp_h=temp_h,
                weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
                mem_emb=mem[0] if mem else None,
                mem_a=mem[1] if mem else None,
                mem_t=mem[2] if mem else None,
                mem_h=mem[3] if mem else None,
                hard_neg_weight=args.hard_neg_weight,
            )
        else:
            infonce = {"loss": torch.tensor(0.0, device=device)}

        # Cross-level InfoNCE (ALL pairs, T+A levels only)
        cross_nce = {"loss": torch.tensor(0.0, device=device)}
        if args.cross_level_weight > 0:
            ta_sum = args.weight_t + args.weight_a
            cw_t = args.weight_t / ta_sum if ta_sum > 0 else 0.5
            cw_a = args.weight_a / ta_sum if ta_sum > 0 else 0.5
            cross_nce = hierarchical_infonce(
                all_emb, all_a, all_t, all_h,
                temp_a=temp_a, temp_t=temp_t, temp_h=temp_h,
                weight_a=cw_a, weight_t=cw_t, weight_h=0.0,
            )

        # SmoothAP
        sap = {"sap_loss": torch.tensor(0.0, device=device)}
        if args.smooth_ap_weight > 0:
            sap = hierarchical_smooth_ap(
                all_emb, all_a, all_t, all_h,
                temp=args.smooth_ap_temp,
                max_samples=args.smooth_ap_max_samples,
                weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
            )

        # Margin ranking
        rank = {"rank_loss": torch.tensor(0.0, device=device)}
        if args.rank_weight > 0 and "probabilities" in batch:
            probs = batch["probabilities"].to(device, non_blocking=True)
            rank = hierarchical_margin_ranking(
                anc_proj, pos_proj, a_anc, t_anc, h_anc, a_pos, t_pos, h_pos,
                probs, margin=args.rank_margin,
                level_weights=tuple(args.rank_level_weights),
            )

        contrastive_loss = (args.contrastive_weight * infonce["loss"]
                            + args.cross_level_weight * cross_nce["loss"]
                            + args.smooth_ap_weight * sap["sap_loss"]
                            + args.rank_weight * rank["rank_loss"])

        # Residue MLM auxiliary loss
        res_loss = torch.tensor(0.0, device=device)
        res_acc = torch.tensor(0.0, device=device)
        if has_residue and getattr(args, "residue_mlm_weight", 0) > 0:
            anc_rl, anc_ra = _compute_residue_mlm(model, anc_embs, batch, device, "anchor")
            pos_rl, pos_ra = _compute_residue_mlm(model, pos_embs, batch, device, "positive")
            res_loss = (anc_rl + pos_rl) / 2
            res_acc = (anc_ra + pos_ra) / 2

            combine_loss = getattr(args, "combine_loss", None)
            if combine_loss is not None:
                loss, _ = combine_loss(
                    contrastive_loss, res_loss,
                    priority_token=args.contrastive_weight,
                    priority_residue=args.residue_mlm_weight,
                )
            else:
                loss = contrastive_loss + args.residue_mlm_weight * res_loss
        else:
            loss = contrastive_loss

    metrics = {
        "loss": loss,
        "con": infonce["loss"],
        "cross": cross_nce["loss"],
        "sap": sap["sap_loss"],
        "rank": rank["rank_loss"],
        "acc_h": infonce.get("acc_h", torch.tensor(0.0, device=device)),
        "tau": tau.detach(),
        "res_loss": res_loss,
        "res_acc": res_acc,
    }
    metrics = {k: v.detach() for k, v in metrics.items()}

    return loss, (all_emb, all_a, all_t, all_h), metrics


# ---------------------------------------------------------------------------
# Code domain forward pass
# ---------------------------------------------------------------------------

def _compute_losses_code(model, batch, device, amp, args, memory_bank=None):
    """Code domain forward pass — single-level algorithm class metrics."""
    anc_tok = batch["anchor_ids"].to(device, non_blocking=True)
    anc_mask = batch["anchor_mask"].to(device, non_blocking=True)
    pos_tok = batch["positive_ids"].to(device, non_blocking=True)
    pos_mask = batch["positive_mask"].to(device, non_blocking=True)
    # For code, all three levels carry the same algorithm label — use h_labels
    labels_anc = batch["h_labels"].to(device, non_blocking=True)
    labels_pos = batch["h_pos_labels"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        anc_proj, pos_proj = _encode_pair_batch(model, anc_tok, anc_mask, pos_tok, pos_mask)

        all_emb = torch.cat([anc_proj, pos_proj])
        all_labels = torch.cat([labels_anc, labels_pos])

        # Single-level InfoNCE with memory bank
        mem = memory_bank.sample(args.memory_bank_samples) if memory_bank else None
        m = _get_model(model)
        tau = m.temperature

        infonce = code_infonce(
            all_emb, all_labels, temp=tau,
            mem_emb=mem[0] if mem else None,
            mem_labels=mem[3] if mem else None,  # h_labels slot in memory bank
            hard_neg_weight=args.hard_neg_weight,
        )

        # Single-level SmoothAP
        sap = {"sap_loss": torch.tensor(0.0, device=device)}
        if args.smooth_ap_weight > 0:
            sap = code_smooth_ap(
                all_emb, all_labels,
                temp=args.smooth_ap_temp,
                max_samples=args.smooth_ap_max_samples,
            )

        loss = (args.contrastive_weight * infonce["loss"]
                + args.smooth_ap_weight * sap["sap_loss"])

    metrics = {
        "loss": loss,
        "con": infonce["loss"],
        "sap": sap["sap_loss"],
        "clone_acc": infonce.get("acc", torch.tensor(0.0, device=device)),
        "map_at_r": infonce.get("map_at_r", torch.tensor(0.0, device=device)),
        "tau": tau.detach(),
    }
    metrics = {k: v.detach() for k, v in metrics.items()}

    # Pack labels for memory bank (reuse a/t/h slots with same label)
    return loss, (all_emb, all_labels, all_labels, all_labels), metrics


# ---------------------------------------------------------------------------
# Domain dispatcher + train / eval wrappers
# ---------------------------------------------------------------------------

def _compute_losses(model, batch, device, amp, args, memory_bank=None):
    """Dispatch to domain-specific loss computation."""
    if getattr(args, "domain", "protein") == "code":
        return _compute_losses_code(model, batch, device, amp, args, memory_bank)
    return _compute_losses_protein(model, batch, device, amp, args, memory_bank)


def train_step(batch, model, device, amp, args, ga, memory_bank):
    loss, embs, metrics = _compute_losses(
        model, batch, device, amp, args, memory_bank,
    )
    (loss / ga).backward()

    if memory_bank:
        memory_bank.enqueue(*embs)

    return metrics


@torch.no_grad()
def eval_step(batch, model, device, amp, args):
    _, _, metrics = _compute_losses(model, batch, device, amp, args)
    return metrics


# Metric key sets per domain
METRIC_KEYS_PROTEIN = ("loss", "con", "cross", "sap", "rank", "acc_h", "tau", "res_loss", "res_acc")
METRIC_KEYS_CODE = ("loss", "con", "sap", "clone_acc", "map_at_r", "tau")


def get_metric_keys(domain: str):
    """Return the metric key tuple for the given domain."""
    if domain == "code":
        return METRIC_KEYS_CODE
    return METRIC_KEYS_PROTEIN
