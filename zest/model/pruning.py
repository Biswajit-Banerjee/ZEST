"""
Layer pruning: extract a subset of transformer layers from a pretrained checkpoint
to create smaller models without retraining from scratch.

Strategies:
  - "uniform": evenly spaced layers (captures features at all depths)
  - "first":   first N layers (preserves low→mid level features)
  - "last":    last N layers (keeps high-level representations)

Usage:
    model, ckpt = prune_checkpoint("path/to/70M.pth", keep_layers=4, strategy="uniform")
    # → ~30M model with 4 layers, ready for hierarchical fine-tuning
"""
import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

from zest.model.trait_encoder import TraitEncoder

_log = logging.getLogger(__name__)


def select_layers(total: int, keep: int, strategy: str = "uniform") -> List[int]:
    """Return indices of layers to keep."""
    if keep >= total:
        return list(range(total))
    if keep <= 0:
        raise ValueError(f"keep_layers must be > 0, got {keep}")

    if strategy == "first":
        return list(range(keep))
    elif strategy == "last":
        return list(range(total - keep, total))
    elif strategy == "uniform":
        # Evenly spaced, always including first and last
        if keep == 1:
            return [total - 1]
        indices = [round(i * (total - 1) / (keep - 1)) for i in range(keep)]
        return sorted(set(indices))
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: uniform, first, last")


def prune_checkpoint(
    checkpoint_path: str,
    keep_layers: int,
    strategy: str = "uniform",
    device: str = "cpu",
    reinit_projector: bool = True,
    **config_overrides,
) -> tuple:
    """
    Load a checkpoint and create a pruned model with fewer transformer layers.

    The embedding table, RoPE, and final_norm are copied from the parent.
    Selected transformer layers are renumbered 0..k-1.
    The projector and downstream heads are re-initialized by default.

    Args:
        checkpoint_path: path to the pretrained checkpoint
        keep_layers: number of transformer layers to keep
        strategy: layer selection strategy ("uniform", "first", "last")
        device: device to load onto
        reinit_projector: if True, projector starts fresh (recommended for fine-tuning)
        **config_overrides: additional config overrides (e.g., num_proj_layers)

    Returns:
        (model, ckpt_dict) — the pruned model and original checkpoint data
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))
    sd = ckpt.get("model_state_dict", {})

    total_layers = config.get("num_layers", 8)
    selected = select_layers(total_layers, keep_layers, strategy)

    _log.info("Pruning %d → %d layers (strategy=%s): %s",
              total_layers, keep_layers, strategy, selected)

    # Build new config with fewer layers
    new_config = dict(config)
    new_config.pop("num_superfamilies", None)
    new_config.pop("cath_counts", None)
    new_config["num_layers"] = keep_layers
    new_config.update(config_overrides)

    # Build new state dict with renumbered layers
    new_sd = {}
    for key, val in sd.items():
        # Skip legacy/downstream heads
        if any(key.startswith(p) for p in ("superfamily_head.", "cath_head.")):
            continue

        # Remap encoder layer indices
        if key.startswith("core.layers."):
            parts = key.split(".", 3)  # core.layers.{idx}.{rest}
            old_idx = int(parts[2])
            if old_idx in selected:
                new_idx = selected.index(old_idx)
                new_key = f"core.layers.{new_idx}.{parts[3]}"
                new_sd[new_key] = val
        elif reinit_projector and any(key.startswith(p) for p in ("projector.", "sim_head.")):
            # Skip — will be randomly initialized
            continue
        else:
            new_sd[key] = val

    model = TraitEncoder(**new_config)

    # Handle size mismatches (e.g., global_pos_emb)
    model_sd = model.state_dict()
    for key in list(new_sd.keys()):
        if key in model_sd and new_sd[key].shape != model_sd[key].shape:
            c, m = new_sd[key], model_sd[key]
            if c.ndim != m.ndim:
                new_sd.pop(key)
                continue
            _log.info("Size mismatch %s: %s → %s, copying overlap",
                      key, list(c.shape), list(m.shape))
            slices = tuple(slice(0, min(a, b)) for a, b in zip(c.shape, m.shape))
            model_sd[key][slices] = c[slices]
            new_sd[key] = model_sd[key]

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    _log.info("Pruned model: %d layers, %.1fM params", keep_layers, total_params / 1e6)
    if missing:
        _log.info("Randomly initialized: %s",
                   list({k.split(".")[0] for k in missing}))
    if unexpected:
        _log.warning("Unexpected keys: %s", unexpected)

    return model, ckpt


def prune_to_sizes(
    checkpoint_path: str,
    layer_counts: List[int],
    strategy: str = "uniform",
    device: str = "cpu",
    **config_overrides,
) -> List[tuple]:
    """
    Create multiple pruned models from a single checkpoint.

    Returns list of (model, layer_count, param_count) tuples.
    """
    results = []
    for n in sorted(layer_counts):
        model, _ = prune_checkpoint(
            checkpoint_path, n, strategy=strategy,
            device=device, **config_overrides,
        )
        params = sum(p.numel() for p in model.parameters())
        results.append((model, n, params))
        _log.info("  %2d layers → %.1fM params", n, params / 1e6)
    return results
