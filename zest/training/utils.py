"""
Shared training utilities: checkpoint management, logging, LR scheduling.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    train_metrics: List[Dict] = field(default_factory=list)
    val_metrics: List[Dict] = field(default_factory=list)


class CheckpointManager:
    """
    Manages model checkpoints within a run directory::

        <run_dir>/
            models/
                <name>_epoch<N>.pth
                <name>_best.pth
                <name>_latest.pth
            <phase>.log
            <phase>_metrics.json
    """

    def __init__(self, save_dir: str, model_name: str = "model",
                 model_size: str = "base", phase: str = "mpm",
                 keep_last_n: int = 3, run_dir: Optional[str] = None):
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(save_dir) / f"{model_size}_{ts}"
        self.models_dir = self.run_dir / "models"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.phase = phase
        self.keep_last_n = keep_last_n
        self.log_path = self.run_dir / f"{phase}.log"
        self.metrics_path = self.run_dir / f"{phase}_metrics.json"
        self.state = TrainingState()
        self._saved: List[Path] = []

    def save(self, model: nn.Module, optimizer: Optimizer,
             scheduler: Optional[LRScheduler] = None,
             epoch: int = 0, global_step: int = 0,
             val_loss: Optional[float] = None) -> Path:
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.state.epoch = epoch
        self.state.global_step = global_step

        is_best = False
        if val_loss is not None and val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            self.state.best_epoch = epoch
            is_best = True

        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "global_step": global_step,
            "best_val_loss": self.state.best_val_loss,
            "best_epoch": self.state.best_epoch,
            "config": getattr(m, "config", {}),
        }
        if scheduler:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        path = self.models_dir / f"{self.model_name}_epoch{epoch}.pth"
        torch.save(ckpt, path)
        self._saved.append(path)

        torch.save(ckpt, self.models_dir / f"{self.model_name}_latest.pth")
        if is_best:
            torch.save(ckpt, self.models_dir / f"{self.model_name}_best.pth")

        self._cleanup()
        self._save_metrics()
        return path

    def load(self, model: nn.Module, optimizer: Optional[Optimizer] = None,
             scheduler: Optional[LRScheduler] = None,
             map_location=None) -> Dict:
        path = self.models_dir / f"{self.model_name}_latest.pth"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                pass
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.state.epoch = ckpt.get("epoch", 0)
        self.state.global_step = ckpt.get("global_step", 0)
        self.state.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.state.best_epoch = ckpt.get("best_epoch", 0)
        self._load_metrics()
        return ckpt

    def add_metrics(self, split: str, epoch: int, metrics: Dict):
        entry = {"epoch": epoch, **metrics}
        if split == "train":
            self.state.train_metrics.append(entry)
        elif split == "val":
            self.state.val_metrics.append(entry)

    def _save_metrics(self):
        with open(self.metrics_path, "w") as f:
            json.dump({
                "config": self.state.config,
                "best_val_loss": self.state.best_val_loss,
                "best_epoch": self.state.best_epoch,
                "train": self.state.train_metrics,
                "val": self.state.val_metrics,
            }, f, indent=2)

    def _load_metrics(self):
        if self.metrics_path.exists():
            try:
                data = json.loads(self.metrics_path.read_text())
                self.state.train_metrics = data.get("train", [])
                self.state.val_metrics = data.get("val", [])
            except (json.JSONDecodeError, KeyError):
                pass

    def _cleanup(self):
        epoch_ckpts = sorted(
            [p for p in self._saved if "epoch" in p.name and p.exists()],
            key=lambda p: p.stat().st_mtime,
        )
        while len(epoch_ckpts) > self.keep_last_n:
            old = epoch_ckpts.pop(0)
            old.unlink(missing_ok=True)
            if old in self._saved:
                self._saved.remove(old)


# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------

def build_cosine_schedule(optimizer: Optimizer, total_steps: int,
                          warmup_steps: int = 0,
                          min_lr_ratio: float = 0.01) -> LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def build_llrd_param_groups(model, base_lr: float, head_lr: float,
                            decay: float, head_prefixes=None):
    """
    Layer-wise learning rate decay: heads get head_lr, encoder layers get
    geometrically decaying LR from top to bottom.
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_layers = m.core.num_layers
    if head_prefixes is None:
        head_prefixes = ("profile_head", "token_mlm_proj", "mlm_head",
                         "global_pos_emb", "aggregator", "projector", "sim_head",
                         "temperature")

    head_params = [p for n, p in m.named_parameters()
                   if p.requires_grad and any(n.startswith(h) for h in head_prefixes)]
    norm_params = [p for p in m.core.final_norm.parameters() if p.requires_grad]

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if norm_params:
        groups.append({"params": norm_params, "lr": base_lr})

    for i in range(n_layers):
        depth = n_layers - 1 - i
        lr_i = base_lr * (decay ** (depth + 1))
        params = [p for p in m.core.layers[i].parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr_i})

    emb_lr = base_lr * (decay ** (n_layers + 1))
    emb_params = [p for n, p in m.core.named_parameters()
                  if p.requires_grad and not n.startswith("layers") and not n.startswith("final_norm")]
    if emb_params:
        groups.append({"params": emb_params, "lr": emb_lr})

    return groups


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str = "training", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
