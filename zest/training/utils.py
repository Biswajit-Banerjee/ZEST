"""Shared training utilities: checkpoint management, logging, LR scheduling.
"""
import json
import logging
import os
import platform
import subprocess
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
# Run directory helpers
# ---------------------------------------------------------------------------

def make_run_dir(output_dir: str, name: str) -> Path:
    """Create a timestamped run directory under *output_dir*."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_environment(run_dir: Union[str, Path]):
    """Snapshot Python / Torch / CUDA / GPU / git info to environment.json."""
    run_dir = Path(run_dir)
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "num_gpus": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    # Find the repo root relative to this source file
    _repo = str(Path(__file__).resolve().parent.parent.parent)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=_repo
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, cwd=_repo
        ).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=_repo
        ).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    with open(run_dir / "environment.json", "w") as f:
        json.dump(info, f, indent=2)


def save_model_arch(run_dir: Union[str, Path], model: nn.Module):
    """Save model architecture config to model_arch.json."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    config = getattr(m, "config", None)
    if config is None:
        return
    info = dict(config)
    info["total_params"] = sum(p.numel() for p in m.parameters())
    info["trainable_params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    with open(Path(run_dir) / "model_arch.json", "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages model checkpoints within a run directory::

        <run_dir>/
            pretrain/
                epoch_1.pth
                epoch_2.pth
                latest.pth  -> epoch_2.pth  (symlink)
                best.pth    -> epoch_1.pth  (symlink)
            finetune/
                ...
            config.json
            pretrain.log
            finetune.log
            pretrain_metrics.json
            finetune_metrics.json
            model_arch.json
            environment.json
    """

    def __init__(self, run_dir: Union[str, Path], phase: str = "pretrain",
                 keep_last_n: int = 3):
        self.run_dir = Path(run_dir)
        self.phase = phase
        self.phase_dir = self.run_dir / phase
        self.keep_last_n = keep_last_n
        self.log_path = self.run_dir / f"{phase}.log"
        self.metrics_path = self.run_dir / f"{phase}_metrics.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.phase_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self._train_epochs: List[Dict] = []
        self._val_epochs: List[Dict] = []
        self._saved: List[Path] = sorted(self.phase_dir.glob("epoch_*.pth"))

        self._load_metrics()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, model: nn.Module, optimizer: Optimizer,
             scheduler: Optional[LRScheduler] = None,
             epoch: int = 0, global_step: int = 0,
             train_metrics: Optional[Dict] = None,
             val_metrics: Optional[Dict] = None,
             lr: float = 0.0,
             train_elapsed_s: float = 0.0,
             val_elapsed_s: float = 0.0) -> Path:
        m = model._orig_mod if hasattr(model, "_orig_mod") else model

        # Best tracking
        is_best = False
        val_loss = val_metrics.get("loss") if val_metrics else None
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            is_best = True

        # Checkpoint dict
        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": getattr(m, "config", {}),
        }
        if scheduler:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        # Save epoch file
        path = self.phase_dir / f"epoch_{epoch}.pth"
        torch.save(ckpt, path)
        self._saved.append(path)

        # Symlink latest
        self._symlink(self.phase_dir / "latest.pth", path.name)

        # Symlink best
        if is_best:
            self._symlink(self.phase_dir / "best.pth", path.name)

        # Append train metrics (every epoch)
        _round = lambda v: round(v, 6) if isinstance(v, float) else v
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if train_metrics:
            self._train_epochs.append({
                "epoch": epoch,
                "wall_time": now,
                "elapsed_s": round(train_elapsed_s, 1),
                "lr": lr,
                **{k: _round(v) for k, v in train_metrics.items()},
            })

        # Append val metrics (only when validation is run)
        if val_metrics:
            self._val_epochs.append({
                "epoch": epoch,
                "wall_time": now,
                "elapsed_s": round(val_elapsed_s, 1),
                **{k: _round(v) for k, v in val_metrics.items()},
            })

        self._save_metrics()
        self._cleanup()
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path], model: nn.Module,
             optimizer: Optional[Optimizer] = None,
             scheduler: Optional[LRScheduler] = None,
             map_location=None) -> Dict:
        """Load a checkpoint and restore model / optimizer / scheduler state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        m.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                pass
        if scheduler and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError):
                pass
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        return ckpt

    # ------------------------------------------------------------------
    # Metrics persistence
    # ------------------------------------------------------------------

    def _save_metrics(self):
        with open(self.metrics_path, "w") as f:
            json.dump({
                "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
                "best_epoch": self.best_epoch,
                "train": {"epochs": self._train_epochs},
                "val": {"epochs": self._val_epochs},
            }, f, indent=2)

    def _load_metrics(self):
        if self.metrics_path.exists():
            try:
                data = json.loads(self.metrics_path.read_text())
                self._train_epochs = data.get("train", {}).get("epochs", [])
                self._val_epochs = data.get("val", {}).get("epochs", [])
                bvl = data.get("best_val_loss")
                self.best_val_loss = bvl if bvl is not None else float("inf")
                self.best_epoch = data.get("best_epoch", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symlink(link: Path, target_name: str):
        """Create or replace a relative symlink."""
        link.unlink(missing_ok=True)
        link.symlink_to(target_name)

    def _cleanup(self):
        """Keep only *keep_last_n* epoch checkpoints, never delete best."""
        epoch_ckpts = sorted(
            [p for p in self._saved if p.name.startswith("epoch_") and p.exists()],
            key=lambda p: p.stat().st_mtime,
        )
        # Find which epoch the best symlink points to
        best_link = self.phase_dir / "best.pth"
        best_target = None
        if best_link.is_symlink():
            best_target = (self.phase_dir / os.readlink(best_link)).name

        while len(epoch_ckpts) > self.keep_last_n:
            old = epoch_ckpts.pop(0)
            if best_target and old.name == best_target:
                continue
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
        head_prefixes = ("aggregator", "projector", "temperature")

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
