"""Shared training utilities: checkpoint management, logging, LR scheduling.
"""
import json
import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    }
    try:
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except RuntimeError:
        info["gpus"] = ["unavailable (CUDA init failed)"]
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
            pretrain.log
            finetune.log
            metrics/
                pretrain.json
                finetune.json
            config.json
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
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_path = self.metrics_dir / f"{phase}.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self._train_epochs: List[Dict] = []
        self._val_epochs: List[Dict] = []
        self._val_steps: List[Dict] = []
        self._saved: List[Path] = sorted(self.phase_dir.glob("epoch_*.pth"))

        self._load_metrics()

    # ------------------------------------------------------------------
    # Split export: model-only + training-state
    # ------------------------------------------------------------------

    def _export_split(self, model: nn.Module, ckpt: Dict,
                      stem: str) -> Tuple[Path, Path]:
        """Write two companion files next to the main checkpoint.

        ``<stem>_model.pth``   — model weights + config (for inference)
        ``<stem>_state.pth``   — optimizer, scheduler, metadata (for resume)

        The main combined checkpoint is still written by the caller so that
        existing resume code keeps working unchanged.
        """
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_path = self.phase_dir / f"{stem}_model.pth"
        state_path = self.phase_dir / f"{stem}_state.pth"

        model_ckpt = {
            "model_state_dict": m.state_dict(),
            "config": getattr(m, "config", {}),
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
        }
        state_ckpt = {k: v for k, v in ckpt.items()
                      if k != "model_state_dict"}

        torch.save(model_ckpt, model_path)
        torch.save(state_ckpt, state_path)
        return model_path, state_path

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
             val_elapsed_s: float = 0.0,
             extra_state: Optional[Dict[str, Any]] = None) -> Path:
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
        if extra_state:
            ckpt.update(extra_state)

        # Save epoch file
        path = self.phase_dir / f"epoch_{epoch}.pth"
        torch.save(ckpt, path)
        self._saved.append(path)
        self._export_split(model, ckpt, f"epoch_{epoch}")

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

    def save_step(self, model: nn.Module, optimizer: Optimizer,
                  scheduler: Optional[LRScheduler] = None,
                  epoch: int = 0, global_step: int = 0,
                  keep_step_ckpts: int = 2,
                  batch_index: int = 0,
                  extra_state: Optional[Dict[str, Any]] = None) -> Path:
        """Save an intra-epoch checkpoint for fault tolerance.

        The checkpoint is marked with ``intra_epoch=True`` so the resume
        logic can restart the same epoch instead of advancing to the next.
        Only the last *keep_step_ckpts* step checkpoints are kept.
        """
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "batch_index": batch_index,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": getattr(m, "config", {}),
            "intra_epoch": True,
        }
        if scheduler:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state:
            ckpt.update(extra_state)

        path = self.phase_dir / f"step_{global_step}.pth"
        torch.save(ckpt, path)
        self._export_split(model, ckpt, f"step_{global_step}")
        self._symlink(self.phase_dir / "latest.pth", path.name)

        # Clean up old step checkpoints (combined + split)
        step_ckpts = sorted(
            self.phase_dir.glob("step_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        # Exclude _model.pth and _state.pth from the combined count
        combined = [p for p in step_ckpts
                    if not p.stem.endswith("_model")
                    and not p.stem.endswith("_state")]
        while len(combined) > keep_step_ckpts:
            old = combined.pop(0)
            step_num = old.stem  # e.g. "step_500"
            old.unlink(missing_ok=True)
            (self.phase_dir / f"{step_num}_model.pth").unlink(missing_ok=True)
            (self.phase_dir / f"{step_num}_state.pth").unlink(missing_ok=True)

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
        result = m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if result.missing_keys:
            _log.warning("Checkpoint missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            _log.warning("Checkpoint unexpected keys: %s", result.unexpected_keys)
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

    def log_step_metrics(self, epoch: int, global_step: int,
                         val_metrics: Dict, lr: float = 0.0) -> bool:
        """Record intra-epoch validation metrics. Returns True if it is a best.

        Long epochs may never complete inside a SLURM wall-time window, in
        which case :meth:`save` is never reached and no validation metric is
        ever persisted. This appends to a ``val.steps`` list so progress is
        still visible, and keeps ``best_val_loss`` up to date.

        The caller is expected to act on the return value by calling
        :meth:`save_best`; otherwise ``best_val_loss`` moves below what any
        epoch-end validation can beat and ``best.pth`` is never written.
        """
        _round = lambda v: round(v, 6) if isinstance(v, float) else v
        self._val_steps.append({
            "epoch": epoch,
            "global_step": global_step,
            "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lr": lr,
            **{k: _round(v) for k, v in val_metrics.items()},
        })

        is_best = False
        val_loss = val_metrics.get("loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            is_best = True

        self._save_metrics()
        return is_best

    def save_best(self, model: nn.Module, optimizer: Optimizer,
                  scheduler: Optional[LRScheduler] = None,
                  epoch: int = 0, global_step: int = 0,
                  **extra) -> Path:
        """Write ``best.pth`` directly (used for intra-epoch bests).

        Unlike :meth:`save`, this writes a real file rather than a symlink to
        an epoch checkpoint, so it survives step-checkpoint rotation.
        Any additional keyword arguments are stored in the checkpoint dict.
        """
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": getattr(m, "config", {}),
            "intra_epoch": True,
        }
        if scheduler:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        ckpt.update(extra)

        path = self.phase_dir / "best.pth"
        tmp = self.phase_dir / "best.pth.tmp"
        torch.save(ckpt, tmp)
        path.unlink(missing_ok=True)   # may be a symlink from save()
        tmp.rename(path)
        self._export_split(model, ckpt, "best")
        return path

    def _save_metrics(self):
        with open(self.metrics_path, "w") as f:
            json.dump({
                "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
                "best_epoch": self.best_epoch,
                "train": {"epochs": self._train_epochs},
                "val": {"epochs": self._val_epochs, "steps": self._val_steps},
            }, f, indent=2)

    def _load_metrics(self):
        if self.metrics_path.exists():
            try:
                data = json.loads(self.metrics_path.read_text())
                self._train_epochs = data.get("train", {}).get("epochs", [])
                self._val_epochs = data.get("val", {}).get("epochs", [])
                self._val_steps = data.get("val", {}).get("steps", [])
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
            # Clean up companion split files
            stem = old.stem  # e.g. "epoch_7"
            (self.phase_dir / f"{stem}_model.pth").unlink(missing_ok=True)
            (self.phase_dir / f"{stem}_state.pth").unlink(missing_ok=True)
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
        head_prefixes = ("aggregator", "projector", "log_temperature",
                         "profile_expansion_head", "global_pos_emb")

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
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger so library modules (e.g. zest.data.*) are visible
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Return a named logger for the caller
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
