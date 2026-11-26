"""Token-level MLM pretraining for ThemeEncoder."""
import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from zest.model.theme_encoder import create_theme_encoder, ThemeEncoder
from zest.tokenizer import ZESTTokenizer
from zest.data.pretrain import MLMDataset, mlm_collate_fn
from zest.training.losses import UncertaintyWeightedLoss
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, setup_logging,
    make_run_dir, save_environment, save_model_arch,
)

GRAD_CLIP = 1.0


def _setup_ddp():
    """Read torchrun's rank variables and initialize NCCL once."""
    ddp_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    present = [key for key in ddp_keys if key in os.environ]
    if not present:
        return 0, 1, 0, False
    if len(present) != len(ddp_keys):
        raise RuntimeError(
            "Incomplete distributed environment; launch with torchrun so "
            "RANK, WORLD_SIZE, and LOCAL_RANK are set together"
        )
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world < 1 or rank < 0 or rank >= world or local_rank < 0:
        raise RuntimeError(
            f"Invalid distributed environment: RANK={rank}, "
            f"WORLD_SIZE={world}, LOCAL_RANK={local_rank}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world, local_rank, True


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _global_count(count, device, is_ddp):
    count = count.to(device=device, dtype=torch.float32)
    if is_ddp:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count.clamp_min(1.0)


def _sync_grads(module, world_size):
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _reduce_metric_sums(sums, n_steps, device, is_ddp):
    """Average accumulated scalar metrics across DDP ranks."""
    if not is_ddp:
        return sums, n_steps
    values = torch.tensor(
        [sums[key] for key in sums] + [float(n_steps)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    reduced_sums = {
        key: values[index].item()
        for index, key in enumerate(sums)
    }
    return reduced_sums, int(round(values[-1].item()))


class EqualizedSampler(Sampler):
    """Shuffle local rank data and pad it to equal length on every rank."""

    def __init__(self, dataset_size, target_size, seed):
        self.dataset_size = dataset_size
        self.target_size = target_size
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        if indices and len(indices) < self.target_size:
            indices.extend(indices[i % len(indices)]
                           for i in range(self.target_size - len(indices)))
        return iter(indices[:self.target_size])

    def __len__(self):
        return self.target_size


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _collect_rng_states(rank, world_size, is_ddp):
    state = _rng_state()
    if not is_ddp:
        return [state]
    states = [None] * world_size if rank == 0 else None
    dist.gather_object(state, states, dst=0)
    return states


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8).cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [torch.as_tensor(cuda_state, dtype=torch.uint8).cpu()
                       for cuda_state in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def train_step(batch, model, device, amp, ga,
               combine_loss=None, token_mlm_weight=0.7, residue_mlm_weight=0.3,
               label_smoothing=0.0,
               token_only=False, residue_only=False, is_ddp=False,
               world_size=1):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = None if residue_only else batch["token_labels"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        tl = batch["token_lengths"].to(device) if "token_lengths" in batch else None
        gi = batch["global_indices"].to(device) if "global_indices" in batch else None
        if residue_only:
            logits = None
            loss_token = None
            _, res_logits = model(ids, mask, token_lengths=tl, global_indices=gi)
        elif tl is not None and gi is not None:
            logits, res_logits = model(ids, mask, token_lengths=tl,
                                       global_indices=gi)
        else:
            # Token-level MLM
            logits = model(ids, mask)
            res_logits = None

        if not residue_only:
            loss_token_sum = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                             labels.view(-1), ignore_index=-100,
                                             reduction="sum",
                                             label_smoothing=label_smoothing)
            loss_token = loss_token_sum * world_size / _global_count(
                (labels != -100).sum(), device, is_ddp)

        if residue_only or (not token_only and "token_lengths" in batch):
            # Residue-level MLM
            aa = batch["aa_labels"].to(device)
            ml = min(res_logits.shape[1], aa.shape[1])
            loss_residue_sum = F.cross_entropy(
                res_logits[:, :ml, :].reshape(-1, 20),
                aa[:, :ml].reshape(-1), ignore_index=-100, reduction="sum")
            loss_residue = loss_residue_sum * world_size / _global_count(
                (aa[:, :ml] != -100).sum(), device, is_ddp)

            if residue_only:
                # Bypass UncertaintyWeightedLoss — use raw residue loss only
                scaled = loss_residue / ga
                loss_stats = {"w_tok": 0.0, "w_res": 1.0,
                              "pct_tok": 0.0, "pct_res": 1.0}
            else:
                balanced, loss_stats = combine_loss(
                    loss_token, loss_residue,
                    priority_token=token_mlm_weight,
                    priority_residue=residue_mlm_weight)
                scaled = balanced / ga
        else:
            scaled = loss_token / ga

        scaled.backward()

    if token_only or "token_lengths" not in batch:
        loss_val = loss_token.detach().item()
    elif residue_only:
        loss_val = loss_residue.detach().item()
    else:
        loss_val = balanced.detach().item()

    metrics = {"loss": loss_val}
    if not residue_only:
        with torch.no_grad():
            valid = labels != -100
            tok_acc = torch.tensor(0.0, device=device)
            if valid.any():
                tok_acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()
        metrics.update({
            "tok_loss": loss_token.detach().item(),
            "tok_acc": tok_acc.item(),
        })

    if not token_only and "token_lengths" in batch:
        with torch.no_grad():
            rp = res_logits[:, :ml, :].argmax(-1)
            rv = aa[:, :ml] != -100
            res_acc = (rp[rv] == aa[:, :ml][rv]).float().mean() if rv.any() else torch.tensor(0.0, device=device)
        metrics.update({
            "res_loss": loss_residue.detach().item(),
            "res_acc": res_acc.item(),
            "w_tok": loss_stats["w_tok"],
            "w_res": loss_stats["w_res"],
            "pct_tok": loss_stats["pct_tok"],
            "pct_res": loss_stats["pct_res"],
        })

    return metrics


@torch.no_grad()
def eval_step(batch, model, device, amp, eval_residue=True, eval_token=True):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    metrics = {}

    if eval_token:
        labels = batch["token_labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(ids, mask)
            loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                         labels.view(-1), ignore_index=-100)

        valid = labels != -100
        tok_acc = torch.tensor(0.0, device=device)
        if valid.any():
            tok_acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()
        metrics.update({"tok_loss": loss_token.item(), "tok_acc": tok_acc.item()})

    if eval_residue and "token_lengths" in batch:
        tl = batch["token_lengths"].to(device)
        gi = batch["global_indices"].to(device)
        aa = batch["aa_labels"].to(device)
        m = getattr(model, "module", model)   # unwrap DDP
        m = getattr(m, "_orig_mod", m)         # unwrap torch.compile
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            res_logits = m.forward_residue_mlm(ids, mask, tl, gi)
        ml = min(res_logits.shape[1], aa.shape[1])
        loss_residue = F.cross_entropy(
            res_logits[:, :ml, :].reshape(-1, 20),
            aa[:, :ml].reshape(-1), ignore_index=-100)
        rp = res_logits[:, :ml, :].argmax(-1)
        rv = aa[:, :ml] != -100
        res_acc = (rp[rv] == aa[:, :ml][rv]).float().mean() if rv.any() else torch.tensor(0.0, device=device)
        metrics.update({"res_loss": loss_residue.item(), "res_acc": res_acc.item()})

    return metrics


@torch.no_grad()
def run_validation(model, val_loader, device, amp, eval_residue=True,
                   eval_token=True, max_batches=None):
    """Run validation and return mean metrics with perplexities.

    ``max_batches`` bounds the cost so this can be called mid-epoch. Returns
    ``None`` if no batch could be evaluated.
    """
    was_training = model.training
    model.eval()
    val_list = []
    for vi, batch in enumerate(val_loader):
        if max_batches and vi >= max_batches:
            break
        try:
            val_list.append(
                eval_step(batch, model, device, amp, eval_residue=eval_residue,
                          eval_token=eval_token))
        except RuntimeError as e:
            if _is_oom(e):
                torch.cuda.empty_cache()
                continue
            raise
    if was_training:
        model.train()

    if not val_list:
        return None

    v = {k: float(np.mean([m[k] for m in val_list])) for k in val_list[0]}
    if "tok_loss" in v:
        v["tok_ppl"] = math.exp(min(v["tok_loss"], 20.0))
    if "res_loss" in v:
        v["res_ppl"] = math.exp(min(v["res_loss"], 20.0))
    # Compat: checkpoint best-tracking keys off "loss"
    v["loss"] = v["tok_loss"] if eval_token else v["res_loss"]
    return v


def build_parser():
    p = argparse.ArgumentParser(description="ThemeEncoder MLM pretraining")
    p.add_argument("--model",            default="base",  choices=["test", "base", "deep"],  help="Model architecture")
    p.add_argument("--epochs",           default=8,       type=int,    help="Number of training epochs")
    p.add_argument("--schedule-epochs",  default=None,   type=int,
                   help="Total epochs used to calculate the LR schedule; defaults to --epochs")
    p.add_argument("--batch-size",       default=128,     type=int,    help="Batch size per GPU")
    p.add_argument("--grad-accum-steps", default=8,       type=int,    help="Gradient accumulation steps") # 128 * 8 = 1024
    p.add_argument("--lr",               default=5e-4,    type=float,  help="Learning rate")
    p.add_argument("--warmup-ratio",     default=0.05,    type=float,  help="Warmup ratio")
    p.add_argument("--mask-prob",        default=0.3,     type=float,  help="Masking probability") # 30% masking
    p.add_argument("--trie-dropout",     default=0.15,    type=float,  help="Trie level dropout")  # 15% trie level dropout
    p.add_argument("--dropout",           default=None,   type=float,  help="Model dropout (defaults to the model preset)")
    p.add_argument("--embedding-dropout", default=0.1, type=float, help="Input embedding dropout")
    p.add_argument("--token-head-dropout", default=0.25, type=float, help="Dropout before tied token logits")
    p.add_argument("--residue-dropout",   default=0.1, type=float, help="Residue prediction head dropout")
    p.add_argument("--label-smoothing",   default=0.1, type=float, help="Token MLM label smoothing")
    p.add_argument("--weight-decay",      default=0.02, type=float, help="AdamW decay excluding bias and norm parameters")
    p.add_argument("--max-tokens",       default=512,     type=int,    help="Maximum sequence length")
    p.add_argument("--num-workers",      default=8,       type=int,    help="Number of data loading workers")
    p.add_argument("--seed",             default=42,      type=int,    help="Random seed for reproducibility")
    p.add_argument("--val-every",        default=1,       type=int,    help="Run validation every N epochs")
    p.add_argument("--val-shards",       default=None, type=int,       help="Limit validation to first K shards (longest seqs)")
    p.add_argument("--tokenizer",        default="data/tokens/tedlh_zest_32K.json",     help="Tokenizer file path")
    p.add_argument("--train-data",       required=True,                help="Training sequences (FASTA dir/file, parquet, or txt)")
    p.add_argument("--val-data",         required=True,                help="Validation sequences (FASTA dir/file, parquet, or txt)")
    p.add_argument("--shards-per-epoch", default=None, type=int,       help="Parquet shards to load per epoch (streaming mode)")
    p.add_argument("--unique-shards",    action="store_true",          help="Each shard seen exactly once (no repeat across epochs)")
    p.add_argument("--output-dir",       default="outputs/models",     help="Output directory for checkpoints")
    p.add_argument("--run-dir",          default=None,                 help="Run directory (inferred from --checkpoint if given)")
    p.add_argument("--checkpoint",       default=None,                 help="Resume from checkpoint (infers run-dir)")
    p.add_argument("--amp",              action="store_true",          help="Use automatic mixed precision")
    p.add_argument("--keep-all-checkpoints", action="store_true",      help="Keep all checkpoints (default: last 3)")
    p.add_argument("--seq-column",       default="sequence",           help="Column name for sequences in parquet")
    p.add_argument("--tokenizer-type",   default="zest", choices=["zest", "bpe"], help="Tokenizer type (zest=trie, bpe=subword baseline)")
    p.add_argument("--token-mlm-weight",  default=0.7,  type=float, help="Priority weight for token-level MLM")
    p.add_argument("--residue-mlm-weight", default=0.3, type=float, help="Priority weight for residue-level MLM (0 disables)")
    p.add_argument("--token-only",        action="store_true",       help="Token MLM only; skip residue head")
    p.add_argument("--freeze-core",       action="store_true",       help="Freeze encoder core; only train residue head + global_pos_emb")
    p.add_argument("--reinit-head",       action="store_true",
                   help="Reinitialise the RecycledResidueDecoder head when loading --checkpoint "
                        "(encoder weights are preserved)")
    p.add_argument("--span-mask",         action="store_true",       help="Mask contiguous runs of tokens instead of independent draws (span infilling)")
    p.add_argument("--min-span",          default=3, type=int,       help="Minimum contiguous token run when --span-mask is set")
    p.add_argument("--max-span",          default=8, type=int,       help="Maximum contiguous token run when --span-mask is set")
    p.add_argument("--optim-checkpoint",  default=None,              help="Load optimizer/scheduler from this checkpoint (model from --checkpoint)")
    p.add_argument("--save-every-steps",  default=0, type=int,       help="Save intra-epoch checkpoint every N optimizer steps (0=epoch-only)")
    p.add_argument("--val-every-steps",   default=0, type=int,       help="Run validation every N optimizer steps (0=epoch-only). Use when an epoch is longer than the SLURM wall-time")
    p.add_argument("--val-max-batches",   default=50, type=int,      help="Max validation batches for intra-epoch validation")

    return p


def parse_args():
    return build_parser().parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def main():
    args = parse_args()

    # DDP setup (no-op when launched with plain `python`)
    rank, world_size, local_rank, is_ddp = _setup_ddp()
    is_main = (rank == 0)

    seed_everything(args.seed + rank)  # per-rank seed for data diversity

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp = args.amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            amp = True

    # Determine run directory (explicit --run-dir takes priority)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.checkpoint:
        run_dir = Path(args.checkpoint).parent.parent
    else:
        run_dir = make_run_dir(args.output_dir, args.model)

    keep_n = 999 if args.keep_all_checkpoints else 3
    ckpt_mgr = CheckpointManager(run_dir, phase="pretrain", keep_last_n=keep_n)
    log = setup_logging("pretrain", str(ckpt_mgr.log_path) if is_main else None)

    if args.tokenizer_type == "bpe":
        from zest.tokenizer_bpe import ProteinBPETokenizer
        tokenizer = ProteinBPETokenizer(args.tokenizer)
        log.info("Using BPE tokenizer: vocab_size=%d", tokenizer.vocab_size)
    else:
        tokenizer = ZESTTokenizer(args.tokenizer)

    encoder = None
    lang_column = None

    residue_mlm = (not args.token_only) and args.residue_mlm_weight > 0

    # Model first: the residue budget of the datasets must match the number of
    # rows in the model's global_pos_emb, which comes from the checkpoint
    # config when resuming (and may differ from --max-tokens).
    loaded_checkpoint = None
    head_swapped = False
    if args.checkpoint:
        model, loaded_checkpoint = ThemeEncoder.load_pretrained(
            args.checkpoint, device="cpu", reinit_head=args.reinit_head,
            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
            token_head_dropout=args.token_head_dropout,
            residue_dropout=args.residue_dropout)
        head_swapped = args.reinit_head
        model = model.to(device)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={model.vocab_size} does not match "
                f"tokenizer vocab_size={tokenizer.vocab_size}. Wrong "
                f"--tokenizer / --tokenizer-type for this checkpoint?")
    else:
        model_kwargs = {
            "embedding_dropout": args.embedding_dropout,
            "token_head_dropout": args.token_head_dropout,
            "residue_dropout": args.residue_dropout,
        }
        if args.dropout is not None:
            model_kwargs["dropout"] = args.dropout
        model = create_theme_encoder(
            args.model, tokenizer.vocab_size, args.max_tokens, **model_kwargs
        ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.1fM params), residue head: %s", args.model,
             total_params / 1e6, type(model.profile_expansion_head).__name__)

    # A reinitialised head starts from noise, so its loss is meaningless unless
    # it is actually being trained. Catching this here avoids burning a GPU
    # reservation on a run that could not have improved anything.
    if args.reinit_head and args.checkpoint:
        if args.token_only or args.residue_mlm_weight <= 0:
            raise ValueError(
                "--reinit-head reinitialises the residue head, but the residue "
                "objective is disabled (--token-only / --residue-mlm-weight=0). "
                "The new head would never receive a gradient.")
        log.warning("RecycledResidueDecoder head reinitialised: encoder weights are "
                    "preserved, head weights are random.")

    max_residues = model.max_residues
    if residue_mlm:
        log.info("Residue MLM: max_tokens=%d, residue budget=%d "
                 "(global_pos_emb rows)", args.max_tokens, max_residues)

    train_ds = MLMDataset(
        args.train_data, tokenizer,
        is_training=True,
        max_tokens=args.max_tokens, trie_dropout=args.trie_dropout,
        shards_per_epoch=args.shards_per_epoch,
        unique_shards=args.unique_shards, seed=args.seed,
        rank=rank, world_size=world_size,
        seq_column=args.seq_column,
        lang_column=lang_column,
        encoder=encoder,
        residue_mlm=residue_mlm,
        max_residues=max_residues,
    )
    val_ds = MLMDataset(
        args.val_data, tokenizer,
        is_training=False,
        max_tokens=args.max_tokens,
        seq_column=args.seq_column,
        lang_column=lang_column,
        encoder=encoder,
        residue_mlm=residue_mlm,
        shards_per_epoch=args.val_shards,
        rank=rank, world_size=world_size,
        max_residues=max_residues,
    )

    collate = lambda b: mlm_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size,
        mask_prob=args.mask_prob,
        span_mask=args.span_mask, min_span=args.min_span, max_span=args.max_span,
    )

    # Freeze core if requested (only residue head + global_pos_emb train)
    if args.freeze_core:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.profile_expansion_head.parameters():
            p.requires_grad = True
        model.global_pos_emb.weight.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Core frozen — training %.1fM / %.1fM params (residue head + pos_emb)",
                 trainable / 1e6, total_params / 1e6)

    # Wrap in DDP (keep raw reference for checkpoint saving)
    raw_model = model
    if is_ddp:
        model = DDP(raw_model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=True)
        if is_main:
            log.info("DDP: %d GPUs, rank %d", world_size, rank)

    bs, ga = args.batch_size, args.grad_accum_steps
    nw = args.num_workers
    pf = 8 if nw > 0 else None

    def make_loader(ds, shuffle=True, drop_last=True, persistent=True, epoch=0,
                    distributed=False):
        pw = nw > 0 and persistent and not ds.streaming
        sampler = None
        if distributed:
            local_size = torch.tensor(len(ds), device=device, dtype=torch.long)
            dist.all_reduce(local_size, op=dist.ReduceOp.MAX)
            target_size = int(local_size.item())
            target_size = ((int(local_size.item()) + bs - 1) // bs) * bs
            sampler = EqualizedSampler(len(ds), target_size, args.seed + epoch)
            shuffle = False  # sampler handles shuffling
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        return DataLoader(
            ds, bs, shuffle=shuffle, num_workers=nw,
            pin_memory=True, collate_fn=collate, drop_last=drop_last,
            prefetch_factor=pf, persistent_workers=pw,
            sampler=sampler, generator=generator,
        )

    # Loss + optimizers
    token_only = args.token_only or args.residue_mlm_weight <= 0
    residue_only = args.freeze_core  # bypass UncertaintyWeightedLoss when core frozen
    combine_loss = None
    opt_loss = None
    if not token_only and not residue_only:
        combine_loss = UncertaintyWeightedLoss(num_tasks=2).to(device)
        opt_loss = optim.AdamW(combine_loss.parameters(), lr=1e-3, weight_decay=0.0)

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(train_params, lr=args.lr)

    loader = make_loader(train_ds, epoch=0, distributed=is_ddp)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)

    # The final partial accumulation is flushed after the loader, so count it
    # in the schedule and in the global-step offset used when resuming.
    steps_ep = max(1, math.ceil(len(loader) / ga))
    schedule_epochs = args.schedule_epochs or args.epochs
    if schedule_epochs < args.epochs:
        raise ValueError("--schedule-epochs must be >= --epochs")
    sched = build_cosine_schedule(
        opt, schedule_epochs * steps_ep,
        int(steps_ep * args.warmup_ratio)
    )

    # Save run metadata (rank 0 only)
    if is_main:
        config = {
            "model": args.model,
            "params_M": round(total_params / 1e6, 2),
            "epochs": args.epochs,
            "batch_size": bs,
            "grad_accum_steps": ga,
            "effective_batch": bs * ga * world_size,
            "world_size": world_size,
            "ddp": is_ddp,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": "cosine",
            "schedule_epochs": schedule_epochs,
            "total_steps": schedule_epochs * steps_ep,
            "warmup_steps": int(steps_ep * args.warmup_ratio),
            "steps_per_epoch": steps_ep,
            "mask_prob": args.mask_prob,
            "span_mask": args.span_mask,
            "min_span": args.min_span,
            "max_span": args.max_span,
            "trie_dropout": args.trie_dropout,
            "max_tokens": args.max_tokens,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "amp": amp,
            "shards_per_epoch": args.shards_per_epoch,
            "streaming": train_ds.streaming,
            "total_train_shards": len(train_ds._shard_paths) if train_ds.streaming else None,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "token_only": token_only,
            "token_mlm_weight": args.token_mlm_weight,
            "residue_mlm_weight": args.residue_mlm_weight,
        }
        with open(ckpt_mgr.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        save_environment(ckpt_mgr.run_dir)
        save_model_arch(ckpt_mgr.run_dir, raw_model)
        log.info("Run directory: %s", ckpt_mgr.run_dir)

    # Resume from checkpoint
    start_epoch = 1
    _resumed_global_step = None
    resume_batch = 0
    if args.checkpoint:
        skip_optim = args.freeze_core or args.optim_checkpoint is not None
        restore_opt = None if skip_optim else opt
        restore_sched = None if skip_optim else sched
        if head_swapped:
            ckpt = loaded_checkpoint
            ckpt_mgr.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            ckpt_mgr.best_epoch = ckpt.get("best_epoch", 0)
            log.info("Head swap already loaded; skipping incompatible model/optimizer reload")
        else:
            ckpt = ckpt_mgr.load(
                args.checkpoint, raw_model, restore_opt, restore_sched,
                map_location=device,
            )

        # A freeze-core run started from a checkpoint in *another* run
        # directory is a new phase, not a continuation: the optimizer and
        # scheduler are already fresh, the residue-only validation loss is a
        # different quantity from the dual-MLM one, and the inherited epoch
        # counter would otherwise trip the "nothing to do" guard below.
        new_phase = (args.freeze_core and
                     Path(args.checkpoint).resolve().parent != ckpt_mgr.phase_dir.resolve())

        # Intra-epoch checkpoints resume the same deterministic batch order
        # with optimizer, scheduler, and model state preserved.
        if new_phase:
            start_epoch = 1
            _resumed_global_step = None
            ckpt_mgr.best_val_loss = float("inf")
            ckpt_mgr.best_epoch = 0
            log.info("freeze-core phase start from %s — epoch counter and "
                     "best-loss tracking reset (weights loaded)", args.checkpoint)
        elif ckpt.get("intra_epoch", False):
            start_epoch = ckpt.get("epoch", 1)
            _resumed_global_step = ckpt.get("global_step", 0)
            resume_batch = ckpt.get("batch_index", 0)
            log.info("Resuming intra-epoch checkpoint (epoch %d, step %d) — "
                     "skipping %d completed batches",
                     start_epoch, _resumed_global_step, resume_batch)
        else:
            _resumed_global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", 0) + 1

        rng_states = ckpt.get("rng_states")
        if rng_states and rank < len(rng_states):
            _restore_rng_state(rng_states[rank])

        # Optionally load optimizer/scheduler from a different checkpoint
        if args.optim_checkpoint:
            ckpt_opt = torch.load(args.optim_checkpoint, map_location=device, weights_only=False)
            opt.load_state_dict(ckpt_opt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_opt:
                sched.load_state_dict(ckpt_opt["scheduler_state_dict"])
            log.info("Optimizer/scheduler restored from %s", args.optim_checkpoint)
            del ckpt_opt
        if opt_loss is not None and "loss_optimizer_state_dict" in ckpt:
            opt_loss.load_state_dict(ckpt["loss_optimizer_state_dict"])

        log.info("Resumed from %s (epoch %d, best_val=%.4f)%s",
                 args.checkpoint, start_epoch - 1, ckpt_mgr.best_val_loss,
                 " [freeze-core: fresh optimizer+scheduler]" if args.freeze_core else "")

        # A resumed checkpoint past --epochs would silently train nothing and
        # report "complete", burning an allocation for no work.
        if start_epoch > args.epochs:
            raise SystemExit(
                f"Nothing to do: checkpoint is at epoch {start_epoch - 1} but "
                f"--epochs is {args.epochs}. Raise --epochs (or use a fresh "
                f"--run-dir) before submitting.")

    # Graceful shutdown on SIGTERM (e.g. SLURM job timeout)
    _shutdown_requested = False
    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        log.info("SIGTERM received — will save checkpoint and exit after current step")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    eval_residue = not token_only
    eval_token = not residue_only
    if args.checkpoint and _resumed_global_step is not None:
        global_step = _resumed_global_step
    else:
        global_step = (start_epoch - 1) * max(1, len(loader) // ga)
    if args.checkpoint:
        # Checkpoints created before --schedule-epochs may contain a scheduler
        # state built for a shorter horizon. Re-anchor it to the new global
        # step using the current full-run schedule.
        sched.step(global_step)
    for ep in range(start_epoch, args.epochs + 1):
        # Rotate shards if streaming (must recreate loader for worker visibility)
        if train_ds.streaming:
            train_ds.load_epoch(ep - 1)
            loader = make_loader(train_ds, epoch=ep, distributed=is_ddp)

        # --- Train ---
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        # Accumulators for epoch metrics
        avg_keys = ["loss"]
        if not residue_only:
            avg_keys += ["tok_loss", "tok_acc"]
        if not token_only:
            avg_keys += ["res_loss", "res_acc", "pct_tok", "pct_res"]
        sums = {k: 0.0 for k in avg_keys}
        n_steps = 0
        oom_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {ep}", leave=False, disable=not is_main)
        for i, batch in enumerate(pbar):
            if ep == start_epoch and i < resume_batch:
                continue
            try:
                sync = not (is_ddp and (i + 1) % ga != 0)
                context = model.no_sync() if is_ddp and not sync else nullcontext()
                with context:
                    m = train_step(
                        batch, model, device, amp, ga,
                        combine_loss=combine_loss,
                        token_mlm_weight=args.token_mlm_weight,
                        residue_mlm_weight=args.residue_mlm_weight,
                        token_only=token_only,
                        residue_only=residue_only,
                        is_ddp=is_ddp,
                        world_size=world_size,
                    )
                if is_ddp and opt_loss is not None:
                    _sync_grads(combine_loss, world_size)
            except RuntimeError as e:
                if _is_oom(e):
                    if is_ddp:
                        log.error("OOM in DDP mode at batch %d; stopping all ranks", i + 1)
                        raise
                    oom_batches += 1
                    log.warning("OOM at batch %d/%d (optimizer step %d; %d this epoch) — skipping batch",
                                i + 1, len(loader), global_step, oom_batches)
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            if math.isnan(m.get("tok_loss", 0)) or math.isnan(m.get("res_loss", 0)):
                opt.zero_grad(set_to_none=True)
                continue

            for k in sums:
                if k in m:
                    sums[k] += m[k]
            n_steps += 1

            if n_steps % 50 == 0:
                d = max(1, n_steps)
                postfix = {
                    "loss": f"{sums['loss']/d:.4f}",
                    "lr": f"{sched.get_last_lr()[0]:.2e}",
                }
                if not residue_only:
                    postfix["t_acc"] = f"{sums['tok_acc']/d:.3f}"
                    postfix["t_ppl"] = f"{math.exp(min(sums['tok_loss']/d, 20.0)):.1f}"
                if not token_only and sums.get("res_acc", 0) > 0:
                    postfix["r_acc"] = f"{sums['res_acc']/d:.3f}"
                    postfix["r_ppl"] = f"{math.exp(min(sums['res_loss']/d, 20.0)):.1f}"
                if not token_only and sums.get("pct_tok", 0) > 0:
                    postfix["split"] = f"{sums['pct_tok']/d:.0%}/{sums['pct_res']/d:.0%}"
                pbar.set_postfix(postfix)

            if (i + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                if opt_loss is not None:
                    opt_loss.step()
                    opt_loss.zero_grad(set_to_none=True)
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % 50 == 0:
                    elapsed = max(1.0, time.time() - t0)
                    log.info("Progress epoch %d/%d: batch %d/%d, optimizer step %d, "
                             "%.1f steps/s, OOM batches=%d",
                             ep, args.epochs, i + 1, len(loader), global_step,
                             n_steps / elapsed, oom_batches)

                # Intra-epoch validation (rank 0 only). Essential when an
                # epoch is longer than the SLURM wall-time, otherwise no
                # validation metric is ever recorded.
                if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                    if is_main:
                        sv = run_validation(raw_model, val_loader, device, amp,
                                            eval_residue=eval_residue,
                                            eval_token=eval_token,
                                            max_batches=args.val_max_batches)
                        if sv:
                            vmsg = f"[step {global_step}] val"
                            if "tok_ppl" in sv:
                                vmsg += (f" tok_ppl={sv['tok_ppl']:.1f} "
                                         f"tok_acc={sv['tok_acc']:.3f}")
                            if "res_ppl" in sv:
                                vmsg += (f" | res_ppl={sv['res_ppl']:.2f} "
                                         f"res_acc={sv['res_acc']:.3f}")
                            log.info(vmsg)
                            if ckpt_mgr.log_step_metrics(
                                    epoch=ep, global_step=global_step,
                                    lr=sched.get_last_lr()[0], val_metrics=sv):
                                ckpt_mgr.save_best(raw_model, opt, sched,
                                                   ep, global_step)
                                log.info("[step %d] new best val_loss=%.4f "
                                         "-> best.pth", global_step,
                                         ckpt_mgr.best_val_loss)
                    if is_ddp:
                        dist.barrier()

                # Intra-epoch checkpoint (rank 0 only)
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    rng_states = _collect_rng_states(rank, world_size, is_ddp)
                    loss_optimizer_state = (
                        opt_loss.state_dict() if opt_loss is not None else None)
                    if is_main:
                        log.info("Saving intra-epoch checkpoint at step %d "
                                 "(epoch %d, batch %d/%d)",
                                 global_step, ep, i + 1, len(loader))
                        ckpt_mgr.save_step(raw_model, opt, sched, ep, global_step,
                                           batch_index=i + 1,
                                           extra_state={"rng_states": rng_states,
                                                        "loss_optimizer_state_dict": loss_optimizer_state})
                    if is_ddp:
                        dist.barrier()

                # Graceful shutdown
                if _shutdown_requested:
                    rng_states = _collect_rng_states(rank, world_size, is_ddp)
                    loss_optimizer_state = (
                        opt_loss.state_dict() if opt_loss is not None else None)
                    if is_main:
                        shutdown_val = run_validation(
                            model, val_loader, device, amp,
                            eval_residue=eval_residue, eval_token=eval_token,
                            max_batches=args.val_max_batches)
                        if shutdown_val:
                            if ckpt_mgr.log_step_metrics(
                                    epoch=ep, global_step=global_step,
                                    lr=sched.get_last_lr()[0],
                                    val_metrics=shutdown_val):
                                ckpt_mgr.save_best(raw_model, opt, sched,
                                                   ep, global_step)
                            log.info("[shutdown step %d] validation saved",
                                     global_step)
                        log.info("Saving checkpoint before shutdown (step %d)...",
                                 global_step)
                        ckpt_mgr.save_step(raw_model, opt, sched, ep, global_step,
                                           batch_index=i + 1,
                                           extra_state={"rng_states": rng_states,
                                                        "loss_optimizer_state_dict": loss_optimizer_state})
                        log.info("Checkpoint saved. Exiting.")
                        log.info("Epoch %d stopped after %d OOM batches", ep,
                                 oom_batches)
                    _cleanup_ddp()
                    return

        # Flush remaining accumulated gradients
        if (i + 1) % ga != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if opt_loss is not None:
                opt_loss.step()
                opt_loss.zero_grad(set_to_none=True)
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        train_elapsed = time.time() - t0
        sums, n_steps = _reduce_metric_sums(sums, n_steps, device, is_ddp)
        if is_main:
            log.info("Epoch %d training loop finished: %d OOM batches",
                     ep, oom_batches)
        current_lr = sched.get_last_lr()[0]
        d = max(1, n_steps)
        t_metrics = {k: sums[k] / d for k in sums if sums[k] > 0}
        if "tok_loss" in t_metrics:
            t_metrics["tok_ppl"] = math.exp(min(t_metrics["tok_loss"], 20.0))
        if "res_loss" in t_metrics:
            t_metrics["res_ppl"] = math.exp(min(t_metrics["res_loss"], 20.0))

        # --- Validation ---
        v_metrics = None
        val_elapsed = 0.0
        if ep % args.val_every == 0 or ep == args.epochs:
            t1 = time.time()
            val_list = []
            if is_main:
                raw_model.eval()
                for batch in val_loader:
                    try:
                        vm = eval_step(batch, raw_model, device, amp,
                                       eval_residue=eval_residue,
                                       eval_token=eval_token)
                        val_list.append(vm)
                    except RuntimeError as e:
                        if _is_oom(e):
                            torch.cuda.empty_cache()
                            continue
                        raise
            val_elapsed = time.time() - t1
            if val_list:
                v_metrics = {k: np.mean([m[k] for m in val_list]) for k in val_list[0]}
                if "tok_loss" in v_metrics:
                    v_metrics["tok_ppl"] = math.exp(min(v_metrics["tok_loss"], 20.0))
                if "res_loss" in v_metrics:
                    v_metrics["res_ppl"] = math.exp(min(v_metrics["res_loss"], 20.0))
                # Compat: put the active objective into "loss" for best-tracking
                v_metrics["loss"] = (v_metrics["tok_loss"] if eval_token
                                     else v_metrics["res_loss"])

        # Log + save (rank 0 only)
        rng_states = _collect_rng_states(rank, world_size, is_ddp)
        loss_optimizer_state = (
            opt_loss.state_dict() if opt_loss is not None else None)
        if is_main:
            msg = f"Epoch {ep:2d}/{args.epochs:2d} [{train_elapsed:.0f}s"
            if v_metrics:
                msg += f"+{val_elapsed:.0f}s"
            msg += f"] loss={t_metrics.get('loss',0):.4f}"
            if not residue_only:
                msg += f" t_acc={t_metrics.get('tok_acc',0):.3f} t_ppl={t_metrics.get('tok_ppl',0):.1f}"
            if not token_only and "res_acc" in t_metrics:
                msg += f" | r_acc={t_metrics['res_acc']:.3f} r_ppl={t_metrics.get('res_ppl',0):.1f}"
            if not token_only and "pct_tok" in t_metrics:
                msg += f" | split={t_metrics['pct_tok']:.0%}/{t_metrics.get('pct_res',0):.0%}"
            msg += f" | lr={current_lr:.2e}"
            if v_metrics:
                msg += f" | val_loss={v_metrics['loss']:.4f}"
                if "tok_acc" in v_metrics:
                    msg += (f" val_acc={v_metrics['tok_acc']:.3f}"
                            f" val_ppl={v_metrics['tok_ppl']:.1f}")
                if "res_acc" in v_metrics:
                    msg += (f" val_r_acc={v_metrics['res_acc']:.3f}"
                            f" val_r_ppl={v_metrics.get('res_ppl', 0):.2f}")
            log.info(msg)

            # Save checkpoint + metrics
            ckpt_mgr.save(
                raw_model, opt, sched, ep, global_step=global_step,
                train_metrics=t_metrics, val_metrics=v_metrics,
                lr=current_lr, train_elapsed_s=train_elapsed,
                val_elapsed_s=val_elapsed,
                extra_state={"rng_states": rng_states,
                             "loss_optimizer_state_dict": loss_optimizer_state},
            )
        if is_ddp:
            dist.barrier()

    if is_main:
        log.info("=" * 50)
        log.info("MLM pretraining complete. Best val loss: %.4f (epoch %d)",
                 ckpt_mgr.best_val_loss, ckpt_mgr.best_epoch)
        log.info("Checkpoints: %s", ckpt_mgr.phase_dir)
    _cleanup_ddp()


if __name__ == "__main__":
    main()