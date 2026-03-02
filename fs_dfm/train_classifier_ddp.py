#
# Fine-tune FS-DFM for IMDB Sentiment Classification (Accelerate DDP)
#
# Three loss modes:
#   cross_entropy   – Vanilla CE loss (teacher-only, simplest)
#   generalized_kl  – MixturePathGeneralizedKL with g(t) weighting (teacher-only)
#   distillation    – Full FS-DFM pipeline: RK-4 shortcut teacher + KL distillation
#                     with budget-aware blending (teacher + student + EMA)
#
# Usage:
#   # Single GPU:
#   python train_classifier_ddp.py \
#       --config configs/config_imdb.yaml \
#       --checkpoint_path /path/to/pretrained/checkpoint.pth
#
#   # Multi-GPU DDP via accelerate:
#   accelerate launch --multi_gpu --num_processes 4 train_classifier_ddp.py \
#       --config configs/config_imdb.yaml \
#       --checkpoint_path /path/to/pretrained/checkpoint.pth
#

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    import wandb
except ImportError:
    wandb = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Transformer
from logic.flow import get_source_distribution, get_path, get_loss_function
from data.imdb_data import get_imdb_datasets

# Losses
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.loss.KL_divergence import ForwardKLDistillationLoss

# Solver (needed for distillation mode)
from flow_matching.solver import get_solver_by_name
from flow_matching.utils import ModelWrapper


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

# Maps YAML nested keys to argparse names where they differ
_YAML_RENAME = {
    "length": "block_size",       # model.length -> --block_size
    "n_iters": "num_iters",       # optim.n_iters -> --num_iters
}

# Keys from YAML sections that need a prefix to match argparse names
_YAML_PREFIX = {
    "ema": {"decay": "ema_decay", "freq": "ema_freq"},
}

# YAML keys to skip (no corresponding argparse arg)
_YAML_SKIP = {}
# YAML sections to skip entirely (inference config belongs to classify_imdb.py)
_YAML_SKIP_SECTIONS = {}


def load_yaml_config(path):
    """Load a YAML config file and flatten its nested structure to argparse-compatible dict.

    Returns a flat dict mapping argparse dest names to values.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    flat = {}
    for section, values in cfg.items():
        if not isinstance(values, dict) or section in _YAML_SKIP_SECTIONS:
            continue
        # Check if this section has special prefix mappings
        prefix_map = _YAML_PREFIX.get(section, {})
        for key, val in values.items():
            if key in _YAML_SKIP:
                continue
            # Apply prefix mapping first, then rename mapping, else use raw key
            if key in prefix_map:
                arg_name = prefix_map[key]
            elif key in _YAML_RENAME:
                arg_name = _YAML_RENAME[key]
            else:
                arg_name = key
            # Coerce list elements for known numeric-list args
            if arg_name == "step_sizes" and isinstance(val, list):
                val = [float(x) for x in val]
            elif arg_name == "dt_weights" and isinstance(val, list):
                val = [int(x) for x in val]
            flat[arg_name] = val
    return flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class WrappedModel(ModelWrapper):
    """Wraps Transformer to output softmax probabilities (required by solver)."""
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return torch.softmax(self.model(x_t=x, time=t, **extras).float(), dim=-1)


def get_lr(lr, step, warmup, n_iters, eta_min_ratio=0.1):
    """Cosine LR schedule with warmup (matches existing FS-DFM schedule)."""
    if step < warmup:
        return lr * (step / warmup)
    eta_min = eta_min_ratio * lr
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup) / (n_iters - warmup)))
    return eta_min + (lr - eta_min) * cosine_decay


def categorical(probs):
    """Sample from categorical distribution (used in RK-4 teacher steps)."""
    return torch.multinomial(probs.flatten(0, -2), 1).reshape(probs.shape[:-1])


def load_pretrained_weights(checkpoint_path, model, device, key_preference=None):
    """Load pre-trained weights from an FS-DFM checkpoint into a model.

    Args:
        key_preference: ordered list of checkpoint keys to try.
            Defaults to ["teacher_model", "model", "student_model"].
    """
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if key_preference is None:
        key_preference = ["teacher_model", "model", "student_model"]

    for key in key_preference:
        if key in loaded:
            missing, unexpected = model.load_state_dict(loaded[key], strict=False)
            print(f"Loaded weights from key '{key}'")
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            return

    raise ValueError(
        f"No recognized model key in checkpoint. Found keys: {list(loaded.keys())}"
    )


def sample_weighted_dt_uniform_t(
    step_sizes, dt_weights, sampling_steps, batch_size, device, time_epsilon=0.0
):
    """Sample (t, dt) pairs with weighted step sizes (from training.py)."""
    step_sizes_t = torch.as_tensor(step_sizes, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(dt_weights, dtype=torch.float32, device=device)

    valid_mask = step_sizes_t <= (1.0 - time_epsilon)
    step_sizes_v = step_sizes_t[valid_mask]
    weights_v = torch.clamp(weights_t[valid_mask], min=0)

    if torch.all(weights_v == 0):
        probs = torch.full_like(weights_v, 1.0 / weights_v.numel())
    else:
        probs = weights_v / weights_v.sum()

    idx = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    dt = step_sizes_v[idx]

    max_indices = torch.floor((1.0 - time_epsilon - dt) * sampling_steps).to(torch.long)
    t_indices = torch.floor(
        torch.rand(batch_size, device=device) * (max_indices.to(torch.float32) + 1.0)
    ).to(torch.long)
    t = t_indices.to(torch.float32) / float(sampling_steps)
    return t, dt


def ema_update(ema_model, model, decay):
    """Exponential moving average update of ema_model from model."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.lerp_(p.data, 1.0 - decay)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)


# ---------------------------------------------------------------------------
# RK-4 Teacher Estimate (from training.py:get_RK_4_estimate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_RK_4_estimate(
    teacher_model, semi_teacher_model, x_t, t, dt, solver, path, vocab_size,
    distill_th=1/513.0, can_apply_dt=True,
):
    """4-stage RK-4 using EMA student as 'semi_teacher'.

    Returns: (artificial_teacher_logits, x_next, artificial_teacher_u)
    """
    dt_half = dt / 2.0
    t_mid = (t + dt_half).clamp_(0.0, 1.0)
    t_next = (t + dt).clamp_(0.0, 1.0)
    shortcut_mask = dt < distill_th

    # Stage 1: k1 at (x_t, t) with dt/2
    k1_logits = semi_teacher_model(x_t, t, dt=dt_half)
    k1_probs = torch.softmax(k1_logits, dim=-1)
    u1 = solver.finite_probs_to_generator(k1_probs, x_t, dt_half, t=t, can_apply_dt=can_apply_dt)
    x_mid_1 = solver._step(x_t, u1, dt_half, dtype=torch.float32, p_1t=k1_probs, t=t)

    # Stage 2: k2 at (x_mid_1, t_mid) with dt/2
    k2_logits = semi_teacher_model(x_mid_1, t_mid, dt=dt_half)
    k2_probs = torch.softmax(k2_logits, dim=-1)
    u2 = solver.finite_probs_to_generator(k2_probs, x_mid_1, dt_half, t=t_mid, can_apply_dt=can_apply_dt)
    x_mid_2 = solver._step(x_mid_1, u2, dt_half, dtype=torch.float32, p_1t=k2_probs, t=t_mid)

    # Stage 3: k3 at (x_mid_2, t_mid) with dt/2
    k3_logits = semi_teacher_model(x_mid_2, t_mid, dt=dt_half)
    k3_probs = torch.softmax(k3_logits, dim=-1)
    u3 = solver.finite_probs_to_generator(k3_probs, x_mid_2, dt_half, t=t_mid, can_apply_dt=can_apply_dt)
    x_mid_3 = solver._step(x_mid_2, u3, dt_half, dtype=torch.float32, p_1t=k3_probs, t=t_mid)

    # Stage 4: k4 at (x_mid_3, t_next) with dt/2
    k4_logits = semi_teacher_model(x_mid_3, t_next, dt=dt_half)
    k4_probs = torch.softmax(k4_logits, dim=-1)
    u4 = solver.finite_probs_to_generator(k4_probs, x_mid_3, dt_half, t=t_next, can_apply_dt=can_apply_dt)

    # RK-4 combination
    artificial_teacher_logits = (k1_logits + 2 * k2_logits + 2 * k3_logits + k4_logits) / 6.0
    artificial_teacher_u = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0

    # Also get single-step teacher prediction (for small dt blending)
    teacher_logit = teacher_model(x_t, t)
    probs_teacher = torch.softmax(teacher_logit, dim=-1)
    x_1_teacher = categorical(probs_teacher.to(dtype=teacher_logit.dtype))

    scheduler_output = path.scheduler(t=t)
    k_t = scheduler_output.alpha_t
    d_k_t = scheduler_output.d_alpha_t
    delta_1 = F.one_hot(x_1_teacher, num_classes=vocab_size).to(k_t.dtype)
    scale = (d_k_t / (1 - k_t)).view(-1, 1, 1)
    u_teacher = scale * delta_1
    delta_t = F.one_hot(x_t, num_classes=vocab_size)
    u_teacher = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u_teacher), u_teacher)

    intensity = u_teacher.sum(dim=-1)
    mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-dt[:, None] * intensity)
    x_t_teacher = x_t.clone()
    if mask_jump.sum() > 0:
        x_t_teacher[mask_jump] = categorical(u_teacher[mask_jump])

    # Compute x_next from RK-4 teacher
    artificial_teacher_probs = torch.softmax(artificial_teacher_logits, dim=-1)
    u_raw = solver.finite_probs_to_generator(
        artificial_teacher_probs, x_t, dt, t=t, can_apply_dt=can_apply_dt
    )
    x_next = solver._step(x_t, u_raw, dt, dtype=torch.float32, p_1t=artificial_teacher_probs, t=t)

    # Blend: for small dt, use single-step teacher; for large dt, use RK-4
    x_next = torch.where(shortcut_mask[:, None], x_t_teacher, x_next)
    artificial_teacher_logits = torch.where(
        shortcut_mask[:, None, None], teacher_logit, artificial_teacher_logits
    )
    artificial_teacher_u = torch.where(
        shortcut_mask[:, None, None], u_teacher, artificial_teacher_u
    )

    return artificial_teacher_logits, x_next, artificial_teacher_u


# ---------------------------------------------------------------------------
# Training functions for each mode
# ---------------------------------------------------------------------------

def compute_label_metrics(logits, batch, vocab_size):
    """Compute classification-relevant metrics at the label token position.

    Returns dict with:
        label_token_acc: fraction of samples where top prediction at label pos is correct
        label_logit_margin: mean (correct_logit - wrong_logit) at label pos
        label_pos_entropy: mean entropy of logit distribution at label pos
    """
    label_token_pos = batch["label_token_pos"]  # [B]
    labels = batch["label"]  # [B] 0=neg, 1=pos

    # GPT2: " positive" -> 3967, " negative" -> 4633
    POS_TOKEN = 3967
    NEG_TOKEN = 4633

    B = logits.shape[0]
    # Gather logits at label position: [B, vocab_size(+1)]
    label_logits = logits[torch.arange(B, device=logits.device), label_token_pos]

    # Only look at the two label tokens
    pos_logit = label_logits[:, POS_TOKEN]  # [B]
    neg_logit = label_logits[:, NEG_TOKEN]  # [B]

    # Predicted label: 1 if pos_logit > neg_logit, else 0
    pred_label = (pos_logit > neg_logit).long()
    acc = (pred_label == labels).float().mean().item()

    # Margin: correct_logit - wrong_logit
    correct_logit = torch.where(labels == 1, pos_logit, neg_logit)
    wrong_logit = torch.where(labels == 1, neg_logit, pos_logit)
    margin = (correct_logit - wrong_logit).mean().item()

    # Entropy at label position (over full vocab, indicates confidence)
    probs = torch.softmax(label_logits[:, :vocab_size].float(), dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

    return {
        "label_token_acc": acc,
        "label_logit_margin": margin,
        "label_pos_entropy": entropy,
    }


def train_step_cross_entropy(model, x_1, batch, source_distribution, path, loss_fn, vocab_size):
    """Mode 1: Vanilla cross-entropy loss (teacher-only)."""
    device = x_1.device
    x_0 = source_distribution.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=device)
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    logits = model(x_t=path_sample.x_t, time=path_sample.t)
    loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()

    # Compute extra metrics (detached)
    mask_token = vocab_size  # mask source distribution uses vocab_size as mask id
    mask_ratio = (path_sample.x_t == mask_token).float().mean().item()
    metrics = {
        "t_mean": t.mean().item(),
        "t_std": t.std().item(),
        "mask_ratio_in_x_t": mask_ratio,
    }
    with torch.no_grad():
        metrics.update(compute_label_metrics(logits, batch, vocab_size))

    return loss, metrics


def train_step_generalized_kl(model, x_1, batch, source_distribution, path, loss_fn, vocab_size):
    """Mode 2: MixturePathGeneralizedKL with g(t) time weighting."""
    device = x_1.device
    x_0 = source_distribution.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=device)
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    logits = model(x_t=path_sample.x_t, time=path_sample.t)
    loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t).mean()

    mask_token = vocab_size
    mask_ratio = (path_sample.x_t == mask_token).float().mean().item()
    metrics = {
        "t_mean": t.mean().item(),
        "t_std": t.std().item(),
        "mask_ratio_in_x_t": mask_ratio,
    }
    with torch.no_grad():
        metrics.update(compute_label_metrics(logits, batch, vocab_size))

    return loss, metrics


def train_step_distillation(
    teacher_model, student_model, student_ema_model,
    x_1, batch, source_distribution, path, solver,
    step_sizes, dt_weights, sampling_steps, vocab_size,
    distill_th, can_apply_dt,
):
    """Mode 3: Full FS-DFM distillation with budget-aware blending.

    For small dt (< distill_th): use GeneralizedKL loss (L_dfm)
    For large dt (>= distill_th): use KL distillation from RK-4 teacher (L_dist)
    """
    device = x_1.device
    x_0 = source_distribution.sample_like(x_1)
    batch_size = x_1.shape[0]

    # Sample weighted (t, dt) pairs
    t, dt = sample_weighted_dt_uniform_t(
        step_sizes=step_sizes,
        dt_weights=dt_weights,
        sampling_steps=sampling_steps,
        batch_size=batch_size,
        device=device,
    )

    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = path_sample.x_t

    # Student forward pass (dt-conditioned)
    logits = student_model(x_t, t, dt=dt)

    # RK-4 teacher estimate (no grad)
    teacher_logits, x_next, teacher_u = get_RK_4_estimate(
        teacher_model=teacher_model,
        semi_teacher_model=student_ema_model,
        x_t=x_t, t=t, dt=dt,
        solver=solver, path=path, vocab_size=vocab_size,
        distill_th=distill_th, can_apply_dt=can_apply_dt,
    )

    shortcut_mask = dt < distill_th  # [B]

    # Loss 1: GeneralizedKL (for small dt)
    loss_fn_1 = MixturePathGeneralizedKL(path=path, reduction="none")
    loss_1 = loss_fn_1(logits=logits, x_1=x_1, x_t=x_t, t=t)

    # Loss 2: KL distillation from RK-4 teacher (for large dt)
    loss_fn_2 = ForwardKLDistillationLoss(reduction="none")
    loss_2 = loss_fn_2(teacher_logits, logits, do_not_apply_softmax=False)

    # Reduce spatial dimensions
    while loss_1.ndim > 1:
        loss_1 = loss_1.mean(dim=-1)
    while loss_2.ndim > 1:
        loss_2 = loss_2.mean(dim=-1)

    # Budget-aware blending (Eq. 4.5): small h -> L_dfm, large h -> L_dist
    per_sample = torch.where(shortcut_mask, loss_1, loss_2)
    loss = per_sample.mean()

    n_shortcut = shortcut_mask.sum().item()
    n_distill = batch_size - n_shortcut
    mask_token = vocab_size
    mask_ratio = (x_t == mask_token).float().mean().item()
    metrics = {
        "t_mean": t.mean().item(),
        "t_std": t.std().item(),
        "dt_mean": dt.mean().item(),
        "dt_median": dt.median().item(),
        "mask_ratio_in_x_t": mask_ratio,
        "loss_dfm": loss_1[shortcut_mask].mean().item() if n_shortcut > 0 else 0.0,
        "loss_dist": loss_2[~shortcut_mask].mean().item() if n_distill > 0 else 0.0,
        "frac_shortcut": n_shortcut / batch_size,
        "frac_distill": n_distill / batch_size,
    }
    with torch.no_grad():
        metrics.update(compute_label_metrics(logits, batch, vocab_size))

    return loss, metrics


def enable_gradient_checkpointing(model):
    """Patch Transformer blocks to use gradient checkpointing (no source changes)."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    for block in model.blocks:
        orig_forward = block.forward

        def _make_ckpt_fn(fn):
            def _ckpt_forward(*args, **kwargs):
                return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return _ckpt_forward

        block.forward = _make_ckpt_fn(orig_forward)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # --- Accelerator (handles DDP, mixed precision, device placement) ---
    accelerator = Accelerator(
        mixed_precision="bf16" if args.precision == "bf16" else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Using device: {device}")
        print(f"Num processes: {accelerator.num_processes}")

    # --- Weights & Biases (main process only) ---
    use_wandb = args.enable_wandb and wandb is not None and is_main
    if args.enable_wandb and wandb is None and is_main:
        print("WARNING: --enable_wandb set but wandb not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=args.wandb_run_name or f"imdb_{args.loss_mode}_lr{args.lr}",
            config=vars(args),
        )

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    if is_main:
        print(f"Vocab size: {vocab_size}")

    # Source distribution and flow path
    source_distribution = get_source_distribution(args.source_distribution, vocab_size)
    path = get_path(scheduler_type="polynomial", exponent=1.0)

    # Model config
    model_config = {
        "hidden_size": args.hidden_size,
        "cond_dim": args.cond_dim,
        "length": args.block_size,
        "n_blocks": args.n_blocks,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }

    # --- Create models based on loss mode ---
    if args.loss_mode in ("cross_entropy", "generalized_kl"):
        model = Transformer(
            config=model_config,
            vocab_size=vocab_size,
            masked=args.source_distribution == "mask",
            dt_conditioned=args.dt_conditioned,
        )

        num_params = sum(p.numel() for p in model.parameters())
        if is_main:
            print(f"Model parameters: {num_params / 1e6:.1f}M")

        # Load weights on CPU before prepare() distributes them
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            load_pretrained_weights(args.checkpoint_path, model, "cpu", key_preference=["student_model"])
        elif is_main:
            print("WARNING: No checkpoint loaded. Training from scratch.")

        if args.gradient_checkpointing:
            enable_gradient_checkpointing(model)
            if is_main:
                print("Gradient checkpointing enabled")

        if args.compile:
            model = torch.compile(model)
            if is_main:
                print("Model compiled with torch.compile")

        teacher_model = None
        student_model = None
        student_ema_model = None
        solver = None

    elif args.loss_mode == "distillation":
        # Full pipeline: teacher (frozen) + student (dt_conditioned) + EMA
        teacher_model = Transformer(
            config=model_config,
            vocab_size=vocab_size,
            masked=True,
            dt_conditioned=False,
        )

        student_model = Transformer(
            config=model_config,
            vocab_size=vocab_size,
            masked=True,
            dt_conditioned=True,
        )

        num_params_t = sum(p.numel() for p in teacher_model.parameters())
        num_params_s = sum(p.numel() for p in student_model.parameters())
        if is_main:
            print(f"Teacher parameters: {num_params_t / 1e6:.1f}M")
            print(f"Student parameters: {num_params_s / 1e6:.1f}M")

        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            load_pretrained_weights(args.checkpoint_path, teacher_model, "cpu")
            load_pretrained_weights(
                args.checkpoint_path, student_model, "cpu",
                key_preference=["student_model", "teacher_model"],
            )
        elif is_main:
            print("WARNING: No checkpoint loaded. Training from scratch.")

        # Freeze teacher
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # EMA copy of student (before prepare, so it's a clean copy)
        student_ema_model = copy.deepcopy(student_model)
        student_ema_model.requires_grad_(False)
        student_ema_model.eval()

        if args.gradient_checkpointing:
            enable_gradient_checkpointing(student_model)
            if is_main:
                print("Gradient checkpointing enabled on student model")

        if args.compile:
            teacher_model = torch.compile(teacher_model)
            student_model = torch.compile(student_model)
            student_ema_model = torch.compile(student_ema_model)
            if is_main:
                print("Models compiled with torch.compile")

        # For distillation, model reference = student (trained)
        model = student_model

    else:
        raise ValueError(f"Unknown loss_mode: {args.loss_mode}")

    # --- Loss function ---
    if args.loss_mode == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_mode == "generalized_kl":
        loss_fn = MixturePathGeneralizedKL(path=path)
    elif args.loss_mode == "distillation":
        loss_fn = None  # handled inside train_step_distillation

    # Data
    train_ds, test_ds = get_imdb_datasets(
        tokenizer=tokenizer,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    if is_main:
        print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer (only trains the model that requires grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    # --- Accelerator prepare (wraps model in DDP, shards dataloader, handles AMP) ---
    if args.loss_mode == "distillation":
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        student_model = model  # model is now DDP-wrapped
        # Teacher and EMA are not trained, just move to device
        teacher_model = teacher_model.to(device)
        student_ema_model = student_ema_model.to(device)

        # Solver (needed for RK-4 estimation) — wraps the unwrapped EMA model
        mask_token = vocab_size
        wrapped_student = WrappedModel(student_ema_model)
        solver_class = get_solver_by_name(args.student_solver)
        solver = solver_class(
            model=wrapped_student,
            path=path,
            vocabulary_size=vocab_size,
            mask_token=mask_token,
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Training loop
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    model.train()
    step = 0
    best_loss = float("inf")
    data_iter = iter(train_loader)

    if is_main:
        print(f"\nStarting training for {args.num_iters} iterations...")
        print(f"  Loss mode: {args.loss_mode}")
        print(f"  Batch size (per GPU): {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Block size: {args.block_size}")
        print(f"  Precision: {args.precision}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print()

    while step < args.num_iters:
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Accelerator's prepared dataloader already places data on device
        x_1 = batch["input_ids"]
        batch_device = batch

        # Forward + loss (with gradient accumulation context)
        with accelerator.accumulate(model):
            optimizer.zero_grad()

            if args.loss_mode == "cross_entropy":
                loss, metrics = train_step_cross_entropy(
                    model, x_1, batch_device, source_distribution, path, loss_fn, vocab_size
                )
            elif args.loss_mode == "generalized_kl":
                loss, metrics = train_step_generalized_kl(
                    model, x_1, batch_device, source_distribution, path, loss_fn, vocab_size
                )
            elif args.loss_mode == "distillation":
                loss, metrics = train_step_distillation(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    student_ema_model=student_ema_model,
                    x_1=x_1,
                    batch=batch_device,
                    source_distribution=source_distribution,
                    path=path,
                    solver=solver,
                    step_sizes=args.step_sizes,
                    dt_weights=args.dt_weights,
                    sampling_steps=args.sampling_steps,
                    vocab_size=vocab_size,
                    distill_th=args.distill_th,
                    can_apply_dt=args.can_apply_dt,
                )

            # Backward (accelerator handles loss scaling for DDP)
            accelerator.backward(loss)

            # Gradient clipping
            max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm)

            # LR schedule
            lr = get_lr(args.lr, step, args.warmup, args.num_iters, args.eta_min_ratio)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.step()

        # EMA update (distillation mode only) — use unwrapped model weights
        if args.loss_mode == "distillation" and step % args.ema_freq == 0:
            ema_update(student_ema_model, accelerator.unwrap_model(student_model), args.ema_decay)

        step += 1

        # Logging (main process only)
        if step % args.log_freq == 0 and is_main:
            loss_val = loss.item()
            print(
                f"Step {step}/{args.num_iters} | "
                f"Loss: {loss_val:.4f} | "
                f"LR: {lr:.2e} | "
                f"Acc: {metrics['label_token_acc']:.3f} | "
                f"GradNorm: {grad_norm:.2f}"
            )

            if use_wandb:
                log_dict = {
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/label_token_acc": metrics["label_token_acc"],
                    "train/label_logit_margin": metrics["label_logit_margin"],
                    "train/label_pos_entropy": metrics["label_pos_entropy"],
                    "train/t_mean": metrics["t_mean"],
                    "train/t_std": metrics["t_std"],
                    "train/mask_ratio_in_x_t": metrics["mask_ratio_in_x_t"],
                    "train/step": step,
                }
                # Distillation-specific metrics
                if args.loss_mode == "distillation":
                    log_dict.update({
                        "train/dt_mean": metrics["dt_mean"],
                        "train/dt_median": metrics["dt_median"],
                        "train/loss_dfm": metrics["loss_dfm"],
                        "train/loss_dist": metrics["loss_dist"],
                        "train/frac_shortcut": metrics["frac_shortcut"],
                        "train/frac_distill": metrics["frac_distill"],
                    })
                wandb.log(log_dict, step=step)

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.num_iters:
            # unwrap DDP wrapper to get raw state_dict
            unwrapped_model = accelerator.unwrap_model(model)
            if is_main:
                save_dict = {"step": step}
                if args.loss_mode == "distillation":
                    save_dict["teacher_model"] = teacher_model.state_dict()
                    save_dict["student_model"] = unwrapped_model.state_dict()
                    save_dict["student_ema_model"] = student_ema_model.state_dict()
                    save_dict["optimizer"] = optimizer.state_dict()
                else:
                    save_dict["teacher_model"] = unwrapped_model.state_dict()
                    save_dict["optimizer"] = optimizer.state_dict()

                save_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pth")
                torch.save(save_dict, save_path)
                print(f"  Saved checkpoint to {save_path}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = os.path.join(args.output_dir, "best_checkpoint.pth")
                    torch.save(save_dict, best_path)

            accelerator.wait_for_everyone()

    if is_main:
        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        print(f"Checkpoints saved to {args.output_dir}")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FS-DFM for IMDB classification (Accelerate DDP)")

    # Config file (YAML defaults, overridden by CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI args override YAML values)")

    # Paths
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to pre-trained FS-DFM checkpoint")
    parser.add_argument("--output_dir", type=str, default="./imdb_finetuned",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for HuggingFace datasets")

    # Loss mode
    parser.add_argument("--loss_mode", type=str, default="cross_entropy",
                        choices=["cross_entropy", "generalized_kl", "distillation"],
                        help="Loss function: cross_entropy (simplest), "
                             "generalized_kl (g(t) weighted), "
                             "distillation (full FS-DFM teacher-student)")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eta_min_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps before optimizer update")

    # Model config (should match pre-trained checkpoint)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=512)

    # Distillation-specific
    parser.add_argument("--student_solver", type=str,
                        default="mixture_euler_with_cumulative_scalar",
                        help="Solver type for distillation mode")
    parser.add_argument("--sampling_steps", type=int, default=1024,
                        help="Number of time discretization steps for dt sampling")
    parser.add_argument("--distill_th", type=float, default=1/513.0,
                        help="Threshold for budget-aware blending: dt < th -> L_dfm, dt >= th -> L_dist")
    parser.add_argument("--can_apply_dt", type=bool, default=True,
                        help="Whether to apply dt scaling in generator computation")
    parser.add_argument("--step_sizes", type=float, nargs="+",
                        default=[1.0, 5e-1, 2.5e-1, 1.25e-1, 6.25e-2, 3.125e-2,
                                 1.5625e-2, 7.8125e-3, 3.90625e-3, 1.953125e-3, 9.765625e-4],
                        help="Step sizes for distillation dt schedule")
    parser.add_argument("--dt_weights", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        help="Weights for sampling dt values")
    parser.add_argument("--ema_decay", type=float, default=0.9995,
                        help="EMA decay rate for student EMA model")
    parser.add_argument("--ema_freq", type=int, default=1,
                        help="EMA update frequency (in steps)")

    # Data
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    # Misc
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations during backward to reduce GPU memory")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Weights & Biases
    parser.add_argument("--enable_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="fs-dfm-imdb",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (user or team)")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="W&B group for organizing runs")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")

    # Two-pass parsing: first extract --config, then override defaults from YAML
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        yaml_defaults = load_yaml_config(preliminary.config)
        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()

    set_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
