# =============================================================================
# step3_train.py — Multi-Organ Tumor Segmentation Training
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Train a MONAI UNet (3D) for multi-organ + multi-tumor segmentation
#   on the filtered AbdomenAtlas 2.0 subset.
#
# ARCHITECTURE:
#   MONAI UNet with (32, 64, 128, 256) channels — chosen over UNETR because:
#   - UNETR uses a ViT encoder that is 10-20× slower on CPU
#   - UNet with residual units gives comparable accuracy for organ segmentation
#   - Much smaller memory footprint (~8M params vs ~92M for UNETR)
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#
#   # Full training (50 epochs, CPU — ~2-4 hours/epoch)
#   python step3_train.py
#
#   # Quick smoke test (verify everything works)
#   python step3_train.py --epochs 1 --max_samples 4
#
#   # Resume from checkpoint
#   python step3_train.py --resume checkpoints/latest_multiorgan.pth
#
#   # With GPU (much faster)
#   python step3_train.py --device cuda --batch_size 2 --num_workers 2
#
# TRAINING COMMANDS EXPLAINED:
#   --epochs N          : Total training epochs (default: 50)
#   --batch_size N      : Samples per GPU step (default: 1 for CPU)
#   --grad_accum N      : Gradient accumulation steps (effective_batch = batch × accum)
#   --lr FLOAT          : Learning rate (default: 1e-4)
#   --max_samples N     : Limit dataset size for testing (default: use all)
#   --device cpu|cuda   : Training device
#   --resume PATH       : Resume from checkpoint
#   --val_interval N    : Validate every N epochs (default: 2)
#   --num_workers N     : DataLoader workers (0 for CPU, 2+ for GPU)
# =============================================================================

import os
import sys
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_CLASSES, CLASS_MAP, PATCH_SIZE,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, EARLY_STOP_PATIENCE, VAL_INTERVAL, NUM_WORKERS,
    SW_BATCH_SIZE, SW_OVERLAP,
    CHECKPOINT_DIR, BEST_MODEL_PATH, LATEST_MODEL_PATH,
)
from step2_preprocess import get_train_transforms, get_val_transforms, build_datalist


# =============================================================================
# MODEL BUILDER
# =============================================================================
def build_model(device="cpu"):
    """
    Build MONAI UNet for 3D multi-organ tumor segmentation.

    Architecture rationale:
    - channels=(32, 64, 128, 256): 4 encoder levels, ~8M parameters
      Deeper (512+) would OOM on CPU; shallower misses fine tumor detail
    - strides=(2, 2, 2): standard 2× downsampling at each level
    - num_res_units=2: residual connections for stable gradient flow
    - dropout=0.2: regularization for the relatively small training set
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: UNet 3D")
    print(f"  Parameters: {n_params:,} total, {n_train:,} trainable")
    print(f"  Output classes: {NUM_CLASSES}")

    return model


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_accum=4):
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["image"].to(device)   # (B, 1, H, W, D)
        labels = batch_data["label"].to(device)    # (B, 1, H, W, D)

        # Forward pass
        outputs = model(images)                     # (B, NUM_CLASSES, H, W, D)
        loss = loss_fn(outputs, labels.long())

        # Scale loss for gradient accumulation
        loss = loss / grad_accum
        loss.backward()

        # Step optimizer every grad_accum batches
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum
        n_batches += 1

        # Progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            avg = running_loss / n_batches
            print(f"    batch [{batch_idx+1}/{len(loader)}]  loss: {avg:.4f}")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, dice_metric, device, patch_size, sw_batch, sw_overlap):
    """Validate using sliding window inference on full volumes."""
    model.eval()
    running_loss = 0.0
    n_batches = 0

    dice_metric.reset()

    for batch_data in loader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # Sliding window inference for full volumes
        outputs = sliding_window_inference(
            images, patch_size, sw_batch, model, overlap=sw_overlap,
        )

        loss = loss_fn(outputs, labels.long())
        running_loss += loss.item()
        n_batches += 1

        # Compute Dice per class
        # Convert outputs to one-hot predictions
        preds = torch.argmax(outputs, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # One-hot encode for DiceMetric
        from monai.networks.utils import one_hot
        preds_oh = one_hot(preds, NUM_CLASSES)        # (B, C, H, W, D)
        labels_oh = one_hot(labels.long(), NUM_CLASSES)

        dice_metric(y_pred=preds_oh, y=labels_oh)

    # Aggregate dice scores
    dice_scores = dice_metric.aggregate()  # (NUM_CLASSES,)
    dice_metric.reset()

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, dice_scores


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice,
                    train_loss, val_dice_per_class):
    """Save full checkpoint with all training state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice,
        "train_loss": train_loss,
        "val_dice_per_class": val_dice_per_class.tolist()
            if hasattr(val_dice_per_class, "tolist") else val_dice_per_class,
        "num_classes": NUM_CLASSES,
        "class_map": CLASS_MAP,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load checkpoint and return start_epoch, best_dice."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_dice = ckpt.get("best_dice", 0.0)
    print(f"  Resumed from epoch {ckpt.get('epoch', '?')}, best_dice={best_dice:.4f}")
    return start_epoch, best_dice


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train multi-organ tumor segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TRAINING COMMANDS — Quick Reference:
─────────────────────────────────────────────────────────────
  # Smoke test (verify pipeline works, ~5 min)
  python step3_train.py --epochs 1 --max_samples 4

  # Short training run (10 epochs)
  python step3_train.py --epochs 10

  # Full training (50 epochs, CPU, ~100-200 hours)
  python step3_train.py --epochs 50

  # Full training with GPU (50 epochs, ~5-10 hours)
  python step3_train.py --epochs 50 --device cuda --batch_size 2 --num_workers 2

  # Resume interrupted training
  python step3_train.py --resume checkpoints/latest_multiorgan.pth --epochs 50

  # Limit dataset for low disk space (use only 50 cases)
  python step3_train.py --max_samples 50 --epochs 30
─────────────────────────────────────────────────────────────
        """
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=None,
                        help="'cpu' or 'cuda' (auto-detect if omitted)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training+val samples (for testing or low disk)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--manifest", default=None,
                        help="Path to filtered_cases.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  MULTI-ORGAN TUMOR SEGMENTATION — Training")
    print("  Model: UNet 3D | Loss: DiceFocalLoss")
    print(f"  Device: {device}")
    print("=" * 65)

    # ── Dataset ───────────────────────────────────────────────────────
    print(f"\n  Loading dataset...")
    train_files, val_files = build_datalist(args.manifest)

    if args.max_samples:
        n = args.max_samples
        train_files = train_files[:int(n * 0.8)]
        val_files = val_files[:int(n * 0.2)] or val_files[:1]
        print(f"  Limited to {len(train_files)} train + {len(val_files)} val samples")

    if not train_files:
        print("  ERROR: No training samples. Run step0/step1 first.")
        sys.exit(1)

    # ── Transforms ────────────────────────────────────────────────────
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # Use CacheDataset for faster repeated access (caches transforms in RAM)
    # cache_rate=0.0 for very low memory; increase to 0.5-1.0 with more RAM
    cache_rate = 0.0 if device.type == "cpu" else 0.5
    print(f"  Cache rate: {cache_rate} (increase with more RAM)")

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=cache_rate,
                            num_workers=args.num_workers)
    val_ds = CacheDataset(val_files, val_transforms, cache_rate=cache_rate,
                          num_workers=args.num_workers)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\n  Building model...")
    model = build_model(device)

    # ── Loss ──────────────────────────────────────────────────────────
    # DiceFocalLoss: combines Dice loss (shape-aware) with Focal loss
    # (handles extreme class imbalance of tiny tumors)
    #   - to_onehot_y=True: auto-converts integer labels to one-hot
    #   - softmax=True: applies softmax to model outputs
    #   - focal_weight: gamma parameter; higher = more focus on hard examples
    loss_fn = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        focal_weight=2.0,      # gamma for focal loss
        lambda_dice=1.0,       # weight for dice component
        lambda_focal=0.5,      # weight for focal component
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Dice metric (per-class) ───────────────────────────────────────
    dice_metric = DiceMetric(
        include_background=False,  # skip background from mean
        reduction="mean_batch",    # per-class scores
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        start_epoch, best_dice = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # ── Training info ─────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    print(f"\n  Training configuration:")
    print(f"    Epochs         : {start_epoch} → {args.epochs}")
    print(f"    Batch size     : {args.batch_size} (effective: {effective_batch} "
          f"with {args.grad_accum}× grad accum)")
    print(f"    Learning rate  : {args.lr}")
    print(f"    Patch size     : {PATCH_SIZE}")
    print(f"    Train samples  : {len(train_files)}")
    print(f"    Val samples    : {len(val_files)}")
    print(f"    Val interval   : every {args.val_interval} epochs")
    print(f"    Early stopping : patience {EARLY_STOP_PATIENCE}")
    print(f"    Checkpoints    : {CHECKPOINT_DIR}/")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    no_improve = 0

    print(f"\n{'=' * 65}")
    print(f"  Starting training...")
    print(f"{'=' * 65}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"  ── Epoch {epoch}/{args.epochs} ──")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            grad_accum=args.grad_accum,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"    Train loss: {train_loss:.4f}  lr: {current_lr:.2e}")

        # Validate
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            print(f"    Validating...")
            val_loss, dice_scores = validate(
                model, val_loader, loss_fn, dice_metric, device,
                PATCH_SIZE, SW_BATCH_SIZE, SW_OVERLAP,
            )

            # Per-class Dice report
            mean_dice = dice_scores.mean().item()
            print(f"    Val loss: {val_loss:.4f}  Mean Dice: {mean_dice:.4f}")
            print(f"    Per-class Dice:")
            for cls_id in range(1, NUM_CLASSES):
                cls_name = CLASS_MAP[cls_id]
                score = dice_scores[cls_id - 1].item()  # offset by 1 (no background)
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"      {cls_name:<20} {bar} {score:.4f}")

            # Save best
            if mean_dice > best_dice:
                best_dice = mean_dice
                no_improve = 0
                save_checkpoint(BEST_MODEL_PATH, model, optimizer, scheduler,
                                epoch, best_dice, train_loss, dice_scores)
                print(f"    ★ New best model! dice={best_dice:.4f}")
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} val cycles)")
                break

        # Save latest checkpoint every epoch
        save_checkpoint(LATEST_MODEL_PATH, model, optimizer, scheduler,
                        epoch, best_dice, train_loss,
                        dice_scores if 'dice_scores' in dir() else [])

        elapsed = time.time() - epoch_start
        remaining = elapsed * (args.epochs - epoch)
        print(f"    Time: {elapsed:.0f}s | ETA: {remaining/60:.0f} min\n")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Training complete!")
    print(f"  Best mean Dice    : {best_dice:.4f}")
    print(f"  Best checkpoint   : {BEST_MODEL_PATH}")
    print(f"  Latest checkpoint : {LATEST_MODEL_PATH}")
    print(f"\n  Next steps:")
    print(f"    python step4_evaluate.py --checkpoint {BEST_MODEL_PATH}")
    print(f"    python step5_inference.py --input patient_ct.nii.gz")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
