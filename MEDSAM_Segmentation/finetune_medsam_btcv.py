"""
finetune_medsam_btcv.py
=======================
Fine-tune MedSAM (ViT-B) on the preprocessed BTCV 2D slices.

Strategy:
  - Freeze the image encoder (ViT)
  - Train the prompt encoder + mask decoder
  - Loss: Dice + Binary Cross-Entropy (focal)
  - Box prompts derived from ground-truth labels

Optimisations for fast training:
  - Data subsetting: train on a configurable fraction (--subset_fraction)
  - Stratified subset: balanced organ representation
  - Image embedding cache: skip encoder after epoch 1
  - Mixed precision (AMP): ~1.5-2x speedup on CUDA
  - Rolling checkpoints: best + latest, old ones overwritten
  - Resume support: --resume to continue from a checkpoint

Usage:
    # Train on 20% subset
    python Box_Prompt/finetune_medsam_btcv.py --subset_fraction 0.2 --epochs 10 --batch_size 4 --device cuda

    # Resume from checkpoint
    python Box_Prompt/finetune_medsam_btcv.py --resume Box_Prompt/latest_medsam_btcv.pth --subset_fraction 0.2 --epochs 10 --device cuda
"""

import os
import glob
import time
import random
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from segment_anything.build_sam import _build_sam
from segment_anything import SamPredictor
from segment_anything.modeling import Sam


# ─── DATASET ──────────────────────────────────────────────────────────────────

class BTCVBoxDataset(Dataset):
    """
    Loads preprocessed BTCV 2D slices + ground-truth masks and bounding boxes.
    Each sample returns ONE (image, mask, box) for a single organ in one slice.

    Supports stratified subsetting to reduce training time while keeping
    all organs represented.
    """

    def __init__(self, data_dir, split="train", subset_fraction=1.0, stratified=True, seed=42):
        super().__init__()
        self.img_dir = os.path.join(data_dir, split, "images")
        self.lbl_dir = os.path.join(data_dir, split, "labels")

        # Build full sample list: (image_path, npz_path, organ_id)
        all_samples = []
        npz_files = sorted(glob.glob(os.path.join(self.lbl_dir, "*.npz")))
        for npz_path in npz_files:
            basename = os.path.splitext(os.path.basename(npz_path))[0]
            img_path = os.path.join(self.img_dir, f"{basename}.png")
            if not os.path.exists(img_path):
                continue

            data = np.load(npz_path)
            organ_ids = data["organ_ids"]
            for oid in organ_ids:
                all_samples.append((img_path, npz_path, int(oid)))

        # Apply subsetting
        if subset_fraction < 1.0 and len(all_samples) > 0:
            rng = random.Random(seed)
            if stratified:
                # Group by organ_id → sample fraction from each group
                organ_groups = defaultdict(list)
                for sample in all_samples:
                    organ_groups[sample[2]].append(sample)

                self.samples = []
                for organ_id, group in sorted(organ_groups.items()):
                    k = max(1, int(len(group) * subset_fraction))
                    self.samples.extend(rng.sample(group, k))

                rng.shuffle(self.samples)
            else:
                k = max(1, int(len(all_samples) * subset_fraction))
                self.samples = rng.sample(all_samples, k)
        else:
            self.samples = all_samples

        print(f"  [{split.upper()}] {len(self.samples)} samples "
              f"(from {len(all_samples)} total, {len(npz_files)} slices"
              f"{f', subset={subset_fraction:.0%}' if subset_fraction < 1.0 else ''})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, npz_path, organ_id = self.samples[idx]

        # Load image → tensor [3, 1024, 1024], float32, [0, 255]
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W

        # Load mask and box
        data = np.load(npz_path)
        mask = data[f"mask_{organ_id}"].astype(np.float32)      # H, W
        box = data[f"box_{organ_id}"].astype(np.float32)        # [4]

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)       # 1, H, W
        box_tensor = torch.from_numpy(box)                      # [4]

        return img_tensor, mask_tensor, box_tensor, img_path


# ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy loss."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """Dice + Focal BCE combined."""

    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalBCELoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.dice_weight * self.dice_loss(pred, target) +
                self.focal_weight * self.focal_loss(pred, target))


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_dice(pred_mask, gt_mask):
    """Compute Dice score between two binary masks (numpy)."""
    intersection = (pred_mask * gt_mask).sum()
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0  # both empty
    return (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum())


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_medsam(checkpoint_path, device="cpu"):
    """Load MedSAM ViT-B model from checkpoint."""
    sam = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Support both full checkpoint dict and raw state_dict
    if "model_state_dict" in state_dict:
        sam.load_state_dict(state_dict["model_state_dict"])
    else:
        sam.load_state_dict(state_dict)
    sam.to(device)
    return sam


def freeze_image_encoder(sam_model):
    """Freeze the image encoder parameters."""
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    print("  Image encoder frozen ✓")


# ─── EMBEDDING CACHE ─────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Caches image embeddings (from the frozen ViT encoder) in CPU memory.
    After the first epoch, all subsequent epochs skip the encoder entirely.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._cache = {}     # img_path → embedding tensor (CPU)
        self._hits = 0
        self._misses = 0

    def get(self, img_path):
        if not self.enabled:
            return None
        embedding = self._cache.get(img_path)
        if embedding is not None:
            self._hits += 1
        else:
            self._misses += 1
        return embedding

    def put(self, img_path, embedding):
        if self.enabled:
            self._cache[img_path] = embedding.detach().cpu()

    def stats(self):
        total = self._hits + self._misses
        if total == 0:
            return "cache: empty"
        return (f"cache: {self._hits}/{total} hits "
                f"({100*self._hits/total:.0f}%), "
                f"{len(self._cache)} entries")


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train_one_epoch(sam_model, dataloader, optimizer, criterion, device,
                    scaler=None, embed_cache=None):
    """Train for one epoch with optional AMP and embedding cache."""
    sam_model.prompt_encoder.train()
    sam_model.mask_decoder.train()
    sam_model.image_encoder.eval()  # always frozen

    running_loss = 0.0
    running_dice = 0.0
    n_samples = 0
    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, (images, masks, boxes, img_paths) in enumerate(dataloader):
        images = images.to(device)  # B, 3, 1024, 1024
        masks = masks.to(device)    # B, 1, 1024, 1024
        boxes = boxes.to(device)    # B, 4

        batch_size = images.shape[0]
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_dice = 0.0

        for i in range(batch_size):
            # Try cache first
            cached = embed_cache.get(img_paths[i]) if embed_cache else None

            if cached is not None:
                image_embedding = cached.to(device)
            else:
                with torch.no_grad():
                    input_image = sam_model.preprocess(images[i])
                    input_image = input_image.unsqueeze(0)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            image_embedding = sam_model.image_encoder(input_image)
                    else:
                        image_embedding = sam_model.image_encoder(input_image)
                # Store in cache
                if embed_cache:
                    embed_cache.put(img_paths[i], image_embedding)

            # Encode prompt (box)
            box_input = boxes[i].unsqueeze(0).unsqueeze(0)  # 1, 1, 4

            if use_amp:
                with torch.cuda.amp.autocast():
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None, boxes=box_input, masks=None,
                    )
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_mask = F.interpolate(
                        low_res_masks, (1024, 1024),
                        mode="bilinear", align_corners=False,
                    )
                    gt_mask = masks[i].unsqueeze(0)
                    loss = criterion(pred_mask, gt_mask)
            else:
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_input, masks=None,
                )
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = F.interpolate(
                    low_res_masks, (1024, 1024),
                    mode="bilinear", align_corners=False,
                )
                gt_mask = masks[i].unsqueeze(0)
                loss = criterion(pred_mask, gt_mask)

            batch_loss += loss

            # Dice metric
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                dice = compute_dice(
                    pred_binary.cpu().numpy().flatten(),
                    gt_mask.cpu().numpy().flatten(),
                )
                batch_dice += dice

        # Average loss over batch
        batch_loss = batch_loss / batch_size

        if use_amp:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()

        running_loss += batch_loss.item() * batch_size
        running_dice += batch_dice
        n_samples += batch_size

        if (batch_idx + 1) % 20 == 0:
            print(f"    batch [{batch_idx+1}/{len(dataloader)}]  "
                  f"loss: {batch_loss.item():.4f}  "
                  f"dice: {batch_dice/batch_size:.4f}")

    return running_loss / n_samples, running_dice / n_samples


@torch.no_grad()
def validate(sam_model, dataloader, criterion, device, embed_cache=None):
    """Validate and return average loss + Dice."""
    sam_model.eval()

    running_loss = 0.0
    running_dice = 0.0
    n_samples = 0
    use_amp = device.type == "cuda"

    for images, masks, boxes, img_paths in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        boxes = boxes.to(device)

        batch_size = images.shape[0]
        batch_loss = 0.0
        batch_dice = 0.0

        for i in range(batch_size):
            # Try cache first
            cached = embed_cache.get(img_paths[i]) if embed_cache else None

            if cached is not None:
                image_embedding = cached.to(device)
            else:
                input_image = sam_model.preprocess(images[i])
                input_image = input_image.unsqueeze(0)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        image_embedding = sam_model.image_encoder(input_image)
                else:
                    image_embedding = sam_model.image_encoder(input_image)
                if embed_cache:
                    embed_cache.put(img_paths[i], image_embedding)

            box_input = boxes[i].unsqueeze(0).unsqueeze(0)

            if use_amp:
                with torch.cuda.amp.autocast():
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None, boxes=box_input, masks=None,
                    )
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
            else:
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_input, masks=None,
                )
                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

            pred_mask = F.interpolate(
                low_res_masks, (1024, 1024),
                mode="bilinear", align_corners=False,
            )
            gt_mask = masks[i].unsqueeze(0)

            loss = criterion(pred_mask, gt_mask)
            batch_loss += loss.item()

            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            dice = compute_dice(
                pred_binary.cpu().numpy().flatten(),
                gt_mask.cpu().numpy().flatten(),
            )
            batch_dice += dice

        running_loss += batch_loss
        running_dice += batch_dice
        n_samples += batch_size

    return running_loss / n_samples, running_dice / n_samples


# ─── CHECKPOINT UTILS ─────────────────────────────────────────────────────────

def save_checkpoint(path, sam_model, optimizer, epoch, best_dice, train_loss, val_dice, scaler=None):
    """Save a full checkpoint (model + optimizer + metadata)."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": sam_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice": best_dice,
        "train_loss": train_loss,
        "val_dice": val_dice,
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, sam_model, optimizer, device, scaler=None):
    """Load a full checkpoint and return the start epoch + best dice."""
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        sam_model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"  ✓ Resumed from epoch {ckpt.get('epoch', '?')}, "
              f"best_dice={best_dice:.4f}")
    else:
        # Raw state dict (e.g. original medsam_vit_b.pth)
        sam_model.load_state_dict(ckpt)
        start_epoch = 1
        best_dice = 0.0
        print(f"  ✓ Loaded raw state dict (starting fresh from epoch 1)")
    return start_epoch, best_dice


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM on BTCV (with subsetting)")
    parser.add_argument("--data_dir", default="data/btcv/processed",
                        help="Processed BTCV dataset directory")
    parser.add_argument("--checkpoint", default="Box_Prompt/medsam_vit_b.pth",
                        help="Pre-trained MedSAM checkpoint")
    parser.add_argument("--output_dir", default="Box_Prompt",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or 'cuda'")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # ── New: Subsetting & optimisation args ───────────────────────
    parser.add_argument("--subset_fraction", type=float, default=0.2,
                        help="Fraction of training data to use (0.0-1.0). "
                             "Default 0.2 = 20%% for fast training.")
    parser.add_argument("--no_stratified", action="store_true",
                        help="Disable stratified organ-balanced subsetting")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--cache_embeddings", action="store_true", default=True,
                        help="Cache image embeddings in CPU RAM (skip encoder after epoch 1)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable embedding cache")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    print("=" * 60)
    print("  Loading MedSAM ...")
    print("=" * 60)
    sam_model = load_medsam(args.checkpoint, device)
    freeze_image_encoder(sam_model)

    # ── Datasets (with subsetting) ────────────────────────────────
    print("\n  Loading datasets ...")
    stratified = not args.no_stratified
    train_ds = BTCVBoxDataset(
        args.data_dir, split="train",
        subset_fraction=args.subset_fraction,
        stratified=stratified, seed=args.seed,
    )
    val_ds = BTCVBoxDataset(
        args.data_dir, split="val",
        subset_fraction=1.0,  # always validate on full val set
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    # ── Optimizer (only trainable params) ─────────────────────────
    trainable_params = [
        p for p in sam_model.parameters() if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)

    # ── AMP scaler (CUDA only) ────────────────────────────────────
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("  Mixed precision (AMP) enabled ✓")

    # ── Embedding cache ───────────────────────────────────────────
    use_cache = args.cache_embeddings and not args.no_cache
    embed_cache = EmbeddingCache(enabled=use_cache)
    if use_cache:
        print("  Embedding cache enabled ✓ (encoder skipped after epoch 1)")

    # ── Resume from checkpoint ────────────────────────────────────
    start_epoch = 1
    best_dice = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_medsam_btcv.pth")
    latest_ckpt_path = os.path.join(args.output_dir, "latest_medsam_btcv.pth")

    if args.resume:
        start_epoch, best_dice = load_checkpoint(
            args.resume, sam_model, optimizer, device, scaler
        )

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in sam_model.parameters())
    print(f"\n  Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    # ── Training loop ─────────────────────────────────────────────
    end_epoch = args.epochs

    print(f"\n{'='*60}")
    print(f"  Starting training: epochs {start_epoch}→{end_epoch}, "
          f"bs={args.batch_size}, lr={args.lr}")
    print(f"  Subset: {args.subset_fraction:.0%} of training data "
          f"({len(train_ds)} samples)")
    print(f"  Checkpoints: best + latest (old ones overwritten)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        print(f"  ── Epoch {epoch}/{end_epoch} ──")

        train_loss, train_dice = train_one_epoch(
            sam_model, train_loader, optimizer, criterion, device,
            scaler=scaler, embed_cache=embed_cache,
        )
        print(f"    Train  →  loss: {train_loss:.4f}  dice: {train_dice:.4f}")

        val_loss, val_dice = validate(
            sam_model, val_loader, criterion, device,
            embed_cache=embed_cache,
        )
        print(f"    Val    →  loss: {val_loss:.4f}  dice: {val_dice:.4f}")

        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (end_epoch - epoch)
        print(f"    Time: {epoch_time:.1f}s  |  "
              f"ETA: {remaining/60:.1f} min  |  "
              f"{embed_cache.stats()}")

        # Save best model (overwritten only when val Dice improves)
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(best_ckpt_path, sam_model, optimizer,
                            epoch, best_dice, train_loss, val_dice, scaler)
            print(f"    ★ New best model saved (dice={best_dice:.4f})")

        # Save latest checkpoint (always overwrites to save space)
        save_checkpoint(latest_ckpt_path, sam_model, optimizer,
                        epoch, best_dice, train_loss, val_dice, scaler)
        print(f"    Latest checkpoint saved → {latest_ckpt_path} (epoch {epoch})")

        print()

    print(f"{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val Dice    : {best_dice:.4f}")
    print(f"  Best checkpoint  : {best_ckpt_path}")
    print(f"  Latest checkpoint: {latest_ckpt_path} (epoch {end_epoch})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
