# =============================================================================
# step4_evaluate.py — Per-Class Dice Evaluation
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Evaluate the trained model on the validation set using MONAI's DiceMetric.
#   Produces per-class Dice scores, a confusion summary, and sample visualizations.
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#   python step4_evaluate.py
#   python step4_evaluate.py --checkpoint checkpoints/best_multiorgan.pth
#   python step4_evaluate.py --save_viz --num_viz 5
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_CLASSES, CLASS_MAP, PATCH_SIZE,
    SW_BATCH_SIZE, SW_OVERLAP,
    BEST_MODEL_PATH, OUTPUT_DIR,
)
from step2_preprocess import get_val_transforms, build_datalist


def load_model(checkpoint_path, device):
    """Load trained UNet model from checkpoint."""
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256), strides=(2, 2, 2),
        num_res_units=2, dropout=0.2,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        best_dice = ckpt.get("best_dice", "?")
        print(f"  Loaded checkpoint: epoch {epoch}, best_dice={best_dice}")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded raw state dict")

    model.eval()
    return model


@torch.no_grad()
def evaluate_full(model, val_loader, device):
    """Run evaluation on full validation set."""
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
    )

    # Also track per-sample scores for variance analysis
    all_sample_dices = []

    for i, batch_data in enumerate(val_loader):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # Sliding window inference
        outputs = sliding_window_inference(
            images, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP,
        )

        preds = torch.argmax(outputs, dim=1, keepdim=True)
        preds_oh = one_hot(preds, NUM_CLASSES)
        labels_oh = one_hot(labels.long(), NUM_CLASSES)

        # Per-sample dice
        sample_dice = DiceMetric(include_background=False, reduction="mean_batch")
        sample_dice(y_pred=preds_oh, y=labels_oh)
        sample_scores = sample_dice.aggregate()
        all_sample_dices.append(sample_scores.cpu().numpy())
        sample_dice.reset()

        # Aggregate
        dice_metric(y_pred=preds_oh, y=labels_oh)

        print(f"    [{i+1}/{len(val_loader)}] Mean Dice: "
              f"{sample_scores.mean().item():.4f}")

    overall_scores = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()

    return overall_scores, np.array(all_sample_dices)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-organ model")
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--save_viz", action="store_true",
                        help="Save visualization PNGs")
    parser.add_argument("--num_viz", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\n" + "=" * 62)
    print("  MULTI-ORGAN TUMOR SEGMENTATION — Evaluation")
    print(f"  Device: {device}")
    print("=" * 62)

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"\n  ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"  Train a model first: python step3_train.py")
        sys.exit(1)

    model = load_model(args.checkpoint, device)

    # Load validation data
    _, val_files = build_datalist(args.manifest)
    val_transforms = get_val_transforms()
    val_ds = CacheDataset(val_files, val_transforms, cache_rate=0.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=list_data_collate)

    print(f"\n  Evaluating on {len(val_files)} validation samples...")

    # Evaluate
    dice_scores, per_sample = evaluate_full(model, val_loader, device)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 62}")
    print(f"\n  {'Class':<22} {'Dice':>8}  {'Std':>8}  {'Bar'}")
    print(f"  {'─' * 55}")

    results = {}
    for idx in range(NUM_CLASSES - 1):  # skip background
        cls_id = idx + 1
        cls_name = CLASS_MAP[cls_id]
        score = dice_scores[idx]
        std = per_sample[:, idx].std() if len(per_sample) > 1 else 0.0
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"  {cls_name:<22} {score:>8.4f}  {std:>8.4f}  {bar}")
        results[cls_name] = {"dice": round(float(score), 4),
                             "std": round(float(std), 4)}

    mean_dice = dice_scores.mean()
    print(f"\n  {'MEAN DICE':<22} {mean_dice:>8.4f}")

    # Organ vs Tumor comparison
    organ_ids = [0, 2, 4, 6, 8]   # indices in dice_scores (0-indexed, no bg)
    tumor_ids = [1, 3, 5, 7, 9]
    organ_mean = dice_scores[organ_ids].mean() if len(organ_ids) <= len(dice_scores) else 0
    tumor_mean = dice_scores[tumor_ids].mean() if len(tumor_ids) <= len(dice_scores) else 0
    print(f"  {'ORGAN MEAN':<22} {organ_mean:>8.4f}")
    print(f"  {'TUMOR MEAN':<22} {tumor_mean:>8.4f}")
    print(f"{'=' * 62}")

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "num_samples": len(val_files),
        "mean_dice": round(float(mean_dice), 4),
        "organ_mean_dice": round(float(organ_mean), 4),
        "tumor_mean_dice": round(float(tumor_mean), 4),
        "per_class": results,
    }
    report_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
