"""
analyze_segmentation.py
=======================
Detailed analysis: compare fine-tuned MedSAM predictions against ground truth
using the EXACT ground-truth bounding boxes.

Generates 4-panel images:
  1. Input + Box Prompt (cyan box)
  2. Ground Truth mask (green)
  3. Predicted mask (red)
  4. Overlap map: GREEN=correct, RED=false positive, BLUE=missed

This helps verify if the model segments the CORRECT region within the box.

Usage:
    python Box_Prompt/analyze_segmentation.py
    python Box_Prompt/analyze_segmentation.py --num_samples 8
"""

import os
import glob
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything.build_sam import _build_sam


ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "portal_vein", 11: "pancreas",
    12: "right_adrenal", 13: "left_adrenal",
}


def load_medsam(checkpoint_path, device="cpu"):
    sam = _build_sam(
        encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sam.load_state_dict(ckpt["model_state_dict"])
        print(f"  Fine-tuned model (epoch {ckpt.get('epoch','?')}, "
              f"best_dice={ckpt.get('best_dice', 0):.4f})")
    else:
        sam.load_state_dict(ckpt)
        print(f"  Pre-trained model (baseline)")
    sam.to(device).eval()
    return sam


@torch.no_grad()
def predict_mask(sam_model, image_np, box, device):
    img_t = torch.from_numpy(image_np).float().permute(2, 0, 1).to(device)
    inp = sam_model.preprocess(img_t).unsqueeze(0)
    emb = sam_model.image_encoder(inp)

    box_t = torch.from_numpy(box).float().to(device).unsqueeze(0).unsqueeze(0)
    sparse, dense = sam_model.prompt_encoder(points=None, boxes=box_t, masks=None)

    low_res, iou = sam_model.mask_decoder(
        image_embeddings=emb,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )

    masks = F.interpolate(low_res, (1024, 1024), mode="bilinear", align_corners=False)
    masks = masks.squeeze(0)
    scores = iou.squeeze(0)

    best = torch.argmax(scores).item()
    pred = (torch.sigmoid(masks[best]) > 0.5).cpu().numpy().astype(np.uint8)
    return pred, scores[best].item()


def dice_score(pred, gt):
    p, g = pred.astype(bool).flatten(), gt.astype(bool).flatten()
    if p.sum() + g.sum() == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


def save_analysis(image_np, gt_mask, pred_mask, box, dice, organ_name, score, save_path):
    """Save a 4-panel analysis figure."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # ─── Panel 1: Input + Box ───────────────────────────────────
    axes[0].imshow(image_np)
    axes[0].add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=3, edgecolor="cyan", facecolor="cyan", alpha=0.1, linestyle="--"
    ))
    axes[0].set_title(f"Input + Box Prompt\n({organ_name})", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # ─── Panel 2: Ground Truth ──────────────────────────────────
    axes[1].imshow(image_np)
    gt_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 220, 0, 160]  # green
    axes[1].imshow(gt_overlay)
    if gt_mask.any():
        axes[1].contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=[2])
    axes[1].add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))
    gt_pixels = gt_mask.sum()
    axes[1].set_title(f"Ground Truth\n({gt_pixels:,} pixels)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # ─── Panel 3: Prediction ───────────────────────────────────
    axes[2].imshow(image_np)
    pred_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    pred_overlay[pred_mask > 0] = [220, 50, 50, 160]  # red
    axes[2].imshow(pred_overlay)
    if pred_mask.any():
        axes[2].contour(pred_mask, levels=[0.5], colors=["yellow"], linewidths=[2])
    axes[2].add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))
    pred_pixels = pred_mask.sum()
    axes[2].set_title(f"Prediction\nDice: {dice:.4f}  |  Score: {score:.3f}\n({pred_pixels:,} pixels)",
                      fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # ─── Panel 4: Error Map ─────────────────────────────────────
    #   GREEN  = True Positive  (correct segmentation)
    #   RED    = False Positive (model predicted, but GT says no)
    #   BLUE   = False Negative (GT says yes, but model missed)
    #   BLACK  = True Negative  (both agree: no organ)
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    error_map = np.zeros((*image_np.shape[:2], 3), dtype=np.uint8)
    error_map[gt_bool & pred_bool] = [0, 200, 0]       # TP = green
    error_map[~gt_bool & pred_bool] = [220, 50, 50]     # FP = red
    error_map[gt_bool & ~pred_bool] = [50, 100, 220]    # FN = blue

    # Blend with original image for context
    blend = (0.4 * image_np.astype(float) + 0.6 * error_map.astype(float))
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    # Only blend where there is an error or TP
    mask_any = gt_bool | pred_bool
    result = image_np.copy()
    result[mask_any] = blend[mask_any]

    axes[3].imshow(result)
    axes[3].add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))

    tp = (gt_bool & pred_bool).sum()
    fp = (~gt_bool & pred_bool).sum()
    fn = (gt_bool & ~pred_bool).sum()

    axes[3].set_title(
        f"Error Map\n"
        f"🟢 Correct: {tp:,}  |  🔴 Extra: {fp:,}  |  🔵 Missed: {fn:,}",
        fontsize=14, fontweight="bold"
    )
    axes[3].axis("off")

    plt.suptitle(
        f"MedSAM Box-Prompt Analysis — {organ_name}  |  Dice: {dice:.4f}",
        fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze MedSAM segmentation vs ground truth")
    parser.add_argument("--data_dir", default="data/btcv/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="Box_Prompt/best_medsam_btcv.pth")
    parser.add_argument("--output_dir", default="Box_Prompt/analysis_outputs")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)  # different seed for variety
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Segmentation Analysis")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)
    sam = load_medsam(args.checkpoint, device)

    lbl_dir = os.path.join(args.data_dir, args.split, "labels")
    img_dir = os.path.join(args.data_dir, args.split, "images")
    npz_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npz")))

    if not npz_files:
        print("  [ERROR] No test data!")
        return

    # Pick diverse samples — try to get different organs
    random.shuffle(npz_files)
    selected = npz_files[:min(args.num_samples * 3, len(npz_files))]

    print(f"\n  Analyzing up to {args.num_samples} samples...\n")

    seen_organs = set()
    count = 0

    for npz_path in selected:
        if count >= args.num_samples:
            break

        basename = os.path.splitext(os.path.basename(npz_path))[0]
        img_path = os.path.join(img_dir, f"{basename}.png")
        if not os.path.exists(img_path):
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        data = np.load(npz_path)
        organ_ids = data["organ_ids"]

        # Prefer organs we haven't seen yet for diversity
        unseen = [oid for oid in organ_ids if int(oid) not in seen_organs]
        pick_from = unseen if unseen else organ_ids
        organ_id = int(random.choice(pick_from))
        seen_organs.add(organ_id)

        gt_mask = data[f"mask_{organ_id}"]
        box = data[f"box_{organ_id}"]
        organ_name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

        print(f"  [{count+1}/{args.num_samples}] {basename} — {organ_name} ...", end=" ", flush=True)

        pred_mask, score = predict_mask(sam, image_np, box, device)
        dice = dice_score(pred_mask, gt_mask)

        save_path = os.path.join(
            args.output_dir,
            f"analysis_{count:02d}_{basename}_{organ_name}.png"
        )
        save_analysis(image_np, gt_mask, pred_mask, box, dice, organ_name, score, save_path)
        count += 1

        # Detailed stats
        gt_bool = gt_mask.astype(bool)
        pred_bool = pred_mask.astype(bool)
        tp = (gt_bool & pred_bool).sum()
        fp = (~gt_bool & pred_bool).sum()
        fn = (gt_bool & ~pred_bool).sum()

        print(f"Dice: {dice:.4f}  Score: {score:.3f}  "
              f"TP:{tp}  FP:{fp}  FN:{fn}  ✓")

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete! {count} images saved to: {args.output_dir}/")
    print(f"{'=' * 60}")
    print(f"\n  Legend for Panel 4 (Error Map):")
    print(f"    🟢 GREEN  = Correctly segmented (True Positive)")
    print(f"    🔴 RED    = Extra area predicted (False Positive)")
    print(f"    🔵 BLUE   = Missed area (False Negative)")
    print(f"    ⬛ DARK   = Correctly ignored (True Negative)\n")


if __name__ == "__main__":
    main()
