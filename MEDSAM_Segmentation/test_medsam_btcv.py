"""
test_medsam_btcv.py
===================
Test a fine-tuned (or pre-trained) MedSAM model on BTCV test slices
using bounding-box prompts.

Outputs:
  - Per-organ and overall Dice scores printed to terminal
  - Side-by-side visualisation images saved to test_outputs/

Usage:
    # Test with the fine-tuned model
    python Box_Prompt/test_medsam_btcv.py \\
        --checkpoint Box_Prompt/best_medsam_btcv.pth \\
        --num_vis 10 --device cpu

    # Test with the original pre-trained MedSAM (baseline)
    python Box_Prompt/test_medsam_btcv.py \\
        --checkpoint Box_Prompt/medsam_vit_b.pth \\
        --num_vis 10 --device cpu
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything.build_sam import _build_sam


# ─── BTCV Organ Names ─────────────────────────────────────────────────────────
ORGAN_NAMES = {
    1:  "spleen",
    2:  "right_kidney",
    3:  "left_kidney",
    4:  "gallbladder",
    5:  "esophagus",
    6:  "liver",
    7:  "stomach",
    8:  "aorta",
    9:  "IVC",
    10: "portal_vein",
    11: "pancreas",
    12: "right_adrenal",
    13: "left_adrenal",
}


def compute_dice(pred, gt):
    """Dice score between two binary numpy masks."""
    pred = pred.astype(bool).flatten()
    gt = gt.astype(bool).flatten()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum())


def load_medsam(checkpoint_path, device="cpu"):
    """Load MedSAM ViT-B."""
    sam = _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu",weights_only = False)
    # Support both full checkpoint dict and raw state_dict
    if "model_state_dict" in state_dict:
        sam.load_state_dict(state_dict["model_state_dict"])
    else:
        sam.load_state_dict(state_dict)
    sam.to(device)
    sam.eval()
    return sam


@torch.no_grad()
def predict_mask(sam_model, image_np, box, device):
    """
    Run MedSAM inference with a single box prompt.

    Args:
        sam_model: loaded SAM model
        image_np: H x W x 3 uint8 RGB image (1024x1024)
        box: [x_min, y_min, x_max, y_max] numpy array
        device: torch device

    Returns:
        pred_mask: H x W binary numpy array
        score: float prediction confidence
    """
    # Prepare image tensor
    img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1)  # 3, H, W
    img_tensor = img_tensor.to(device)

    # Encode image
    input_image = sam_model.preprocess(img_tensor)          # 3, 1024, 1024
    input_image = input_image.unsqueeze(0)                  # 1, 3, 1024, 1024
    image_embedding = sam_model.image_encoder(input_image)  # 1, 256, 64, 64

    # Encode box prompt
    box_tensor = torch.from_numpy(box).float().to(device)
    box_input = box_tensor.unsqueeze(0).unsqueeze(0)  # 1, 1, 4

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None, boxes=box_input, masks=None,
    )

    # Decode mask
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )

    # Upscale to original resolution
    masks = F.interpolate(
        low_res_masks, (1024, 1024),
        mode="bilinear", align_corners=False,
    )  # 1, N, 1024, 1024

    masks = masks.squeeze(0)  # N, 1024, 1024
    scores = iou_predictions.squeeze(0)  # N

    # Pick best mask
    best_idx = torch.argmax(scores).item()
    pred_mask = (torch.sigmoid(masks[best_idx]) > 0.5).cpu().numpy().astype(np.uint8)
    score = scores[best_idx].item()

    return pred_mask, score


def save_visualisation(image_np, gt_mask, pred_mask, box, dice_score,
                       organ_name, save_path):
    """Save a 3-panel side-by-side visualisation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Image + bounding box
    axes[0].imshow(image_np)
    rect = Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"Input + Box Prompt\n({organ_name})", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Ground truth mask
    axes[1].imshow(image_np)
    gt_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 200, 0, 140]  # green
    axes[1].imshow(gt_overlay)
    if gt_mask.any():
        axes[1].contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=[1.5])
    axes[1].set_title("Ground Truth", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Predicted mask
    axes[2].imshow(image_np)
    pred_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    pred_overlay[pred_mask > 0] = [220, 50, 50, 140]  # red
    axes[2].imshow(pred_overlay)
    if pred_mask.any():
        axes[2].contour(pred_mask, levels=[0.5], colors=["yellow"], linewidths=[1.5])
    rect2 = Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    )
    axes[2].add_patch(rect2)
    axes[2].set_title(f"Prediction  |  Dice: {dice_score:.4f}", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(f"MedSAM Box-Prompt Segmentation — {organ_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test MedSAM on BTCV with box prompts")
    parser.add_argument("--data_dir", default="data/btcv/processed",
                        help="Processed BTCV dataset directory")
    parser.add_argument("--split", default="test",
                        help="Which split to evaluate: train, val, or test")
    parser.add_argument("--checkpoint", default="Box_Prompt/best_medsam_btcv.pth",
                        help="MedSAM checkpoint to evaluate")
    parser.add_argument("--output_dir", default="Box_Prompt/test_outputs",
                        help="Directory to save visualisation images")
    parser.add_argument("--num_vis", type=int, default=10,
                        help="Number of samples to save as visualisation images")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Loading MedSAM from: {args.checkpoint}")
    print("=" * 60)
    sam_model = load_medsam(args.checkpoint, device)

    # ── Load test samples ─────────────────────────────────────────
    img_dir = os.path.join(args.data_dir, args.split, "images")
    lbl_dir = os.path.join(args.data_dir, args.split, "labels")

    npz_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npz")))
    print(f"\n  Found {len(npz_files)} test slices in '{args.split}' split\n")

    if len(npz_files) == 0:
        print("  [ERROR] No test data found. Run prepare_btcv_dataset.py first.")
        return

    # ── Evaluate ──────────────────────────────────────────────────
    organ_dices = defaultdict(list)
    all_dices = []
    vis_count = 0

    for idx, npz_path in enumerate(npz_files):
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        img_path = os.path.join(img_dir, f"{basename}.png")

        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found for {basename}, skipping")
            continue

        # Load image
        image_np = np.array(Image.open(img_path).convert("RGB"))

        # Load labels
        data = np.load(npz_path)
        organ_ids = data["organ_ids"]

        for organ_id in organ_ids:
            organ_id = int(organ_id)
            gt_mask = data[f"mask_{organ_id}"]
            box = data[f"box_{organ_id}"]

            organ_name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

            # Predict
            pred_mask, score = predict_mask(sam_model, image_np, box, device)

            # Dice
            dice = compute_dice(pred_mask, gt_mask)
            organ_dices[organ_id].append(dice)
            all_dices.append(dice)

            # Visualise first N
            if vis_count < args.num_vis:
                save_path = os.path.join(
                    args.output_dir,
                    f"vis_{vis_count:03d}_{basename}_{organ_name}.png"
                )
                save_visualisation(
                    image_np, gt_mask, pred_mask, box, dice,
                    organ_name, save_path
                )
                vis_count += 1

        if (idx + 1) % 10 == 0:
            running_dice = np.mean(all_dices) if all_dices else 0
            print(f"  [{idx+1}/{len(npz_files)}] "
                  f"Running mean Dice: {running_dice:.4f}")

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {args.split.upper()} split")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    print(f"  {'Organ':<25s}  {'Mean Dice':>10s}  {'Samples':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}")

    for organ_id in sorted(organ_dices.keys()):
        name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")
        dices = organ_dices[organ_id]
        mean_dice = np.mean(dices)
        print(f"  {name:<25s}  {mean_dice:>10.4f}  {len(dices):>8d}")

    overall_dice = np.mean(all_dices) if all_dices else 0
    print(f"\n  {'OVERALL':<25s}  {overall_dice:>10.4f}  {len(all_dices):>8d}")

    print(f"\n  Visualisations saved to: {args.output_dir}/")
    print(f"  ({vis_count} images saved)\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
