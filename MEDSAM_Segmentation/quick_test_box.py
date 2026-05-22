"""
quick_test_box.py
=================
Quick visual test: pick a few BTCV test samples, run box-prompt segmentation
with the fine-tuned MedSAM, and SAVE side-by-side images.

Processes only N samples (default 5) so it finishes in minutes on CPU.

Usage:
    python Box_Prompt/quick_test_box.py
    python Box_Prompt/quick_test_box.py --num_samples 10 --checkpoint Box_Prompt/best_medsam_btcv.pth
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
matplotlib.use("Agg")  # Non-interactive — saves to file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything.build_sam import _build_sam


# ─── BTCV Organ Names ────────────────────────────────────────────────────────
ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "portal_vein", 11: "pancreas",
    12: "right_adrenal", 13: "left_adrenal",
}


def load_medsam(checkpoint_path, device="cpu"):
    """Load MedSAM, supporting both raw state_dict and full checkpoint."""
    sam = _build_sam(
        encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sam.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        dice = ckpt.get("best_dice", "?")
        print(f"  Loaded fine-tuned model (epoch {epoch}, best_dice={dice})")
    else:
        sam.load_state_dict(ckpt)
        print(f"  Loaded raw pre-trained model")
    sam.to(device).eval()
    return sam


@torch.no_grad()
def predict_mask(sam_model, image_np, box, device):
    """Run MedSAM inference with a box prompt. Returns (mask, score)."""
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


def save_result(image_np, gt_mask, pred_mask, box, dice, organ_name, score, save_path):
    """Save a 3-panel figure: Input+Box | Ground Truth | Prediction."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Input + Bounding Box
    ax1.imshow(image_np)
    ax1.add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=2.5, edgecolor="cyan", facecolor="cyan", alpha=0.1, linestyle="--"
    ))
    ax1.set_title(f"Input + Box Prompt\n({organ_name})", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Ground Truth
    ax2.imshow(image_np)
    gt_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 200, 0, 140]
    ax2.imshow(gt_overlay)
    if gt_mask.any():
        ax2.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=[2])
    ax2.set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Prediction
    ax3.imshow(image_np)
    pred_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    pred_overlay[pred_mask > 0] = [220, 50, 50, 140]
    ax3.imshow(pred_overlay)
    if pred_mask.any():
        ax3.contour(pred_mask, levels=[0.5], colors=["yellow"], linewidths=[2])
    ax3.add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))
    ax3.set_title(f"Prediction  |  Dice: {dice:.4f}  |  Score: {score:.3f}",
                  fontsize=13, fontweight="bold")
    ax3.axis("off")

    plt.suptitle(f"MedSAM Box-Prompt Segmentation — {organ_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quick visual test of box-prompt segmentation")
    parser.add_argument("--data_dir", default="data/btcv/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="Box_Prompt/best_medsam_btcv.pth")
    parser.add_argument("--output_dir", default="Box_Prompt/quick_test_outputs")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of test samples to process (default: 5)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Quick Box-Prompt Test")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)
    sam = load_medsam(args.checkpoint, device)

    # ── Pick random test samples ──────────────────────────────────
    lbl_dir = os.path.join(args.data_dir, args.split, "labels")
    img_dir = os.path.join(args.data_dir, args.split, "images")
    npz_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npz")))

    if not npz_files:
        print("  [ERROR] No test data found!")
        return

    # Pick random slices (but deterministic with seed)
    selected = random.sample(npz_files, min(args.num_samples, len(npz_files)))

    print(f"\n  Testing {len(selected)} samples from '{args.split}' split")
    print(f"  Saving results to: {args.output_dir}/\n")

    all_dices = []
    img_count = 0

    for i, npz_path in enumerate(selected):
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        img_path = os.path.join(img_dir, f"{basename}.png")

        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found: {basename}")
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        data = np.load(npz_path)
        organ_ids = data["organ_ids"]

        # Pick one random organ from this slice
        organ_id = int(random.choice(organ_ids))
        gt_mask = data[f"mask_{organ_id}"]
        box = data[f"box_{organ_id}"]
        organ_name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

        print(f"  [{i+1}/{len(selected)}] {basename} — {organ_name} ...", end=" ", flush=True)

        pred_mask, score = predict_mask(sam, image_np, box, device)
        dice = dice_score(pred_mask, gt_mask)
        all_dices.append(dice)

        save_path = os.path.join(
            args.output_dir,
            f"test_{img_count:02d}_{basename}_{organ_name}.png"
        )
        save_result(image_np, gt_mask, pred_mask, box, dice, organ_name, score, save_path)
        img_count += 1

        print(f"Dice: {dice:.4f}  Score: {score:.3f}  ✓ Saved")

    # ── Summary ───────────────────────────────────────────────────
    mean_dice = np.mean(all_dices) if all_dices else 0
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"  Samples tested : {len(all_dices)}")
    print(f"  Mean Dice      : {mean_dice:.4f}")
    print(f"  Min Dice       : {min(all_dices):.4f}" if all_dices else "")
    print(f"  Max Dice       : {max(all_dices):.4f}" if all_dices else "")
    print(f"  Images saved to: {args.output_dir}/")
    print(f"{'=' * 60}\n")

    # List saved files
    print("  Saved files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            fpath = os.path.join(args.output_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    📷 {f}  ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
