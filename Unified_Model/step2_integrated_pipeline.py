# =============================================================================
# step2_integrated_pipeline.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   UNIFIED pipeline that combines:
#              1. TotalSegmentator  → classifies WHICH organ is in the box
#              2. MedSAM (fine-tuned) → segments the organ boundary
#
#            Uses precomputed TotalSegmentator masks from step1.
#
# Run:       conda activate medsam_env
#            cd ~/Desktop/intern/Box_Prompt
#            python step2_integrated_pipeline.py
#            python step2_integrated_pipeline.py --num_samples 10
# =============================================================================

import os, sys, glob, argparse, random, json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything.build_sam import _build_sam

# =============================================================================
# CONFIG
# =============================================================================
PROCESSED_DIR = "btcv/processed"
CACHE_DIR     = "totalseg_cache"
OUTPUT_DIR    = "integrated_outputs"

ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "portal_vein", 11: "pancreas",
    12: "right_adrenal", 13: "left_adrenal",
}

# TotalSegmentator structure name → BTCV organ ID
TOTALSEG_TO_BTCV = {
    "spleen": 1, "kidney_right": 2, "kidney_left": 3,
    "gallbladder": 4, "esophagus": 5, "liver": 6,
    "stomach": 7, "aorta": 8, "inferior_vena_cava": 9,
    "portal_vein_and_splenic_vein": 10, "pancreas": 11,
    "adrenal_gland_right": 12, "adrenal_gland_left": 13,
}

ORGAN_COLORS = {
    1:"#E74C3C", 2:"#3498DB", 3:"#2ECC71", 4:"#F39C12",
    5:"#9B59B6", 6:"#1ABC9C", 7:"#E67E22", 8:"#E91E63",
    9:"#00BCD4", 10:"#8BC34A", 11:"#FF5722",
    12:"#607D8B", 13:"#795548",
}


def parse_args():
    p = argparse.ArgumentParser(description="Integrated MedSAM + TotalSegmentator pipeline")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default="best_medsam_btcv.pth")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of test samples to process")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# MedSAM — Load & Predict
# =============================================================================
def load_medsam(checkpoint_path, device="cpu"):
    """Load MedSAM model."""
    sam = _build_sam(
        encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sam.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        dice  = ckpt.get("best_dice", "?")
        print(f"    Loaded fine-tuned (epoch {epoch}, best_dice={dice})")
    else:
        sam.load_state_dict(ckpt)
        print(f"    Loaded raw pre-trained model")
    sam.to(device).eval()
    return sam


@torch.no_grad()
def medsam_predict(sam_model, image_np, box, device):
    """Run MedSAM box-prompt inference. Returns (mask, score)."""
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


# =============================================================================
# TotalSegmentator — Classify organ from precomputed masks
# =============================================================================
def classify_organ_totalseg(case_name, slice_idx, box, orig_shape):
    """
    Look up precomputed TotalSegmentator masks for this case,
    extract the slice, check which organ has the most voxels in the box.

    IMPORTANT: TotalSegmentator masks are in RAS space.
    The preprocessed PNGs are from the original LAS volume.
    LAS → RAS flips the x-axis, so we flip the mask slice horizontally.
    The masks are in original voxel space (e.g. 512×512), but the box
    coordinates are in resized 1024×1024 space, so we scale accordingly.
    """
    cache_path = os.path.join(CACHE_DIR, case_name)
    if not os.path.isdir(cache_path):
        return None, []

    orig_h, orig_w = orig_shape[0], orig_shape[1]
    scale_h = 1024.0 / orig_h
    scale_w = 1024.0 / orig_w

    # Box in 1024×1024 space
    bx1, by1, bx2, by2 = box
    # Convert box to original voxel space
    ox1 = int(bx1 / scale_w)
    oy1 = int(by1 / scale_h)
    ox2 = int(bx2 / scale_w)
    oy2 = int(by2 / scale_h)

    results = []
    for ts_name, btcv_id in TOTALSEG_TO_BTCV.items():
        seg_path = os.path.join(cache_path, f"{ts_name}.nii.gz")
        if not os.path.exists(seg_path):
            continue

        mask_arr = nib.load(seg_path).get_fdata()
        if slice_idx >= mask_arr.shape[2]:
            continue

        # Extract the slice in RAS space
        mask_slice = mask_arr[:, :, slice_idx]

        # Flip horizontally to match LAS (original) orientation
        mask_slice = np.flip(mask_slice, axis=0).copy()

        # Count voxels in the box region (in original voxel coords)
        box_region = mask_slice[oy1:oy2, ox1:ox2]
        vox = int(box_region.sum())
        if vox == 0:
            continue

        total_box = max(1, (oy2-oy1) * (ox2-ox1))
        pct = 100.0 * vox / total_box
        results.append({
            "btcv_id": btcv_id,
            "organ_name": ORGAN_NAMES.get(btcv_id, f"organ_{btcv_id}"),
            "ts_name": ts_name,
            "voxels": vox,
            "pct": round(pct, 2),
        })

    results.sort(key=lambda x: -x["voxels"])
    winner = results[0]["organ_name"] if results else "unknown"
    return winner, results


def dice_score(pred, gt):
    p, g = pred.astype(bool).flatten(), gt.astype(bool).flatten()
    if p.sum() + g.sum() == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


# =============================================================================
# Visualization — 4-panel figure
# =============================================================================
def save_result_figure(image_np, box, gt_mask, pred_mask, gt_organ, ts_organ,
                       ts_results, dice, score, save_path, slice_name):
    """
    4-panel figure:
      1. Input + Box + GT organ label
      2. TotalSegmentator classification (bar chart)
      3. Ground Truth mask
      4. MedSAM Prediction
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor("#0d0d0d")

    # ─── Panel 1: Input + Box ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d0d0d")
    ax.imshow(image_np)
    ax.add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=2.5, edgecolor="#00FF88", facecolor="#00FF88",
        alpha=0.1, linestyle="--"
    ))
    ax.set_title(f"Input + Box Prompt\nGT: {gt_organ}", color="white",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # ─── Panel 2: TotalSeg Classification ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    if ts_results:
        top = ts_results[:6]
        names = [r["organ_name"] for r in top]
        pcts  = [r["pct"]       for r in top]
        colors_bar = ["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B"
                      for i in range(len(top))]
        bars = ax2.barh(names[::-1], pcts[::-1], color=colors_bar[::-1], height=0.55)
        for bar, val in zip(bars, pcts[::-1]):
            ax2.text(min(val+0.5, max(pcts)*0.95 if max(pcts) > 0 else 1),
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="white", fontsize=8)
        match_str = "✓ MATCH" if ts_organ == gt_organ else f"✗ (GT={gt_organ})"
        ax2.set_title(f"TotalSeg: {ts_organ.upper()}\n{match_str}",
                      color="#4CAF50" if ts_organ == gt_organ else "#FF5252",
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("% of box", color="white", fontsize=8)
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#333")
        for lbl in ax2.get_xticklabels()+ax2.get_yticklabels():
            lbl.set_color("white")
    else:
        ax2.text(0.5, 0.5, "No TotalSeg\nmasks available",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="white", fontsize=11)
        ax2.axis("off")

    # ─── Panel 3: Ground Truth ────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0d0d0d")
    ax3.imshow(image_np)
    gt_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 200, 0, 140]
    ax3.imshow(gt_overlay)
    if gt_mask.any():
        ax3.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=[1.5])
    ax3.set_title(f"Ground Truth\n{gt_organ}", color="white",
                  fontsize=11, fontweight="bold")
    ax3.axis("off")

    # ─── Panel 4: MedSAM Prediction ──────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor("#0d0d0d")
    ax4.imshow(image_np)
    pred_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    pred_overlay[pred_mask > 0] = [220, 50, 50, 140]
    ax4.imshow(pred_overlay)
    if pred_mask.any():
        ax4.contour(pred_mask, levels=[0.5], colors=["yellow"], linewidths=[1.5])
    ax4.add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))
    ax4.set_title(f"MedSAM Prediction\nDice: {dice:.4f} | Score: {score:.3f}",
                  color="white", fontsize=11, fontweight="bold")
    ax4.axis("off")

    plt.suptitle(f"{slice_name}  |  TotalSeg → {ts_organ}  |  MedSAM Dice: {dice:.4f}",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  STEP 2 — Integrated Pipeline")
    print("  TotalSegmentator (classify) + MedSAM (segment)")
    print("="*60)

    # ── Load MedSAM ──────────────────────────────────────────────
    print(f"\n  Loading MedSAM from {args.checkpoint} ...")
    sam = load_medsam(args.checkpoint, device)

    # ── Find test samples ────────────────────────────────────────
    lbl_dir = os.path.join(PROCESSED_DIR, args.split, "labels")
    img_dir = os.path.join(PROCESSED_DIR, args.split, "images")
    npz_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npz")))

    if not npz_files:
        print(f"  ERROR: No test labels in {lbl_dir}")
        sys.exit(1)

    selected = random.sample(npz_files, min(args.num_samples, len(npz_files)))
    print(f"\n  Testing {len(selected)} samples from '{args.split}' split")
    print(f"  Results dir: {OUTPUT_DIR}/\n")

    # ── Check TotalSeg cache ─────────────────────────────────────
    if not os.path.isdir(CACHE_DIR):
        print(f"  WARNING: TotalSegmentator cache not found at {CACHE_DIR}/")
        print(f"           Run step1_precompute_totalseg.py first!")
        print(f"           Classification will show 'unknown' for all samples.\n")

    # ── Get original volume shapes (for coordinate mapping) ──────
    orig_shapes = {}
    raw_dir = "btcv/raw/data/kaggle_btcv/images"
    for f in os.listdir(raw_dir):
        if f.endswith(".nii"):
            case = f.replace(".nii", "")
            nii = nib.load(os.path.join(raw_dir, f))
            orig_shapes[case] = nii.shape  # (H, W, D)

    # ── Process each sample ──────────────────────────────────────
    all_dices = []
    all_classifications = []

    for i, npz_path in enumerate(selected, 1):
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        img_path = os.path.join(img_dir, f"{basename}.png")

        if not os.path.exists(img_path):
            print(f"  [{i}/{len(selected)}] {basename} — IMAGE NOT FOUND, skip")
            continue

        # Load image and labels
        image_np = np.array(Image.open(img_path).convert("RGB"))
        data = np.load(npz_path)
        organ_ids = data["organ_ids"]

        # Pick one organ (random)
        organ_id = int(random.choice(organ_ids))
        gt_mask = data[f"mask_{organ_id}"]
        box = data[f"box_{organ_id}"]
        gt_organ = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

        # Parse case name and slice index from filename
        # e.g. img0008-0002_s0067 → case=img0008-0002, slice=67
        parts = basename.rsplit("_s", 1)
        case_name = parts[0]
        slice_idx = int(parts[1]) if len(parts) > 1 else 0

        # Get original volume shape
        orig_shape = orig_shapes.get(case_name, (512, 512, 139))

        # ── CLASSIFY via TotalSegmentator ────────────────────────
        ts_organ, ts_results = classify_organ_totalseg(
            case_name, slice_idx, box, orig_shape
        )

        # ── SEGMENT via MedSAM ──────────────────────────────────
        pred_mask, score = medsam_predict(sam, image_np, box, device)
        dice = dice_score(pred_mask, gt_mask)

        all_dices.append(dice)
        is_match = (ts_organ == gt_organ)
        all_classifications.append(is_match)

        status = "✓" if is_match else "✗"
        print(f"  [{i}/{len(selected)}] {basename}")
        print(f"    GT organ     : {gt_organ}")
        print(f"    TotalSeg     : {ts_organ} {status}")
        print(f"    MedSAM Dice  : {dice:.4f}  (score: {score:.3f})")

        # ── Save 4-panel figure ─────────────────────────────────
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{i:02d}_{basename}_{gt_organ}_result.png"
        )
        save_result_figure(
            image_np, box, gt_mask, pred_mask,
            gt_organ, ts_organ, ts_results,
            dice, score, save_path, basename
        )
        print(f"    Saved        : {save_path}\n")

    # ── Summary ───────────────────────────────────────────────────
    mean_dice = np.mean(all_dices) if all_dices else 0
    n_correct = sum(all_classifications)
    n_total   = len(all_classifications)
    cls_acc   = 100.0 * n_correct / n_total if n_total > 0 else 0

    print("="*60)
    print("  INTEGRATED PIPELINE RESULTS")
    print("="*60)
    print(f"  Samples tested        : {n_total}")
    print(f"  MedSAM Mean Dice      : {mean_dice:.4f}")
    if all_dices:
        print(f"  MedSAM Min/Max Dice   : {min(all_dices):.4f} / {max(all_dices):.4f}")
    print(f"  TotalSeg Classification")
    print(f"    Accuracy            : {cls_acc:.1f}% ({n_correct}/{n_total})")
    print(f"  Output dir            : {OUTPUT_DIR}/")
    print("="*60 + "\n")

    # Save JSON summary
    summary = {
        "num_samples": n_total,
        "medsam_mean_dice": round(float(mean_dice), 4),
        "totalseg_accuracy": round(cls_acc, 1),
        "totalseg_correct": n_correct,
        "totalseg_total": n_total,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
