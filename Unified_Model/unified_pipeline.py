# =============================================================================
# unified_pipeline.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   UNIFIED 3-MODEL pipeline that combines:
#              1. SegResNet        → liver/tumor DETECTION (MONAI)
#              2. TotalSegmentator → organ CLASSIFICATION (precomputed masks)
#              3. MedSAM           → box-prompt SEGMENTATION (fine-tuned)
#
# Run:       conda activate medsam_env
#            cd ~/Desktop/intern/Box_Prompt
#            python unified_pipeline.py
#            python unified_pipeline.py --num_samples 10 --device cuda
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
PROCESSED_DIR   = "btcv/processed"
CACHE_DIR       = "totalseg_cache"
OUTPUT_DIR      = "unified_outputs"
SEGRESNET_MODEL = "SegResNet_Project/best_segresnet_model.pth"
SEGRESNET_CFG   = "SegResNet_Project/model_config.json"

ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "portal_vein", 11: "pancreas",
    12: "right_adrenal", 13: "left_adrenal",
}

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
    p = argparse.ArgumentParser(
        description="Unified Pipeline: SegResNet + TotalSegmentator + MedSAM"
    )
    p.add_argument("--split", default="test")
    p.add_argument("--medsam_checkpoint", default="best_medsam_btcv.pth")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of test samples to process")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# MODEL 1 — SegResNet (MONAI) for Liver + Tumor Detection
# =============================================================================
def load_segresnet(model_path, config_path, device):
    """Load SegResNet model using MONAI."""
    from monai.networks.nets import SegResNet

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=cfg["NUM_CLASSES"],
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"    SegResNet loaded: {cfg['NUM_CLASSES']} classes, "
          f"img_size={cfg['IMAGE_SIZE']}")
    return model, cfg


@torch.no_grad()
def segresnet_predict(model, image_np, cfg, device):
    """
    Run SegResNet inference on a 2D CT slice.
    Input:  RGB image (H, W, 3) in 0-255 range
    Output: (pred_mask, tumor_info)
      pred_mask: (H, W) with 0=bg, 1=liver, 2=tumor
      tumor_info: dict with pixel counts and flags
    """
    img_size = cfg["IMAGE_SIZE"]  # 128

    # Convert RGB to grayscale (single channel CT-like)
    gray = np.mean(image_np, axis=2).astype(np.float32)
    orig_h, orig_w = gray.shape

    # Resize to model input size
    gray_pil = Image.fromarray(gray).resize((img_size, img_size), Image.BILINEAR)
    gray_resized = np.array(gray_pil, dtype=np.float32)

    # Normalize to [0, 1]
    if gray_resized.max() > 0:
        gray_resized = gray_resized / gray_resized.max()

    # To tensor: (1, 1, H, W)
    inp = torch.from_numpy(gray_resized).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    logits = model(inp)  # (1, 3, 128, 128)

    # Upsample to original size
    logits_up = F.interpolate(logits, size=(orig_h, orig_w),
                              mode="bilinear", align_corners=False)
    pred = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Compute tumor info
    total_pixels = pred.size
    liver_pixels = int((pred == 1).sum())
    tumor_pixels = int((pred == 2).sum())
    tumor_pct = 100.0 * tumor_pixels / total_pixels if total_pixels > 0 else 0

    tumor_info = {
        "liver_pixels": liver_pixels,
        "tumor_pixels": tumor_pixels,
        "tumor_percentage": round(tumor_pct, 4),
        "tumor_detected": tumor_pixels > 0,
        "total_pixels": total_pixels,
    }

    return pred, tumor_info


# =============================================================================
# MODEL 2 — MedSAM for Box-Prompt Segmentation
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
        print(f"    MedSAM loaded: fine-tuned (epoch {epoch}, best_dice={dice})")
    else:
        sam.load_state_dict(ckpt)
        print(f"    MedSAM loaded: raw pre-trained model")
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
# MODEL 3 — TotalSegmentator Organ Classification (precomputed masks)
# =============================================================================
def classify_organ_totalseg(case_name, slice_idx, box, orig_shape):
    """
    Look up precomputed TotalSegmentator masks for this case,
    extract the slice, check which organ has the most voxels in the box.

    Because TotalSeg --fast mode resamples to 3mm and back, organ boundaries
    can shift by ±15-20 slices. We search a window around the target slice
    and pick the slice with maximum organ overlap for each organ.

    Handles LAS→RAS x-axis flip and 1024→native coordinate scaling.
    """
    cache_path = os.path.join(CACHE_DIR, case_name)
    if not os.path.isdir(cache_path):
        return None, []

    orig_h, orig_w = orig_shape[0], orig_shape[1]
    scale_h = 1024.0 / orig_h
    scale_w = 1024.0 / orig_w

    bx1, by1, bx2, by2 = box
    ox1 = int(bx1 / scale_w)
    oy1 = int(by1 / scale_h)
    ox2 = int(bx2 / scale_w)
    oy2 = int(by2 / scale_h)

    SEARCH_RADIUS = 20  # search ±20 slices to handle fast-mode offset

    results = []
    for ts_name, btcv_id in TOTALSEG_TO_BTCV.items():
        seg_path = os.path.join(cache_path, f"{ts_name}.nii.gz")
        if not os.path.exists(seg_path):
            continue

        mask_arr = nib.load(seg_path).get_fdata()
        max_z = mask_arr.shape[2]

        # Search window around target slice for best overlap
        best_vox = 0
        z_start = max(0, slice_idx - SEARCH_RADIUS)
        z_end = min(max_z, slice_idx + SEARCH_RADIUS + 1)

        for z in range(z_start, z_end):
            mask_slice = mask_arr[:, :, z]
            # Flip x-axis: LAS→RAS
            mask_slice = np.flip(mask_slice, axis=0).copy()

            box_region = mask_slice[oy1:oy2, ox1:ox2]
            vox = int(box_region.sum())
            if vox > best_vox:
                best_vox = vox

        if best_vox == 0:
            continue

        total_box = max(1, (oy2 - oy1) * (ox2 - ox1))
        pct = 100.0 * best_vox / total_box
        results.append({
            "btcv_id": btcv_id,
            "organ_name": ORGAN_NAMES.get(btcv_id, f"organ_{btcv_id}"),
            "ts_name": ts_name,
            "voxels": best_vox,
            "pct": round(pct, 2),
        })

    results.sort(key=lambda x: -x["voxels"])
    winner = results[0]["organ_name"] if results else "unknown"
    return winner, results


# =============================================================================
# METRICS
# =============================================================================
def dice_score(pred, gt):
    p, g = pred.astype(bool).flatten(), gt.astype(bool).flatten()
    if p.sum() + g.sum() == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


# =============================================================================
# VISUALIZATION — 5-Panel Figure
# =============================================================================
def save_result_figure(image_np, box, gt_mask, pred_mask, segresnet_mask,
                       tumor_info, gt_organ, ts_organ, ts_results,
                       medsam_dice, medsam_score, save_path, slice_name):
    """
    5-panel figure:
      1. Input + Box + GT organ label
      2. SegResNet Tumor Detection (liver=gold, tumor=red)
      3. TotalSegmentator Classification (bar chart)
      4. MedSAM Prediction with boundary
      5. Ground Truth + Metrics Summary
    """
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
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
                 fontsize=10, fontweight="bold")
    ax.axis("off")

    # ─── Panel 2: SegResNet Tumor Detection ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d0d")
    ax2.imshow(image_np)

    # Overlay: liver = gold, tumor = red
    seg_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    seg_overlay[segresnet_mask == 1] = [255, 200, 50, 120]   # liver = gold
    seg_overlay[segresnet_mask == 2] = [255, 30, 30, 180]    # tumor = red
    ax2.imshow(seg_overlay)

    # Draw contours for liver and tumor
    if (segresnet_mask == 1).any():
        ax2.contour(segresnet_mask == 1, levels=[0.5], colors=["gold"],
                    linewidths=[1.0])
    if (segresnet_mask == 2).any():
        ax2.contour(segresnet_mask == 2, levels=[0.5], colors=["red"],
                    linewidths=[1.5])

    tumor_flag = "⚠ TUMOR DETECTED" if tumor_info["tumor_detected"] else "✓ No Tumor"
    tumor_color = "#FF5252" if tumor_info["tumor_detected"] else "#4CAF50"
    ax2.set_title(f"SegResNet Detection\n{tumor_flag}\n"
                  f"Liver: {tumor_info['liver_pixels']}px | "
                  f"Tumor: {tumor_info['tumor_pixels']}px",
                  color=tumor_color, fontsize=9, fontweight="bold")
    ax2.axis("off")

    # ─── Panel 3: TotalSeg Classification ─────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#1a1a2e")
    if ts_results:
        top = ts_results[:6]
        names = [r["organ_name"] for r in top]
        pcts  = [r["pct"]       for r in top]
        colors_bar = ["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B"
                      for i in range(len(top))]
        bars = ax3.barh(names[::-1], pcts[::-1], color=colors_bar[::-1], height=0.55)
        for bar, val in zip(bars, pcts[::-1]):
            ax3.text(min(val+0.5, max(pcts)*0.95 if max(pcts) > 0 else 1),
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="white", fontsize=8)
        match_str = "✓ MATCH" if ts_organ == gt_organ else f"✗ (GT={gt_organ})"
        ax3.set_title(f"TotalSeg: {ts_organ.upper()}\n{match_str}",
                      color="#4CAF50" if ts_organ == gt_organ else "#FF5252",
                      fontsize=10, fontweight="bold")
        ax3.set_xlabel("% of box", color="white", fontsize=8)
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("#333")
        for lbl in ax3.get_xticklabels()+ax3.get_yticklabels():
            lbl.set_color("white")
    else:
        ax3.text(0.5, 0.5, "No TotalSeg\nmasks available\n\nRun step1 to\npopulate cache",
                 transform=ax3.transAxes, ha="center", va="center",
                 color="#888", fontsize=10)
        ax3.set_title("TotalSeg: N/A", color="#888",
                      fontsize=10, fontweight="bold")
        ax3.axis("off")

    # ─── Panel 4: MedSAM Prediction ──────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor("#0d0d0d")
    ax4.imshow(image_np)
    pred_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    pred_overlay[pred_mask > 0] = [50, 150, 255, 140]   # blue for MedSAM
    ax4.imshow(pred_overlay)
    if pred_mask.any():
        ax4.contour(pred_mask, levels=[0.5], colors=["cyan"], linewidths=[1.5])
    ax4.add_patch(Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
    ))
    ax4.set_title(f"MedSAM Segmentation\nDice: {medsam_dice:.4f} | "
                  f"Score: {medsam_score:.3f}",
                  color="white", fontsize=10, fontweight="bold")
    ax4.axis("off")

    # ─── Panel 5: Ground Truth + Summary ──────────────────────────────────────
    ax5 = axes[4]
    ax5.set_facecolor("#0d0d0d")
    ax5.imshow(image_np)
    gt_overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 220, 0, 140]
    ax5.imshow(gt_overlay)
    if gt_mask.any():
        ax5.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=[1.5])
    ax5.set_title(f"Ground Truth\n{gt_organ}", color="white",
                  fontsize=10, fontweight="bold")
    ax5.axis("off")

    # ─── Super Title ──────────────────────────────────────────────────────────
    ts_label = ts_organ if ts_organ else "N/A"
    plt.suptitle(
        f"UNIFIED PIPELINE  |  {slice_name}\n"
        f"SegResNet: {'TUMOR' if tumor_info['tumor_detected'] else 'CLEAN'}  |  "
        f"TotalSeg: {ts_label}  |  "
        f"MedSAM Dice: {medsam_dice:.4f}",
        color="white", fontsize=12, fontweight="bold", y=1.04
    )
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

    print("\n" + "=" * 70)
    print("  UNIFIED 3-MODEL PIPELINE")
    print("  SegResNet (tumor) + TotalSegmentator (organ) + MedSAM (segment)")
    print("=" * 70)

    # ── Load Model 1: SegResNet ──────────────────────────────────
    print(f"\n  [1/3] Loading SegResNet from {SEGRESNET_MODEL} ...")
    segresnet, seg_cfg = load_segresnet(SEGRESNET_MODEL, SEGRESNET_CFG, device)

    # ── Load Model 2: MedSAM ────────────────────────────────────
    print(f"  [2/3] Loading MedSAM from {args.medsam_checkpoint} ...")
    medsam = load_medsam(args.medsam_checkpoint, device)

    # ── Model 3: TotalSegmentator (precomputed masks) ────────────
    print(f"  [3/3] TotalSegmentator masks: {CACHE_DIR}/")
    if os.path.isdir(CACHE_DIR):
        cached_cases = [d for d in os.listdir(CACHE_DIR)
                        if os.path.isdir(os.path.join(CACHE_DIR, d))]
        print(f"    Cached cases: {len(cached_cases)}")
    else:
        print(f"    WARNING: Cache not found — organ classification unavailable")
        print(f"    Run: python step1_precompute_totalseg.py")

    # ── Find test samples ────────────────────────────────────────
    lbl_dir = os.path.join(PROCESSED_DIR, args.split, "labels")
    img_dir = os.path.join(PROCESSED_DIR, args.split, "images")
    npz_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npz")))

    if not npz_files:
        print(f"\n  ERROR: No test labels in {lbl_dir}")
        sys.exit(1)

    selected = random.sample(npz_files, min(args.num_samples, len(npz_files)))
    print(f"\n  Testing {len(selected)} samples from '{args.split}' split")
    print(f"  Output dir: {OUTPUT_DIR}/\n")
    print("-" * 70)

    # ── Get original volume shapes ───────────────────────────────
    orig_shapes = {}
    raw_dir = "btcv/raw/data/kaggle_btcv/images"
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".nii"):
                case = f.replace(".nii", "")
                nii = nib.load(os.path.join(raw_dir, f))
                orig_shapes[case] = nii.shape

    # ── Process each sample ──────────────────────────────────────
    all_medsam_dices = []
    all_classifications = []
    all_tumor_flags = []
    sample_reports = []

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

        # Pick one organ
        organ_id = int(random.choice(organ_ids))
        gt_mask = data[f"mask_{organ_id}"]
        box = data[f"box_{organ_id}"]
        gt_organ = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

        # Parse case name and slice index
        parts = basename.rsplit("_s", 1)
        case_name = parts[0]
        slice_idx = int(parts[1]) if len(parts) > 1 else 0
        orig_shape = orig_shapes.get(case_name, (512, 512, 139))

        print(f"\n  [{i}/{len(selected)}] {basename}")
        print(f"    GT organ      : {gt_organ} (id={organ_id})")

        # ── MODEL 1: SegResNet Tumor Detection ──────────────────
        segresnet_mask, tumor_info = segresnet_predict(
            segresnet, image_np, seg_cfg, device
        )
        tumor_flag = "⚠ TUMOR" if tumor_info["tumor_detected"] else "✓ CLEAN"
        print(f"    SegResNet      : {tumor_flag} "
              f"(liver={tumor_info['liver_pixels']}px, "
              f"tumor={tumor_info['tumor_pixels']}px)")
        all_tumor_flags.append(tumor_info["tumor_detected"])

        # ── MODEL 2: TotalSegmentator Classification ────────────
        ts_organ, ts_results = classify_organ_totalseg(
            case_name, slice_idx, box, orig_shape
        )
        ts_organ = ts_organ or "unknown"
        is_match = (ts_organ == gt_organ)
        all_classifications.append(is_match)
        print(f"    TotalSeg       : {ts_organ} "
              f"{'✓' if is_match else '✗'}")

        # ── MODEL 3: MedSAM Segmentation ───────────────────────
        pred_mask, medsam_score = medsam_predict(medsam, image_np, box, device)
        medsam_dice = dice_score(pred_mask, gt_mask)
        all_medsam_dices.append(medsam_dice)
        print(f"    MedSAM         : Dice={medsam_dice:.4f} "
              f"(score={medsam_score:.3f})")

        # ── Save 5-panel figure ─────────────────────────────────
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{i:02d}_{basename}_{gt_organ}_unified.png"
        )
        save_result_figure(
            image_np, box, gt_mask, pred_mask, segresnet_mask,
            tumor_info, gt_organ, ts_organ, ts_results,
            medsam_dice, medsam_score, save_path, basename
        )
        print(f"    Saved          : {save_path}")

        # ── Per-sample report ───────────────────────────────────
        sample_reports.append({
            "slice": basename,
            "gt_organ": gt_organ,
            "segresnet": tumor_info,
            "totalseg_organ": ts_organ,
            "totalseg_match": is_match,
            "medsam_dice": round(float(medsam_dice), 4),
            "medsam_score": round(float(medsam_score), 4),
        })

    # ── SUMMARY ───────────────────────────────────────────────────
    mean_dice = np.mean(all_medsam_dices) if all_medsam_dices else 0
    n_correct = sum(all_classifications)
    n_total   = len(all_classifications)
    cls_acc   = 100.0 * n_correct / n_total if n_total > 0 else 0
    n_tumors  = sum(all_tumor_flags)

    print("\n" + "=" * 70)
    print("  UNIFIED PIPELINE RESULTS")
    print("=" * 70)
    print(f"  Samples tested          : {n_total}")
    print(f"")
    print(f"  MODEL 1 — SegResNet (Tumor Detection)")
    print(f"    Tumor flagged         : {n_tumors}/{n_total} samples")
    print(f"")
    print(f"  MODEL 2 — TotalSegmentator (Organ Classification)")
    print(f"    Classification acc    : {cls_acc:.1f}% ({n_correct}/{n_total})")
    print(f"")
    print(f"  MODEL 3 — MedSAM (Box-Prompt Segmentation)")
    print(f"    Mean Dice             : {mean_dice:.4f}")
    if all_medsam_dices:
        print(f"    Min/Max Dice          : "
              f"{min(all_medsam_dices):.4f} / {max(all_medsam_dices):.4f}")
    print(f"")
    print(f"  Output dir              : {OUTPUT_DIR}/")
    print("=" * 70 + "\n")

    # ── Save JSON summary ─────────────────────────────────────────
    summary = {
        "num_samples": n_total,
        "segresnet_tumors_detected": n_tumors,
        "totalseg_accuracy": round(cls_acc, 1),
        "totalseg_correct": n_correct,
        "medsam_mean_dice": round(float(mean_dice), 4),
        "medsam_min_dice": round(float(min(all_medsam_dices)), 4) if all_medsam_dices else 0,
        "medsam_max_dice": round(float(max(all_medsam_dices)), 4) if all_medsam_dices else 0,
        "samples": sample_reports,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
