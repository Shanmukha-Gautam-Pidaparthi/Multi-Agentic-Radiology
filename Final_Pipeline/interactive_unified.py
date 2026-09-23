# =============================================================================
# interactive_unified.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   INTERACTIVE version of the unified 3-model pipeline.
#            Draw a box on a CT slice → see all 3 models' results at once:
#              1. SegResNet        → liver/tumor DETECTION
#              2. TotalSegmentator → organ CLASSIFICATION
#              3. MedSAM           → box-prompt SEGMENTATION
#
# Run:       conda activate medsam_env
#            cd ~/Desktop/intern/Box_Prompt
#            python interactive_unified.py
#            python interactive_unified.py --image btcv/processed/test/images/img0009-0002_s0100.png
# =============================================================================

import matplotlib
matplotlib.use('TkAgg')

import os, sys, argparse, glob, json
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from PIL import Image
from segment_anything.build_sam import _build_sam

# =============================================================================
# CONFIG
# =============================================================================
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


def parse_args():
    p = argparse.ArgumentParser(description="Interactive Unified 3-Model Pipeline")
    p.add_argument("--checkpoint", default="best_medsam_btcv.pth")
    p.add_argument("--image", default=None)
    p.add_argument("--save-dir", default="unified_interactive_outputs")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# =============================================================================
# MODEL 1 — SegResNet
# =============================================================================
def load_segresnet(model_path, config_path, device):
    from monai.networks.nets import SegResNet
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = SegResNet(
        spatial_dims=2, in_channels=1, out_channels=cfg["NUM_CLASSES"],
        init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"    SegResNet loaded: {cfg['NUM_CLASSES']} classes")
    return model, cfg


@torch.no_grad()
def segresnet_predict(model, image_np, cfg, device):
    img_size = cfg["IMAGE_SIZE"]
    gray = np.mean(image_np, axis=2).astype(np.float32)
    orig_h, orig_w = gray.shape
    gray_pil = Image.fromarray(gray).resize((img_size, img_size), Image.BILINEAR)
    gray_resized = np.array(gray_pil, dtype=np.float32)
    if gray_resized.max() > 0:
        gray_resized = gray_resized / gray_resized.max()
    inp = torch.from_numpy(gray_resized).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(inp)
    logits_up = F.interpolate(logits, size=(orig_h, orig_w),
                              mode="bilinear", align_corners=False)
    pred = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    liver_px = int((pred == 1).sum())
    tumor_px = int((pred == 2).sum())
    return pred, {
        "liver_pixels": liver_px, "tumor_pixels": tumor_px,
        "tumor_detected": tumor_px > 0,
    }


# =============================================================================
# MODEL 2 — MedSAM
# =============================================================================
def load_medsam(checkpoint_path, device):
    sam = _build_sam(
        encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sam.load_state_dict(ckpt["model_state_dict"])
        print(f"    MedSAM fine-tuned (epoch {ckpt.get('epoch','?')}, dice={ckpt.get('best_dice','?')})")
    else:
        sam.load_state_dict(ckpt)
        print(f"    MedSAM pre-trained")
    sam.to(device).eval()
    return sam


# =============================================================================
# MODEL 3 — TotalSegmentator (precomputed masks)
# =============================================================================
def load_totalseg_masks(seg_dir):
    masks = {}
    if seg_dir is None or not os.path.isdir(seg_dir):
        return masks
    for ts_name, btcv_id in TOTALSEG_TO_BTCV.items():
        path = os.path.join(seg_dir, f"{ts_name}.nii.gz")
        if os.path.exists(path):
            arr = nib.load(path).get_fdata()
            if arr.sum() > 0:
                masks[btcv_id] = arr
    return masks


def classify_box(ts_masks, slice_idx, box, orig_shape, img_shape):
    if not ts_masks or slice_idx is None:
        return "unknown", []
    orig_h, orig_w = orig_shape[:2]
    scale_h = img_shape[0] / orig_h
    scale_w = img_shape[1] / orig_w
    bx1, by1, bx2, by2 = [int(b) for b in box]
    ox1, oy1 = int(bx1 / scale_w), int(by1 / scale_h)
    ox2, oy2 = int(bx2 / scale_w), int(by2 / scale_h)

    SEARCH_RADIUS = 20
    results = []
    for btcv_id, mask_3d in ts_masks.items():
        max_z = mask_3d.shape[2]
        best_vox = 0
        for z in range(max(0, slice_idx - SEARCH_RADIUS),
                       min(max_z, slice_idx + SEARCH_RADIUS + 1)):
            ms = np.flip(mask_3d[:, :, z], axis=0).copy()
            vox = int(ms[oy1:oy2, ox1:ox2].sum())
            if vox > best_vox:
                best_vox = vox
        if best_vox == 0:
            continue
        total_box = max(1, (oy2 - oy1) * (ox2 - ox1))
        pct = 100.0 * best_vox / total_box
        results.append({
            "btcv_id": btcv_id,
            "name": ORGAN_NAMES.get(btcv_id, f"organ_{btcv_id}"),
            "voxels": best_vox, "pct": round(pct, 2),
        })
    results.sort(key=lambda x: -x["voxels"])
    winner = results[0]["name"] if results else "unknown"
    return winner, results


# =============================================================================
# Helpers
# =============================================================================
def auto_pick_image():
    for d in ["btcv/processed/test/images"]:
        if os.path.isdir(d):
            pngs = sorted([p for p in glob.glob(os.path.join(d, "*.png"))
                           if "annotated" not in p])
            if pngs:
                return pngs[len(pngs) // 2]
    return None


def infer_case_and_slice(image_path):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    if "_s" in basename:
        parts = basename.rsplit("_s", 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return parts[0], None
    return None, None


def find_seg_dir(case_name):
    for d in [os.path.join("totalseg_cache", case_name), "totalseg_test_output"]:
        if os.path.isdir(d):
            masks = [f for f in os.listdir(d)
                     if f.endswith(".nii.gz") and f != "input_RAS.nii.gz"]
            if len(masks) > 5:
                return d
    return None


def get_orig_shape(case_name):
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join("btcv/raw/data/kaggle_btcv/images", case_name + ext)
        if os.path.exists(p):
            return nib.load(p).shape
    return (512, 512, 139)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  INTERACTIVE UNIFIED 3-MODEL PIPELINE")
    print("  SegResNet (tumor) + TotalSeg (organ) + MedSAM (segment)")
    print("  Draw a box on the left panel → see all results instantly")
    print("=" * 65)

    # ── Pick image ────────────────────────────────────────────
    image_path = args.image or auto_pick_image()
    if image_path is None or not os.path.exists(image_path):
        print(f"  ERROR: No image found. Use --image <path>")
        sys.exit(1)
    print(f"\n  Image: {image_path}")

    case_name, slice_idx = infer_case_and_slice(image_path)
    print(f"  Case : {case_name or 'unknown'}")
    print(f"  Slice: {slice_idx if slice_idx is not None else 'unknown'}")

    # ── Load 3 models ─────────────────────────────────────────
    print(f"\n  [1/3] Loading SegResNet ...")
    segresnet, seg_cfg = load_segresnet(SEGRESNET_MODEL, SEGRESNET_CFG, device)

    print(f"  [2/3] Loading MedSAM ...")
    sam = load_medsam(args.checkpoint, device)

    print(f"  [3/3] Loading TotalSeg masks ...")
    seg_dir = find_seg_dir(case_name) if case_name else None
    ts_masks = load_totalseg_masks(seg_dir) if seg_dir else {}
    print(f"    {'Loaded ' + str(len(ts_masks)) + ' organ masks' if ts_masks else 'No masks found (run step1_precompute_totalseg.py)'}")

    orig_shape = get_orig_shape(case_name) if case_name else (512, 512, 139)

    # ── Load image + compute embedding ────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)
    img_h, img_w = image_rgb.shape[:2]
    print(f"\n  Image shape: {image_rgb.shape}")
    print("  Computing MedSAM embedding (~30s on CPU) ...")

    with torch.no_grad():
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).to(device)
        input_image = sam.preprocess(img_tensor).unsqueeze(0)
        image_embedding = sam.image_encoder(input_image)
    print("  Embedding ready!")

    # ── Run SegResNet once on full image ──────────────────────
    print("  Running SegResNet on full slice ...")
    segresnet_mask, tumor_info = segresnet_predict(segresnet, image_rgb, seg_cfg, device)
    tumor_flag = "⚠ TUMOR DETECTED" if tumor_info["tumor_detected"] else "✓ No Tumor"
    print(f"  SegResNet: {tumor_flag} (liver={tumor_info['liver_pixels']}px, tumor={tumor_info['tumor_pixels']}px)")

    scale_x = 1024.0 / img_w
    scale_y = 1024.0 / img_h

    @torch.no_grad()
    def predict_mask(box_np):
        box_1024 = box_np.copy().astype(np.float32)
        box_1024[0] *= scale_x
        box_1024[1] *= scale_y
        box_1024[2] *= scale_x
        box_1024[3] *= scale_y
        box_t = torch.from_numpy(box_1024).float().to(device).unsqueeze(0).unsqueeze(0)
        sparse, dense = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
        low_res, iou = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        masks = F.interpolate(low_res, (1024, 1024),
                              mode="bilinear", align_corners=False).squeeze(0)
        scores = iou.squeeze(0)
        best = torch.argmax(scores).item()
        pred_1024 = (torch.sigmoid(masks[best]) > 0.5).cpu().numpy().astype(np.uint8)
        pred_resized = np.array(
            Image.fromarray(pred_1024 * 255).resize((img_w, img_h), Image.NEAREST)
        ) > 127
        return pred_resized.astype(np.uint8), scores[best].item()

    # ── Setup 4-panel figure ──────────────────────────────────
    fig, (ax_input, ax_tumor, ax_class, ax_seg) = plt.subplots(1, 4, figsize=(26, 7))
    fig.patch.set_facecolor("#0d0d0d")

    # Panel 1: Input
    ax_input.imshow(image_rgb)
    ax_input.set_title("Draw a box (click + drag)", color="white", fontsize=12)
    ax_input.axis("off"); ax_input.set_facecolor("#0d0d0d")

    # Panel 2: SegResNet (static — pre-computed)
    ax_tumor.imshow(image_rgb)
    seg_ov = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    seg_ov[segresnet_mask == 1] = [255, 200, 50, 120]
    seg_ov[segresnet_mask == 2] = [255, 30, 30, 180]
    ax_tumor.imshow(seg_ov)
    if (segresnet_mask == 1).any():
        ax_tumor.contour(segresnet_mask == 1, levels=[0.5], colors=["gold"], linewidths=[1.0])
    if (segresnet_mask == 2).any():
        ax_tumor.contour(segresnet_mask == 2, levels=[0.5], colors=["red"], linewidths=[1.5])
    tc = "#FF5252" if tumor_info["tumor_detected"] else "#4CAF50"
    ax_tumor.set_title(f"SegResNet: {tumor_flag}\nLiver: {tumor_info['liver_pixels']}px | "
                       f"Tumor: {tumor_info['tumor_pixels']}px",
                       color=tc, fontsize=10, fontweight="bold")
    ax_tumor.axis("off"); ax_tumor.set_facecolor("#0d0d0d")

    # Panel 3: Classification (updated per box)
    ax_class.set_facecolor("#1a1a2e")
    ax_class.set_title("TotalSeg Classification", color="white", fontsize=12)
    ax_class.axis("off")

    # Panel 4: MedSAM (updated per box)
    ax_seg.imshow(image_rgb)
    ax_seg.set_title("MedSAM Segmentation", color="white", fontsize=12)
    ax_seg.axis("off"); ax_seg.set_facecolor("#0d0d0d")

    plt.suptitle("INTERACTIVE UNIFIED PIPELINE | SegResNet + TotalSeg + MedSAM | Draw a box to analyze",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    state = {"drawing": False, "start": None, "rect": None, "count": 0}

    def on_press(event):
        if event.inaxes != ax_input: return
        state["drawing"] = True
        state["start"] = (event.xdata, event.ydata)
        if state["rect"]:
            state["rect"].set_visible(False)
            state["rect"] = None

    def on_motion(event):
        if not state["drawing"] or event.inaxes != ax_input: return
        if event.xdata is None: return
        x0, y0 = state["start"]
        x1, y1 = event.xdata, event.ydata
        if state["rect"]:
            state["rect"].set_visible(False)
        rect = Rectangle((min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0),
                         lw=2, edgecolor="#00FF88", facecolor="#00FF88",
                         alpha=0.12, linestyle="--")
        state["rect"] = ax_input.add_patch(rect)
        fig.canvas.draw_idle()

    def on_release(event):
        if not state["drawing"]: return
        state["drawing"] = False
        if event.inaxes != ax_input or event.xdata is None: return
        x0, y0 = state["start"]
        x1, y1 = event.xdata, event.ydata
        if abs(x1-x0) < 5 or abs(y1-y0) < 5:
            print("  Box too small."); return

        box = np.array([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)], dtype=np.float32)
        state["count"] += 1
        sid = state["count"]
        print(f"\n  === Box #{sid} → x:[{box[0]:.0f}→{box[2]:.0f}]  y:[{box[1]:.0f}→{box[3]:.0f}] ===")

        # ── TotalSeg classify ─────────────────────────────────
        winner, ts_results = classify_box(ts_masks, slice_idx, box, orig_shape, image_rgb.shape[:2])
        print(f"  TotalSeg → {winner.upper()}")

        # ── MedSAM segment ────────────────────────────────────
        mask, score = predict_mask(box)
        print(f"  MedSAM   → Score: {score:.3f}  Mask: {mask.sum():,}px")
        print(f"  SegResNet→ {'TUMOR' if tumor_info['tumor_detected'] else 'CLEAN'}")

        # ── Update Panel 1 ────────────────────────────────────
        ax_input.clear(); ax_input.set_facecolor("#0d0d0d")
        ax_input.imshow(image_rgb)
        ax_input.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     lw=2.5, edgecolor="#00FF88", facecolor="#00FF88",
                                     alpha=0.1, linestyle="--"))
        ax_input.set_title(f"Box #{sid} — draw again to re-analyze", color="white", fontsize=11)
        ax_input.axis("off")

        # ── Update Panel 3: Classification ────────────────────
        ax_class.clear(); ax_class.set_facecolor("#1a1a2e")
        if ts_results:
            top = ts_results[:6]
            names = [r["name"] for r in top]
            pcts = [r["pct"] for r in top]
            colors_bar = ["#4CAF50" if i == 0 else "#2196F3" if i < 3 else "#607D8B"
                          for i in range(len(top))]
            bars = ax_class.barh(names[::-1], pcts[::-1], color=colors_bar[::-1], height=0.55)
            for bar, val in zip(bars, pcts[::-1]):
                ax_class.text(min(val + 0.5, max(pcts) * 0.95 if max(pcts) > 0 else 1),
                              bar.get_y() + bar.get_height() / 2,
                              f"{val:.1f}%", va="center", color="white", fontsize=9)
            ax_class.set_title(f"TotalSeg: {winner.upper()}", color="#4CAF50",
                               fontsize=13, fontweight="bold")
            ax_class.set_xlabel("% of box", color="white", fontsize=9)
            ax_class.tick_params(colors="white")
            ax_class.spines[:].set_color("#333")
            for lbl in ax_class.get_xticklabels() + ax_class.get_yticklabels():
                lbl.set_color("white")
        else:
            ax_class.text(0.5, 0.5, "No TotalSeg masks\navailable\n\nRun:\nstep1_precompute_totalseg.py",
                          transform=ax_class.transAxes, ha="center", va="center",
                          color="#FF5252", fontsize=11)
            ax_class.set_title("Classification N/A", color="#FF5252", fontsize=11)
            ax_class.axis("off")

        # ── Update Panel 4: MedSAM ───────────────────────────
        ax_seg.clear(); ax_seg.set_facecolor("#0d0d0d")
        ax_seg.imshow(image_rgb)
        overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        overlay[mask > 0] = [220, 50, 50, 160]
        ax_seg.imshow(overlay)
        if mask.any():
            ax_seg.contour(mask.astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=[2])
        ax_seg.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                   lw=1.5, edgecolor="cyan", facecolor="none", linestyle="--"))
        organ_label = winner.upper() if winner != "unknown" else "?"
        ax_seg.set_title(f"MedSAM: {organ_label} | Score: {score:.3f}",
                         color="white", fontsize=12, fontweight="bold")
        ax_seg.axis("off")

        fig.canvas.draw_idle()
        state["rect"] = None

        # ── Auto-save ─────────────────────────────────────────
        ts = datetime.now().strftime("%H%M%S")
        sp = os.path.join(args.save_dir, f"unified_{sid:03d}_{winner}_{ts}.png")
        fig.savefig(sp, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved → {sp}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print("\n" + "=" * 65)
    print("  READY! Draw a box on the LEFT panel.")
    print("  Panel 2 shows SegResNet tumor detection (pre-computed).")
    print("  Panel 3 shows TotalSeg organ classification (per box).")
    print("  Panel 4 shows MedSAM segmentation (per box).")
    print("=" * 65 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
