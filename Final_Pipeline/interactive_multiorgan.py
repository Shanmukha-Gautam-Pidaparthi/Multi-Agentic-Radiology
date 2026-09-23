#!/usr/bin/env python3
"""
interactive_multiorgan.py — UNIFIED 4-MODEL PIPELINE
  Panel 1: Input (draw box)
  Panel 2: Multi-organ Tumor UNet (3D, pre-computed per volume)
  Panel 3: TotalSeg organ classification (per box)
  Panel 4: MedSAM segmentation (per box)

Run:
  conda activate medsam_env
  cd ~/Desktop/intern/Box_Prompt
  python interactive_multiorgan.py --image btcv/processed/test/images/img0008-0002_s0092.png
"""

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

# ── Config ──
MULTIORGAN_MODEL = "weights/best_multiorgan.pth"
SEGRESNET_MODEL  = "SegResNet_Project/best_segresnet_model.pth"
SEGRESNET_CFG    = "SegResNet_Project/model_config.json"

NUM_CLASSES = 11
CLASS_NAMES = {
    0:"background", 1:"liver", 2:"liver_tumor", 3:"kidney", 4:"kidney_tumor",
    5:"pancreas", 6:"pancreas_tumor", 7:"colon", 8:"colon_tumor",
    9:"esophagus", 10:"esophagus_tumor"
}
TUMOR_IDS = {2, 4, 6, 8, 10}
ORGAN_IDS = {1, 3, 5, 7, 9}

TUMOR_COLORS = {
    2: [255, 50, 50, 180],    # liver_tumor → red
    4: [255, 165, 0, 180],    # kidney_tumor → orange
    6: [255, 255, 0, 180],    # pancreas_tumor → yellow
    8: [0, 255, 100, 180],    # colon_tumor → green
    10: [200, 50, 255, 180],  # esophagus_tumor → purple
}
ORGAN_COLORS = {
    1: [255, 200, 50, 80],    # liver
    3: [255, 140, 0, 80],     # kidney
    5: [255, 255, 100, 80],   # pancreas
    7: [100, 255, 100, 80],   # colon
    9: [180, 100, 255, 80],   # esophagus
}

ORGAN_NAMES_BTCV = {
    1:"spleen", 2:"right_kidney", 3:"left_kidney", 4:"gallbladder",
    5:"esophagus", 6:"liver", 7:"stomach", 8:"aorta",
    9:"IVC", 10:"portal_vein", 11:"pancreas",
    12:"right_adrenal", 13:"left_adrenal",
}
TOTALSEG_TO_BTCV = {
    "spleen":1, "kidney_right":2, "kidney_left":3,
    "gallbladder":4, "esophagus":5, "liver":6,
    "stomach":7, "aorta":8, "inferior_vena_cava":9,
    "portal_vein_and_splenic_vein":10, "pancreas":11,
    "adrenal_gland_right":12, "adrenal_gland_left":13,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Path to a 2D PNG slice")
    p.add_argument("--checkpoint", default="best_medsam_btcv.pth")
    p.add_argument("--save-dir", default="unified_multiorgan_outputs")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ── Model 1: Multi-organ 3D UNet ──
def load_multiorgan_unet(model_path, device):
    from monai.networks.nets import UNet
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    Multi-organ UNet loaded ({NUM_CLASSES} classes)")
    return model


@torch.no_grad()
def predict_3d_volume(model, nifti_path, device):
    """Run sliding-window inference on the full 3D volume."""
    from monai.inferers import sliding_window_inference
    from monai.transforms import Spacing, ScaleIntensityRange

    vol = nib.load(nifti_path)
    data = vol.get_fdata().astype(np.float32)

    # Clip HU range and normalize
    data = np.clip(data, -175, 250)
    data = (data - (-175)) / (250 - (-175))

    # Add batch+channel dims: (1, 1, H, W, D)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().to(device)

    # Sliding window inference with small patches for CPU
    patch = (96, 96, 96)
    pred = sliding_window_inference(tensor, patch, sw_batch_size=1, predictor=model,
                                     overlap=0.25, mode="gaussian")
    pred_class = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_class  # shape: (H, W, D)


# ── Model 2: SegResNet (liver-only, 2D) ──
def load_segresnet(model_path, config_path, device):
    from monai.networks.nets import SegResNet
    with open(config_path) as f:
        cfg = json.load(f)
    model = SegResNet(
        spatial_dims=2, in_channels=1, out_channels=cfg["NUM_CLASSES"],
        init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.to(device).eval()
    print(f"    SegResNet loaded ({cfg['NUM_CLASSES']} classes, liver-only)")
    return model, cfg


@torch.no_grad()
def segresnet_predict(model, image_np, cfg, device):
    sz = cfg["IMAGE_SIZE"]
    gray = np.mean(image_np, axis=2).astype(np.float32)
    h, w = gray.shape
    g_r = np.array(Image.fromarray(gray).resize((sz, sz), Image.BILINEAR), dtype=np.float32)
    if g_r.max() > 0: g_r /= g_r.max()
    inp = torch.from_numpy(g_r).unsqueeze(0).unsqueeze(0).to(device)
    logits = F.interpolate(model(inp), size=(h, w), mode="bilinear", align_corners=False)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, {"liver_px": int((pred==1).sum()), "tumor_px": int((pred==2).sum()),
                  "tumor_detected": int((pred==2).sum()) > 0}


# ── Model 3: MedSAM ──
def load_medsam(ckpt_path, device):
    sam = _build_sam(encoder_embed_dim=768, encoder_depth=12,
                     encoder_num_heads=12, encoder_global_attn_indexes=[2,5,8,11], checkpoint=None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sam.load_state_dict(ckpt.get("model_state_dict", ckpt))
    sam.to(device).eval()
    print(f"    MedSAM loaded")
    return sam


# ── Model 4: TotalSegmentator (precomputed) ──
def load_totalseg(seg_dir):
    masks = {}
    if not seg_dir or not os.path.isdir(seg_dir): return masks
    for ts_name, btcv_id in TOTALSEG_TO_BTCV.items():
        p = os.path.join(seg_dir, f"{ts_name}.nii.gz")
        if os.path.exists(p):
            arr = nib.load(p).get_fdata()
            if arr.sum() > 0: masks[btcv_id] = arr
    return masks


def classify_box(ts_masks, slice_idx, box, orig_shape, img_shape):
    if not ts_masks or slice_idx is None: return "unknown", []
    oh, ow = orig_shape[:2]
    sh, sw = img_shape[0]/oh, img_shape[1]/ow
    bx1, by1, bx2, by2 = [int(b) for b in box]
    ox1, oy1, ox2, oy2 = int(bx1/sw), int(by1/sh), int(bx2/sw), int(by2/sh)
    results = []
    for bid, m3d in ts_masks.items():
        best = 0
        for z in range(max(0,slice_idx-20), min(m3d.shape[2], slice_idx+21)):
            ms = np.flip(m3d[:,:,z], axis=0).copy()
            v = int(ms[oy1:oy2, ox1:ox2].sum())
            if v > best: best = v
        if best == 0: continue
        total = max(1, (oy2-oy1)*(ox2-ox1))
        results.append({"btcv_id":bid, "name":ORGAN_NAMES_BTCV.get(bid,f"organ_{bid}"),
                         "voxels":best, "pct":round(100*best/total, 2)})
    results.sort(key=lambda x: -x["voxels"])
    return (results[0]["name"] if results else "unknown"), results


# ── Helpers ──
def infer_case_slice(path):
    bn = os.path.splitext(os.path.basename(path))[0]
    if "_s" in bn:
        parts = bn.rsplit("_s", 1)
        try: return parts[0], int(parts[1])
        except: return parts[0], None
    return None, None

def find_seg_dir(case):
    for d in [os.path.join("totalseg_cache", case), "totalseg_test_output"]:
        if os.path.isdir(d) and len([f for f in os.listdir(d) if f.endswith(".nii.gz")]) > 5:
            return d
    return None

def find_nifti(case):
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join("btcv/raw/data/kaggle_btcv/images", case + ext)
        if os.path.exists(p): return p
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  UNIFIED MULTI-ORGAN PIPELINE")
    print("  Tumor UNet + SegResNet + TotalSeg + MedSAM")
    print("="*65)

    # Pick image
    img_path = args.image
    if not img_path:
        pngs = sorted(glob.glob("btcv/processed/test/images/*.png"))
        img_path = pngs[len(pngs)//2] if pngs else None
    if not img_path or not os.path.exists(img_path):
        print("ERROR: use --image <path>"); sys.exit(1)

    case_name, slice_idx = infer_case_slice(img_path)
    print(f"  Image: {img_path}")
    print(f"  Case:  {case_name}  Slice: {slice_idx}")

    # Load models
    print("\n  Loading models...")
    multiorgan = load_multiorgan_unet(MULTIORGAN_MODEL, device)
    segresnet, seg_cfg = load_segresnet(SEGRESNET_MODEL, SEGRESNET_CFG, device)
    sam = load_medsam(args.checkpoint, device)

    seg_dir = find_seg_dir(case_name) if case_name else None
    ts_masks = load_totalseg(seg_dir) if seg_dir else {}
    print(f"    TotalSeg: {len(ts_masks)} organ masks" if ts_masks else "    TotalSeg: no masks")

    # Load image
    pil_img = Image.open(img_path).convert("RGB")
    image_rgb = np.array(pil_img)
    img_h, img_w = image_rgb.shape[:2]

    # ── Run 3D tumor model on NIfTI volume ──
    nifti_path = find_nifti(case_name) if case_name else None
    tumor_3d = None
    tumor_slice = None
    if nifti_path and os.path.exists(MULTIORGAN_MODEL):
        print(f"\n  Running 3D Multi-organ UNet on {os.path.basename(nifti_path)}...")
        print("  (This takes 1-3 min on CPU, ~30s on CUDA)")
        tumor_3d = predict_3d_volume(multiorgan, nifti_path, device)
        if slice_idx is not None and slice_idx < tumor_3d.shape[2]:
            tumor_slice = np.flip(tumor_3d[:, :, slice_idx], axis=0).copy()
            # Count detections
            for cid, cname in CLASS_NAMES.items():
                if cid == 0: continue
                count = int((tumor_slice == cid).sum())
                if count > 0:
                    tag = "🔴 TUMOR" if cid in TUMOR_IDS else "organ"
                    print(f"    {cname:>18}: {count:>6} px  ({tag})")
        print("  ✓ 3D inference complete")
    else:
        print("  ⚠ NIfTI volume not found — skipping 3D tumor model")

    # ── Run 2D SegResNet ──
    print("  Running SegResNet (liver-only) ...")
    seg_mask, seg_info = segresnet_predict(segresnet, image_rgb, seg_cfg, device)

    # ── Compute MedSAM embedding ──
    print("  Computing MedSAM embedding ...")
    with torch.no_grad():
        img_t = torch.from_numpy(image_rgb).float().permute(2,0,1).to(device)
        embedding = sam.image_encoder(sam.preprocess(img_t).unsqueeze(0))
    print("  ✓ Embedding ready")

    orig_shape = nib.load(nifti_path).shape if nifti_path else (512,512,139)
    sx, sy = 1024.0/img_w, 1024.0/img_h

    @torch.no_grad()
    def predict_mask(box_np):
        b = box_np.copy().astype(np.float32)
        b[0]*=sx; b[1]*=sy; b[2]*=sx; b[3]*=sy
        bt = torch.from_numpy(b).float().to(device).unsqueeze(0).unsqueeze(0)
        sp, dn = sam.prompt_encoder(points=None, boxes=bt, masks=None)
        lr, iou = sam.mask_decoder(image_embeddings=embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp, dense_prompt_embeddings=dn, multimask_output=True)
        masks = F.interpolate(lr, (1024,1024), mode="bilinear", align_corners=False).squeeze(0)
        best = torch.argmax(iou.squeeze(0)).item()
        p = (torch.sigmoid(masks[best]) > 0.5).cpu().numpy().astype(np.uint8)
        pr = np.array(Image.fromarray(p*255).resize((img_w,img_h), Image.NEAREST)) > 127
        return pr.astype(np.uint8), iou.squeeze(0)[best].item()

    # ── 4-panel figure ──
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor("#0d0d0d")

    # Panel 1: Input
    ax1.imshow(image_rgb); ax1.set_title("Draw a box (click+drag)", color="white", fontsize=12)
    ax1.axis("off"); ax1.set_facecolor("#0d0d0d")

    # Panel 2: Multi-organ tumor (pre-computed)
    ax2.imshow(image_rgb)
    if tumor_slice is not None:
        ov = np.zeros((*tumor_slice.shape[:2], 4), dtype=np.uint8)
        # Resize tumor_slice to match image
        ts_resized = np.array(Image.fromarray(tumor_slice).resize((img_w, img_h), Image.NEAREST))
        ov_r = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        detected_tumors = []
        for cid, color in ORGAN_COLORS.items():
            ov_r[ts_resized == cid] = color
        for cid, color in TUMOR_COLORS.items():
            mask = ts_resized == cid
            if mask.any():
                ov_r[mask] = color
                detected_tumors.append(CLASS_NAMES[cid])
                ax2.contour(mask.astype(np.uint8), levels=[0.5],
                           colors=[[c/255 for c in color[:3]]], linewidths=[2])
        ax2.imshow(ov_r)
        if detected_tumors:
            title = "🔴 " + ", ".join(detected_tumors).upper()
            ax2.set_title(f"Tumor UNet: {title}", color="#FF5252", fontsize=11, fontweight="bold")
        else:
            ax2.set_title("Tumor UNet: ✓ No tumors", color="#4CAF50", fontsize=11, fontweight="bold")
    else:
        ax2.set_title("Tumor UNet: N/A (no NIfTI)", color="#FF9800", fontsize=11)
    ax2.axis("off"); ax2.set_facecolor("#0d0d0d")

    # Panel 3: TotalSeg (updated per box)
    ax3.set_facecolor("#1a1a2e"); ax3.set_title("TotalSeg Classification", color="white")
    ax3.axis("off")

    # Panel 4: MedSAM (updated per box)
    ax4.imshow(image_rgb); ax4.set_title("MedSAM Segmentation", color="white")
    ax4.axis("off"); ax4.set_facecolor("#0d0d0d")

    plt.suptitle("MULTI-ORGAN TUMOR PIPELINE | Tumor UNet + SegResNet + TotalSeg + MedSAM",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    state = {"drawing": False, "start": None, "rect": None, "count": 0}

    def on_press(e):
        if e.inaxes != ax1: return
        state["drawing"] = True; state["start"] = (e.xdata, e.ydata)
        if state["rect"]: state["rect"].set_visible(False); state["rect"] = None

    def on_motion(e):
        if not state["drawing"] or e.inaxes != ax1 or e.xdata is None: return
        x0,y0 = state["start"]; x1,y1 = e.xdata, e.ydata
        if state["rect"]: state["rect"].set_visible(False)
        state["rect"] = ax1.add_patch(Rectangle((min(x0,x1),min(y0,y1)),abs(x1-x0),abs(y1-y0),
            lw=2, edgecolor="#00FF88", facecolor="#00FF88", alpha=0.12, linestyle="--"))
        fig.canvas.draw_idle()

    def on_release(e):
        if not state["drawing"]: return
        state["drawing"] = False
        if e.inaxes != ax1 or e.xdata is None: return
        x0,y0 = state["start"]; x1,y1 = e.xdata, e.ydata
        if abs(x1-x0)<5 or abs(y1-y0)<5: return
        box = np.array([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)], dtype=np.float32)
        state["count"] += 1; sid = state["count"]
        print(f"\n  === Box #{sid} ===")

        # TotalSeg
        winner, ts_res = classify_box(ts_masks, slice_idx, box, orig_shape, image_rgb.shape[:2])
        print(f"  TotalSeg → {winner.upper()}")

        # Tumor UNet in box region
        tumor_in_box = []
        if tumor_slice is not None:
            ts_r = np.array(Image.fromarray(tumor_slice).resize((img_w,img_h), Image.NEAREST))
            bx1,by1,bx2,by2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            roi = ts_r[by1:by2, bx1:bx2]
            for cid in TUMOR_IDS:
                cnt = int((roi == cid).sum())
                if cnt > 0:
                    tumor_in_box.append(f"{CLASS_NAMES[cid]}({cnt}px)")
        if tumor_in_box:
            print(f"  Tumor   → 🔴 {', '.join(tumor_in_box)}")
        else:
            print(f"  Tumor   → ✓ Clean")

        # MedSAM
        mask, score = predict_mask(box)
        print(f"  MedSAM  → Score: {score:.3f}  Mask: {mask.sum():,}px")

        # Update panels
        ax1.clear(); ax1.imshow(image_rgb); ax1.set_facecolor("#0d0d0d")
        ax1.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
            lw=2.5, edgecolor="#00FF88", facecolor="#00FF88", alpha=0.1, linestyle="--"))
        ax1.set_title(f"Box #{sid}", color="white"); ax1.axis("off")

        # Panel 3
        ax3.clear(); ax3.set_facecolor("#1a1a2e")
        if ts_res:
            top = ts_res[:6]
            names = [r["name"] for r in top]; pcts = [r["pct"] for r in top]
            cols = ["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B" for i in range(len(top))]
            bars = ax3.barh(names[::-1], pcts[::-1], color=cols[::-1], height=0.55)
            for bar, val in zip(bars, pcts[::-1]):
                ax3.text(min(val+0.5, max(pcts)*0.95 if max(pcts)>0 else 1),
                         bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", color="white", fontsize=9)
            tumor_str = f" | 🔴 {', '.join(tumor_in_box)}" if tumor_in_box else ""
            ax3.set_title(f"Organ: {winner.upper()}{tumor_str}",
                         color="#FF5252" if tumor_in_box else "#4CAF50", fontsize=12, fontweight="bold")
            ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333")
            for l in ax3.get_xticklabels()+ax3.get_yticklabels(): l.set_color("white")
        else:
            ax3.text(0.5, 0.5, "No TotalSeg masks\n\nRun:\npython step1_precompute_totalseg.py",
                     transform=ax3.transAxes, ha="center", va="center", color="#FF5252", fontsize=11)
            ax3.axis("off")

        # Panel 4
        ax4.clear(); ax4.imshow(image_rgb); ax4.set_facecolor("#0d0d0d")
        ov = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        ov[mask > 0] = [220, 50, 50, 160]
        ax4.imshow(ov)
        if mask.any():
            ax4.contour(mask.astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=[2])
        ax4.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
            lw=1.5, edgecolor="cyan", facecolor="none", linestyle="--"))
        organ_label = winner.upper() if winner != "unknown" else "?"
        ax4.set_title(f"MedSAM: {organ_label} | Score: {score:.3f}",
                      color="white", fontsize=12, fontweight="bold")
        ax4.axis("off")

        fig.canvas.draw_idle()

        ts = datetime.now().strftime("%H%M%S")
        sp = os.path.join(args.save_dir, f"multi_{sid:03d}_{winner}_{ts}.png")
        fig.savefig(sp, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved → {sp}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print("\n" + "="*65)
    print("  READY! Draw a box on the LEFT panel.")
    print("  Panel 2: Multi-organ tumor detection (pre-computed)")
    print("  Panel 3: TotalSeg organ classification (per box)")
    print("  Panel 4: MedSAM segmentation (per box)")
    print("="*65 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
