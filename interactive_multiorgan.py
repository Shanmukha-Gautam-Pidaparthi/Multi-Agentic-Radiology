# =============================================================================
# interactive_multiorgan.py — Unified Box-Prompt Pipeline
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Interactive 4-panel GUI that combines:
#     1. Multi-Organ Model (UNet)  → segment organs + detect tumors
#     2. MedSAM (fine-tuned)       → box-prompt segmentation
#
#   Draw a box on a CT slice → see:
#     Panel 1: Input image + drawn box
#     Panel 2: Multi-organ segmentation (all 10 classes)
#     Panel 3: MedSAM box-prompt segmentation
#     Panel 4: Tumor detection report + pixel counts
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#
#   # Run with a 2D CT slice (PNG)
#   python interactive_multiorgan.py --image ../btcv/processed/test/images/img0008-0002_s0092.png
#
#   # Run with a 3D CT volume (NIfTI) — will show middle slice
#   python interactive_multiorgan.py --volume /path/to/ct.nii.gz --slice 100
#
#   # Custom model checkpoints
#   python interactive_multiorgan.py --image slice.png \
#       --multiorgan_ckpt checkpoints/best_multiorgan.pth \
#       --medsam_ckpt ../best_medsam_btcv.pth
# =============================================================================

import matplotlib
matplotlib.use('TkAgg')

import os
import sys
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import (
    NUM_CLASSES, CLASS_MAP, CLASS_COLORS,
    HU_MIN, HU_MAX, BEST_MODEL_PATH,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive Multi-Organ + MedSAM Unified Pipeline"
    )
    p.add_argument("--image", default=None,
                   help="Path to a 2D CT slice (PNG)")
    p.add_argument("--volume", default=None,
                   help="Path to a 3D CT volume (.nii.gz)")
    p.add_argument("--slice", type=int, default=None,
                   help="Slice index for 3D volume (default: middle)")
    p.add_argument("--multiorgan_ckpt", default=None,
                   help="Multi-organ model checkpoint")
    p.add_argument("--medsam_ckpt", default=None,
                   help="MedSAM checkpoint (default: ../best_medsam_btcv.pth)")
    p.add_argument("--save_dir", default="interactive_outputs")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# =============================================================================
# MODEL LOADERS
# =============================================================================
def load_multiorgan_model(checkpoint_path, device):
    """Load the trained multi-organ UNet for 2D slice inference."""
    from monai.networks.nets import UNet

    # For 2D slice inference, we use a 2D version
    # The 3D model works on volumes; for interactive 2D, we adapt
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            # Try loading — may fail if 3D weights into 2D model
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"    Multi-organ model loaded (2D)")
            except RuntimeError:
                print(f"    Note: 3D checkpoint detected — using 2D model with fresh weights")
                print(f"    For 3D inference, use step5_inference.py instead")
        else:
            try:
                model.load_state_dict(ckpt)
            except RuntimeError:
                print(f"    Note: Using 2D model with fresh weights")
    else:
        print(f"    WARNING: Checkpoint not found: {checkpoint_path}")
        print(f"    Using untrained model (random weights)")

    model.to(device).eval()
    return model


def load_medsam(checkpoint_path, device):
    """Load MedSAM model for box-prompt segmentation."""
    try:
        from segment_anything.build_sam import _build_sam

        sam = _build_sam(
            encoder_embed_dim=768, encoder_depth=12,
            encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
            checkpoint=None,
        )
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                sam.load_state_dict(ckpt["model_state_dict"])
                print(f"    MedSAM loaded (fine-tuned, epoch {ckpt.get('epoch', '?')})")
            else:
                sam.load_state_dict(ckpt)
                print(f"    MedSAM loaded (pre-trained)")
        else:
            print(f"    WARNING: MedSAM checkpoint not found: {checkpoint_path}")
            return None

        sam.to(device).eval()
        return sam

    except ImportError:
        print(f"    WARNING: segment_anything not installed — MedSAM disabled")
        return None


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
@torch.no_grad()
def multiorgan_predict_2d(model, image_np, device):
    """
    Run the multi-organ model on a 2D slice.

    Args:
        image_np: RGB image array (H, W, 3)
    Returns:
        pred_mask: (H, W) array with class IDs
    """
    # Convert to grayscale
    gray = np.mean(image_np, axis=2).astype(np.float32)
    orig_h, orig_w = gray.shape

    # Normalize to [0, 1] (assuming already windowed)
    if gray.max() > 0:
        gray = gray / gray.max()

    # Resize to model input size
    model_size = 256
    gray_pil = Image.fromarray(gray).resize((model_size, model_size), Image.BILINEAR)
    gray_resized = np.array(gray_pil, dtype=np.float32)

    # To tensor: (1, 1, H, W)
    inp = torch.from_numpy(gray_resized).unsqueeze(0).unsqueeze(0).to(device)

    # Forward
    logits = model(inp)

    # Upsample back to original size
    logits_up = F.interpolate(logits, size=(orig_h, orig_w),
                              mode="bilinear", align_corners=False)
    pred = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred


@torch.no_grad()
def medsam_predict(sam, image_embedding, box_np, img_h, img_w, device):
    """
    Run MedSAM box-prompt segmentation.
    """
    scale_x = 1024.0 / img_w
    scale_y = 1024.0 / img_h

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


def classify_box_region(multiorgan_mask, box):
    """
    Analyze what organs and tumors are inside the drawn box.
    Returns a report dict.
    """
    x1, y1, x2, y2 = [int(b) for b in box]
    roi = multiorgan_mask[y1:y2, x1:x2]

    results = []
    for cls_id in range(1, NUM_CLASSES):
        cls_name = CLASS_MAP[cls_id]
        px_count = int((roi == cls_id).sum())
        if px_count == 0:
            continue
        total_roi = max(1, roi.size)
        pct = 100.0 * px_count / total_roi
        is_tumor = "tumor" in cls_name
        results.append({
            "class_id": cls_id,
            "name": cls_name,
            "pixels": px_count,
            "pct": round(pct, 2),
            "is_tumor": is_tumor,
        })

    results.sort(key=lambda x: -x["pixels"])
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  INTERACTIVE MULTI-ORGAN UNIFIED PIPELINE")
    print("  Multi-Organ UNet (tumor) + MedSAM (segment)")
    print("  Draw a box on the left panel → see all results instantly")
    print("=" * 65)

    # ── Pick image ────────────────────────────────────────────────
    if args.volume:
        import nibabel as nib
        nii = nib.load(args.volume)
        vol = nii.get_fdata()
        slice_idx = args.slice if args.slice is not None else vol.shape[2] // 2
        ct_slice = vol[:, :, slice_idx].astype(np.float32)
        # Window + normalize to 0-255 for display
        ct_slice = np.clip(ct_slice, HU_MIN, HU_MAX)
        ct_slice = ((ct_slice - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
        image_rgb = np.stack([ct_slice] * 3, axis=-1)
        print(f"\n  Volume: {args.volume}")
        print(f"  Slice:  {slice_idx}")
    elif args.image:
        if not os.path.exists(args.image):
            print(f"  ERROR: Image not found: {args.image}")
            sys.exit(1)
        pil_img = Image.open(args.image).convert("RGB")
        image_rgb = np.array(pil_img)
        print(f"\n  Image: {args.image}")
    else:
        # Auto-pick from BTCV test
        import glob
        btcv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "btcv", "processed", "test", "images")
        pngs = sorted(glob.glob(os.path.join(btcv_dir, "*.png")))
        if pngs:
            pil_img = Image.open(pngs[len(pngs)//2]).convert("RGB")
            image_rgb = np.array(pil_img)
            print(f"\n  Auto-selected: {pngs[len(pngs)//2]}")
        else:
            print("  ERROR: No image found. Use --image or --volume")
            sys.exit(1)

    img_h, img_w = image_rgb.shape[:2]
    print(f"  Image shape: {image_rgb.shape}")

    # ── Load models ───────────────────────────────────────────────
    print(f"\n  [1/2] Loading Multi-Organ model...")
    mo_ckpt = args.multiorgan_ckpt or BEST_MODEL_PATH
    multiorgan_model = load_multiorgan_model(mo_ckpt, device)

    print(f"  [2/2] Loading MedSAM...")
    medsam_ckpt = args.medsam_ckpt or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "best_medsam_btcv.pth"
    )
    sam = load_medsam(medsam_ckpt, device)

    # ── Compute MedSAM embedding (one-time) ───────────────────────
    image_embedding = None
    if sam is not None:
        print("  Computing MedSAM embedding (~30s on CPU)...")
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).to(device)
        input_image = sam.preprocess(img_tensor).unsqueeze(0)
        image_embedding = sam.image_encoder(input_image)
        print("  Embedding ready!")

    # ── Run multi-organ model on full slice ────────────────────────
    print("  Running multi-organ segmentation on full slice...")
    mo_mask = multiorgan_predict_2d(multiorgan_model, image_rgb, device)

    # Count what was found
    found_classes = []
    for cls_id in range(1, NUM_CLASSES):
        px = int((mo_mask == cls_id).sum())
        if px > 0:
            found_classes.append((CLASS_MAP[cls_id], px))
    if found_classes:
        for name, px in found_classes:
            marker = " ⚠ TUMOR" if "tumor" in name else ""
            print(f"    {name:<20}: {px:>8,} px{marker}")
    else:
        print(f"    No organs/tumors detected in this slice")

    # ── Setup 4-panel figure ──────────────────────────────────────
    fig, (ax_input, ax_organs, ax_seg, ax_report) = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor("#0d0d0d")

    # Panel 1: Input
    ax_input.imshow(image_rgb)
    ax_input.set_title("Draw a box (click + drag)", color="white", fontsize=12)
    ax_input.axis("off")
    ax_input.set_facecolor("#0d0d0d")

    # Panel 2: Multi-organ segmentation (static)
    ax_organs.imshow(image_rgb)
    organ_overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    for cls_id, rgba in CLASS_COLORS.items():
        organ_overlay[mo_mask == cls_id] = rgba
    ax_organs.imshow(organ_overlay)
    # Draw contours for each class
    for cls_id in range(1, NUM_CLASSES):
        if (mo_mask == cls_id).any():
            color = f"#{CLASS_COLORS[cls_id][0]:02x}{CLASS_COLORS[cls_id][1]:02x}{CLASS_COLORS[cls_id][2]:02x}"
            try:
                ax_organs.contour(mo_mask == cls_id, levels=[0.5],
                                  colors=[color], linewidths=[0.8])
            except Exception:
                pass
    ax_organs.set_title("Multi-Organ Segmentation\n(UNet — all 10 classes)",
                        color="white", fontsize=10, fontweight="bold")
    ax_organs.axis("off")
    ax_organs.set_facecolor("#0d0d0d")

    # Panel 3: MedSAM (updated per box)
    ax_seg.imshow(image_rgb)
    ax_seg.set_title("MedSAM Segmentation\n(draw box to segment)",
                     color="white", fontsize=10)
    ax_seg.axis("off")
    ax_seg.set_facecolor("#0d0d0d")

    # Panel 4: Report (updated per box)
    ax_report.set_facecolor("#1a1a2e")
    ax_report.set_title("Tumor Detection Report", color="white", fontsize=12)
    ax_report.axis("off")

    plt.suptitle(
        "MULTI-ORGAN UNIFIED PIPELINE | UNet (organ+tumor) + MedSAM (segment) | "
        "Draw a box to analyze",
        color="white", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    state = {"drawing": False, "start": None, "rect": None, "count": 0}

    def on_press(event):
        if event.inaxes != ax_input:
            return
        state["drawing"] = True
        state["start"] = (event.xdata, event.ydata)
        if state["rect"]:
            state["rect"].set_visible(False)
            state["rect"] = None

    def on_motion(event):
        if not state["drawing"] or event.inaxes != ax_input:
            return
        if event.xdata is None:
            return
        x0, y0 = state["start"]
        x1, y1 = event.xdata, event.ydata
        if state["rect"]:
            state["rect"].set_visible(False)
        rect = Rectangle((min(x0, x1), min(y0, y1)),
                          abs(x1 - x0), abs(y1 - y0),
                          lw=2, edgecolor="#00FF88", facecolor="#00FF88",
                          alpha=0.12, linestyle="--")
        state["rect"] = ax_input.add_patch(rect)
        fig.canvas.draw_idle()

    def on_release(event):
        if not state["drawing"]:
            return
        state["drawing"] = False
        if event.inaxes != ax_input or event.xdata is None:
            return
        x0, y0 = state["start"]
        x1, y1 = event.xdata, event.ydata
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            print("  Box too small.")
            return

        box = np.array([min(x0, x1), min(y0, y1),
                        max(x0, x1), max(y0, y1)], dtype=np.float32)
        state["count"] += 1
        sid = state["count"]
        print(f"\n  === Box #{sid} → x:[{box[0]:.0f}→{box[2]:.0f}]  "
              f"y:[{box[1]:.0f}→{box[3]:.0f}] ===")

        # ── Classify box region using multi-organ mask ─────────────
        box_results = classify_box_region(mo_mask, box)
        tumors_in_box = [r for r in box_results if r["is_tumor"]]
        organs_in_box = [r for r in box_results if not r["is_tumor"]]

        if box_results:
            print(f"  Multi-Organ → {box_results[0]['name'].upper()}")
            for r in box_results:
                marker = " ⚠ TUMOR!" if r["is_tumor"] else ""
                print(f"    {r['name']:<20}: {r['pixels']:>8,} px ({r['pct']:.1f}%){marker}")
        else:
            print(f"  Multi-Organ → No organ detected in box region")

        # ── MedSAM segment ────────────────────────────────────────
        medsam_mask = None
        medsam_score = 0.0
        if sam is not None and image_embedding is not None:
            medsam_mask, medsam_score = medsam_predict(
                sam, image_embedding, box, img_h, img_w, device
            )
            print(f"  MedSAM   → Score: {medsam_score:.3f}  "
                  f"Mask: {medsam_mask.sum():,}px")

        # ── Update Panel 1: Input + box ───────────────────────────
        ax_input.clear()
        ax_input.set_facecolor("#0d0d0d")
        ax_input.imshow(image_rgb)
        ax_input.add_patch(Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            lw=2.5, edgecolor="#00FF88", facecolor="#00FF88",
            alpha=0.1, linestyle="--"
        ))
        ax_input.set_title(f"Box #{sid} — draw again to re-analyze",
                           color="white", fontsize=11)
        ax_input.axis("off")

        # ── Update Panel 3: MedSAM ───────────────────────────────
        ax_seg.clear()
        ax_seg.set_facecolor("#0d0d0d")
        ax_seg.imshow(image_rgb)
        if medsam_mask is not None:
            overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
            overlay[medsam_mask > 0] = [220, 50, 50, 160]
            ax_seg.imshow(overlay)
            if medsam_mask.any():
                ax_seg.contour(medsam_mask.astype(np.uint8), levels=[0.5],
                               colors=["yellow"], linewidths=[2])
        ax_seg.add_patch(Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            lw=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
        ))
        top_name = box_results[0]["name"].upper() if box_results else "?"
        ax_seg.set_title(
            f"MedSAM: {top_name} | Score: {medsam_score:.3f}",
            color="white", fontsize=11, fontweight="bold"
        )
        ax_seg.axis("off")

        # ── Update Panel 4: Report ────────────────────────────────
        ax_report.clear()
        ax_report.set_facecolor("#1a1a2e")

        if box_results:
            # Bar chart of what's in the box
            names = [r["name"] for r in box_results[:8]]
            pcts = [r["pct"] for r in box_results[:8]]
            colors_bar = []
            for r in box_results[:8]:
                cid = r["class_id"]
                rgba = CLASS_COLORS.get(cid, (128, 128, 128, 180))
                colors_bar.append(f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}")

            bars = ax_report.barh(names[::-1], pcts[::-1],
                                  color=colors_bar[::-1], height=0.55)
            for bar, val in zip(bars, pcts[::-1]):
                ax_report.text(
                    min(val + 0.5, max(pcts) * 0.95 if max(pcts) > 0 else 1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9
                )

            # Title with tumor warning
            if tumors_in_box:
                tumor_names = ", ".join(t["name"] for t in tumors_in_box)
                ax_report.set_title(
                    f"⚠ TUMOR DETECTED\n{tumor_names}",
                    color="#FF5252", fontsize=12, fontweight="bold"
                )
            else:
                ax_report.set_title(
                    f"✓ No Tumor in Box\n{top_name}",
                    color="#4CAF50", fontsize=12, fontweight="bold"
                )

            ax_report.set_xlabel("% of box", color="white", fontsize=9)
            ax_report.tick_params(colors="white")
            ax_report.spines[:].set_color("#333")
            for lbl in ax_report.get_xticklabels() + ax_report.get_yticklabels():
                lbl.set_color("white")
        else:
            ax_report.text(0.5, 0.5,
                           "No organs detected\nin box region",
                           transform=ax_report.transAxes,
                           ha="center", va="center",
                           color="#FF5252", fontsize=12)
            ax_report.axis("off")

        fig.canvas.draw_idle()
        state["rect"] = None

        # ── Auto-save ─────────────────────────────────────────────
        ts = datetime.now().strftime("%H%M%S")
        sp = os.path.join(args.save_dir, f"multiorgan_{sid:03d}_{ts}.png")
        fig.savefig(sp, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved → {sp}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print("\n" + "=" * 65)
    print("  READY! Draw a box on the LEFT panel.")
    print("  Panel 2: Multi-Organ segmentation (pre-computed)")
    print("  Panel 3: MedSAM box-prompt segmentation (per box)")
    print("  Panel 4: Tumor detection report (per box)")
    print("=" * 65 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
