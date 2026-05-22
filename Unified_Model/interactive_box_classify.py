# =============================================================================
# interactive_box_classify.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   Interactive box-prompting with BOTH:
#              • TotalSegmentator organ classification
#              • MedSAM segmentation
#            Draw a box → instantly see which organ it is + segmented mask.
#
# Run:       conda activate medsam_env
#            cd ~/Desktop/intern/Box_Prompt
#            python interactive_box_classify.py
#            python interactive_box_classify.py --image btcv/processed/test/images/img0008-0002_s0092.png
#            python interactive_box_classify.py --seg-dir totalseg_test_output --case img0001-0002 --slice 104
# =============================================================================

import matplotlib
matplotlib.use('TkAgg')

import os, sys, argparse, glob
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
    p = argparse.ArgumentParser(description="Interactive Box-Prompt: Classify + Segment")
    p.add_argument("--checkpoint", default="best_medsam_btcv.pth",
                   help="MedSAM checkpoint")
    p.add_argument("--image", default=None,
                   help="2D slice PNG to segment (auto-picks if not set)")
    p.add_argument("--seg-dir", default=None,
                   help="TotalSegmentator output dir (e.g. totalseg_test_output)")
    p.add_argument("--case", default=None,
                   help="Case name for TotalSeg lookup (e.g. img0001-0002)")
    p.add_argument("--slice", type=int, default=None,
                   help="Slice index for TotalSeg lookup")
    p.add_argument("--save-dir", default="interactive_outputs",
                   help="Where to auto-save results")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# =============================================================================
# Load MedSAM
# =============================================================================
def load_medsam(checkpoint_path, device):
    print("  Loading MedSAM ...")
    sam = _build_sam(
        encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sam.load_state_dict(ckpt["model_state_dict"])
        print(f"    Fine-tuned (epoch {ckpt.get('epoch','?')}, "
              f"best_dice={ckpt.get('best_dice','?')})")
    else:
        sam.load_state_dict(ckpt)
        print(f"    Raw pre-trained model")
    sam.to(device).eval()
    return sam


# =============================================================================
# Load TotalSegmentator masks (all organs for the case)
# =============================================================================
def load_totalseg_masks(seg_dir):
    """Load all organ masks into a dict {btcv_id: 3D_array}."""
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


def classify_box(ts_masks, slice_idx, box, orig_shape, img_shape_1024):
    """
    Given preloaded TotalSeg masks, classify which organ is in the box
    on the given slice. Returns (winner_name, results_list).

    Handles coordinate mapping:
    - Box is in 1024×1024 space, masks are in native voxel space
    - LAS→RAS flip: x-axis is flipped in masks
    """
    if not ts_masks or slice_idx is None:
        return "unknown", []

    orig_h, orig_w = orig_shape[:2]
    scale_h = img_shape_1024[0] / orig_h
    scale_w = img_shape_1024[1] / orig_w

    bx1, by1, bx2, by2 = [int(b) for b in box]
    ox1 = int(bx1 / scale_w)
    oy1 = int(by1 / scale_h)
    ox2 = int(bx2 / scale_w)
    oy2 = int(by2 / scale_h)

    results = []
    for btcv_id, mask_3d in ts_masks.items():
        if slice_idx >= mask_3d.shape[2]:
            continue
        mask_slice = mask_3d[:, :, slice_idx]
        # Flip x-axis (LAS→RAS difference)
        mask_slice = np.flip(mask_slice, axis=0).copy()

        box_region = mask_slice[oy1:oy2, ox1:ox2]
        vox = int(box_region.sum())
        if vox == 0:
            continue

        total_box = max(1, (oy2-oy1) * (ox2-ox1))
        pct = 100.0 * vox / total_box
        results.append({
            "btcv_id": btcv_id,
            "name": ORGAN_NAMES.get(btcv_id, f"organ_{btcv_id}"),
            "voxels": vox,
            "pct": round(pct, 2),
        })

    results.sort(key=lambda x: -x["voxels"])
    winner = results[0]["name"] if results else "unknown"
    return winner, results


# =============================================================================
# Auto-pick an image if not specified
# =============================================================================
def auto_pick_image():
    """Find a good default test image."""
    candidates = [
        "btcv/processed/test/images",
        "slice_output",
    ]
    for d in candidates:
        if os.path.isdir(d):
            pngs = sorted(glob.glob(os.path.join(d, "*.png")))
            # Skip annotated ones
            pngs = [p for p in pngs if "annotated" not in p]
            if pngs:
                return pngs[len(pngs)//2]  # pick a middle one
    return None


def infer_case_and_slice(image_path):
    """Try to extract case name and slice index from filename."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    case, slice_idx = None, None
    # Pattern: img0008-0002_s0067 or img0001-0002_slice_z104
    if "_s" in basename:
        parts = basename.rsplit("_s", 1)
        case = parts[0]
        try:
            slice_idx = int(parts[1])
        except ValueError:
            pass
    elif "_slice_z" in basename:
        parts = basename.split("_slice_z")
        case = parts[0]
        try:
            slice_idx = int(parts[1])
        except ValueError:
            pass
    return case, slice_idx


def find_seg_dir(case_name):
    """Auto-find TotalSegmentator output for this case."""
    candidates = [
        os.path.join("totalseg_cache", case_name),
        "totalseg_test_output",
    ]
    for d in candidates:
        if os.path.isdir(d):
            masks = [f for f in os.listdir(d)
                     if f.endswith(".nii.gz") and f != "input_RAS.nii.gz"]
            if len(masks) > 5:
                return d
    return None


def get_orig_shape(case_name):
    """Get original volume shape for coordinate mapping."""
    raw_dir = "btcv/raw/data/kaggle_btcv/images"
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join(raw_dir, case_name + ext)
        if os.path.exists(p):
            return nib.load(p).shape
    return (512, 512, 139)  # fallback


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  Interactive Box-Prompt: Classify + Segment")
    print("  Draw a box → see organ name + segmentation")
    print("="*60)

    # ── Pick image ────────────────────────────────────────────────
    if args.image:
        image_path = args.image
    else:
        image_path = auto_pick_image()
        if image_path is None:
            print("  ERROR: No image found. Use --image <path>")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"  ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n  Image: {image_path}")

    # ── Infer case/slice for TotalSeg lookup ──────────────────────
    case_name = args.case
    slice_idx = args.slice
    if case_name is None or slice_idx is None:
        auto_case, auto_slice = infer_case_and_slice(image_path)
        if case_name is None:
            case_name = auto_case
        if slice_idx is None:
            slice_idx = auto_slice

    print(f"  Case : {case_name or 'unknown'}")
    print(f"  Slice: {slice_idx if slice_idx is not None else 'unknown'}")

    # ── Load TotalSegmentator masks ──────────────────────────────
    seg_dir = args.seg_dir or (find_seg_dir(case_name) if case_name else None)
    if seg_dir:
        print(f"  TotalSeg masks: {seg_dir}/")
        ts_masks = load_totalseg_masks(seg_dir)
        print(f"    Loaded {len(ts_masks)} organ masks")
    else:
        print(f"  TotalSeg masks: NOT FOUND")
        print(f"    Run step1_precompute_totalseg.py first, or use --seg-dir")
        ts_masks = {}

    orig_shape = get_orig_shape(case_name) if case_name else (512, 512, 139)

    # ── Load MedSAM ──────────────────────────────────────────────
    sam = load_medsam(args.checkpoint, device)

    # ── Load image + compute embedding ───────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)
    print(f"\n  Image shape: {image_rgb.shape}")
    print("  Computing image embedding (30s on CPU) ...")

    with torch.no_grad():
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).to(device)
        input_image = sam.preprocess(img_tensor).unsqueeze(0)
        image_embedding = sam.image_encoder(input_image)
    print("  Embedding ready!\n")

    # ── MedSAM inference function ─────────────────────────────────
    # MedSAM always works in 1024×1024 space internally.
    # We scale the box to 1024 space, get the 1024 mask, then resize
    # back to the display image size.
    img_h, img_w = image_rgb.shape[:2]
    scale_to_1024_x = 1024.0 / img_w
    scale_to_1024_y = 1024.0 / img_h

    @torch.no_grad()
    def predict_mask(box_np):
        # Scale box from display coords → 1024×1024 coords
        box_1024 = box_np.copy()
        box_1024[0] *= scale_to_1024_x
        box_1024[1] *= scale_to_1024_y
        box_1024[2] *= scale_to_1024_x
        box_1024[3] *= scale_to_1024_y

        box_t = torch.from_numpy(box_1024).float().to(device)
        box_input = box_t.unsqueeze(0).unsqueeze(0)
        sparse, dense = sam.prompt_encoder(points=None, boxes=box_input, masks=None)
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

        # Resize mask from 1024×1024 → display image size
        pred_resized = np.array(
            Image.fromarray(pred_1024 * 255).resize((img_w, img_h), Image.NEAREST)
        ) > 127
        return pred_resized.astype(np.uint8), scores[best].item()

    # ── Setup figure (3 panels) ───────────────────────────────────
    fig, (ax_input, ax_class, ax_seg) = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#0d0d0d")

    ax_input.imshow(image_rgb)
    ax_input.set_title("Draw a box (click + drag)", color="white", fontsize=12)
    ax_input.axis("off")
    ax_input.set_facecolor("#0d0d0d")

    ax_class.set_facecolor("#1a1a2e")
    ax_class.set_title("TotalSegmentator Classification", color="white", fontsize=12)
    ax_class.axis("off")

    ax_seg.imshow(image_rgb)
    ax_seg.set_title("MedSAM Segmentation", color="white", fontsize=12)
    ax_seg.axis("off")
    ax_seg.set_facecolor("#0d0d0d")

    plt.suptitle("Interactive Box-Prompt  |  TotalSeg + MedSAM  |  Click + Drag to draw",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ── Box drawing state ─────────────────────────────────────────
    state = {"drawing": False, "start": None, "rect": None, "count": 0}

    def on_press(event):
        if event.inaxes != ax_input:
            return
        state["drawing"] = True
        state["start"] = (event.xdata, event.ydata)
        if state["rect"] is not None:
            state["rect"].set_visible(False)
            state["rect"] = None

    def on_motion(event):
        if not state["drawing"] or event.inaxes != ax_input:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = state["start"]
        x1, y1 = event.xdata, event.ydata
        if state["rect"] is not None:
            state["rect"].set_visible(False)
        rect = Rectangle(
            (min(x0,x1), min(y0,y1)), abs(x1-x0), abs(y1-y0),
            linewidth=2, edgecolor="#00FF88", facecolor="#00FF88",
            alpha=0.12, linestyle="--"
        )
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
        if abs(x1-x0) < 5 or abs(y1-y0) < 5:
            print("  Box too small — drag a larger region.")
            return

        box = np.array([min(x0,x1), min(y0,y1),
                        max(x0,x1), max(y0,y1)], dtype=np.float32)

        print(f"\n  Box → x:[{box[0]:.0f}→{box[2]:.0f}]  y:[{box[1]:.0f}→{box[3]:.0f}]")

        # ── CLASSIFY via TotalSegmentator ──────────────────────────
        winner, ts_results = classify_box(
            ts_masks, slice_idx, box, orig_shape, image_rgb.shape[:2]
        )
        print(f"  TotalSeg → {winner.upper()}")
        if ts_results:
            for r in ts_results[:3]:
                print(f"    {r['name']:<18} {r['pct']:5.1f}%  ({r['voxels']:,} vox)")

        # ── SEGMENT via MedSAM ────────────────────────────────────
        mask, score = predict_mask(box)
        print(f"  MedSAM  → Score: {score:.3f}  Mask pixels: {mask.sum():,}")

        # ── Update Panel 1: Input + Box ───────────────────────────
        ax_input.clear()
        ax_input.set_facecolor("#0d0d0d")
        ax_input.imshow(image_rgb)
        ax_input.add_patch(Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2.5, edgecolor="#00FF88", facecolor="#00FF88",
            alpha=0.1, linestyle="--"
        ))
        ax_input.set_title("Input + box  (draw again to re-prompt)",
                           color="white", fontsize=11)
        ax_input.axis("off")

        # ── Update Panel 2: Classification bar chart ──────────────
        ax_class.clear()
        ax_class.set_facecolor("#1a1a2e")
        if ts_results:
            top = ts_results[:6]
            names = [r["name"] for r in top]
            pcts  = [r["pct"]  for r in top]
            colors_bar = ["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B"
                          for i in range(len(top))]
            bars = ax_class.barh(names[::-1], pcts[::-1],
                                 color=colors_bar[::-1], height=0.55)
            for bar, val in zip(bars, pcts[::-1]):
                ax_class.text(
                    min(val+0.5, max(pcts)*0.95 if max(pcts) > 0 else 1),
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", color="white", fontsize=9
                )
            ax_class.set_title(
                f"TotalSeg: {winner.upper()}",
                color="#4CAF50", fontsize=13, fontweight="bold"
            )
            ax_class.set_xlabel("% of box voxels", color="white", fontsize=9)
            ax_class.tick_params(colors="white")
            ax_class.spines[:].set_color("#333")
            for lbl in ax_class.get_xticklabels()+ax_class.get_yticklabels():
                lbl.set_color("white")
        else:
            ax_class.text(0.5, 0.5, "No TotalSeg masks\navailable",
                          transform=ax_class.transAxes, ha="center", va="center",
                          color="#FF5252", fontsize=13)
            ax_class.set_title("Classification unavailable",
                               color="#FF5252", fontsize=11)
            ax_class.axis("off")

        # ── Update Panel 3: MedSAM segmentation ──────────────────
        ax_seg.clear()
        ax_seg.set_facecolor("#0d0d0d")
        ax_seg.imshow(image_rgb)

        overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        overlay[mask > 0] = [220, 50, 50, 160]
        ax_seg.imshow(overlay)

        if mask.any():
            ax_seg.contour(mask.astype(np.uint8), levels=[0.5],
                           colors=["yellow"], linewidths=[2])

        ax_seg.add_patch(Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--"
        ))
        organ_label = winner.upper() if winner != "unknown" else "?"
        ax_seg.set_title(
            f"MedSAM: {organ_label}  |  Score: {score:.3f}",
            color="white", fontsize=12, fontweight="bold"
        )
        ax_seg.axis("off")

        fig.canvas.draw_idle()
        state["rect"] = None
        state["count"] += 1

        # ── Auto-save ────────────────────────────────────────────
        ts = datetime.now().strftime("%H%M%S")
        save_path = os.path.join(
            args.save_dir,
            f"interactive_{state['count']:03d}_{winner}_{ts}.png"
        )
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved → {save_path}")

    # ── Connect events ────────────────────────────────────────────
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print("="*60)
    print("  READY! Draw a box on the left panel.")
    print("  Close the window to exit.")
    print("="*60 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
