# =============================================================================
# file3_totalseg_classify_2d.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   Given a 2D slice index + bounding box on a BTCV CT volume,
#            run TotalSegmentator on the FULL 3D volume (RAS-reoriented),
#            then classify which organ is inside the box on that slice.
#
# Why full-3D?  TotalSegmentator is a 3D model — it needs real anatomical
#               context across adjacent slices. Pseudo-3D (stacking one
#               2D slice N times) produces empty masks.
#
# Run:
#   # Auto-box (picks brightest region on the best slice):
#   python file3_totalseg_classify_2d.py
#
#   # Specify a case:
#   python file3_totalseg_classify_2d.py --case img0001-0002.nii
#
#   # Manual box (x1 y1 x2 y2 in pixel coords on the slice):
#   python file3_totalseg_classify_2d.py --case img0001-0002.nii --slice 104 --box 150 200 320 350
# =============================================================================

import os, sys, argparse, json, datetime
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIG
# =============================================================================
RAW_IMG_DIR   = "btcv/raw/data/kaggle_btcv/images"
RAW_LABEL_DIR = "btcv/raw/data/kaggle_btcv/labels"
OUT_DIR       = "totalseg_classify_output"

# TotalSegmentator structure names → BTCV label IDs
TOTALSEG_TO_BTCV = {
    "spleen":       {"id":1,  "name":"spleen"},
    "kidney_right": {"id":2,  "name":"right_kidney"},
    "kidney_left":  {"id":3,  "name":"left_kidney"},
    "gallbladder":  {"id":4,  "name":"gallbladder"},
    "esophagus":    {"id":5,  "name":"esophagus"},
    "liver":        {"id":6,  "name":"liver"},
    "stomach":      {"id":7,  "name":"stomach"},
    "aorta":        {"id":8,  "name":"aorta"},
    "inferior_vena_cava": {"id":9,  "name":"ivc"},
    "portal_vein_and_splenic_vein": {"id":10, "name":"portal_vein"},
    "pancreas":     {"id":11, "name":"pancreas"},
    "adrenal_gland_right": {"id":12, "name":"adrenal_R"},
    "adrenal_gland_left":  {"id":13, "name":"adrenal_L"},
}

ORGAN_COLORS = {
    "spleen":"#E74C3C", "right_kidney":"#3498DB", "left_kidney":"#2ECC71",
    "gallbladder":"#F39C12", "esophagus":"#9B59B6", "liver":"#1ABC9C",
    "stomach":"#E67E22", "aorta":"#E91E63", "ivc":"#00BCD4",
    "portal_vein":"#8BC34A", "pancreas":"#FF5722",
    "adrenal_R":"#607D8B", "adrenal_L":"#795548",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default=None,
                   help="e.g. img0001-0002.nii  (auto-picks best if not set)")
    p.add_argument("--slice", type=int, default=None,
                   help="Axial slice index. Auto-picks best if not set.")
    p.add_argument("--box", nargs=4, type=int, default=None,
                   metavar=("X1","Y1","X2","Y2"),
                   help="Manual box in pixel coords. Auto-detect if not set.")
    p.add_argument("--fast", action="store_true", default=False,
                   help="Use TotalSegmentator fast mode (default: False)")
    p.add_argument("--seg-dir", default=None,
                   help="Path to existing TotalSegmentator output (skip re-run)")
    return p.parse_args()


# =============================================================================
# STEP 1 — Pick a BTCV case and find the best slice
# =============================================================================
def pick_case_and_slice(args):
    import glob
    imgs = sorted(glob.glob(os.path.join(RAW_IMG_DIR, "*.nii")))
    if not imgs:
        print(f"  ERROR: No .nii files in {RAW_IMG_DIR}")
        sys.exit(1)

    if args.case:
        matched = [p for p in imgs if args.case in os.path.basename(p)]
        if not matched:
            print(f"  ERROR: '{args.case}' not found."); sys.exit(1)
        img_path = matched[0]
    else:
        # Auto-pick richest case
        best_img, best_score = None, 0
        print("  Auto-selecting richest case …")
        for ip in imgs[:20]:
            lp = ip.replace("images","labels").replace("img","label")
            if not os.path.exists(lp): continue
            la = nib.load(lp).get_fdata().astype(np.int16)
            n  = len(np.unique(la[la > 0]))
            if n > best_score:
                best_score = n
                best_img   = ip
        img_path = best_img or imgs[0]
        print(f"  Selected: {os.path.basename(img_path)} ({best_score} organs)")

    label_path = img_path.replace("images","labels").replace("img","label")
    return img_path, label_path


def find_best_slice(label_arr):
    """Find axial slice with most organ labels visible."""
    best_z, best_count = label_arr.shape[2]//2, 0
    for z in range(label_arr.shape[2]):
        n = len(np.unique(label_arr[:,:,z][label_arr[:,:,z] > 0]))
        if n > best_count:
            best_count = n
            best_z = z
    return best_z, best_count


# =============================================================================
# STEP 2 — Auto-detect a bounding box on a 2D slice
# =============================================================================
def auto_detect_box(img_slice, percentile=85):
    from scipy import ndimage
    # Normalise to 0-255 for thresholding
    sl = np.clip(img_slice, -175, 250)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255

    thresh = np.percentile(sl, percentile)
    binary = (sl >= thresh).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    if n == 0:
        H, W = sl.shape
        return (W//4, H//4, 3*W//4, 3*H//4)
    sizes = ndimage.sum(binary, labeled, range(1, n+1))
    largest = int(np.argmax(sizes)) + 1
    mask = (labeled == largest)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    pad = 10
    x1 = max(0,     cols[0]  - pad)
    y1 = max(0,     rows[0]  - pad)
    x2 = min(sl.shape[1]-1, cols[-1] + pad)
    y2 = min(sl.shape[0]-1, rows[-1] + pad)
    return (x1, y1, x2, y2)


# =============================================================================
# STEP 3 — Run TotalSegmentator on the FULL 3D volume
# =============================================================================
def run_totalseg_full_volume(img_path, seg_output_dir, fast=False):
    from totalsegmentator.python_api import totalsegmentator

    # Load and reorient to RAS
    orig = nib.load(img_path)
    orig_orient = nib.aff2axcodes(orig.affine)
    print(f"\n  Original orientation : {orig_orient}")

    ras = nib.as_closest_canonical(orig)
    ras_orient = nib.aff2axcodes(ras.affine)
    print(f"  Reoriented to        : {ras_orient}")

    ras_path = os.path.join(seg_output_dir, "input_RAS.nii.gz")
    nib.save(ras, ras_path)
    print(f"  RAS volume saved     : {ras_path}  shape={ras.shape}")

    print(f"\n  Running TotalSegmentator (fast={fast}) …")
    print(f"  Output dir : {seg_output_dir}/")
    if not fast:
        print(f"  Full-res mode — takes 5–15 min on CPU, ~2 min on GPU\n")
    else:
        print(f"  Fast mode — takes 2–5 min on CPU\n")

    totalsegmentator(
        input  = ras_path,
        output = seg_output_dir,
        task   = "total",
        fast   = fast,
        ml     = False,     # individual per-organ .nii.gz files
        quiet  = False,
    )
    print(f"\n  TotalSegmentator finished.")
    return ras


# =============================================================================
# STEP 4 — Classify: which organs overlap the box on the given slice?
# =============================================================================
def classify_box_on_slice(seg_dir, ras_img, slice_z, box):
    """
    For each organ mask from TotalSegmentator, check how many voxels
    fall inside the (x1,y1)→(x2,y2) box on slice z.
    """
    x1, y1, x2, y2 = box
    results = []

    for ts_name, btcv_info in TOTALSEG_TO_BTCV.items():
        seg_path = os.path.join(seg_dir, f"{ts_name}.nii.gz")
        if not os.path.exists(seg_path):
            continue
        mask_nib = nib.load(seg_path)
        mask_arr = mask_nib.get_fdata()

        # Ensure the mask has the slice dimension
        if slice_z >= mask_arr.shape[2]:
            continue

        # Extract the slice and crop to the box
        mask_slice = mask_arr[:, :, slice_z]
        box_region = mask_slice[y1:y2, x1:x2]
        vox = int(box_region.sum())
        if vox == 0:
            continue

        total_box = (y2-y1) * (x2-x1)
        pct = 100.0 * vox / total_box
        results.append({
            "ts_name"  : ts_name,
            "btcv_name": btcv_info["name"],
            "btcv_id"  : btcv_info["id"],
            "voxels"   : vox,
            "pct"      : round(pct, 2),
        })

    results.sort(key=lambda x: -x["voxels"])
    return results


# =============================================================================
# STEP 5 — Visualisation
# =============================================================================
def save_result_overlay(img_arr, slice_z, box, results, seg_dir, out_path, case_name):
    x1, y1, x2, y2 = box

    # Normalise the slice for display
    img_slice = img_arr[:, :, slice_z].astype(np.float32)
    sl = np.clip(img_slice, -175, 250)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d0d")

    # ─── Left panel: slice + box + organ overlays ─────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d0d0d")
    ax.imshow(np.rot90(sl), cmap="gray", origin="upper", aspect="equal")

    # Draw organ contours from TotalSegmentator masks
    for r in results[:5]:  # top 5 organs
        seg_path = os.path.join(seg_dir, f"{r['ts_name']}.nii.gz")
        if not os.path.exists(seg_path):
            continue
        mask_slice = nib.load(seg_path).get_fdata()[:, :, slice_z]
        mask_rot = np.rot90(mask_slice)
        color = ORGAN_COLORS.get(r["btcv_name"], "#FFFFFF")
        if mask_rot.any():
            ov = np.zeros((*mask_rot.shape, 4), dtype=np.float32)
            rgb = [int(color.lstrip('#')[i:i+2], 16)/255 for i in (0,2,4)]
            ov[mask_rot > 0] = [*rgb, 0.35]
            ax.imshow(ov, origin="upper", aspect="equal")
            ax.contour(mask_rot.astype(float), levels=[0.5],
                       colors=[color], linewidths=1.2)

    # Draw bounding box (coordinates need rotation adjustment)
    H = sl.shape[0]
    # After rot90: new_x = y_old, new_y = H - 1 - x_old
    rx1, ry1 = y1, H - 1 - x2
    rx2, ry2 = y2, H - 1 - x1
    rect = mpatches.Rectangle((rx1, ry1), rx2-rx1, ry2-ry1,
                               linewidth=2.5, edgecolor="#00FF88",
                               facecolor="none")
    ax.add_patch(rect)
    ax.text(rx1+4, ry1-4, f"Box ({x1},{y1})→({x2},{y2})",
            color="#00FF88", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d0d0d",
                      edgecolor="#00FF88", alpha=0.85))
    ax.set_title(f"Slice z={slice_z} + box + organ masks", color="white", fontsize=10)
    ax.axis("off")

    # ─── Right panel: bar chart ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    if results:
        top = results[:8]
        names = [r["btcv_name"] for r in top]
        pcts  = [r["pct"]       for r in top]
        colors_bar = ["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B"
                      for i in range(len(top))]
        bars = ax2.barh(names[::-1], pcts[::-1], color=colors_bar[::-1], height=0.55)
        for bar, val in zip(bars, pcts[::-1]):
            ax2.text(min(val+0.5, max(pcts)*0.95),
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="white", fontsize=9)

        winner = results[0]["btcv_name"].upper()
        ax2.set_title(
            f"TotalSegmentator verdict: {winner}",
            color="#4CAF50", fontsize=11, fontweight="bold"
        )
        ax2.set_xlabel("% of box pixels", color="white", fontsize=9)
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#333")
        for label in ax2.get_xticklabels()+ax2.get_yticklabels():
            label.set_color("white")
    else:
        ax2.text(0.5, 0.5, "No organs detected in this box",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="white", fontsize=12)
        ax2.axis("off")

    plt.suptitle(f"{case_name} z={slice_z}  |  Box-prompt → TotalSegmentator",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Result PNG    → {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  FILE 3 — 2D box-prompt → TotalSegmentator classification")
    print("  (runs on FULL 3D volume, then classifies on one slice)")
    print("="*60)

    # 1. Pick case
    img_path, label_path = pick_case_and_slice(args)
    case_name = os.path.basename(img_path).replace(".nii", "")
    print(f"\n  Case : {case_name}")
    print(f"  Image: {img_path}")

    # 2. Load volume + find best slice
    img_nib = nib.load(img_path)
    img_arr = img_nib.get_fdata().astype(np.float32)
    print(f"  Shape: {img_arr.shape}")

    if args.slice is not None:
        slice_z = args.slice
        print(f"  Slice: z={slice_z} (user-specified)")
    elif os.path.exists(label_path):
        label_arr = nib.load(label_path).get_fdata().astype(np.int16)
        slice_z, n = find_best_slice(label_arr)
        print(f"  Slice: z={slice_z} (auto-selected, {n} organs)")
    else:
        slice_z = img_arr.shape[2] // 2
        print(f"  Slice: z={slice_z} (middle, no labels)")

    # 3. Box
    img_slice = img_arr[:, :, slice_z]
    if args.box:
        box = tuple(args.box)
        print(f"  Box  : manual ({box[0]},{box[1]})→({box[2]},{box[3]})")
    else:
        box = auto_detect_box(img_slice)
        print(f"  Box  : auto   ({box[0]},{box[1]})→({box[2]},{box[3]})")

    # 4. Run TotalSegmentator on full 3D volume (or reuse existing output)
    seg_dir = args.seg_dir or os.path.join(OUT_DIR, f"{case_name}_seg")
    os.makedirs(seg_dir, exist_ok=True)

    # Check if we already have segmentation output
    existing_masks = [f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")
                      and f != "input_RAS.nii.gz"]

    if existing_masks:
        print(f"\n  Found {len(existing_masks)} existing masks in {seg_dir}/")
        print(f"  Skipping TotalSegmentator (use --seg-dir to point elsewhere)")
        # Load the RAS volume for visualization
        ras_path = os.path.join(seg_dir, "input_RAS.nii.gz")
        if os.path.exists(ras_path):
            ras_img = nib.load(ras_path)
        else:
            ras_img = nib.as_closest_canonical(img_nib)
    else:
        ras_img = run_totalseg_full_volume(img_path, seg_dir, fast=args.fast)

    # Use RAS-reoriented volume for consistent coordinates
    ras_arr = ras_img.get_fdata().astype(np.float32)
    print(f"  RAS shape: {ras_arr.shape}")

    # Adjust slice index if shape changed during reorientation
    if ras_arr.shape[2] != img_arr.shape[2]:
        print(f"  NOTE: Z-dimension changed after RAS reorientation "
              f"({img_arr.shape[2]} → {ras_arr.shape[2]})")
        slice_z = min(slice_z, ras_arr.shape[2] - 1)

    # 5. Classify — which organs are in the box on this slice?
    results = classify_box_on_slice(seg_dir, ras_img, slice_z, box)

    print(f"\n{'─'*60}")
    print(f"  CLASSIFICATION RESULT  (slice z={slice_z})")
    print(f"{'─'*60}")
    if results:
        print(f"  Winner  : {results[0]['btcv_name'].upper()}  "
              f"(BTCV label {results[0]['btcv_id']})  "
              f"{results[0]['pct']:.1f}% of box")
        print(f"\n  Full ranking (organs in box):")
        for i, r in enumerate(results, 1):
            bar = "█"*int(r["pct"]/2) + "░"*(50-int(r["pct"]/2))
            print(f"  {i:>2}. {r['btcv_name']:<18} {bar} {r['pct']:5.1f}%  "
                  f"({r['voxels']:,} vox)")
    else:
        print("  No organs found in this box.")
        print("  Try a larger box or a different slice.")

    # 6. Result overlay
    result_png = os.path.join(OUT_DIR, f"{case_name}_z{slice_z}_result.png")
    save_result_overlay(ras_arr, slice_z, box, results,
                        seg_dir, result_png, case_name)

    # 7. Save JSON
    report = {
        "timestamp"   : datetime.datetime.now().isoformat(),
        "case"        : case_name,
        "image_path"  : img_path,
        "slice_z"     : int(slice_z),
        "box"         : [int(b) for b in box],
        "ras_shape"   : [int(s) for s in ras_arr.shape],
        "winner"      : results[0]["btcv_name"] if results else None,
        "winner_id"   : int(results[0]["btcv_id"]) if results else None,
        "all_results" : [{k: (int(v) if isinstance(v, (np.integer,)) else v)
                          for k, v in r.items()} for r in results],
    }
    json_path = os.path.join(OUT_DIR, f"{case_name}_z{slice_z}_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Winner      : {results[0]['btcv_name'].upper() if results else 'none'}")
    print(f"  Result PNG  : {result_png}")
    print(f"  JSON report : {json_path}")
    print(f"  Seg masks   : {seg_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
