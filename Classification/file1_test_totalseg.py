# =============================================================================
# file1_test_totalseg.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   Test TotalSegmentator on ONE real BTCV .nii image.
#            Confirms the tool installs and runs correctly before
#            connecting it to the box-prompt pipeline.
#
# Run:       python file1_test_totalseg.py
# =============================================================================

import os, sys, subprocess

# =============================================================================
# STEP 0 — install TotalSegmentator if not present
# =============================================================================
def ensure_installed():
    try:
        import totalsegmentator
        print(f"  TotalSegmentator already installed.")
    except ImportError:
        print("  TotalSegmentator not found — installing …")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "totalsegmentator"
        ])
        print("  Installed successfully.")

# =============================================================================
# CONFIG
# =============================================================================
RAW_IMG_DIR = "btcv/raw/data/kaggle_btcv/images"
OUT_DIR     = "totalseg_test_output"

# BTCV has 13 organ labels — these are the corresponding TotalSegmentator
# structure names (subset of its 117 classes that overlap with BTCV)
BTCV_TO_TOTALSEG = {
    "spleen"        : "spleen",
    "right_kidney"  : "kidney_right",
    "left_kidney"   : "kidney_left",
    "gallbladder"   : "gallbladder",
    "liver"         : "liver",
    "stomach"       : "stomach",
    "aorta"         : "aorta",
    "pancreas"      : "pancreas",
}

def main():
    ensure_installed()

    import glob
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from totalsegmentator.python_api import totalsegmentator

    print("\n" + "="*60)
    print("  FILE 1 — TotalSegmentator single-image test")
    print("="*60)

    # ── pick first available BTCV image ──────────────────────────────────────
    imgs = sorted(glob.glob(os.path.join(RAW_IMG_DIR, "*.nii")))
    if not imgs:
        print(f"\n  ERROR: No .nii files found in {RAW_IMG_DIR}")
        print("  Make sure you ran the Kaggle download and are in Box_Prompt/")
        sys.exit(1)

    input_nii = imgs[0]
    case_name = os.path.basename(input_nii).replace(".nii", "")
    print(f"\n  Input  : {input_nii}")
    print(f"  Case   : {case_name}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── reorient to RAS (TotalSegmentator requirement) ────────────────────────
    orig_img = nib.load(input_nii)
    orig_orient = nib.aff2axcodes(orig_img.affine)
    print(f"  Original orientation: {orig_orient}")

    ras_img = nib.as_closest_canonical(orig_img)   # reorient to RAS
    ras_orient = nib.aff2axcodes(ras_img.affine)
    print(f"  Reoriented to       : {ras_orient}")

    ras_path = os.path.join(OUT_DIR, f"{case_name}_RAS.nii.gz")
    nib.save(ras_img, ras_path)
    print(f"  Saved RAS volume    : {ras_path}  shape={ras_img.shape}")

    # ── run TotalSegmentator ──────────────────────────────────────────────────
    # task="total"    → segment all 117 structures
    # fast=False      → full-resolution model (better detection)
    # ml=False        → individual per-organ .nii.gz output files
    print(f"\n  Running TotalSegmentator (full-resolution mode) …")
    print(f"  Output dir : {OUT_DIR}/")
    print(f"  This takes 5–15 min on CPU, ~2 min on GPU. Grab a coffee ☕\n")

    totalsegmentator(
        input  = ras_path,
        output = OUT_DIR,
        task   = "total",
        fast   = False,      # full-resolution model for accurate detection
        ml     = False,      # individual per-organ .nii.gz files
        quiet  = False,
    )

    # ── check which BTCV-relevant organs were found ───────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Organs detected (BTCV-relevant subset):")
    found = {}
    for btcv_name, ts_name in BTCV_TO_TOTALSEG.items():
        seg_path = os.path.join(OUT_DIR, f"{ts_name}.nii.gz")
        if os.path.exists(seg_path):
            arr      = nib.load(seg_path).get_fdata()
            vox      = int(arr.sum())
            found[btcv_name] = {"path": seg_path, "voxels": vox}
            status = f"{vox:>10,} voxels"
            print(f"    {'✓':<3} {btcv_name:<18} → {ts_name:<20} {status}")
        else:
            print(f"    {'–':<3} {btcv_name:<18} → {ts_name:<20} not found")

    # ── save a quick 3-panel verification PNG ─────────────────────────────────
    print(f"\n  Saving verification overlay PNG …")

    img_nib = nib.load(ras_path)
    img_arr = img_nib.get_fdata().astype(np.float32)
    # Window for display
    img_w   = np.clip(img_arr, -175, 250)
    img_n   = (img_w - img_w.min()) / (img_w.max() - img_w.min() + 1e-8)

    # Pick the spleen mask for overlay (most reliable organ)
    overlay_organ = "spleen" if "spleen" in found else (list(found.keys())[0] if found else None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#111")

    if overlay_organ:
        mask_arr = nib.load(found[overlay_organ]["path"]).get_fdata().astype(bool)
        # BTCV nii is (H,W,D) in nibabel; find centre of organ
        coords = np.where(mask_arr)
        if len(coords[0]) > 0:
            cx = int(np.mean(coords[0]))
            cy = int(np.mean(coords[1]))
            cz = int(np.mean(coords[2]))
        else:
            cx = img_arr.shape[0]//2
            cy = img_arr.shape[1]//2
            cz = img_arr.shape[2]//2
    else:
        cx = img_arr.shape[0]//2
        cy = img_arr.shape[1]//2
        cz = img_arr.shape[2]//2
        mask_arr = None

    views = [
        (img_n[:, :, cz],    mask_arr[:, :, cz]    if mask_arr is not None else None, f"Axial z={cz}"),
        (img_n[:, cy, :],    mask_arr[:, cy, :]    if mask_arr is not None else None, f"Coronal y={cy}"),
        (img_n[cx, :, :],    mask_arr[cx, :, :]    if mask_arr is not None else None, f"Sagittal x={cx}"),
    ]

    for ax, (vol_sl, mask_sl, title) in zip(axes, views):
        ax.set_facecolor("#111")
        ax.imshow(np.rot90(vol_sl), cmap="gray", origin="upper", aspect="auto")
        if mask_sl is not None and mask_sl.any():
            ov = np.zeros((*np.rot90(mask_sl).shape, 4), dtype=np.float32)
            ov[np.rot90(mask_sl)] = [1.0, 0.3, 0.1, 0.55]
            ax.imshow(ov, origin="upper", aspect="auto")
            ax.contour(np.rot90(mask_sl).astype(float),
                       levels=[0.5], colors=["#FFD700"], linewidths=1.5)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    title_str = (f"{case_name}  |  TotalSegmentator fast mode  |  "
                 f"overlay: {overlay_organ or 'none'}")
    plt.suptitle(title_str, color="white", fontsize=11, y=1.01)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{case_name}_totalseg_verify.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  PNG saved → {out_png}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TotalSegmentator test COMPLETE")
    print(f"  Organs found : {len(found)} / {len(BTCV_TO_TOTALSEG)}")
    print(f"  Output dir   : {OUT_DIR}/")
    print(f"  Verify PNG   : {out_png}")
    print(f"\n  Next step:")
    print(f"    python file2_btcv_slice_to_png.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
