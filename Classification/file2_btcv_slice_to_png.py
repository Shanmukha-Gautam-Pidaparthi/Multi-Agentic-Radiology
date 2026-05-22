# =============================================================================
# file2_btcv_slice_to_png.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   Extract the best 2D axial slice from a BTCV CT volume and save
#            as a PNG. Finds the slice where the most organ labels are visible
#            so you get a rich test image for box-prompting.
#            Also saves a labelled PNG showing which organs are on the slice.
#
# Run:       python file2_btcv_slice_to_png.py
#            python file2_btcv_slice_to_png.py --case img0007-0001.nii
# =============================================================================

import os, sys, argparse
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
OUT_DIR       = "slice_output"

ORGAN_NAMES = {
    1:"spleen", 2:"right_kidney", 3:"left_kidney", 4:"gallbladder",
    5:"esophagus", 6:"liver", 7:"stomach", 8:"aorta", 9:"ivc",
    10:"portal_vein", 11:"pancreas", 12:"adrenal_R", 13:"adrenal_L"
}

# Distinct colours for each organ overlay
ORGAN_COLORS = [
    None,
    "#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6",
    "#1ABC9C","#E67E22","#E91E63","#00BCD4","#8BC34A",
    "#FF5722","#607D8B","#795548"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default=None,
                   help="e.g. img0001-0002.nii  (auto-picks best case if not set)")
    return p.parse_args()


def find_best_slice(label_arr):
    """
    Find the axial slice (z-index) where the most different organ labels
    are visible. This gives the richest single-slice test image.
    """
    best_z     = label_arr.shape[2] // 2
    best_count = 0
    for z in range(label_arr.shape[2]):
        sl      = label_arr[:, :, z]
        n_organs= len(np.unique(sl[sl > 0]))
        if n_organs > best_count:
            best_count = n_organs
            best_z     = z
    return best_z, best_count


def save_plain_png(img_slice, out_path):
    """Save clean CT slice as 8-bit greyscale PNG (no annotations)."""
    # Soft-tissue window
    sl = np.clip(img_slice, -175, 250)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    sl_uint8 = (sl * 255).astype("uint8")
    from PIL import Image
    Image.fromarray(sl_uint8).save(out_path)
    print(f"  Plain PNG saved    → {out_path}")


def save_annotated_png(img_slice, label_slice, present_organs, out_path):
    """Save CT slice with coloured organ overlays and bounding boxes."""
    sl = np.clip(img_slice, -175, 250)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.imshow(np.rot90(sl), cmap="gray", origin="upper", aspect="equal")

    legend_patches = []
    for organ_id in present_organs:
        color = ORGAN_COLORS[organ_id]
        name  = ORGAN_NAMES[organ_id]
        mask  = (label_slice == organ_id)
        mask_rot = np.rot90(mask)

        # Colour fill overlay
        ov = np.zeros((*mask_rot.shape, 4), dtype=np.float32)
        ov[mask_rot] = [*[int(color.lstrip('#')[i:i+2], 16)/255 for i in (0,2,4)], 0.40]
        ax.imshow(ov, origin="upper", aspect="equal")

        # Contour
        ax.contour(mask_rot.astype(float), levels=[0.5],
                   colors=[color], linewidths=1.5)

        # Bounding box + label
        rows = np.where(np.any(mask_rot, axis=1))[0]
        cols = np.where(np.any(mask_rot, axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            r1,r2 = rows[0], rows[-1]
            c1,c2 = cols[0], cols[-1]
            rect = mpatches.Rectangle(
                (c1, r1), c2-c1, r2-r1,
                linewidth=1.5, edgecolor=color,
                facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(c1+2, r1-4, name, color=color, fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d0d0d",
                              edgecolor=color, alpha=0.85))

        legend_patches.append(mpatches.Patch(color=color, label=name))

    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a1a", labelcolor="white",
              fontsize=7, framealpha=0.9, ncol=2)
    ax.axis("off")
    ax.set_title(f"BTCV axial slice — {len(present_organs)} organs visible",
                 color="white", fontsize=11, pad=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  Annotated PNG saved→ {out_path}")


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  FILE 2 — BTCV slice → PNG extractor")
    print("="*60)

    import glob
    imgs = sorted(glob.glob(os.path.join(RAW_IMG_DIR, "*.nii")))
    if not imgs:
        print(f"  ERROR: No .nii files in {RAW_IMG_DIR}")
        sys.exit(1)

    # Select case
    if args.case:
        matched = [ip for ip in imgs if args.case in os.path.basename(ip)]
        if not matched:
            print(f"  ERROR: '{args.case}' not found."); sys.exit(1)
        img_path = matched[0]
    else:
        # Auto-pick: find case with most organs
        best_img   = None
        best_score = 0
        print("  Auto-selecting richest case …")
        for ip in imgs[:20]:   # check first 20 cases
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
    case_name  = os.path.basename(img_path).replace(".nii","")

    # Load
    img_nib   = nib.load(img_path)
    img_arr   = img_nib.get_fdata().astype(np.float32)   # (H, W, D)
    label_arr = nib.load(label_path).get_fdata().astype(np.int16) \
                if os.path.exists(label_path) else None

    print(f"\n  Case      : {case_name}")
    print(f"  Volume    : {img_arr.shape}  (H × W × D)")
    print(f"  HU range  : [{img_arr.min():.0f}, {img_arr.max():.0f}]")

    # Find best slice
    if label_arr is not None:
        best_z, n_organs = find_best_slice(label_arr)
        print(f"  Best slice: z={best_z}  ({n_organs} organs visible)")
    else:
        best_z = img_arr.shape[2] // 2
        n_organs = 0
        print(f"  No label file — using middle slice z={best_z}")

    img_slice   = img_arr[:, :, best_z]
    label_slice = label_arr[:, :, best_z] if label_arr is not None else None

    # Which organs are on this slice?
    present = []
    if label_slice is not None:
        present = sorted([int(i) for i in np.unique(label_slice) if i > 0])
        print(f"\n  Organs on slice z={best_z}:")
        for oid in present:
            vox = int((label_slice == oid).sum())
            print(f"    {oid:2d}. {ORGAN_NAMES.get(oid,'?'):<18} {vox:>6,} px")

    # Save plain PNG (input for TotalSegmentator / box-prompt)
    plain_path = os.path.join(OUT_DIR, f"{case_name}_slice_z{best_z}.png")
    save_plain_png(img_slice, plain_path)

    # Save annotated PNG (reference for what organs should be found)
    if label_slice is not None and present:
        ann_path = os.path.join(OUT_DIR, f"{case_name}_slice_z{best_z}_annotated.png")
        save_annotated_png(img_slice, label_slice, present, ann_path)

    # Also save the slice as a standalone NIfTI for TotalSegmentator
    sl_3d    = img_arr[:, :, max(0,best_z-1):best_z+2]   # 3 slices (min depth TotalSeg needs)
    sl_nib   = nib.Nifti1Image(sl_3d, img_nib.affine)
    nii_path = os.path.join(OUT_DIR, f"{case_name}_slice_z{best_z}.nii.gz")
    nib.save(sl_nib, nii_path)
    print(f"  Slice NIfTI saved  → {nii_path}  shape={sl_3d.shape}")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Plain PNG  : {plain_path}")
    print(f"  Slice NIfTI: {nii_path}")
    print(f"\n  Next step:")
    print(f"    python file3_totalseg_classify_2d.py --png {plain_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
