# =============================================================================
# step0_use_btcv.py — Use Existing BTCV Data (ZERO extra download)
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   AbdomenAtlas 2.0 is too large (~18 GB labels + ~400 GB CTs).
#   Your BTCV dataset ALREADY has all 5 target organs labeled:
#     - liver (BTCV class 6)
#     - right kidney (class 2) + left kidney (class 3)
#     - pancreas (class 11)
#     - esophagus (class 5)
#     - stomach (class 7) — used as stomach/colon proxy
#
#   This script scans your existing BTCV data, remaps the labels to
#   our 11-class format, and produces the same filtered_cases.json
#   manifest that all downstream training scripts expect.
#
#   ZERO additional download required.  Uses your existing ~6 GB BTCV data.
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#   python step0_use_btcv.py
#   python step0_use_btcv.py --dry_run           # preview without writing
#   python step0_use_btcv.py --max_cases 20      # limit cases
#
# AFTER THIS:
#   python step3_train.py --epochs 10            # train on BTCV data
# =============================================================================

import os
import sys
import json
import argparse
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASS_MAP, MANIFEST_PATH

# =============================================================================
# BTCV → Our Class Mapping
#
# BTCV labels:                    Our labels:
#   0: background                   0: background
#   1: spleen                       (skip)
#   2: right_kidney          →      3: kidney
#   3: left_kidney           →      3: kidney
#   4: gallbladder                  (skip)
#   5: esophagus             →      9: esophagus
#   6: liver                 →      1: liver
#   7: stomach               →      7: colon (stomach/colon proxy)
#   8: aorta                        (skip)
#   9: inferior_vena_cava           (skip)
#  10: portal_vein                  (skip)
#  11: pancreas              →      5: pancreas
#  12: right_adrenal                (skip)
#  13: left_adrenal                 (skip)
#
# NOTE: BTCV does NOT have tumor labels.  The model will learn organ
# segmentation.  For tumor detection, use the existing SegResNet model
# (liver tumors) or add tumor pseudo-labels later.
# =============================================================================
BTCV_TO_OURS = {
    0: 0,    # background → background
    2: 3,    # right_kidney → kidney
    3: 3,    # left_kidney  → kidney
    5: 9,    # esophagus → esophagus
    6: 1,    # liver → liver
    7: 7,    # stomach → colon/stomach
    11: 5,   # pancreas → pancreas
}


def find_btcv_pairs(btcv_root):
    """
    Find matching (image, label) pairs in the BTCV directory.

    BTCV naming: img0001-0002.nii  ↔  label0001-0002.nii
    """
    img_dir = os.path.join(btcv_root, "images")
    lbl_dir = os.path.join(btcv_root, "labels")

    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        # Try alternate paths
        for alt in [btcv_root, os.path.join(btcv_root, "data")]:
            if os.path.isdir(os.path.join(alt, "images")):
                img_dir = os.path.join(alt, "images")
                lbl_dir = os.path.join(alt, "labels")
                break

    images = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    pairs = []

    for img_path in images:
        basename = os.path.basename(img_path)
        # img0001-0002.nii → label0001-0002.nii
        lbl_name = basename.replace("img", "label")
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))

    return pairs


def remap_and_save_labels(lbl_path, output_path):
    """
    Load a BTCV label, remap to our class IDs, and save.
    Returns dict of organ voxel counts.
    """
    import nibabel as nib

    nii = nib.load(lbl_path)
    data = nii.get_fdata().astype(np.int16)

    # Remap
    remapped = np.zeros_like(data)
    for btcv_id, our_id in BTCV_TO_OURS.items():
        remapped[data == btcv_id] = our_id

    # Save remapped label
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_nii = nib.Nifti1Image(remapped, nii.affine, nii.header)
    nib.save(out_nii, output_path)

    # Count voxels per class
    counts = {}
    for our_id, name in CLASS_MAP.items():
        if our_id == 0:
            continue
        c = int((remapped == our_id).sum())
        if c > 0:
            counts[name] = c

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Use existing BTCV data for multi-organ training (ZERO download)"
    )
    parser.add_argument("--btcv_root", default=None,
                        help="Path to BTCV data (default: auto-detect)")
    parser.add_argument("--output", default=None,
                        help="Output manifest path")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit number of cases")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()

    # Auto-detect BTCV path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_btcv = os.path.join(script_dir, "..", "btcv", "raw", "data", "kaggle_btcv")
    btcv_root = args.btcv_root or default_btcv
    output_path = args.output or MANIFEST_PATH

    print("\n" + "=" * 62)
    print("  BTCV → Multi-Organ Dataset (ZERO download)")
    print("  Remapping BTCV labels to our 11-class format")
    print("=" * 62)

    # Find pairs
    pairs = find_btcv_pairs(btcv_root)
    if not pairs:
        print(f"\n  ERROR: No BTCV data found at: {btcv_root}")
        print(f"  Expected: {btcv_root}/images/img*.nii")
        sys.exit(1)

    if args.max_cases:
        pairs = pairs[:args.max_cases]

    print(f"\n  BTCV root:     {btcv_root}")
    print(f"  Pairs found:   {len(pairs)}")

    # Remap labels
    remapped_dir = os.path.join(script_dir, "data", "btcv_remapped")
    cases = []
    organ_totals = {}

    for i, (img_path, lbl_path) in enumerate(pairs):
        basename = os.path.basename(img_path).replace(".nii.gz", "").replace(".nii", "")
        case_name = basename.replace("img", "case")

        # Output paths
        remapped_lbl_path = os.path.join(remapped_dir, case_name, "label_remapped.nii.gz")

        if args.dry_run:
            # Just check the label content
            import nibabel as nib
            data = nib.load(lbl_path).get_fdata().astype(np.int16)
            counts = {}
            for btcv_id, our_id in BTCV_TO_OURS.items():
                if our_id == 0:
                    continue
                c = int((data == btcv_id).sum())
                if c > 0:
                    name = CLASS_MAP[our_id]
                    counts[name] = c
        else:
            counts = remap_and_save_labels(lbl_path, remapped_lbl_path)

        for name, c in counts.items():
            organ_totals[name] = organ_totals.get(name, 0) + c

        cases.append({
            "case_name": case_name,
            "image_path": img_path,
            "label_path": remapped_lbl_path,
            "original_label": lbl_path,
            "source": "btcv",
            "organs": counts,
            "has_ct": True,
            "tumors": {},       # BTCV has no tumor labels
            "num_tumors": 0,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"    Processed {i+1}/{len(pairs)}")

    # Summary
    print(f"\n{'=' * 62}")
    print(f"  RESULTS")
    print(f"{'=' * 62}")
    print(f"  Cases processed: {len(cases)}")
    print(f"\n  Organ coverage across all cases:")
    for name, total in sorted(organ_totals.items()):
        bar = "█" * min(40, total // 50000) + "░" * max(0, 40 - total // 50000)
        print(f"    {name:<18} {total:>12,} voxels  {bar}")

    print(f"\n  ⚠ NOTE: BTCV has NO tumor labels.")
    print(f"    The model will learn organ segmentation.")
    print(f"    For tumor detection: use your existing SegResNet model")
    print(f"    or add AbdomenAtlas 2.0 data later when disk space allows.")

    if args.dry_run:
        print(f"\n  [DRY RUN] — No files written.")
        print(f"  Run without --dry_run to create remapped labels (~500 MB)")
        return

    # Estimate remapped size
    remap_size_mb = len(cases) * 8  # ~8 MB per remapped label
    print(f"\n  Remapped labels saved to: {remapped_dir}")
    print(f"  Estimated size: ~{remap_size_mb} MB")

    # Write manifest
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manifest = {
        "source": "btcv_remapped",
        "total_scanned": len(pairs),
        "total_filtered": len(cases),
        "tumor_type_counts": {},
        "organ_totals": organ_totals,
        "note": "BTCV organ labels only — no tumor annotations",
        "cases": cases,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Manifest saved: {output_path}")
    print(f"\n  Next steps:")
    print(f"    python step3_train.py --epochs 10   # train on BTCV organs")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
