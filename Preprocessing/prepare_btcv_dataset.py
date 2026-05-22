"""
prepare_btcv_dataset.py
=======================
Converts 3D NIfTI BTCV volumes into 2D PNG slices + NPZ label/box metadata
suitable for MedSAM fine-tuning with box prompts.

Usage:
    python Box_Prompt/prepare_btcv_dataset.py

Output structure:
    data/btcv/processed/{train,val,test}/
        images/  — PNG files (1024x1024 RGB)
        labels/  — NPZ files (masks + bounding boxes per organ)
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

# ─── BTCV Organ Label Map ─────────────────────────────────────────────────────
ORGAN_NAMES = {
    1:  "spleen",
    2:  "right_kidney",
    3:  "left_kidney",
    4:  "gallbladder",
    5:  "esophagus",
    6:  "liver",
    7:  "stomach",
    8:  "aorta",
    9:  "inferior_vena_cava",
    10: "portal_vein_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}

MEDSAM_IMG_SIZE = 1024


def ct_windowing(img_arr, window_width=400, window_level=40):
    """Apply CT soft-tissue window and normalise to [0, 255]."""
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    img_arr = np.clip(img_arr, lower, upper)
    img_arr = (img_arr - lower) / (upper - lower) * 255.0
    return img_arr.astype(np.uint8)


def get_bbox_from_mask(mask_2d, img_h, img_w, margin=5):
    """
    Compute tight bounding box [x_min, y_min, x_max, y_max] from a binary mask.
    Adds a small margin and clips to image bounds.
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add margin
    x_min = max(0, int(cmin) - margin)
    y_min = max(0, int(rmin) - margin)
    x_max = min(img_w - 1, int(cmax) + margin)
    y_max = min(img_h - 1, int(rmax) + margin)

    return [x_min, y_min, x_max, y_max]


def process_volume(img_path, lbl_path, out_img_dir, out_lbl_dir, vol_name):
    """Process a single NIfTI volume into 2D slices."""
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img_data = img_nii.get_fdata().astype(np.float32)
    lbl_data = lbl_nii.get_fdata().astype(np.int16)

    # Iterate over axial slices (last axis)
    n_slices = img_data.shape[2]
    count = 0

    for s in range(n_slices):
        img_slice = img_data[:, :, s]
        lbl_slice = lbl_data[:, :, s]

        # Skip empty slices (no foreground organ)
        unique_labels = np.unique(lbl_slice)
        fg_labels = [l for l in unique_labels if l > 0]
        if len(fg_labels) == 0:
            continue

        # CT windowing → grayscale → 3-channel RGB
        img_uint8 = ct_windowing(img_slice)
        img_rgb = np.stack([img_uint8] * 3, axis=-1)  # H x W x 3

        orig_h, orig_w = img_rgb.shape[:2]

        # Resize image to 1024x1024
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE), Image.BILINEAR)
        img_resized = np.array(pil_img)

        # Scale factors for bounding boxes and masks
        scale_h = MEDSAM_IMG_SIZE / orig_h
        scale_w = MEDSAM_IMG_SIZE / orig_w

        # Process each organ in this slice
        masks = {}
        boxes = {}
        organ_ids = []

        for organ_id in fg_labels:
            organ_id = int(organ_id)
            if organ_id not in ORGAN_NAMES:
                continue

            # Binary mask for this organ
            organ_mask = (lbl_slice == organ_id).astype(np.uint8)

            # Resize mask to 1024x1024 (nearest neighbour to preserve labels)
            pil_mask = Image.fromarray(organ_mask * 255)
            pil_mask = pil_mask.resize(
                (MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE), Image.NEAREST
            )
            mask_resized = (np.array(pil_mask) > 127).astype(np.uint8)

            # Compute bounding box on resized mask
            bbox = get_bbox_from_mask(mask_resized, MEDSAM_IMG_SIZE, MEDSAM_IMG_SIZE)
            if bbox is None:
                continue

            masks[organ_id] = mask_resized
            boxes[organ_id] = np.array(bbox, dtype=np.float32)
            organ_ids.append(organ_id)

        if len(organ_ids) == 0:
            continue

        # Save image as PNG
        slice_name = f"{vol_name}_s{s:04d}"
        img_save_path = os.path.join(out_img_dir, f"{slice_name}.png")
        Image.fromarray(img_resized).save(img_save_path)

        # Save labels + boxes as NPZ
        lbl_save_path = os.path.join(out_lbl_dir, f"{slice_name}.npz")
        save_dict = {"organ_ids": np.array(organ_ids, dtype=np.int32)}
        for oid in organ_ids:
            save_dict[f"mask_{oid}"] = masks[oid]
            save_dict[f"box_{oid}"] = boxes[oid]
        np.savez_compressed(lbl_save_path, **save_dict)

        count += 1

    return count


def parse_split_file(split_file, data_root):
    """Parse a BTCV split file and return list of (img_path, lbl_path, vol_name)."""
    pairs = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_rel, lbl_rel = parts[0], parts[1]

            # The paths in the split files use "01_btcv/" prefix
            # but actual files are directly under kaggle_btcv/
            img_rel = img_rel.replace("01_btcv/", "")
            lbl_rel = lbl_rel.replace("01_btcv/", "")

            img_path = os.path.join(data_root, "kaggle_btcv", img_rel)
            lbl_path = os.path.join(data_root, "kaggle_btcv", lbl_rel)

            vol_name = os.path.splitext(os.path.basename(img_rel))[0]

            if os.path.exists(img_path) and os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path, vol_name))
            else:
                print(f"  [WARN] Skipping missing pair: {img_path} / {lbl_path}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare BTCV for MedSAM fine-tuning")
    parser.add_argument(
        "--data_root",
        default="data/btcv/raw/data",
        help="Root of BTCV data (contains kaggle_btcv/ and dataset_test_list2/)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/btcv/processed",
        help="Output directory for processed slices",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir

    split_dir = os.path.join(data_root, "dataset_test_list2")

    splits = {
        "train": os.path.join(split_dir, "btcv_train.txt"),
        "val": os.path.join(split_dir, "btcv_val.txt"),
        "test": os.path.join(split_dir, "btcv_test.txt"),
    }

    total_slices = 0
    for split_name, split_file in splits.items():
        if not os.path.exists(split_file):
            print(f"[WARN] Split file not found: {split_file}, skipping {split_name}")
            continue

        out_img_dir = os.path.join(output_dir, split_name, "images")
        out_lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        pairs = parse_split_file(split_file, data_root)
        print(f"\n{'='*60}")
        print(f"  Processing [{split_name.upper()}] — {len(pairs)} volumes")
        print(f"{'='*60}")

        split_count = 0
        for i, (img_path, lbl_path, vol_name) in enumerate(pairs):
            print(f"  [{i+1}/{len(pairs)}] {vol_name} ...", end=" ", flush=True)
            n = process_volume(img_path, lbl_path, out_img_dir, out_lbl_dir, vol_name)
            print(f"→ {n} slices")
            split_count += n

        print(f"  Total {split_name} slices: {split_count}")
        total_slices += split_count

    print(f"\n{'='*60}")
    print(f"  ALL DONE — {total_slices} total slices saved to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
