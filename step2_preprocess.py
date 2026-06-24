# =============================================================================
# step2_preprocess.py — MONAI Preprocessing & Label Merging
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Defines the MONAI transform pipelines for training and validation, and
#   the custom label-merging transform that combines per-organ .nii.gz files
#   into a single multi-class volume.
#
#   This module is imported by step3_train.py and step4_evaluate.py.
#   Can also be run standalone to verify the pipeline works:
#
# USAGE:
#   python step2_preprocess.py --verify --num_samples 2
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np

import torch
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Spacingd,
    Orientationd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    SpatialPadd,
    MapTransform,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEGMENTATION_FILES, HU_MIN, HU_MAX, PATCH_SIZE,
    TARGET_SPACING, NUM_SAMPLES_PER_VOLUME, MANIFEST_PATH,
    RAW_DATA_DIR, NUM_CLASSES,
)


# =============================================================================
# CUSTOM TRANSFORM: Merge per-organ masks into one multi-class volume
# =============================================================================
class MergeAbdomenAtlasLabelsd(MapTransform):
    """
    Custom MONAI dict-transform that reads individual per-organ .nii.gz
    segmentation files from AbdomenAtlas 2.0 and merges them into a single
    multi-class label volume.

    The merge order matters: tumors are applied AFTER organs, so tumor
    voxels overwrite the organ voxels at that location.  This ensures
    a tumor inside a liver is labeled as liver_tumor (2), not liver (1).

    Input:
        data_dict["label"] = path to the case's segmentations/ directory

    Output:
        data_dict["label"] = numpy array of shape (H, W, D) with integer
                             class IDs from config.CLASS_MAP
    """

    def __init__(self, keys=("label",), seg_files=None):
        super().__init__(keys)
        self.seg_files = seg_files or SEGMENTATION_FILES

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg_dir = d[key]

            # We need the reference image shape — load from the CT
            # The CT should already be loaded at this point by LoadImaged
            # but the label is a directory path, not a file.
            # We'll load the first available mask to get the shape.
            ref_shape = None
            ref_affine = None

            for filename, class_id in self.seg_files:
                fpath = os.path.join(seg_dir, filename)
                if os.path.exists(fpath):
                    ref_nii = nib.load(fpath)
                    ref_shape = ref_nii.shape
                    ref_affine = ref_nii.affine
                    break

            if ref_shape is None:
                # No masks found — return zeros
                # Try to get shape from CT
                ct_path = os.path.join(os.path.dirname(seg_dir), "ct.nii.gz")
                if os.path.exists(ct_path):
                    ref_nii = nib.load(ct_path)
                    ref_shape = ref_nii.shape
                    ref_affine = ref_nii.affine
                else:
                    raise ValueError(f"No reference shape found for {seg_dir}")

            # Create empty label volume
            merged = np.zeros(ref_shape, dtype=np.int16)

            # Apply organ masks first, then tumor masks (order from config)
            for filename, class_id in self.seg_files:
                fpath = os.path.join(seg_dir, filename)
                if not os.path.exists(fpath):
                    continue
                try:
                    mask_data = nib.load(fpath).get_fdata()
                    # Overwrite where this mask has nonzero voxels
                    merged[mask_data > 0] = class_id
                except Exception:
                    continue

            # Replace the label path with the actual merged array
            # Save as a temporary NIfTI in memory for MONAI LoadImaged compatibility
            # Actually, since LoadImaged already ran on "image", we need to handle
            # this differently.  We'll store the merged array directly.
            d[key] = merged
            d[f"{key}_meta_dict"] = {
                "filename_or_obj": seg_dir,
                "affine": ref_affine,
                "spatial_shape": ref_shape,
            }

        return d


# =============================================================================
# CUSTOM LOADER: Load CT + merge/load labels in one step
# Supports TWO data formats:
#   1. AbdomenAtlas: label = path to segmentations/ directory
#   2. BTCV remapped: label = path to a single .nii.gz file
# =============================================================================
class LoadAbdomenAtlasData(MapTransform):
    """
    Combined loader that auto-detects data format.

    Format 1 (AbdomenAtlas 2.0):
        "image": path to ct.nii.gz
        "label": path to segmentations/ directory → merges per-organ masks

    Format 2 (BTCV remapped / single label file):
        "image": path to img*.nii
        "label": path to label_remapped.nii.gz → loads directly
    """

    def __init__(self, keys=("image", "label")):
        super().__init__(keys)
        self.merge_transform = MergeAbdomenAtlasLabelsd(keys=("label",))

    def __call__(self, data):
        d = dict(data)

        # Load CT volume
        ct_path = d["image"]
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata().astype(np.float32)
        d["image"] = ct_data
        d["image_meta_dict"] = {
            "filename_or_obj": ct_path,
            "affine": ct_nii.affine,
            "spatial_shape": ct_data.shape,
        }

        # Auto-detect label format
        label_path = d["label"]
        if os.path.isdir(label_path):
            # Format 1: AbdomenAtlas — directory of per-organ masks
            d = self.merge_transform(d)
        else:
            # Format 2: BTCV — single NIfTI label file
            lbl_nii = nib.load(label_path)
            lbl_data = lbl_nii.get_fdata().astype(np.int16)
            d["label"] = lbl_data
            d["label_meta_dict"] = {
                "filename_or_obj": label_path,
                "affine": lbl_nii.affine,
                "spatial_shape": lbl_data.shape,
            }

        return d


# =============================================================================
# CUSTOM CHANNEL-FIRST TRANSFORM for our manual loading
# =============================================================================
class EnsureChannelFirstManual(MapTransform):
    """Add channel dimension to manually loaded numpy arrays."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            if isinstance(arr, np.ndarray):
                d[key] = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, D)
            elif isinstance(arr, torch.Tensor) and arr.ndim == 3:
                d[key] = arr.unsqueeze(0)
        return d


# =============================================================================
# BUILD TRANSFORMS
# =============================================================================
def get_train_transforms():
    """
    Training transform pipeline.

    Key design decisions:
    1. ScaleIntensityRange: HU [-175, 250] → [0, 1] for abdominal soft tissue
    2. Spacing: Resample to (1.5, 1.5, 2.0) mm for consistency across scanners
    3. CropForeground: Remove air/table to reduce volume size
    4. RandCropByPosNegLabel: Extract patches with 2:1 pos:neg ratio
       (ensures tumor voxels appear in most patches despite being tiny)
    5. Light augmentation: flips + rotation + noise (no heavy warps on CPU)
    """
    return Compose([
        # ── Load & prepare ────────────────────────────────────────────
        LoadAbdomenAtlasData(keys=["image", "label"]),
        EnsureChannelFirstManual(keys=["image", "label"]),

        # ── Intensity: HU windowing for abdominal soft tissue ─────────
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        # ── Spatial: standardize voxel spacing ────────────────────────
        Spacingd(
            keys=["image", "label"],
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),

        # ── Spatial: ensure consistent orientation ────────────────────
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # ── Crop: remove surrounding air/table to shrink volume ───────
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # ── Pad: ensure volume is at least PATCH_SIZE ─────────────────
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        # ── Patch extraction: random crops with tumor bias ────────────
        # pos=2, neg=1 means 2/3 of patches will contain tumor voxels
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=2, neg=1,
            num_samples=NUM_SAMPLES_PER_VOLUME,
        ),

        # ── Augmentation: lightweight for CPU ─────────────────────────
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
    ])


def get_val_transforms():
    """
    Validation transforms — NO augmentation, NO random cropping.
    Full volumes are processed via SlidingWindowInferer at inference time.
    """
    return Compose([
        LoadAbdomenAtlasData(keys=["image", "label"]),
        EnsureChannelFirstManual(keys=["image", "label"]),

        # Same intensity + spatial normalization as training
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
    ])


# =============================================================================
# BUILD DATA LIST from manifest
# Supports both AbdomenAtlas and BTCV-remapped manifests.
# =============================================================================
def build_datalist(manifest_path=None, split_ratio=0.8, seed=42):
    """
    Read the filtered_cases.json manifest and build MONAI-format data dicts.
    Auto-detects whether the manifest is from AbdomenAtlas or BTCV.

    Returns:
        train_files: list of {"image": path, "label": path_or_dir}
        val_files:   list of {"image": path, "label": path_or_dir}
    """
    manifest_path = manifest_path or MANIFEST_PATH

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}")
        print(f"Run step0_use_btcv.py or step1_subset_filter.py first.")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    source = manifest.get("source", "abdomenatlas")
    print(f"  Data source: {source}")

    # Build data dicts
    all_files = []
    skipped = 0

    for case in manifest["cases"]:
        if source == "btcv_remapped":
            # BTCV format: image_path and label_path are direct file paths
            img_path = case.get("image_path", "")
            lbl_path = case.get("label_path", "")
            if not os.path.exists(img_path):
                skipped += 1
                continue
            if not os.path.exists(lbl_path):
                skipped += 1
                continue
            all_files.append({"image": img_path, "label": lbl_path})
        else:
            # AbdomenAtlas format: case_dir with ct.nii.gz + segmentations/
            case_dir = case.get("case_dir", "")
            ct_path = os.path.join(case_dir, "ct.nii.gz")
            seg_dir = os.path.join(case_dir, "segmentations")
            if not os.path.exists(ct_path) or not os.path.isdir(seg_dir):
                skipped += 1
                continue
            all_files.append({"image": ct_path, "label": seg_dir})

    if skipped > 0:
        print(f"  Warning: Skipped {skipped} cases (files not found)")

    # Deterministic split
    import random
    rng = random.Random(seed)
    rng.shuffle(all_files)

    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"  Dataset split: {len(train_files)} train, {len(val_files)} val "
          f"(from {len(all_files)} total)")

    return train_files, val_files


# =============================================================================
# STANDALONE VERIFICATION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Verify preprocessing pipeline")
    parser.add_argument("--verify", action="store_true",
                        help="Load a few samples and verify transforms work")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    if not args.verify:
        print("Use --verify to test the pipeline.")
        print("This module is primarily imported by step3_train.py")
        return

    print("\n" + "=" * 62)
    print("  Preprocessing Pipeline Verification")
    print("=" * 62)

    train_files, val_files = build_datalist(args.manifest)

    if not train_files:
        print("  ERROR: No training samples found.")
        return

    # Test training transforms
    train_tfm = get_train_transforms()
    test_files = train_files[:args.num_samples]

    for i, data_dict in enumerate(test_files):
        print(f"\n  [{i+1}/{len(test_files)}] {data_dict['image']}")
        try:
            result = train_tfm(data_dict)
            # RandCropByPosNegLabel returns a list of patches
            if isinstance(result, list):
                for j, patch in enumerate(result):
                    img_shape = patch["image"].shape
                    lbl_shape = patch["label"].shape
                    unique_labels = torch.unique(patch["label"]).tolist()
                    print(f"    Patch {j}: image={img_shape}, label={lbl_shape}, "
                          f"classes={unique_labels}")
            else:
                img_shape = result["image"].shape
                lbl_shape = result["label"].shape
                unique_labels = torch.unique(result["label"]).tolist()
                print(f"    image={img_shape}, label={lbl_shape}, "
                      f"classes={unique_labels}")
            print(f"    ✓ Success")
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 62}")
    print(f"  Verification complete.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
