# =============================================================================
# step5_inference.py — Inference + Pixel Count Report for Text Bot
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Run the trained multi-organ model on a new patient CT scan.
#   Produces:
#     1. Per-organ binary masks (saved as prediction.nii.gz)
#     2. Pixel/voxel count report (for text-based reporting bot)
#     3. Summary JSON (machine-readable)
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#   python step5_inference.py --input /path/to/patient_ct.nii.gz
#   python step5_inference.py --input /path/to/patient_ct.nii.gz --save_masks
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np

import torch
import nibabel as nib

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensityRange,
    Spacing, Orientation, CropForeground, SpatialPad,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_CLASSES, CLASS_MAP, PATCH_SIZE, TARGET_SPACING,
    HU_MIN, HU_MAX, SW_BATCH_SIZE, SW_OVERLAP,
    BEST_MODEL_PATH, OUTPUT_DIR, CLASS_COLORS,
)


def load_model(checkpoint_path, device):
    """Load trained UNet model."""
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256), strides=(2, 2, 2),
        num_res_units=2, dropout=0.2,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


@torch.no_grad()
def run_inference(model, ct_path, device):
    """
    Run multi-organ inference on a CT volume.

    Returns:
        pred_mask: numpy array (H, W, D) with class IDs
        ct_data:   preprocessed CT volume
        original_nii: original NIfTI for header/affine
    """
    print(f"  Loading CT: {ct_path}")
    original_nii = nib.load(ct_path)
    ct_data = original_nii.get_fdata().astype(np.float32)

    # Apply the same preprocessing as training
    # HU windowing
    ct_data = np.clip(ct_data, HU_MIN, HU_MAX)
    ct_data = (ct_data - HU_MIN) / (HU_MAX - HU_MIN)

    # Convert to tensor: (1, 1, H, W, D)
    ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).to(device)

    print(f"  Volume shape: {ct_data.shape}")
    print(f"  Running sliding window inference...")

    # Sliding window inference
    outputs = sliding_window_inference(
        ct_tensor, PATCH_SIZE, SW_BATCH_SIZE, model, overlap=SW_OVERLAP,
    )

    # Argmax → class predictions
    pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.int16)

    return pred_mask, ct_data, original_nii


def generate_pixel_report(pred_mask, voxel_spacing=None):
    """
    Generate a pixel/voxel count report for the text-based reporting bot.

    Returns:
        report_str: formatted text report
        report_dict: machine-readable dict
    """
    # Calculate voxel volume in mm³ if spacing is available
    voxel_vol_mm3 = 1.0
    if voxel_spacing is not None:
        voxel_vol_mm3 = float(np.prod(voxel_spacing))

    # Pair organs and their tumors
    organ_pairs = [
        ("liver",     1, "liver_tumor",     2),
        ("kidney",    3, "kidney_tumor",    4),
        ("pancreas",  5, "pancreas_tumor",  6),
        ("colon",     7, "colon_tumor",     8),
        ("esophagus", 9, "esophagus_tumor", 10),
    ]

    divider = "=" * 78
    thin = "-" * 78

    lines = []
    lines.append(f"\n{divider}")
    lines.append(f"  MULTI-ORGAN TUMOR SEGMENTATION REPORT")
    lines.append(f"{divider}")
    lines.append(f"")
    lines.append(f"  {'ORGAN':<16} {'DETECTED':>10} {'TUMOR':>8} "
                 f"{'ORGAN_PX':>12} {'TUMOR_PX':>12} {'TUMOR_%':>10}")
    lines.append(f"  {thin}")

    report_dict = {"organs": {}}
    total_tumor_voxels = 0
    tumors_detected = []

    for organ_name, organ_id, tumor_name, tumor_id in organ_pairs:
        organ_px = int((pred_mask == organ_id).sum())
        tumor_px = int((pred_mask == tumor_id).sum())
        total_px = organ_px + tumor_px  # total organ region including tumor

        organ_detected = "YES" if total_px > 0 else "NO"
        tumor_detected = "YES" if tumor_px > 0 else "NO"
        tumor_pct = (100.0 * tumor_px / total_px) if total_px > 0 else 0.0

        status_marker = " ⚠" if tumor_px > 0 else ""
        lines.append(f"  {organ_name:<16} {organ_detected:>10} {tumor_detected:>8} "
                     f"{organ_px:>12,} {tumor_px:>12,} {tumor_pct:>9.2f}%{status_marker}")

        total_tumor_voxels += tumor_px
        if tumor_px > 0:
            tumors_detected.append(organ_name)

        # Volume in mm³
        organ_vol_mm3 = total_px * voxel_vol_mm3
        tumor_vol_mm3 = tumor_px * voxel_vol_mm3

        report_dict["organs"][organ_name] = {
            "organ_voxels": organ_px,
            "tumor_voxels": tumor_px,
            "tumor_percentage": round(tumor_pct, 2),
            "organ_volume_mm3": round(organ_vol_mm3, 1),
            "tumor_volume_mm3": round(tumor_vol_mm3, 1),
            "organ_detected": organ_detected == "YES",
            "tumor_detected": tumor_detected == "YES",
        }

    lines.append(f"  {thin}")

    # Summary
    lines.append(f"")
    if tumors_detected:
        lines.append(f"  ⚠ TUMORS DETECTED in: {', '.join(tumors_detected)}")
        lines.append(f"    Total tumor voxels: {total_tumor_voxels:,}")
        lines.append(f"    Total tumor volume: {total_tumor_voxels * voxel_vol_mm3:.1f} mm³")
    else:
        lines.append(f"  ✓ No tumors detected — all organs appear normal.")

    lines.append(f"\n{divider}\n")

    report_dict["tumors_detected"] = tumors_detected
    report_dict["total_tumor_voxels"] = total_tumor_voxels
    report_dict["total_tumor_volume_mm3"] = round(total_tumor_voxels * voxel_vol_mm3, 1)

    report_str = "\n".join(lines)
    return report_str, report_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-organ inference on a patient CT scan"
    )
    parser.add_argument("--input", required=True,
                        help="Path to patient CT volume (.nii or .nii.gz)")
    parser.add_argument("--checkpoint", default=BEST_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_masks", action="store_true",
                        help="Save per-organ mask files")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 62)
    print("  MULTI-ORGAN TUMOR SEGMENTATION — Inference")
    print(f"  Device: {device}")
    print("=" * 62)

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"\n  ERROR: Input not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"\n  ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"  Train a model first: python step3_train.py")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, device)

    # Run inference
    pred_mask, ct_data, original_nii = run_inference(model, args.input, device)

    # Get voxel spacing
    try:
        spacing = original_nii.header.get_zooms()[:3]
    except Exception:
        spacing = (1.0, 1.0, 1.0)

    # Generate report
    report_str, report_dict = generate_pixel_report(pred_mask, spacing)
    print(report_str)

    # Save prediction mask
    pred_path = os.path.join(output_dir, "prediction.nii.gz")
    pred_nii = nib.Nifti1Image(pred_mask, original_nii.affine, original_nii.header)
    nib.save(pred_nii, pred_path)
    print(f"  Prediction mask saved: {pred_path}")

    # Save JSON report
    report_dict["input_file"] = args.input
    report_dict["checkpoint"] = args.checkpoint
    report_dict["voxel_spacing_mm"] = list(spacing)
    report_path = os.path.join(output_dir, "inference_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  JSON report saved:     {report_path}")

    # Save per-organ masks if requested
    if args.save_masks:
        for cls_id in range(1, NUM_CLASSES):
            cls_name = CLASS_MAP[cls_id]
            mask = (pred_mask == cls_id).astype(np.int16)
            if mask.sum() == 0:
                continue
            mask_path = os.path.join(output_dir, f"{cls_name}.nii.gz")
            mask_nii = nib.Nifti1Image(mask, original_nii.affine, original_nii.header)
            nib.save(mask_nii, mask_path)
            print(f"  Saved: {mask_path} ({mask.sum():,} voxels)")

    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    main()
