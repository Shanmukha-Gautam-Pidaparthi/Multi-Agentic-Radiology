# =============================================================================
# step1_subset_filter.py — Scan AbdomenAtlas 2.0 Labels & Build Case Manifest
# Place at: ~/Desktop/intern/Box_Prompt/AbdomenAtlas/
#
# PURPOSE:
#   Walk the extracted label directories, check which cases contain
#   tumor annotations for our 5 target organs, and produce a JSON
#   manifest of only those cases.  This manifest drives all downstream
#   scripts (training, evaluation, etc.).
#
# USAGE:
#   conda activate medsam_env
#   cd ~/Desktop/intern/Box_Prompt/AbdomenAtlas
#   python step1_subset_filter.py --data_dir data/AbdomenAtlas2.0
#   python step1_subset_filter.py --data_dir data/AbdomenAtlas2.0 --dry_run
#
# OUTPUT:
#   data/filtered_cases.json — JSON with per-case metadata
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np

# Add parent for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_TUMOR_FILES, SEGMENTATION_FILES, MANIFEST_PATH


def scan_case(case_dir, case_name):
    """
    Check a single AbdomenAtlas case directory for our target tumor masks.

    AbdomenAtlas 2.0 per-case structure:
        BDMAP_XXXXXXXX/
        ├── ct.nii.gz
        └── segmentations/
            ├── liver.nii.gz
            ├── liver_tumor.nii.gz
            ├── kidney_left.nii.gz
            ├── kidney_tumor.nii.gz
            ├── pancreas.nii.gz
            ├── pancreas_tumor.nii.gz
            ├── colon.nii.gz
            ├── colon_tumor.nii.gz
            ├── esophagus.nii.gz
            ├── esophagus_tumor.nii.gz
            └── ... (other organs we don't use)

    Returns:
        dict with case metadata, or None if case has no relevant tumors.
    """
    seg_dir = os.path.join(case_dir, "segmentations")
    ct_path = os.path.join(case_dir, "ct.nii.gz")

    # Must have a segmentations folder
    if not os.path.isdir(seg_dir):
        return None

    # Check which of our target tumor files exist and have non-zero voxels
    tumors_found = {}
    organs_found = {}

    for tumor_file in TARGET_TUMOR_FILES:
        tumor_path = os.path.join(seg_dir, tumor_file)
        if os.path.exists(tumor_path):
            try:
                import nibabel as nib
                mask = nib.load(tumor_path).get_fdata()
                voxel_count = int((mask > 0).sum())
                if voxel_count > 0:
                    tumor_name = tumor_file.replace(".nii.gz", "")
                    tumors_found[tumor_name] = voxel_count
            except Exception as e:
                # Skip corrupted files silently
                pass

    # If no tumors found, this case is a "control" — skip it for training
    # to keep our subset minimal.  We want cases WITH tumors.
    if not tumors_found:
        return None

    # Also check which organ masks exist (for the organ segmentation part)
    organ_files = [
        "liver.nii.gz", "kidney_left.nii.gz", "kidney_right.nii.gz",
        "pancreas.nii.gz", "colon.nii.gz", "esophagus.nii.gz",
    ]
    for organ_file in organ_files:
        organ_path = os.path.join(seg_dir, organ_file)
        if os.path.exists(organ_path):
            organ_name = organ_file.replace(".nii.gz", "")
            organs_found[organ_name] = True

    has_ct = os.path.exists(ct_path)

    return {
        "case_name": case_name,
        "case_dir": case_dir,
        "has_ct": has_ct,
        "tumors": tumors_found,       # e.g. {"liver_tumor": 12340, "pancreas_tumor": 3210}
        "organs": list(organs_found.keys()),
        "num_tumors": len(tumors_found),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scan AbdomenAtlas 2.0 labels and filter cases with target tumors"
    )
    parser.add_argument("--data_dir", default="data/AbdomenAtlas2.0",
                        help="Path to extracted AbdomenAtlas2.0 directory")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: from config.py)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print stats without writing JSON")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit total filtered cases (for disk space)")
    args = parser.parse_args()

    output_path = args.output or MANIFEST_PATH
    data_dir = args.data_dir

    print("\n" + "=" * 62)
    print("  AbdomenAtlas 2.0 — Subset Filter")
    print("  Scanning for: liver, kidney, pancreas, colon, esophagus tumors")
    print("=" * 62)

    if not os.path.isdir(data_dir):
        print(f"\n  ERROR: Data directory not found: {data_dir}")
        print(f"  Run step0_download_subset.sh first to download labels.")
        sys.exit(1)

    # Find all case directories (BDMAP_XXXXXXXX pattern)
    all_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith(".")
    ])

    print(f"\n  Total case directories found: {len(all_dirs)}")
    print(f"  Scanning for tumor annotations...\n")

    filtered_cases = []
    tumor_type_counts = {t.replace(".nii.gz", ""): 0 for t in TARGET_TUMOR_FILES}
    skipped = 0

    for i, case_name in enumerate(all_dirs):
        case_dir = os.path.join(data_dir, case_name)
        result = scan_case(case_dir, case_name)

        if result is None:
            skipped += 1
            continue

        filtered_cases.append(result)

        # Update per-tumor-type counts
        for tumor_name in result["tumors"]:
            if tumor_name in tumor_type_counts:
                tumor_type_counts[tumor_name] += 1

        # Progress
        if (i + 1) % 500 == 0 or (i + 1) == len(all_dirs):
            print(f"    Scanned {i+1}/{len(all_dirs)} — "
                  f"found {len(filtered_cases)} cases with tumors")

    # Apply max_cases limit if specified (for low disk space)
    if args.max_cases and len(filtered_cases) > args.max_cases:
        # Prioritize cases with multiple tumor types
        filtered_cases.sort(key=lambda x: -x["num_tumors"])
        filtered_cases = filtered_cases[:args.max_cases]
        print(f"\n  Limited to {args.max_cases} cases (prioritized multi-tumor cases)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  FILTER RESULTS")
    print(f"{'=' * 62}")
    print(f"  Total cases scanned  : {len(all_dirs)}")
    print(f"  Cases with tumors    : {len(filtered_cases)}")
    print(f"  Cases without tumors : {skipped}")
    print(f"  Cases with CT file   : {sum(1 for c in filtered_cases if c['has_ct'])}")
    print(f"\n  Per tumor type:")
    for tumor_name, count in sorted(tumor_type_counts.items()):
        bar = "█" * min(50, count // 10) + "░" * max(0, 50 - count // 10)
        print(f"    {tumor_name:<22} : {count:5d} cases  {bar}")

    # Estimate disk usage (rough: ~40 MB per CT volume)
    est_gb = len(filtered_cases) * 0.04
    print(f"\n  Estimated CT download : ~{est_gb:.1f} GB")
    print(f"{'=' * 62}")

    if args.dry_run:
        print("\n  [DRY RUN] — No files written.")
        return

    # ── Write manifest ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manifest = {
        "total_scanned": len(all_dirs),
        "total_filtered": len(filtered_cases),
        "tumor_type_counts": tumor_type_counts,
        "cases": filtered_cases,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest saved: {output_path}")
    print(f"  Next step:  python step3_train.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
