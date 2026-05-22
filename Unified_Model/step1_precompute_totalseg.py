# =============================================================================
# step1_precompute_totalseg.py
# Place at:  ~/Desktop/intern/Box_Prompt/
# Purpose:   Run TotalSegmentator on BTCV test volumes (ONE-TIME).
#            Saves per-organ masks to totalseg_cache/<case>/ so the
#            integrated pipeline (step2) can classify organs instantly.
#
# Run:       conda activate medsam_env
#            cd ~/Desktop/intern/Box_Prompt
#            python step1_precompute_totalseg.py
#            python step1_precompute_totalseg.py --fast   # (quicker, lower accuracy)
# =============================================================================

import os, sys, glob, argparse, time
import numpy as np
import nibabel as nib

# =============================================================================
# CONFIG
# =============================================================================
RAW_IMG_DIR = "btcv/raw/data/kaggle_btcv/images"
CACHE_DIR   = "totalseg_cache"

# These are the test-set case prefixes (from processed/test/images/)
# We'll auto-detect them from the processed data.
PROCESSED_TEST_DIR = "btcv/processed/test/images"


def get_test_cases():
    """Find unique case names from the processed test images."""
    if not os.path.isdir(PROCESSED_TEST_DIR):
        print(f"  ERROR: {PROCESSED_TEST_DIR} not found")
        sys.exit(1)
    pngs = os.listdir(PROCESSED_TEST_DIR)
    cases = set()
    for f in pngs:
        # e.g. img0008-0002_s0067.png → img0008-0002
        if f.endswith(".png"):
            case = f.rsplit("_s", 1)[0]
            cases.add(case)
    return sorted(cases)


def find_raw_nii(case_name):
    """Find the raw .nii file for a case name (e.g. img0008-0002)."""
    candidates = [
        os.path.join(RAW_IMG_DIR, f"{case_name}.nii"),
        os.path.join(RAW_IMG_DIR, f"{case_name}.nii.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_totalseg_on_case(nii_path, case_name, cache_dir, fast=False):
    """Reorient to RAS and run TotalSegmentator. Returns time taken."""
    from totalsegmentator.python_api import totalsegmentator

    out_dir = os.path.join(cache_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    # Check if already done
    existing = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")
                and f != "input_RAS.nii.gz"]
    if len(existing) > 10:
        print(f"    SKIP (already has {len(existing)} masks)")
        return 0

    # Load and reorient to RAS
    orig = nib.load(nii_path)
    ras = nib.as_closest_canonical(orig)
    ras_path = os.path.join(out_dir, "input_RAS.nii.gz")
    nib.save(ras, ras_path)

    orig_orient = nib.aff2axcodes(orig.affine)
    ras_orient = nib.aff2axcodes(ras.affine)
    print(f"    Orientation: {orig_orient} → {ras_orient}  shape={ras.shape}")

    # Save the orientation mapping so step2 can reverse it
    # Store as a simple text file: original_orient, ras_shape
    meta = {
        "orig_orient": list(orig_orient),
        "ras_orient": list(ras_orient),
        "orig_shape": list(orig.shape),
        "ras_shape": list(ras.shape),
    }
    import json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Run TotalSegmentator
    t0 = time.time()
    totalsegmentator(
        input  = ras_path,
        output = out_dir,
        task   = "total",
        fast   = fast,
        ml     = False,      # individual per-organ .nii.gz files
        quiet  = False,
    )
    elapsed = time.time() - t0
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Use fast (lower-res) model")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Specific cases to process (default: all test cases)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  STEP 1 — Precompute TotalSegmentator on BTCV test volumes")
    print("="*60)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Get test cases
    test_cases = args.cases or get_test_cases()
    print(f"\n  Test cases found: {len(test_cases)}")
    for c in test_cases:
        print(f"    • {c}")

    # Process each
    total_time = 0
    for i, case in enumerate(test_cases, 1):
        nii_path = find_raw_nii(case)
        if nii_path is None:
            print(f"\n  [{i}/{len(test_cases)}] {case} — RAW NII NOT FOUND, skipping")
            continue

        print(f"\n  [{i}/{len(test_cases)}] {case}")
        print(f"    Input: {nii_path}")
        elapsed = run_totalseg_on_case(nii_path, case, CACHE_DIR, fast=args.fast)
        total_time += elapsed
        if elapsed > 0:
            print(f"    Done in {elapsed:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  PRECOMPUTE COMPLETE")
    print(f"  Cases processed : {len(test_cases)}")
    print(f"  Total time      : {total_time/60:.1f} min")
    print(f"  Cache dir       : {CACHE_DIR}/")
    print(f"\n  Next step:")
    print(f"    python step2_integrated_pipeline.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
