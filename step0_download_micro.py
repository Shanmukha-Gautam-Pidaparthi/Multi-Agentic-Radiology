# =============================================================================
# step0_download_micro.py — TRUE SUBSET: Stream & Extract Only Tumor Cases
#
# STRATEGY:
#   Stream the 18 GB tar.gz DIRECTLY over HTTP (never saved to disk).
#   For each case in the stream:
#     - Extract only our 11 target files (5 organs + 5 tumors + kidney_right)
#     - When we move to the next case, check: did the previous case have tumors?
#       YES → keep it     NO → delete it immediately
#   Stop when we have >= N cases for EACH of the 5 tumor types.
#
# DISK USAGE: Only the kept tumor cases (~200-500 MB for ~30 cases)
# NETWORK:    Streams through the archive (~3-8 min on Colab)
# THE TAR.GZ IS NEVER SAVED TO DISK.
#
# USAGE (Colab):
#   !python step0_download_micro.py --per_type 6       # ~30 cases
#   !python step0_download_micro.py --per_type 10      # ~50 cases, better model
#   !python step0_download_micro.py --skip_labels      # re-download CTs only
# =============================================================================

import os
import sys
import json
import argparse
import tarfile
import urllib.request
import ssl
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEGMENTATION_FILES, MANIFEST_PATH

LABEL_URL = "https://www.cs.jhu.edu/~zongwei/dataset/AbdomenAtlas2.0Mini_label.tar.gz"
CT_URL_TEMPLATE = (
    "https://huggingface.co/datasets/MrGiovanni/AbdomenAtlas2.0Mini/"
    "resolve/main/AbdomenAtlas2.0Mini_ct_{start}_{end}.tar.gz?download=true"
)

# Only these files get extracted per case (11 files, not all 25+)
WANTED_FILES = set()
for fname, _ in SEGMENTATION_FILES:
    WANTED_FILES.add(fname)

TUMOR_FILES = {
    "liver_tumor.nii.gz",
    "kidney_tumor.nii.gz",
    "pancreas_tumor.nii.gz",
    "colon_tumor.nii.gz",
    "esophagus_tumor.nii.gz",
}


def stream_and_filter(output_dir, per_type=6):
    """
    Stream the 18 GB label tar.gz over HTTP.
    Extract only our 11 target files per case.
    Keep only cases that have tumor annotations.
    Stop when all 5 tumor types have >= per_type cases.

    THE TAR.GZ IS NEVER SAVED TO DISK.
    """
    print(f"\n  Streaming label archive over HTTP...")
    print(f"  URL: {LABEL_URL}")
    print(f"  Target: {per_type} cases per tumor type (5 types)")
    print(f"  Only extracting: {len(WANTED_FILES)} files per case (not all 25+)")
    print(f"  The .tar.gz is NEVER saved to disk.\n")

    ctx = ssl.create_default_context()
    try:
        response = urllib.request.urlopen(LABEL_URL, context=ctx)
    except Exception as e:
        print(f"  ERROR connecting: {e}")
        sys.exit(1)

    tar = tarfile.open(fileobj=response, mode="r|gz")
    os.makedirs(output_dir, exist_ok=True)

    # Track per-type counts
    type_counts = {t: 0 for t in TUMOR_FILES}
    kept_cases = {}        # case_name -> {"tumors": set, "organs": set}
    current_case = None
    current_tumors = set()
    current_organs = set()
    cases_scanned = 0
    cases_kept = 0
    cases_deleted = 0
    total_bytes_kept = 0

    def finalize_case(case_name, tumors, organs):
        """Check if previous case had tumors. Keep or delete."""
        nonlocal cases_kept, cases_deleted, total_bytes_kept

        case_dir = os.path.join(output_dir, case_name)
        if not os.path.isdir(case_dir):
            return

        if tumors:
            # KEEP — has tumor annotations
            cases_kept += 1
            kept_cases[case_name] = {"tumors": tumors.copy(), "organs": organs.copy()}
            for t in tumors:
                type_counts[t] = type_counts.get(t, 0) + 1

            tumor_names = ", ".join(t.replace(".nii.gz", "") for t in sorted(tumors))
            coverage = " | ".join(
                f"{t.replace('.nii.gz','').replace('_tumor','')[:3]}:"
                f"{type_counts.get(t,0)}/{per_type}"
                for t in sorted(TUMOR_FILES)
            )
            print(f"    ✓ [{cases_kept}] {case_name} — {tumor_names}  [{coverage}]")

            # Count kept bytes
            for dp, _, fns in os.walk(case_dir):
                for fn in fns:
                    total_bytes_kept += os.path.getsize(os.path.join(dp, fn))
        else:
            # DELETE — no tumors, remove from disk immediately
            shutil.rmtree(case_dir, ignore_errors=True)
            cases_deleted += 1

    for member in tar:
        if not member.isfile():
            continue

        parts = member.name.split("/")
        filename = parts[-1]

        # Find case name (BDMAP_XXXXXXXX)
        case_name = None
        for p in parts:
            if p.startswith("BDMAP_"):
                case_name = p
                break
        if not case_name:
            continue

        # ── New case started → finalize previous one ──
        if case_name != current_case:
            if current_case is not None:
                finalize_case(current_case, current_tumors, current_organs)
                cases_scanned += 1

                # ── STOP: all 5 types covered? ──
                all_covered = all(
                    type_counts.get(t, 0) >= per_type for t in TUMOR_FILES
                )
                if all_covered:
                    print(f"\n  ✓ ALL 5 tumor types have >= {per_type} cases!")
                    print(f"    Stopping stream early (scanned {cases_scanned} cases)")
                    break

            # Reset for new case
            current_case = case_name
            current_tumors = set()
            current_organs = set()

            # Progress every 500 cases
            if cases_scanned > 0 and cases_scanned % 500 == 0:
                print(f"    ... scanned {cases_scanned} cases, "
                      f"kept {cases_kept}, disk: {total_bytes_kept/1e6:.0f} MB")

        # ── Only extract our target files (skip aorta, bladder, etc.) ──
        if filename not in WANTED_FILES:
            continue

        # Extract to disk
        seg_dir = os.path.join(output_dir, case_name, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)
        out_path = os.path.join(seg_dir, filename)

        fobj = tar.extractfile(member)
        if fobj:
            with open(out_path, "wb") as f:
                f.write(fobj.read())

            if filename in TUMOR_FILES:
                current_tumors.add(filename)
            else:
                current_organs.add(filename)

    # Finalize last case
    if current_case:
        finalize_case(current_case, current_tumors, current_organs)
        cases_scanned += 1

    tar.close()
    response.close()

    # ── Report ──
    print(f"\n  ═══════════════════════════════════════════")
    print(f"  LABEL DOWNLOAD SUMMARY")
    print(f"  Cases scanned:  {cases_scanned}")
    print(f"  Cases kept:     {cases_kept} (with tumors)")
    print(f"  Cases deleted:  {cases_deleted} (no tumors)")
    print(f"  Disk used:      {total_bytes_kept / 1e6:.1f} MB")
    print(f"\n  Per tumor type:")
    all_ok = True
    for t in sorted(TUMOR_FILES):
        name = t.replace(".nii.gz", "")
        count = type_counts.get(t, 0)
        status = "✓" if count >= per_type else "⚠ needs more"
        if count < per_type:
            all_ok = False
        print(f"    {name:<22} : {count:3d} / {per_type}  {status}")

    if not all_ok:
        print(f"\n  ⚠ Some tumor types have fewer than {per_type} cases.")
        print(f"    This is normal — the dataset may have fewer cases for rare tumors.")
        print(f"    Training will still work with available data.")
    print(f"  ═══════════════════════════════════════════")

    return kept_cases


def download_cts(output_dir, case_names):
    """Stream CT chunks from HuggingFace, extract only our cases."""
    chunk_cases = {}
    for name in case_names:
        try:
            num = int(name.replace("BDMAP_", ""))
            chunk_id = ((num - 1) // 500) + 1
            chunk_cases.setdefault(chunk_id, []).append(name)
        except ValueError:
            pass

    needed = sorted(chunk_cases.keys())
    print(f"\n  Downloading CTs from {len(needed)} chunk(s): {needed}")

    ctx = ssl.create_default_context()
    total = 0

    for ci, cid in enumerate(needed):
        start = f"{(cid - 1) * 500 + 1:08d}"
        end = f"{cid * 500:08d}"
        url = CT_URL_TEMPLATE.format(start=start, end=end)
        targets = set(chunk_cases[cid])

        print(f"\n  [{ci+1}/{len(needed)}] Chunk {cid} ({len(targets)} cases)...")

        try:
            response = urllib.request.urlopen(url, context=ctx)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        tar = tarfile.open(fileobj=response, mode="r|gz")
        found = set()

        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            cn = None
            for p in parts:
                if p.startswith("BDMAP_"):
                    cn = p
                    break
            if not cn or cn not in targets or parts[-1] != "ct.nii.gz":
                continue

            out = os.path.join(output_dir, cn, "ct.nii.gz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fobj = tar.extractfile(member)
            if fobj:
                data = fobj.read()
                with open(out, "wb") as f:
                    f.write(data)
                found.add(cn)
                total += 1
                print(f"    [{total}] {cn} ({len(data)/1e6:.1f} MB)")
            if found == targets:
                break

        tar.close()
        response.close()

    print(f"\n  CTs downloaded: {total} / {len(case_names)}")
    return total


def build_manifest(output_dir, kept_cases, manifest_path):
    """Build manifest with only cases that have BOTH labels + CT."""
    cases = []
    counts = {t.replace(".nii.gz", ""): 0 for t in TUMOR_FILES}

    for cn in sorted(kept_cases):
        cd = os.path.join(output_dir, cn)
        if not os.path.exists(os.path.join(cd, "ct.nii.gz")):
            continue
        tumors = {}
        for t in kept_cases[cn]["tumors"]:
            tn = t.replace(".nii.gz", "")
            tumors[tn] = 1
            if tn in counts:
                counts[tn] += 1
        cases.append({
            "case_name": cn, "case_dir": cd, "has_ct": True,
            "tumors": tumors,
            "organs": list(kept_cases[cn].get("organs", set())),
            "num_tumors": len(tumors),
        })

    manifest = {
        "source": "abdomenatlas2.0_micro",
        "total_filtered": len(cases),
        "tumor_type_counts": counts,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest: {manifest_path}")
    print(f"  Trainable cases: {len(cases)}")
    for n, c in sorted(counts.items()):
        print(f"    {n:<22}: {c} cases")


def main():
    parser = argparse.ArgumentParser(
        description="Download SUBSET of AbdomenAtlas 2.0 (all 5 tumor types)"
    )
    parser.add_argument("--per_type", type=int, default=6,
                        help="Cases to keep per tumor type (default: 6)")
    parser.add_argument("--skip_labels", action="store_true",
                        help="Skip label download, use existing data")
    parser.add_argument("--skip_cts", action="store_true",
                        help="Skip CT download")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "data", "AbdomenAtlas2.0")

    print("\n" + "=" * 65)
    print("  AbdomenAtlas 2.0 — TRUE SUBSET DOWNLOAD")
    print(f"  Streams 18 GB archive over HTTP (NEVER saved to disk)")
    print(f"  Keeps only tumor cases ({args.per_type} per type × 5 types)")
    print(f"  Expected disk: ~{args.per_type * 5 * 0.05:.1f} GB (subset only)")
    print("=" * 65)

    if not args.skip_labels:
        kept = stream_and_filter(output_dir, per_type=args.per_type)
    else:
        print("\n  Scanning existing labels...")
        kept = {}
        for cn in sorted(os.listdir(output_dir)):
            sd = os.path.join(output_dir, cn, "segmentations")
            if not os.path.isdir(sd):
                continue
            files = set(os.listdir(sd))
            tumors = files & TUMOR_FILES
            organs = files & WANTED_FILES - TUMOR_FILES
            if tumors:
                kept[cn] = {"tumors": tumors, "organs": organs}
        print(f"  Found {len(kept)} tumor cases on disk.")

    if not kept:
        print("\n  ERROR: No tumor cases found.")
        sys.exit(1)

    if not args.skip_cts:
        download_cts(output_dir, sorted(kept.keys()))

    build_manifest(output_dir, kept, MANIFEST_PATH)

    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(output_dir) for f in fs)
    print(f"\n{'=' * 65}")
    print(f"  ✓ SUBSET DOWNLOAD COMPLETE")
    print(f"  Cases: {len(kept)}  |  Disk: {total / 1e9:.2f} GB")
    print(f"  Next: !python step3_train.py --epochs 30 --device cuda --batch_size 2")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
