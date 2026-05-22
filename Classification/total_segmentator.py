import os
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import numpy as np

# 1. Define paths
input_path = "input_ct_scan.nii.gz"
output_dir = "segmentation_results"

# 2. Run TotalSegmentator
# 'task="total"' identifies all 117 classes
# 'ml=True' saves one single file with all organ labels (multilabel)
print("Starting TotalSegmentator classification...")
totalsegmentator(input_path, output_dir, task="total", ml=True, fast=True)

# 3. Read the classification results
# TotalSegmentator maps pixels to specific organ IDs
label_map_path = os.path.join(output_dir, "class_map.nii.gz")
if os.path.exists(label_map_path):
    img = nib.load(label_map_path)
    data = img.get_fdata()
    
    # Get unique IDs present in the image
    unique_ids = np.unique(data)
    print(f"\nClassification Successful! Organs found: {len(unique_ids) - 1}")
    
    # Example organ mapping (refer to TotalSegmentator documentation for full 117 list)
    organ_dict = {1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 5: "Liver"}
    
    for oid in unique_ids:
        if oid in organ_dict:
            print(f" - ID {int(oid)}: {organ_dict[oid]}")