# =============================================================================
# train_organ_classifier.py
# Place at: ~/Desktop/intern/
# Run:      python train_organ_classifier.py
#
# Pipeline:
#   BTCV CT volumes → extract organ crops → DenseNet121 classifier (13 organs)
#   → saves best model to  classifier/best_organ_classifier.pth
# =============================================================================

import os, sys, json, csv, datetime, random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# MONAI imports
from monai.transforms import (
    Compose, ScaleIntensityRange, Resize,
    RandRotate90, RandFlip, RandGaussianNoise,
    ToTensor, AddChannel, NormalizeIntensity
)
from monai.networks.nets import DenseNet121
from monai.metrics import ConfusionMatrixMetric

# =============================================================================
# CONFIG
# =============================================================================
RAW_IMG_DIR   = "data/btcv/raw/data/kaggle_btcv/images"
RAW_LABEL_DIR = "data/btcv/raw/data/kaggle_btcv/labels"
OUT_DIR       = "classifier"

CROP_SIZE     = (64, 64, 64)    # size of each organ crop fed to the model
EPOCHS        = 20
BATCH_SIZE    = 4
LR            = 1e-4
VAL_SPLIT     = 0.2
EARLY_STOP    = 7
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# BTCV has labels 1-13 (0 = background, ignored)
NUM_CLASSES   = 13
ORGAN_NAMES   = [
    "spleen","right_kidney","left_kidney","gallbladder","esophagus",
    "liver","stomach","aorta","ivc","portal_vein",
    "pancreas","adrenal_R","adrenal_L"
]  # index 0 → label 1, index 12 → label 13

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — EXTRACT ORGAN CROPS FROM BTCV VOLUMES
# =============================================================================
def extract_crops(img_path, label_path, min_voxels=200):
    """
    For each organ present in the label file, extract a tight bounding-box
    crop from the CT image. Returns list of (crop_array, label_id) tuples.

    Why crops?
      Training on full 512×512×139 volumes is slow and wastes capacity on
      background. Crops let the model focus on organ-specific texture and shape.
    """
    img_arr   = nib.load(img_path).get_fdata().astype(np.float32)
    label_arr = nib.load(label_path).get_fdata().astype(np.int16)

    crops = []
    for organ_id in range(1, NUM_CLASSES + 1):
        mask = (label_arr == organ_id)
        if mask.sum() < min_voxels:
            continue   # organ absent or too small — skip

        # Bounding box
        z_idx = np.where(np.any(mask, axis=(0,1)))[0]
        y_idx = np.where(np.any(mask, axis=(0,2)))[0]
        x_idx = np.where(np.any(mask, axis=(1,2)))[0]
        z1,z2 = z_idx[0],  z_idx[-1]+1
        y1,y2 = y_idx[0],  y_idx[-1]+1
        x1,x2 = x_idx[0],  x_idx[-1]+1

        # Add small padding (10%) so the model sees context at the boundary
        def pad(lo, hi, maxval, frac=0.10):
            p = max(2, int((hi-lo)*frac))
            return max(0, lo-p), min(maxval, hi+p)

        x1,x2 = pad(x1,x2, img_arr.shape[0])
        y1,y2 = pad(y1,y2, img_arr.shape[1])
        z1,z2 = pad(z1,z2, img_arr.shape[2])

        crop = img_arr[x1:x2, y1:y2, z1:z2]
        crops.append((crop, organ_id - 1))  # label 1 → class index 0

    return crops


def build_crop_dataset(max_cases=None):
    """
    Walk all BTCV pairs, extract crops for every organ found.
    Returns list of (crop_array, class_index) tuples.
    """
    import glob
    imgs   = sorted(glob.glob(os.path.join(RAW_IMG_DIR,   "*.nii")))
    labels = sorted(glob.glob(os.path.join(RAW_LABEL_DIR, "*.nii")))

    label_map = {}
    for lp in labels:
        key = os.path.basename(lp).replace("label","").replace(".nii","")
        label_map[key] = lp

    pairs = []
    for ip in imgs:
        key = os.path.basename(ip).replace("img","").replace(".nii","")
        if key in label_map:
            pairs.append((ip, label_map[key]))

    if max_cases:
        pairs = pairs[:max_cases]

    print(f"\n[STEP 1] Extracting crops from {len(pairs)} cases …")
    all_crops = []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    for i,(ip,lp) in enumerate(pairs):
        case = os.path.basename(ip)
        crops = extract_crops(ip, lp)
        for crop, cls in crops:
            all_crops.append((crop, cls))
            class_counts[cls] += 1
        print(f"  [{i+1:02d}/{len(pairs)}] {case}  → {len(crops)} crops")

    print(f"\n  Total crops : {len(all_crops)}")
    print(f"  Per class   :")
    for i,name in enumerate(ORGAN_NAMES):
        print(f"    {i+1:2d}. {name:<18} : {class_counts[i]} crops")

    return all_crops

# =============================================================================
# STEP 2 — DATASET CLASS
# =============================================================================
class OrganCropDataset(Dataset):
    """
    Takes a list of (crop_array, class_index) and applies MONAI transforms.

    Transform chain:
      1. ScaleIntensityRange  — clip CT HU values to [-175, 250] → [0,1]
      2. AddChannel           — add channel dim: (D,H,W) → (1,D,H,W)
      3. Resize               — standardise all crops to CROP_SIZE
      4. Augmentation         — random flips, rotations, noise  (train only)
      5. NormalizeIntensity   — zero-mean unit-variance per sample
    """
    def __init__(self, samples, augment=True):
        self.samples = samples

        base = [
            ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            AddChannel(),
            Resize(CROP_SIZE),
        ]
        aug = [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate90(prob=0.4, max_k=3),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
        ]
        post = [NormalizeIntensity(nonzero=True)]

        self.transform = Compose(base + (aug if augment else []) + post)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        crop, cls = self.samples[idx]
        # MONAI transforms expect numpy array with no channel dim here
        tensor = self.transform(crop.astype(np.float32))
        return tensor, torch.tensor(cls, dtype=torch.long)

# =============================================================================
# STEP 3 — BUILD MODEL
# =============================================================================
def build_model():
    """
    DenseNet121 from MONAI — pretrained on ImageNet features, adapted for
    3D grayscale medical volumes.

    spatial_dims=3  → 3D convolutions
    in_channels=1   → single channel (CT)
    out_channels=13 → one score per organ class
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
    )
    return model.to(DEVICE)

# =============================================================================
# STEP 4 — TRAINING LOOP
# =============================================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0; correct = 0; n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss/len(loader), 100*correct/n


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0; correct = 0; n = 0
    per_class  = {i:{"correct":0,"total":0} for i in range(NUM_CLASSES)}

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        n          += len(labels)
        for p,l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            per_class[l]["total"]   += 1
            per_class[l]["correct"] += int(p == l)

    acc = 100*correct/n
    return total_loss/len(loader), acc, per_class

# =============================================================================
# STEP 5 — MAIN
# =============================================================================
def main():
    print("\n" + "="*62)
    print("  MONAI ORGAN CLASSIFIER — BTCV 13-class")
    print("="*62)
    print(f"  Device    : {DEVICE}")
    print(f"  Crop size : {CROP_SIZE}")
    print(f"  Epochs    : {EPOCHS}  |  LR : {LR}  |  Batch : {BATCH_SIZE}")
    print("="*62)

    # 1. Build crop dataset
    all_crops = build_crop_dataset(max_cases=None)   # use all 58 cases

    if len(all_crops) < 10:
        print("ERROR: Too few crops found. Check RAW_IMG_DIR / RAW_LABEL_DIR.")
        sys.exit(1)

    # 2. Split train/val
    random.shuffle(all_crops)
    n_val      = max(1, int(len(all_crops) * VAL_SPLIT))
    train_data = all_crops[n_val:]
    val_data   = all_crops[:n_val]
    print(f"\n  Train crops : {len(train_data)}   Val crops : {len(val_data)}")

    train_ds = OrganCropDataset(train_data, augment=True)
    val_ds   = OrganCropDataset(val_data,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model, loss, optimiser
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {params:,}")

    # 4. Training loop
    best_acc    = 0.0
    best_epoch  = 0
    no_improve  = 0
    best_path   = os.path.join(OUT_DIR, "best_organ_classifier.pth")
    log_rows    = []

    print(f"\n  {'Ep':>4}  {'T-Loss':>8}  {'T-Acc%':>8}  {'V-Loss':>8}  {'V-Acc%':>8}")
    print("  " + "-"*50)

    for epoch in range(1, EPOCHS+1):
        t_loss, t_acc       = train_epoch(model, train_dl, optimizer, criterion)
        v_loss, v_acc, pcls = val_epoch(model, val_dl, criterion)
        scheduler.step()

        print(f"  {epoch:>4}  {t_loss:>8.4f}  {t_acc:>7.2f}%  {v_loss:>8.4f}  {v_acc:>7.2f}%")
        log_rows.append({"epoch":epoch,"t_loss":round(t_loss,4),
                         "t_acc":round(t_acc,2),"v_loss":round(v_loss,4),
                         "v_acc":round(v_acc,2)})

        if v_acc > best_acc:
            best_acc   = v_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_acc": v_acc,
                        "organ_names": ORGAN_NAMES,
                        "num_classes": NUM_CLASSES,
                        "crop_size": CROP_SIZE},
                       best_path)
            print(f"         ↑ best saved  val_acc={v_acc:.2f}%")

            # Print per-class accuracy when best model updates
            print("         Per-class accuracy:")
            for i,name in enumerate(ORGAN_NAMES):
                d = pcls[i]
                if d["total"] > 0:
                    pct = 100*d["correct"]/d["total"]
                    bar = "█"*int(pct/10) + "░"*(10-int(pct/10))
                    print(f"           {name:<18} {bar} {pct:.0f}%  ({d['correct']}/{d['total']})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # 5. Save log
    with open(os.path.join(OUT_DIR,"train_log.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    print(f"\n{'='*62}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Best checkpoint   : {best_path}")
    print(f"  Training log      : {OUT_DIR}/train_log.csv")
    print(f"\n  Next step:")
    print(f"    python test_organ_classifier.py")
    print("="*62 + "\n")

if __name__ == "__main__":
    main()