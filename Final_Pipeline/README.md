# Multi-Agentic Radiology: Final Pipeline

Interactive 3D multi-organ tumor segmentation and diagnostic pipeline combining 4 deep learning models.

---

## Quick Start

```bash
# From the repository root. On Debian/Ubuntu a venv is mandatory - PEP 668
# blocks system-wide installs, and `python` does not exist until it is active.
python3 -m venv .venv && source .venv/bin/activate

pip install -r Final_Pipeline/requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download best_medsam_btcv.pth and medsam_vit_b.pth into models/
# (Drive links in the repository root README.md)

cd Final_Pipeline

# Web UI: upload NIfTI, draw box, download PDF report -> localhost:5000
python webapp/app.py

# 3D Interactive Pipeline (scroll slices, draw box, 3D mesh, clinical report)
python interactive_3d_pipeline.py --nifti <path_to_nifti_volume.nii>

# 4-Panel Multi-organ Pipeline (2D PNG slice input)
python interactive_multiorgan.py --image <path_to_slice.png>

# Original 3-Model Pipeline (SegResNet + TotalSeg + MedSAM)
python interactive_unified.py --image <path_to_slice.png>
```

Model and config paths resolve relative to `Final_Pipeline/`, so these run from
any working directory. `webapp/app.py` exposes `GET /health` listing which
models loaded and which files are missing.

---

## Folder Structure

> **Note:** the tree below is the intended layout. These entries are **not
> committed to this repository** and must be obtained separately:
> `models/best_medsam_btcv.pth` and `models/medsam_vit_b.pth` (Drive links in
> the root README), `config/config.py`, `sample_outputs/`,
> `pipeline_architecture.png`, `class_mapping_schema.png`,
> `step0_download_micro.py`, `step3_train.py`.
> Nothing under `Final_Pipeline/` imports `config.py`, so the three
> interactive scripts and the web UI run without it.

```
Final_Pipeline/
├── README.md                          ← This file
│
├── interactive_3d_pipeline.py         ← PRIMARY: 3D interactive clinical pipeline
├── interactive_multiorgan.py          ← 4-panel multi-organ pipeline (2D input)
├── interactive_unified.py             ← Original 3-model unified pipeline (2D input)
│
├── step0_download_micro.py            ← Dataset streaming downloader (Colab)
├── step3_train.py                     ← 3D UNet training script (Colab)
│
├── models/                            ← All trained model weights
│   ├── best_multiorgan.pth            ← 3D Multi-organ Tumor UNet (19 MB)
│   ├── best_segresnet_model.pth       ← 2D SegResNet liver/tumor (6.1 MB)
│   ├── best_medsam_btcv.pth           ← MedSAM fine-tuned on BTCV (388 MB) [DOWNLOAD]
│   └── medsam_vit_b.pth              ← MedSAM pre-trained ViT-B (358 MB) [DOWNLOAD]
│
├── config/
│   └── model_config.json              ← SegResNet architecture config
│
├── requirements.txt                   ← Python dependencies
│
├── sample_outputs/
│   ├── 3d_img0008-0002.png            ← 3D mesh render sample
│   ├── s0094.png                      ← 4-panel interactive output sample
│   ├── multi_002_liver_192740.png     ← Multi-organ pipeline output sample
│   └── report_img0008-0002.txt        ← Auto-generated clinical report sample
│
├── pipeline_architecture.png          ← Architecture flow diagram
└── class_mapping_schema.png           ← Dataset-to-class mapping diagram
```

---

## File Descriptions

### Core Pipeline Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `interactive_3d_pipeline.py` | **Primary pipeline.** Loads 3D NIfTI volume, runs all 4 models, provides scrollable slice viewer with box-prompt segmentation, 3D mesh rendering, and clinical report generation. | NIfTI `.nii` volume | Interactive 4-panel GUI + 3D mesh + clinical report |
| `interactive_multiorgan.py` | 4-panel pipeline for 2D PNG slices. Runs multi-organ UNet + SegResNet + TotalSeg + MedSAM on a single slice with box prompts. | 2D PNG slice | Interactive 4-panel GUI with saved screenshots |
| `interactive_unified.py` | Original 3-model pipeline (SegResNet + TotalSeg + MedSAM). Liver-focused tumor detection with organ classification. | 2D PNG slice | Interactive 4-panel GUI |

### Training Scripts (Google Colab)

| File | Purpose |
|------|---------|
| `step0_download_micro.py` | Streams and extracts balanced subsets from MSD (Liver/Pancreas/Colon) and AbdomenAtlas (Esophagus) datasets on Colab without saving raw archives to disk. |
| `step3_train.py` | Trains the 3D Multi-organ UNet on Google Colab T4 GPU using MONAI with DiceFocalLoss and cosine annealing. |

### Models

| Model File | Architecture | Size | Classes | Purpose |
|-----------|-------------|------|---------|---------|
| `best_multiorgan.pth` | MONAI 3D UNet (4.7M params) | 19 MB | 11 (5 organs + 5 tumors + bg) | Volumetric multi-organ tumor detection |
| `best_segresnet_model.pth` | MONAI SegResNet 2D | 6.1 MB | 3 (bg + liver + tumor) | 2D liver-specific tumor detection (Dice: 0.819) |
| `best_medsam_btcv.pth` | SAM ViT-B (fine-tuned) | 388 MB | Binary mask | Box-prompt precision segmentation (fine-tuned on BTCV) |
| `medsam_vit_b.pth` | SAM ViT-B (pre-trained) | 358 MB | Binary mask | Original MedSAM weights (fallback) |

### Config

| File | Purpose |
|------|---------|
| `config.py` | Unified 11-class mapping, target organ/tumor file lists, hyperparameters |
| `model_config.json` | SegResNet architecture config (image size, num classes, best Dice score) |

---

## 11-Class Schema

| ID | Class | Type | Source Dataset |
|----|-------|------|---------------|
| 0 | Background | — | — |
| 1 | Liver | Organ | MSD Task03 |
| 2 | Liver Tumor | **Tumor** | MSD Task03 |
| 3 | Kidney | Organ | AbdomenAtlas |
| 4 | Kidney Tumor | **Tumor** | KiTS19 (unavailable) |
| 5 | Pancreas | Organ | MSD Task07 |
| 6 | Pancreas Tumor | **Tumor** | MSD Task07 |
| 7 | Colon | Organ | AbdomenAtlas |
| 8 | Colon Tumor | **Tumor** | MSD Task10 |
| 9 | Esophagus | Organ | AbdomenAtlas |
| 10 | Esophagus Tumor | **Tumor** | AbdomenAtlas |

---

## Controls (interactive_3d_pipeline.py)

| Action | Control |
|--------|---------|
| Navigate slices | Scroll wheel |
| Segment region | Click + drag box on Panel 1 |
| Render 3D mesh | Press **R** key |
| Print clinical report | Press **P** key |

---

## Dependencies

```
Python 3.10, PyTorch, MONAI 1.5.2, NiBabel, scikit-image,
SciPy, Matplotlib, Pillow, segment-anything
```

Install: `conda activate medsam_env` (pre-configured environment)
