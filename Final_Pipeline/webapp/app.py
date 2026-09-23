#!/usr/bin/env python3
"""
Multi-Agentic Radiology Web UI
Upload NIfTI → Run 4-model pipeline → View results → Download PDF report

Run:
  conda activate medsam_env
  cd ~/Desktop/intern/Final_Pipeline/webapp
  python app.py
  Open http://localhost:5000
"""
import os, sys, json, uuid, io, torch, numpy as np, nibabel as nib
import torch.nn.functional as F, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
from PIL import Image
from datetime import datetime
from fpdf import FPDF

# Add parent for imports
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PARENT)
# Legacy dev layout: a sibling Box_Prompt/ checkout holding segment_anything.
# Kept on sys.path so existing working copies still resolve, but the package
# is normally pip-installed now (see requirements.txt).
BP = os.path.join(REPO_ROOT, "Box_Prompt")
if os.path.isdir(BP):
    sys.path.insert(0, BP)

try:
    from segment_anything.build_sam import _build_sam
    SAM_IMPORT_ERROR = None
except ImportError as e:
    _build_sam = None
    SAM_IMPORT_ERROR = str(e)
    print(f"  WARNING: segment_anything not importable ({e}).\n"
          f"           MedSAM box-prompt segmentation will be unavailable.\n"
          f"           Install: pip install "
          f"git+https://github.com/facebookresearch/segment-anything.git")

# templates/ is a sibling of webapp/, not a child, so point Flask at it
# explicitly - the default (webapp/templates) does not exist.
app = Flask(__name__, template_folder=os.path.join(PARENT, "templates"))
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_DIR = os.path.join(PARENT, "models")
CONFIG_DIR = os.path.join(PARENT, "config")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLS = 11
CLS = {0:"background",1:"liver",2:"liver_tumor",3:"kidney",4:"kidney_tumor",
       5:"pancreas",6:"pancreas_tumor",7:"colon",8:"colon_tumor",9:"esophagus",10:"esophagus_tumor"}
TUMOR_IDS = {2,4,6,8,10}
COLORS = {1:[255,200,50],2:[255,30,30],3:[255,140,0],4:[255,100,0],
          5:[255,255,100],6:[200,200,0],7:[100,255,100],8:[0,200,50],
          9:[180,100,255],10:[150,0,200]}

BTCV_NAMES = {1:"spleen",2:"right_kidney",3:"left_kidney",4:"gallbladder",5:"esophagus",
              6:"liver",7:"stomach",8:"aorta",9:"IVC",10:"portal_vein",11:"pancreas",
              12:"right_adrenal",13:"left_adrenal"}
TS2BTCV = {"spleen":1,"kidney_right":2,"kidney_left":3,"gallbladder":4,"esophagus":5,
           "liver":6,"stomach":7,"aorta":8,"inferior_vena_cava":9,
           "portal_vein_and_splenic_vein":10,"pancreas":11,"adrenal_gland_right":12,"adrenal_gland_left":13}

# Global model cache
models = {}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight files are distributed out-of-band (see the Drive links in the repo
# README) because two of them exceed GitHub's file size limit. A missing file
# must not take down the whole app: each model loads independently and records
# why it is unavailable, so the UI can report the gap instead of 500-ing.
MODEL_FILES = {
    "unet":      "best_multiorgan.pth",
    "segresnet": "best_segresnet_model.pth",
    "sam":       "best_medsam_btcv.pth",
}
load_errors = {}


def model_status():
    """Per-model availability, for /health and the UI banner."""
    out = {}
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        out[key] = {
            "file": fname,
            "present": os.path.isfile(path),
            "loaded": key in models,
            "error": load_errors.get(key),
        }
    return out


def load_models():
    """Load whatever is available. Never raises - inspect load_errors after."""
    dev = get_device()

    def _missing(key):
        path = os.path.join(MODEL_DIR, MODEL_FILES[key])
        if os.path.isfile(path):
            return None
        msg = (f"{MODEL_FILES[key]} not found in {MODEL_DIR}. "
               f"Download it from the Drive link in the repo README.")
        print(f"  WARNING: {msg}")
        load_errors[key] = msg
        return msg

    if "unet" not in models and not _missing("unet"):
        try:
            print("  Loading 3D UNet...")
            from monai.networks.nets import UNet
            m = UNet(spatial_dims=3, in_channels=1, out_channels=NUM_CLS,
                     channels=(32,64,128,256), strides=(2,2,2), num_res_units=2)
            st = torch.load(os.path.join(MODEL_DIR, MODEL_FILES["unet"]),
                            map_location="cpu", weights_only=False)
            m.load_state_dict(st); m.to(dev).eval()
            models["unet"] = m; load_errors.pop("unet", None)
        except Exception as e:
            print(f"  ERROR loading 3D UNet: {e}"); load_errors["unet"] = str(e)

    if "segresnet" not in models and not _missing("segresnet"):
        try:
            print("  Loading SegResNet...")
            from monai.networks.nets import SegResNet
            cfg = json.load(open(os.path.join(CONFIG_DIR, "model_config.json")))
            m = SegResNet(spatial_dims=2, in_channels=1, out_channels=cfg["NUM_CLASSES"],
                          init_filters=16, blocks_down=(1,2,2,4), blocks_up=(1,1,1), dropout_prob=0.2)
            m.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_FILES["segresnet"]),
                              map_location="cpu", weights_only=False))
            m.to(dev).eval()
            models["segresnet"] = m; models["seg_cfg"] = cfg
            load_errors.pop("segresnet", None)
        except Exception as e:
            print(f"  ERROR loading SegResNet: {e}"); load_errors["segresnet"] = str(e)

    if "sam" not in models:
        if _build_sam is None:
            load_errors["sam"] = f"segment_anything not importable: {SAM_IMPORT_ERROR}"
        elif not _missing("sam"):
            try:
                print("  Loading MedSAM...")
                sam = _build_sam(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                                 encoder_global_attn_indexes=[2,5,8,11], checkpoint=None)
                ck = torch.load(os.path.join(MODEL_DIR, MODEL_FILES["sam"]),
                                map_location="cpu", weights_only=False)
                sam.load_state_dict(ck.get("model_state_dict", ck)); sam.to(dev).eval()
                models["sam"] = sam; load_errors.pop("sam", None)
            except Exception as e:
                print(f"  ERROR loading MedSAM: {e}"); load_errors["sam"] = str(e)

    return dev

# ── Pipeline Functions ──
@torch.no_grad()
def run_3d_unet(vol, dev):
    from monai.inferers import sliding_window_inference
    d = np.clip(vol.astype(np.float32), -175, 250)
    d = (d + 175) / 425.0
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).float().to(dev)
    p = sliding_window_inference(t, (96,96,96), 1, models["unet"], overlap=0.25, mode="gaussian")
    return torch.argmax(p, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

@torch.no_grad()
def run_segresnet(gray_slice, dev):
    cfg = models["seg_cfg"]; sz = cfg["IMAGE_SIZE"]
    g = np.array(Image.fromarray(gray_slice).resize((sz,sz), Image.BILINEAR), dtype=np.float32)
    if g.max() > 0: g /= g.max()
    inp = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(dev)
    logits = F.interpolate(models["segresnet"](inp), size=gray_slice.shape, mode="bilinear", align_corners=False)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

@torch.no_grad()
def run_medsam(rgb, box, dev):
    sam = models["sam"]; ih, iw = rgb.shape[:2]
    img_t = torch.from_numpy(rgb).float().permute(2,0,1).to(dev)
    emb = sam.image_encoder(sam.preprocess(img_t).unsqueeze(0))
    sx, sy = 1024.0/iw, 1024.0/ih
    b = np.array(box, dtype=np.float32)
    b[0]*=sx; b[1]*=sy; b[2]*=sx; b[3]*=sy
    bt = torch.from_numpy(b).float().to(dev).unsqueeze(0).unsqueeze(0)
    sp, dn = sam.prompt_encoder(points=None, boxes=bt, masks=None)
    lr, iou = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp, dense_prompt_embeddings=dn, multimask_output=True)
    ms = F.interpolate(lr, (1024,1024), mode="bilinear", align_corners=False).squeeze(0)
    bi = torch.argmax(iou.squeeze(0)).item()
    p = (torch.sigmoid(ms[bi]) > 0.5).cpu().numpy().astype(np.uint8)
    mask = (np.array(Image.fromarray(p*255).resize((iw,ih), Image.NEAREST)) > 127).astype(np.uint8)
    return mask, iou.squeeze(0)[bi].item()

# Where precomputed TotalSegmentator masks may live. The legacy Box_Prompt/
# location is kept first for existing working copies; the repo-relative paths
# are what a fresh clone uses after running
# Unified_Model/step1_precompute_totalseg.py. Override with MEDAI_TOTALSEG_DIR.
TOTALSEG_SEARCH_DIRS = [d for d in [
    os.environ.get("MEDAI_TOTALSEG_DIR"),
    os.path.join(BP, "totalseg_cache"),
    os.path.join(BP, "totalseg_test_output"),
    os.path.join(PARENT, "totalseg_cache"),
    os.path.join(REPO_ROOT, "totalseg_cache"),
    os.path.join(REPO_ROOT, "totalseg_test_output"),
] if d]


def load_totalseg_masks(case_name):
    """Load precomputed TotalSegmentator organ masks.

    Returns {} when no cache is found - organ classification is then reported
    as unavailable rather than silently wrong.
    """
    masks = {}
    ts_dir = None
    for base in TOTALSEG_SEARCH_DIRS:
        for cand in (os.path.join(base, case_name), base):
            if os.path.isdir(cand) and any(
                f.endswith(".nii.gz") for f in os.listdir(cand)
            ):
                ts_dir = cand
                break
        if ts_dir:
            break
    if ts_dir is None:
        print(f"  WARNING: no TotalSegmentator cache for {case_name!r}. Searched: "
              f"{TOTALSEG_SEARCH_DIRS}. Organ classification will be unavailable - "
              f"generate it with Unified_Model/step1_precompute_totalseg.py")
        return masks
    for ts_name, btcv_id in TS2BTCV.items():
        p = os.path.join(ts_dir, f"{ts_name}.nii.gz")
        if os.path.exists(p):
            arr = nib.load(p).get_fdata()
            if arr.sum() > 0:
                masks[btcv_id] = arr
    return masks

def classify_box(ts_masks, slice_idx, box, orig_shape, img_shape):
    """Classify which organ is inside the drawn box using TotalSeg masks."""
    if not ts_masks or slice_idx is None:
        return "unknown", []
    oh, ow = orig_shape[:2]
    sh, sw = img_shape[0] / oh, img_shape[1] / ow
    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    ox1, oy1 = int(bx1 / sw), int(by1 / sh)
    ox2, oy2 = int(bx2 / sw), int(by2 / sh)
    results = []
    for bid, m3d in ts_masks.items():
        best = 0
        for z in range(max(0, slice_idx - 15), min(m3d.shape[2], slice_idx + 16)):
            ms = np.flip(m3d[:, :, z], axis=0).copy()
            v = int(ms[oy1:oy2, ox1:ox2].sum())
            if v > best: best = v
        if best == 0: continue
        total = max(1, (oy2 - oy1) * (ox2 - ox1))
        results.append({"btcv_id": bid, "name": BTCV_NAMES.get(bid, f"organ_{bid}"),
                         "voxels": best, "pct": round(100 * best / total, 2)})
    results.sort(key=lambda x: -x["voxels"])
    return (results[0]["name"] if results else "unknown"), results

def get_slice_img(vol, z):
    s = np.clip(vol[:,:,z].astype(np.float32), -175, 250)
    return np.flip(((s+175)/425*255).astype(np.uint8), axis=0).copy()

def overlay_pred(img, pred_s):
    rgb = np.stack([img]*3, axis=-1).astype(np.float32)
    for cid, col in COLORS.items():
        m = pred_s == cid
        if not m.any(): continue
        a = 0.6 if cid in TUMOR_IDS else 0.25
        for c in range(3): rgb[:,:,c] = np.where(m, (1-a)*rgb[:,:,c]+a*col[c], rgb[:,:,c])
    return np.clip(rgb, 0, 255).astype(np.uint8)

def calc_volumes(pred_3d, pixdim):
    vv = float(np.prod(pixdim[:3])); r = {}
    for cid in sorted(CLS.keys()):
        if cid == 0: continue
        cnt = int((pred_3d == cid).sum())
        if cnt > 0:
            r[CLS[cid]] = {"voxels":cnt, "mm3":round(cnt*vv,1), "cm3":round(cnt*vv/1000,2),
                            "is_tumor": cid in TUMOR_IDS}
    return r

def save_panel_image(vol, pred_3d, z, session_dir, box=None, medsam_mask=None):
    ct = get_slice_img(vol, z)
    ps = np.flip(pred_3d[:,:,z], axis=0).copy()
    fig, axes = plt.subplots(1, 4 if medsam_mask is not None else 3, figsize=(24 if medsam_mask is not None else 18, 6))
    fig.patch.set_facecolor("#0a0a0a")

    # Panel 1: CT
    axes[0].imshow(ct, cmap="gray"); axes[0].set_title(f"CT Slice {z}", color="white", fontsize=12)
    axes[0].axis("off"); axes[0].set_facecolor("#0a0a0a")
    if box:
        from matplotlib.patches import Rectangle
        axes[0].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
            lw=2, edgecolor="#00FF88", facecolor="none", linestyle="--"))

    # Panel 2: UNet overlay
    axes[1].imshow(overlay_pred(ct, ps)); axes[1].axis("off"); axes[1].set_facecolor("#0a0a0a")
    found = [CLS[c] for c in range(1,NUM_CLS) if (ps==c).any()]
    has_t = any(c in TUMOR_IDS for c in range(1,NUM_CLS) if (ps==c).any())
    axes[1].set_title("UNet: " + (", ".join(found[:3]) or "bg"),
                       color="#FF5252" if has_t else "#4CAF50", fontsize=11, fontweight="bold")

    # Panel 3: SegResNet. Missing weights degrade this panel only - panel 4
    # (MedSAM) below must still render, so do not bail out of the figure.
    axes[2].axis("off"); axes[2].set_facecolor("#0a0a0a")
    if "segresnet" not in models:
        axes[2].imshow(ct, cmap="gray")
        axes[2].set_title("SegResNet unavailable", color="#FF5252",
                          fontsize=11, fontweight="bold")
    else:
        seg_pred = run_segresnet(ct, get_device())
        seg_ov = np.stack([ct]*3, axis=-1).astype(np.float32)
        seg_ov[seg_pred==1] = [255,200,50]; seg_ov[seg_pred==2] = [255,30,30]
        axes[2].imshow(np.clip(seg_ov,0,255).astype(np.uint8))
        t_px = int((seg_pred==2).sum())
        axes[2].set_title(f"SegResNet: {'TUMOR' if t_px>0 else 'Clean'} ({t_px}px)",
                           color="#FF5252" if t_px>0 else "#4CAF50", fontsize=11, fontweight="bold")

    # Panel 4: MedSAM (if box provided)
    if medsam_mask is not None and len(axes) > 3:
        rgb = np.stack([ct]*3, axis=-1)
        axes[3].imshow(rgb); axes[3].set_facecolor("#0a0a0a"); axes[3].axis("off")
        ov = np.zeros((*medsam_mask.shape[:2],4), dtype=np.uint8)
        ov[medsam_mask>0] = [0,255,100,170]
        axes[3].imshow(ov)
        if medsam_mask.any():
            axes[3].contour(medsam_mask, levels=[0.5], colors=["yellow"], linewidths=[1.5])
        axes[3].set_title(f"MedSAM: {medsam_mask.sum():,}px", color="#00FF88", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(session_dir, f"panel_z{z}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    return path

def generate_pdf(session_dir, case_name, volumes, panel_paths):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_fill_color(10, 10, 30)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_text_color(0, 255, 136)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, "", ln=True)
    pdf.cell(0, 15, "Clinical Diagnostic Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 10, "Multi-Agentic Radiology Pipeline", ln=True, align="C")
    pdf.cell(0, 20, "", ln=True)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Case: {case_name}", ln=True, align="C")
    pdf.cell(0, 8, f"Date: {datetime.now():%Y-%m-%d %H:%M}", ln=True, align="C")
    pdf.cell(0, 8, f"Models: 3D UNet + SegResNet + TotalSeg + MedSAM", ln=True, align="C")

    # Findings page
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "Findings", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(5)

    # Table header
    pdf.set_fill_color(40, 40, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Structure", 1, 0, "C", True)
    pdf.cell(35, 8, "Volume (cm3)", 1, 0, "C", True)
    pdf.cell(35, 8, "Voxels", 1, 0, "C", True)
    pdf.cell(30, 8, "Type", 1, 1, "C", True)

    # Table rows
    pdf.set_font("Helvetica", "", 10)
    for name, info in sorted(volumes.items()):
        if info["is_tumor"]:
            pdf.set_fill_color(255, 230, 230)
            pdf.set_text_color(200, 0, 0)
        else:
            pdf.set_fill_color(240, 255, 240)
            pdf.set_text_color(0, 100, 0)
        pdf.cell(60, 7, name, 1, 0, "L", True)
        pdf.cell(35, 7, f"{info['cm3']:.2f}", 1, 0, "C", True)
        pdf.cell(35, 7, f"{info['voxels']:,}", 1, 0, "C", True)
        pdf.cell(30, 7, "TUMOR" if info["is_tumor"] else "organ", 1, 1, "C", True)

    # Summary
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    tumors = [f"{n} ({i['cm3']:.2f} cm3)" for n, i in volumes.items() if i["is_tumor"]]
    if tumors:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 10, f"TUMORS DETECTED: {len(tumors)}", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for t in tumors:
            pdf.cell(0, 7, f"  - {t}", ln=True)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 10, "No tumors detected.", ln=True)

    # Panel images
    for pp in panel_paths:
        if os.path.exists(pp):
            pdf.add_page('L')
            pdf.set_fill_color(10, 10, 30)
            pdf.rect(0, 0, 297, 210, 'F')
            pdf.image(pp, x=5, y=10, w=287)

    pdf_path = os.path.join(session_dir, f"report_{case_name}.pdf")
    pdf.output(pdf_path)
    return pdf_path

# ── Session storage ──
sessions = {}

# ── Routes ──
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    """Component availability - check this first when something is missing."""
    return jsonify({
        "models": model_status(),
        "model_dir": MODEL_DIR,
        "segment_anything": SAM_IMPORT_ERROR or "ok",
        "totalseg_cache": TOTALSEG_SEARCH_DIRS,
        "device": str(get_device()),
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith((".nii", ".nii.gz")):
        return jsonify({"error": "Only .nii or .nii.gz files accepted"}), 400

    sid = str(uuid.uuid4())[:8]
    session_dir = os.path.join(OUTPUT_DIR, sid)
    os.makedirs(session_dir, exist_ok=True)
    fpath = os.path.join(UPLOAD_DIR, f"{sid}_{f.filename}")
    f.save(fpath)

    case_name = f.filename.replace(".nii.gz","").replace(".nii","")
    nii = nib.load(fpath)
    vol = nii.get_fdata()
    pixdim = nii.header.get_zooms()

    # Load TotalSeg masks
    ts_masks = load_totalseg_masks(case_name)
    print(f"  TotalSeg: {len(ts_masks)} organ masks loaded for {case_name}")

    sessions[sid] = {"vol": vol, "pixdim": pixdim, "case": case_name, "dir": session_dir,
                     "D": vol.shape[2], "pred_3d": None, "volumes": None, "panels": [],
                     "ts_masks": ts_masks, "orig_shape": vol.shape}

    return jsonify({"session_id": sid, "case": case_name, "shape": list(vol.shape),
                    "spacing": [float(p) for p in pixdim[:3]], "total_slices": vol.shape[2],
                    "totalseg_organs": len(ts_masks)})

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    sid = request.json.get("session_id")
    if sid not in sessions:
        return jsonify({"error": "Session not found"}), 404

    s = sessions[sid]; dev = load_models()
    if "unet" not in models:
        return jsonify({
            "error": "3D UNet unavailable - cannot run the pipeline.",
            "detail": load_errors.get("unet"),
            "models": model_status(),
        }), 503
    print(f"  Running 3D UNet for {s['case']}...")
    pred_3d = run_3d_unet(s["vol"], dev)
    volumes = calc_volumes(pred_3d, s["pixdim"])
    s["pred_3d"] = pred_3d; s["volumes"] = volumes

    # Generate panels for 3 representative slices
    D = s["D"]
    slices = [D//4, D//2, 3*D//4]
    panels = []
    for z in slices:
        pp = save_panel_image(s["vol"], pred_3d, z, s["dir"])
        panels.append(pp); s["panels"].append(pp)

    return jsonify({"volumes": volumes, "slices": slices,
                    "panel_urls": [url_for('get_image', sid=sid, fname=os.path.basename(p)) for p in panels]})

@app.route("/get_slice", methods=["POST"])
def get_slice_route():
    sid = request.json.get("session_id")
    z = request.json.get("slice", 0)
    box = request.json.get("box")

    if sid not in sessions: return jsonify({"error": "Not found"}), 404
    s = sessions[sid]
    if s["pred_3d"] is None: return jsonify({"error": "Run pipeline first"}), 400

    dev = get_device()
    medsam_mask = None
    medsam_score = 0
    organ_name = "unknown"
    organ_results = []

    if box and len(box) == 4 and "sam" not in models:
        print(f"  Slice {z}: MedSAM unavailable ({load_errors.get('sam')}) - "
              f"skipping box-prompt segmentation.")
    elif box and len(box) == 4:
        ct = get_slice_img(s["vol"], z)
        rgb = np.stack([ct]*3, axis=-1)
        medsam_mask, medsam_score = run_medsam(rgb, box, dev)

        # TotalSeg classification — identify which organ is in the box
        organ_name, organ_results = classify_box(
            s["ts_masks"], z, box, s["orig_shape"], ct.shape[:2])
        print(f"  Slice {z}: TotalSeg={organ_name.upper()} | MedSAM={medsam_mask.sum():,}px")

    pp = save_panel_image(s["vol"], s["pred_3d"], z, s["dir"], box=box, medsam_mask=medsam_mask)
    s["panels"].append(pp)

    return jsonify({
        "panel_url": url_for('get_image', sid=sid, fname=os.path.basename(pp)),
        "medsam_score": medsam_score,
        "medsam_pixels": int(medsam_mask.sum()) if medsam_mask is not None else 0,
        "organ_name": organ_name,
        "organ_results": [{"name": r["name"], "pct": r["pct"]} for r in organ_results[:5]]
    })

@app.route("/download_report", methods=["POST"])
def download_report():
    sid = request.json.get("session_id") if request.is_json else request.form.get("session_id")
    if sid not in sessions: return "Not found", 404
    s = sessions[sid]
    if s["volumes"] is None: return "Run pipeline first", 400

    # Use last 3 panels
    recent = s["panels"][-3:] if len(s["panels"]) >= 3 else s["panels"]
    pdf_path = generate_pdf(s["dir"], s["case"], s["volumes"], recent)
    return send_file(pdf_path, as_attachment=True, download_name=f"report_{s['case']}.pdf")

@app.route("/images/<sid>/<fname>")
def get_image(sid, fname):
    return send_file(os.path.join(OUTPUT_DIR, sid, fname))

@app.route("/raw_slice", methods=["POST"])
def raw_slice():
    """Return a CT slice as PNG for canvas rendering."""
    sid = request.json.get("session_id")
    z = int(request.json.get("slice", 0))
    mode = request.json.get("mode", "ct")  # 'ct' or 'overlay'
    if sid not in sessions: return jsonify({"error": "Not found"}), 404
    s = sessions[sid]
    ct = get_slice_img(s["vol"], z)
    if mode == "overlay" and s["pred_3d"] is not None:
        ps = np.flip(s["pred_3d"][:,:,z], axis=0).copy()
        img = Image.fromarray(overlay_pred(ct, ps))
    else:
        img = Image.fromarray(np.stack([ct]*3, axis=-1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/slice_info", methods=["POST"])
def slice_info():
    """Return detected structures on a slice."""
    sid = request.json.get("session_id")
    z = int(request.json.get("slice", 0))
    if sid not in sessions: return jsonify({"error": "Not found"}), 404
    s = sessions[sid]
    if s["pred_3d"] is None: return jsonify({"structures": []})
    ps = np.flip(s["pred_3d"][:,:,z], axis=0).copy()
    structs = []
    for cid in range(1, NUM_CLS):
        cnt = int((ps == cid).sum())
        if cnt > 0:
            structs.append({"name": CLS[cid], "pixels": cnt, "is_tumor": cid in TUMOR_IDS})
    return jsonify({"structures": structs, "shape": [int(ps.shape[0]), int(ps.shape[1])]})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Multi-Agentic Radiology Web UI")
    print("  Open http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
