# medsam_box.py — Interactive Box-Prompt Segmentation
#
# Uses the SAME direct inference pipeline as quick_test_box.py
# (image_encoder → prompt_encoder → mask_decoder)
# so results match exactly when using the same box coordinates.
#
# Usage:
#   python Box_Prompt/box_prompting.py
#   python Box_Prompt/box_prompting.py --image path/to/image.png
#   python Box_Prompt/box_prompting.py --checkpoint Box_Prompt/medsam_vit_b.pth  (baseline)

import matplotlib
matplotlib.use('TkAgg')  # Must be before pyplot import

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from PIL import Image
from segment_anything.build_sam import _build_sam

# ─── PARSE ARGUMENTS ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive MedSAM box-prompt segmentation")
parser.add_argument("--checkpoint", default="Box_Prompt/best_medsam_btcv.pth",
                    help="MedSAM checkpoint (default: fine-tuned best model)")
parser.add_argument("--image", default="data/btcv/processed/test/images/img0008-0002_s0092.png",
                    help="Image to segment (default: a BTCV test slice)")
parser.add_argument("--save_dir", default="Box_Prompt/box_prompt_outputs",
                    help="Directory to auto-save segmentation results")
args = parser.parse_args()

CHECKPOINT = args.checkpoint
IMAGE_PATH = args.image
SAVE_DIR   = args.save_dir
DEVICE     = "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 1. LOAD MODEL ────────────────────────────────────────────────────────────
print("Loading MedSAM on CPU...")
sam = _build_sam(
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_global_attn_indexes=[2, 5, 8, 11],
    checkpoint=None
)
state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
# Support both full checkpoint dict and raw state_dict
if "model_state_dict" in state_dict:
    sam.load_state_dict(state_dict["model_state_dict"])
    print(f"  Loaded fine-tuned model (epoch {state_dict.get('epoch','?')}, "
          f"best_dice={state_dict.get('best_dice','?')})")
else:
    sam.load_state_dict(state_dict)
    print("  Loaded raw pre-trained model")
sam.to(DEVICE)
sam.eval()
print("Model ready.")

# ─── 2. LOAD IMAGE & PRE-COMPUTE EMBEDDING ───────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}\nCWD: {os.getcwd()}")

pil_img   = Image.open(IMAGE_PATH).convert("RGB")
image_rgb = np.array(pil_img)
print(f"Image loaded: {image_rgb.shape}")

# Pre-compute the image embedding once (same pipeline as quick_test_box.py)
print("Computing image embedding (this takes ~30s on CPU)...")
with torch.no_grad():
    img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1)  # 3, H, W
    img_tensor = img_tensor.to(DEVICE)
    input_image = sam.preprocess(img_tensor)       # 3, 1024, 1024
    input_image = input_image.unsqueeze(0)          # 1, 3, 1024, 1024
    image_embedding = sam.image_encoder(input_image) # 1, 256, 64, 64
print("Embedding ready — draw a box to segment!\n")


# ─── 3. DIRECT INFERENCE (same as quick_test_box.py) ──────────────────────────
@torch.no_grad()
def predict_mask_direct(box_np):
    """
    Run MedSAM inference using the SAME direct pipeline as quick_test_box.py.
    Uses pre-computed image_embedding.
    """
    box_t = torch.from_numpy(box_np).float().to(DEVICE)
    box_input = box_t.unsqueeze(0).unsqueeze(0)  # 1, 1, 4

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None, boxes=box_input, masks=None,
    )

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )

    # Upsample to 1024x1024
    masks = F.interpolate(
        low_res_masks, (1024, 1024),
        mode="bilinear", align_corners=False,
    )  # 1, N, 1024, 1024

    masks = masks.squeeze(0)        # N, 1024, 1024
    scores = iou_predictions.squeeze(0)  # N

    # Pick best mask
    best_idx = torch.argmax(scores).item()
    pred_mask = (torch.sigmoid(masks[best_idx]) > 0.5).cpu().numpy().astype(np.uint8)
    score = scores[best_idx].item()

    return pred_mask, score


# ─── 4. BOX DRAWING STATE ─────────────────────────────────────────────────────
state = {
    "drawing": False,
    "start_xy": None,
    "rect_patch": None,
}

# ─── 5. SETUP FIGURE ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax_main, ax_result = axes

ax_main.imshow(image_rgb)
ax_main.set_title("Draw a box around any structure", fontsize=12)
ax_main.axis("off")

ax_result.imshow(image_rgb)
ax_result.set_title("Segmentation result will appear here", fontsize=12)
ax_result.axis("off")

plt.suptitle("MedSAM Box Prompting (direct pipeline) | Click + Drag to draw, release to segment", fontsize=13)
plt.tight_layout()

# ─── 6. EVENT HANDLERS ────────────────────────────────────────────────────────

def on_press(event):
    if event.inaxes != ax_main:
        return
    state["drawing"]  = True
    state["start_xy"] = (event.xdata, event.ydata)

    if state["rect_patch"] is not None:
        state["rect_patch"].remove()
        state["rect_patch"] = None

def on_motion(event):
    if not state["drawing"] or event.inaxes != ax_main:
        return
    if event.xdata is None or event.ydata is None:
        return

    x0, y0 = state["start_xy"]
    x1, y1 = event.xdata, event.ydata

    if state["rect_patch"] is not None:
        state["rect_patch"].remove()

    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (min(x0, x1), min(y0, y1)),
        abs(x1 - x0), abs(y1 - y0),
        linewidth=2, edgecolor="cyan",
        facecolor="cyan", alpha=0.15, linestyle="--"
    )
    state["rect_patch"] = ax_main.add_patch(rect)
    fig.canvas.draw_idle()

def on_release(event):
    if not state["drawing"]:
        return
    state["drawing"] = False

    if event.inaxes != ax_main or event.xdata is None:
        return

    x0, y0 = state["start_xy"]
    x1, y1 = event.xdata, event.ydata

    if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
        print("Box too small — drag a larger region.")
        return

    # Box as [x_min, y_min, x_max, y_max]
    box = np.array([
        min(x0, x1), min(y0, y1),
        max(x0, x1), max(y0, y1)
    ], dtype=np.float32)

    print(f"Box → x:[{box[0]:.0f}→{box[2]:.0f}]  y:[{box[1]:.0f}→{box[3]:.0f}]")

    # ── Run prediction (SAME pipeline as quick_test_box.py) ─────
    mask, score = predict_mask_direct(box)
    print(f"  Score: {score:.3f}  |  Mask pixels: {mask.sum():,}")

    # ── Update LEFT panel ───────────────────────────────────────
    ax_main.clear()
    ax_main.imshow(image_rgb)
    from matplotlib.patches import Rectangle
    ax_main.add_patch(Rectangle(
        (box[0], box[1]),
        box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor="cyan",
        facecolor="cyan", alpha=0.15, linestyle="--"
    ))
    ax_main.set_title("Prompt box (draw a new one to re-segment)", fontsize=11)
    ax_main.axis("off")

    # ── Update RIGHT panel ──────────────────────────────────────
    ax_result.clear()
    ax_result.imshow(image_rgb)

    overlay = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    overlay[mask > 0] = [220, 50, 50, 160]
    ax_result.imshow(overlay)

    if mask.any():
        ax_result.contour(mask.astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=[2])

    ax_result.add_patch(Rectangle(
        (box[0], box[1]),
        box[2] - box[0], box[3] - box[1],
        linewidth=1.5, edgecolor="cyan",
        facecolor="none", linestyle="--"
    ))

    ax_result.set_title(f"Segmented region  |  Score: {score:.3f}", fontsize=12)
    ax_result.axis("off")

    fig.canvas.draw_idle()
    state["rect_patch"] = None

    # ── Auto-save ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%H%M%S")
    save_path = os.path.join(SAVE_DIR, f"box_seg_{timestamp}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {save_path}")

# ─── 7. CONNECT EVENTS ────────────────────────────────────────────────────────
fig.canvas.mpl_connect("button_press_event",   on_press)
fig.canvas.mpl_connect("motion_notify_event",  on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

plt.show()