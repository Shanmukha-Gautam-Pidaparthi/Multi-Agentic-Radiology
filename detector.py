"""MONAI-based pathology detection (stage 2 of the BTCV pipeline).

Sits between the organ segmentor and the MedSAM segmentor: takes a
preprocessed volume + organ mask + TaskParams and returns candidate
pathology bboxes constrained to the organ region.

This is a STUB. Real swap-in notes:
  - The real MONAI RetinaNet expects (C, H, W, D) tensors and per-organ
    fine-tuned weights — we'll need an intensity normalization step and
    an organ→model lookup table (one detector per organ, or a multi-class
    head).
  - Real model returns many overlapping boxes; we'll need NMS and a score
    threshold. The stub already returns a small, score-sorted set so
    downstream code shouldn't assume NMS has run.
  - Real `organ_mask` use: crop the volume to the mask's bbox and zero
    voxels outside the mask before inference. The stub uses the bbox to
    constrain box generation but doesn't crop.
  - Real model returns class indices; we'll need a per-organ class→label
    table. The stub takes the label from `params.pathology` directly.
"""
import hashlib
import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from parser import TaskParams

logger = logging.getLogger(__name__)


# Canonical pathology labels for abdominal CT detection. The LLM parser
# fills `params.pathology` from this vocabulary; the stub uses it as a
# fallback when params.pathology is empty/unknown.
ABDOMINAL_PATHOLOGY_LABELS = (
    "lesion",
    "cyst",
    "mass",
    "tumor",
    "calcification",
)


class Detection(BaseModel):
    # 4 floats for 2D ([x1,y1,x2,y2]) or 6 for 3D ([x1,y1,z1,x2,y2,z2]).
    bbox: List[float]
    label: str
    score: float


class MONAIDetector:
    """Wraps a MONAI RetinaNet detector. Stubbed until real weights land."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._is_stub = weights_path is None
        if self._is_stub:
            logger.info("MONAIDetector running in STUB mode — no weights loaded")

    def detect(
        self,
        image: np.ndarray,
        params: TaskParams,
        is_3d: bool = False,
        organ_mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Run pathology detection on the input image/volume.

        Args:
          image: 2D slice (H, W) / RGB (H, W, 3) when is_3d=False, or
                 3D volume (H, W, D) when is_3d=True.
          params: parsed TaskParams — `pathology` is used as the label.
          is_3d: if True, return 6-element bboxes; otherwise 4-element.
          organ_mask: optional bool ndarray with the same shape as `image`.
                 Real detectors will only search within the True voxels;
                 the stub uses the mask's bbox to constrain candidate boxes.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
        if is_3d:
            if image.ndim != 3 or image.shape[2] <= 3:
                raise ValueError(
                    f"is_3d=True requires a 3D volume (H,W,D) with D>3, "
                    f"got shape {image.shape}"
                )
        else:
            valid_2d = image.ndim == 2 or (image.ndim == 3 and image.shape[2] <= 3)
            if not valid_2d:
                raise ValueError(
                    f"is_3d=False requires a 2D slice (H,W) or RGB (H,W,3), "
                    f"got shape {image.shape}"
                )

        if organ_mask is not None:
            if organ_mask.shape != image.shape[:organ_mask.ndim]:
                raise ValueError(
                    f"organ_mask shape {organ_mask.shape} does not match "
                    f"image shape {image.shape}"
                )

        if self._is_stub:
            return self._stub_detect(image, params, is_3d, organ_mask)
        raise NotImplementedError("Real MONAI inference not implemented yet")

    # ------------------------------------------------------------------
    def _stub_detect(
        self,
        image: np.ndarray,
        params: TaskParams,
        is_3d: bool,
        organ_mask: Optional[np.ndarray],
    ) -> List[Detection]:
        # Constrain the search region: if an organ mask was supplied, draw
        # boxes inside its bbox; otherwise span the whole image. The real
        # detector will crop the volume by the mask before inference.
        if organ_mask is not None and organ_mask.any():
            search_bounds = self._mask_bbox(organ_mask, is_3d)
        else:
            if is_3d:
                h, w, d = image.shape
                search_bounds = (0, 0, 0, h - 1, w - 1, d - 1)
            else:
                h, w = image.shape[:2]
                search_bounds = (0, 0, h - 1, w - 1)

        if is_3d:
            y_lo, x_lo, z_lo, y_hi, x_hi, z_hi = search_bounds
            seed_str = f"3d:{search_bounds}:{params.organ}:{params.pathology}"
        else:
            y_lo, x_lo, y_hi, x_hi = search_bounds
            z_lo = z_hi = 0
            seed_str = f"2d:{search_bounds}:{params.organ}:{params.pathology}"

        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        n = int(rng.integers(1, 4))   # 1, 2, or 3 candidate detections
        label = self._pick_label(params, rng)

        # Box sizes scale to the search region so we don't generate boxes
        # larger than the organ bbox.
        max_w = max(8, (x_hi - x_lo) // 2)
        max_h = max(8, (y_hi - y_lo) // 2)
        max_d = max(2, (z_hi - z_lo) // 2) if is_3d else 0
        min_xy = max(4, min(8, max_w, max_h))

        detections: List[Detection] = []
        for _ in range(n):
            box_w = int(rng.integers(min_xy, max_w + 1))
            box_h = int(rng.integers(min_xy, max_h + 1))
            x1 = int(rng.integers(x_lo, max(x_lo + 1, x_hi - box_w + 1)))
            y1 = int(rng.integers(y_lo, max(y_lo + 1, y_hi - box_h + 1)))
            x2 = min(x1 + box_w, x_hi)
            y2 = min(y1 + box_h, y_hi)
            score = float(rng.uniform(0.55, 0.92))
            if is_3d:
                box_d = int(rng.integers(2, max_d + 1))
                z1 = int(rng.integers(z_lo, max(z_lo + 1, z_hi - box_d + 1)))
                z2 = min(z1 + box_d, z_hi)
                bbox = [float(x1), float(y1), float(z1),
                        float(x2), float(y2), float(z2)]
            else:
                bbox = [float(x1), float(y1), float(x2), float(y2)]
            detections.append(Detection(
                bbox=bbox, label=label, score=round(score, 3),
            ))

        # MONAI/RetinaNet returns boxes sorted by score descending — match it.
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    @staticmethod
    def _mask_bbox(mask: np.ndarray, is_3d: bool):
        nz = np.argwhere(mask)
        if is_3d:
            ymin, xmin, zmin = nz.min(axis=0)
            ymax, xmax, zmax = nz.max(axis=0)
            return (int(ymin), int(xmin), int(zmin),
                    int(ymax), int(xmax), int(zmax))
        ymin, xmin = nz.min(axis=0)
        ymax, xmax = nz.max(axis=0)
        return int(ymin), int(xmin), int(ymax), int(xmax)

    @staticmethod
    def _pick_label(params: TaskParams, rng: np.random.Generator) -> str:
        path = (params.pathology or "").strip().lower()
        if path and path != "unknown":
            return path
        # Fallback for stubs only — the real detector will always have a
        # specific class label from the model head.
        return str(rng.choice(ABDOMINAL_PATHOLOGY_LABELS))
