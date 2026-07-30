"""Stage-1 organ segmentation for the two-stage BTCV pipeline.

Isolates a target organ in a 3D CT volume so the downstream pathology
detector only searches within the relevant anatomy.  Organ-first
gives the detector a much smaller search space and stops it firing on
look-alike pathology in neighboring organs.

THIS IS A STUB. The real implementation is MedSAM fine-tuned on BTCV
organ masks (the intern has already trained these weights). To wire the
real model in:
  - Replace `_stub_organ_mask` with a call to the fine-tuned MedSAM.
  - Pass the full volume + organ-class index (not name) to the model.
  - Post-process to a single connected component if MedSAM returns multi-CC.
"""
import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from parser import BTCV_ORGANS

logger = logging.getLogger(__name__)


# Anatomical anchors per BTCV organ as fractions of the volume axes:
#   (cz_frac, cy_frac, cx_frac, radius_frac)
# These are *rough* normalized centers for a supine abdominal CT, used only
# by the stub to produce visually plausible blobs in the right anatomical
# region for tests and debugging. They are NOT clinically accurate.
_ORGAN_ANCHORS = {
    "spleen":              (0.55, 0.45, 0.85, 0.10),
    "right kidney":        (0.45, 0.55, 0.20, 0.08),
    "left kidney":         (0.45, 0.55, 0.80, 0.08),
    "gallbladder":         (0.60, 0.45, 0.40, 0.05),
    "esophagus":           (0.85, 0.50, 0.50, 0.03),
    "liver":               (0.65, 0.45, 0.30, 0.18),
    "stomach":             (0.65, 0.40, 0.65, 0.10),
    "aorta":               (0.50, 0.65, 0.50, 0.04),
    "inferior vena cava":  (0.50, 0.60, 0.45, 0.04),
    "portal vein":         (0.60, 0.55, 0.40, 0.03),
    "pancreas":            (0.55, 0.55, 0.50, 0.07),
    "right adrenal gland": (0.55, 0.55, 0.30, 0.03),
    "left adrenal gland":  (0.55, 0.55, 0.70, 0.03),
}

# Sanity check: the stub anchors must cover every BTCV organ exactly.
assert set(_ORGAN_ANCHORS) == set(BTCV_ORGANS), (
    "OrganSegmentor anchors and parser BTCV_ORGANS are out of sync"
)


class OrganMask(BaseModel):
    organ: str
    bbox: List[int]      # [x1, y1, z1, x2, y2, z2] of the mask
    voxel_count: int
    mask: np.ndarray     # 3D bool array, same shape as input volume

    model_config = {"arbitrary_types_allowed": True}


class OrganSegmentor:
    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self._is_stub = weights_path is None
        if self._is_stub:
            logger.info(
                "OrganSegmentor running in STUB mode — replace with intern's "
                "MedSAM-on-BTCV weights when wiring real model"
            )

    def segment(self, volume_3d: np.ndarray, organ_name: str) -> OrganMask:
        if not isinstance(volume_3d, np.ndarray):
            raise TypeError(
                f"volume_3d must be np.ndarray, got {type(volume_3d).__name__}"
            )
        if volume_3d.ndim != 3:
            raise ValueError(
                f"volume_3d must be 3D (H, W, D), got shape {volume_3d.shape}"
            )
        if organ_name not in _ORGAN_ANCHORS:
            raise ValueError(
                f"Unknown organ {organ_name!r}; must be one of "
                f"{sorted(_ORGAN_ANCHORS)}"
            )
        if self._is_stub:
            return self._stub_organ_mask(volume_3d, organ_name)
        raise NotImplementedError(
            "Real MedSAM-on-BTCV organ segmentation not wired in yet"
        )

    def _stub_organ_mask(
        self, volume: np.ndarray, organ_name: str,
    ) -> OrganMask:
        h, w, d = volume.shape
        cz_f, cy_f, cx_f, r_f = _ORGAN_ANCHORS[organ_name]
        cz = cz_f * d
        cy = cy_f * h
        cx = cx_f * w
        # Organs tend to extend more in z than the simple radius — give them
        # 1.5x in z so single-slice depths don't produce empty masks.
        rxy = max(2.0, r_f * max(h, w))
        rz = max(2.0, r_f * d * 1.5)

        ys = np.arange(h)[:, None, None]
        xs = np.arange(w)[None, :, None]
        zs = np.arange(d)[None, None, :]
        mask = (
            ((ys - cy) / rxy) ** 2
            + ((xs - cx) / rxy) ** 2
            + ((zs - cz) / rz) ** 2
        ) <= 1.0

        nz = np.argwhere(mask)
        if nz.size == 0:
            # Degenerate: organ anchor fell entirely outside the volume.
            # Return an empty mask with a zero-volume bbox at the center.
            bbox = [int(cx), int(cy), int(cz), int(cx), int(cy), int(cz)]
            voxel_count = 0
        else:
            ymin, xmin, zmin = nz.min(axis=0)
            ymax, xmax, zmax = nz.max(axis=0)
            bbox = [int(xmin), int(ymin), int(zmin),
                    int(xmax), int(ymax), int(zmax)]
            voxel_count = int(mask.sum())

        return OrganMask(
            organ=organ_name,
            bbox=bbox,
            voxel_count=voxel_count,
            mask=mask,
        )
