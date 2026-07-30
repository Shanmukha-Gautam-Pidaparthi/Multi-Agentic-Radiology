"""MedSAM-based 2D segmentation wrapper.

Sits downstream of the detector. Takes a 2D slice plus a bounding-box prompt
and returns a binary mask of the same H×W shape. Stub for now — real MedSAM
weights aren't fine-tuned for our brain-MRI task. The interface (image +
2D bbox → mask + score) matches MedSAM's box-prompted API so the real model
drops in without changes to the orchestrator.
"""
import hashlib
import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SegmentationResult(BaseModel):
    bbox: List[float]   # 2D prompt bbox: [x1, y1, x2, y2]
    label: str
    score: float        # 0.0 - 1.0
    mask: np.ndarray    # 2D bool array, same H×W as input slice

    model_config = {"arbitrary_types_allowed": True}


class MedSAMSegmentor:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._is_stub = weights_path is None
        if self._is_stub:
            logger.info("MedSAMSegmentor running in STUB mode — no weights loaded")

    def segment(
        self,
        image: np.ndarray,
        bbox_2d: List[float],
        label: str = "lesion",
    ) -> SegmentationResult:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
        if image.ndim == 2:
            h, w = image.shape
        elif image.ndim == 3 and image.shape[2] == 3:
            h, w = image.shape[:2]  # accept RGB; stub doesn't care about channels
        else:
            raise ValueError(
                f"MedSAMSegmentor expects 2D slice, got shape {image.shape}"
            )
        if len(bbox_2d) != 4:
            raise ValueError(
                f"bbox_2d must be [x1,y1,x2,y2] (length 4), got length {len(bbox_2d)}"
            )
        if self._is_stub:
            return self._stub_segment(h, w, bbox_2d, label)
        raise NotImplementedError("Real MedSAM inference not implemented yet")

    def _stub_segment(
        self, h: int, w: int, bbox_2d: List[float], label: str
    ) -> SegmentationResult:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_2d]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        mask = np.zeros((h, w), dtype=bool)
        if x2 > x1 and y2 > y1:
            # Stub: ellipse inscribed in the bbox — visually plausible for
            # round-ish lesions and gives non-trivial mask geometry for tests.
            cy = (y1 + y2) / 2.0
            cx = (x1 + x2) / 2.0
            ry = max(0.5, (y2 - y1) / 2.0)
            rx = max(0.5, (x2 - x1) / 2.0)
            ys, xs = np.ogrid[:h, :w]
            mask[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0] = True

        seed = int(
            hashlib.sha256(f"{h}x{w}:{bbox_2d}".encode()).hexdigest()[:8], 16
        )
        rng = np.random.default_rng(seed)
        score = float(round(rng.uniform(0.6, 0.95), 3))

        return SegmentationResult(
            bbox=[float(v) for v in bbox_2d],
            label=label,
            score=score,
            mask=mask,
        )
