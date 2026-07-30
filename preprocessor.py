"""CT preprocessing: HU windowing, normalization, resampling.

Reads NIfTI volumes and produces inputs ready for the BTCV pipeline:
  1. Hounsfield-unit clip to soft-tissue window [-150, 250]
  2. Normalize to [0, 1]
  3. Resample in-plane to 512x512 (depth preserved)
  4. Return both the full 3D volume and a list of axial 2D slices

If `nibabel` is not installed we fall back to a synthetic stub volume so
downstream modules remain testable. THIS IS A STUB — real production must
have nibabel installed; the warning is logged loudly on every fallback.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nib = None
    NIBABEL_AVAILABLE = False
    # NOTE: stub-only mode. Real BTCV preprocessing requires nibabel.
    logger.warning(
        "nibabel not installed — CTPreprocessor will return SYNTHETIC STUB volumes. "
        "Install with: pip install nibabel"
    )


# Soft-tissue window for abdominal CT. These bounds are the standard for
# liver/spleen/kidney protocols; if you need bone or lung windows, change
# them at instantiation.
HU_MIN_DEFAULT = -150.0
HU_MAX_DEFAULT = 250.0

TARGET_HW_DEFAULT = 512


class CTPreprocessor:
    def __init__(
        self,
        hu_min: float = HU_MIN_DEFAULT,
        hu_max: float = HU_MAX_DEFAULT,
        target_hw: int = TARGET_HW_DEFAULT,
    ):
        if hu_max <= hu_min:
            raise ValueError(f"hu_max ({hu_max}) must exceed hu_min ({hu_min})")
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.target_hw = int(target_hw)

    def preprocess(
        self, nifti_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run the full preprocessing pipeline.

        Returns (volume_3d, axial_slices). volume_3d shape is
        (target_hw, target_hw, D); axial_slices is a list of D arrays each
        shaped (target_hw, target_hw).
        """
        if NIBABEL_AVAILABLE and nifti_path is not None:
            volume = self._load_nifti(nifti_path)
        else:
            if nifti_path is not None:
                logger.warning(
                    "STUB: nibabel unavailable; ignoring path %r and returning "
                    "synthetic volume", nifti_path,
                )
            volume = self._stub_volume()

        volume = self._window_hu(volume)
        volume = self._normalize(volume)
        volume = self._resize_xy(volume, self.target_hw)

        slices = [volume[:, :, z] for z in range(volume.shape[2])]
        return volume, slices

    # ------------------------------------------------------------------
    def _load_nifti(self, path: str) -> np.ndarray:
        img = nib.load(path)
        # nibabel returns shape (X, Y, Z); we treat the first two as (H, W)
        # and the third as depth. Real BTCV volumes use this convention.
        return np.asarray(img.get_fdata(), dtype=np.float32)

    def _stub_volume(
        self, h: int = 512, w: int = 512, d: int = 64,
    ) -> np.ndarray:
        # Synthesize HU-like values across the soft-tissue range so the rest
        # of the pipeline (windowing/normalization) has something realistic
        # to operate on. Deterministic seed for stable tests.
        rng = np.random.default_rng(42)
        return rng.uniform(-200, 300, size=(h, w, d)).astype(np.float32)

    def _window_hu(self, volume: np.ndarray) -> np.ndarray:
        return np.clip(volume, self.hu_min, self.hu_max)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        return (volume - self.hu_min) / (self.hu_max - self.hu_min)

    def _resize_xy(self, volume: np.ndarray, target_hw: int) -> np.ndarray:
        h, w, d = volume.shape
        if h == target_hw and w == target_hw:
            return volume
        # Pure-numpy nearest-neighbor resampling — we deliberately don't
        # take a scipy dependency for this stub. Real production should use
        # scipy.ndimage.zoom or monai.transforms.Resized for proper
        # bilinear/bicubic interpolation.
        ys = np.linspace(0, h - 1, target_hw).astype(np.int64)
        xs = np.linspace(0, w - 1, target_hw).astype(np.int64)
        return volume[np.ix_(ys, xs, np.arange(d))]
