"""Mask post-processing — stage 6 of the abdominal CT pipeline.

After MedSAM produces a binary pathology mask, this stage cleans it up and
computes the clinical measurements the report layer needs:
  - volume (cm³)
  - longest diameter (mm) — RECIST-style
  - mean / std HU inside the mask
  - sphericity
  - surface area (mm²)

THIS IS A STUB. Gautam owns the real implementation. When wiring it in:
  - Cleanup: morphological open/close, hole-fill, keep largest connected
    component (scipy.ndimage.label + binary_fill_holes).
  - Volume: voxel_count × voxel_spacing product (the stub already does this
    properly for 3D masks).
  - Longest diameter: maximum pairwise distance over the mask boundary
    points (use scipy.spatial.distance.pdist on the surface voxels).
  - HU stats: ct_image[mask].mean() / .std() — needs the original CT
    intensity values, not the binary mask alone.
  - Sphericity: π^(1/3) × (6V)^(2/3) / SA where V=volume_mm3, SA=surface_mm2.
  - Surface area: skimage.measure.marching_cubes on the 3D mask, sum the
    triangle areas. For 2D masks, perimeter × slice_thickness is a rough
    proxy.
"""
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MaskPostProcessor:
    def __init__(self):
        # Real impl will hold cleanup-config (kernel sizes, min CC volume, etc.).
        logger.info("MaskPostProcessor running in STUB mode — Gautam's real impl pending")

    def process(
        self,
        mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ct_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Clean the mask and compute clinical measurements.

        Args:
          mask: binary 2D or 3D numpy array.
          voxel_spacing: (x_mm, y_mm, z_mm) — pulled from the NIfTI header.
          ct_image: original CT in HU for intensity stats; stub ignores it.

        Returns a dict with: cleaned_mask, volume_cm3, longest_diameter_mm,
        mean_hu, std_hu, sphericity, surface_area_mm2.
        """
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask must be np.ndarray, got {type(mask).__name__}")
        if mask.ndim not in (2, 3):
            raise ValueError(f"mask must be 2D or 3D, got shape {mask.shape}")
        if len(voxel_spacing) != 3:
            raise ValueError(
                f"voxel_spacing must be (x, y, z), got length {len(voxel_spacing)}"
            )

        cleaned = mask.astype(bool)
        voxel_count = int(cleaned.sum())

        # Volume: trivially correct from voxel count × spacing. Real impl
        # would do this on the CLEANED mask (after morphology) — same here
        # since the stub cleanup is a no-op.
        if cleaned.ndim == 3:
            vox_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
        else:
            # 2D mask: multiply by z spacing as a single-slice approximation.
            vox_mm3 = (
                voxel_spacing[0] * voxel_spacing[1]
                * (voxel_spacing[2] if voxel_spacing[2] > 0 else 1.0)
            )
        volume_cm3 = round(voxel_count * vox_mm3 / 1000.0, 3)

        # Other fields are stubbed: deterministic fake values keyed off mask
        # shape + voxel count so test assertions are stable.
        seed_str = f"{cleaned.shape}:{voxel_count}:{voxel_spacing}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        return {
            "cleaned_mask": cleaned,
            "volume_cm3": volume_cm3,
            "longest_diameter_mm": round(float(rng.uniform(5.0, 60.0)), 1),
            "mean_hu": round(float(rng.uniform(30.0, 80.0)), 1),
            "std_hu": round(float(rng.uniform(5.0, 25.0)), 1),
            "sphericity": round(float(rng.uniform(0.4, 0.95)), 3),
            "surface_area_mm2": round(float(rng.uniform(50.0, 600.0)), 1),
        }
