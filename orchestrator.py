"""Pipeline orchestrator: parser → router → organ_segmentor → detector → MedSAM → post-process.

Three input modes are supported:

  click_only    — radiologist clicks a region on a slice. Skip parser, organ
                  segmentation, and detection. Hand the click bbox straight
                  to MedSAM, then post-process the mask.

  text_only     — radiologist types a free-text instruction. Run the full
                  two-stage pipeline: parser → organ-seg → detector → MedSAM
                  → post-process.

  combined      — instruction AND a click. Parse the instruction for context
                  (organ, task, urgency) but trust the click as ground truth
                  — skip organ-seg and detection, hand the click bbox to
                  MedSAM, then post-process.

For all modes, the OrchestratorResult carries every intermediate output so
the report layer can show provenance for each stage.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from parser import TaskParams, parse_instruction, confidence_router
from detector import Detection, MONAIDetector
from segmentor import MedSAMSegmentor, SegmentationResult
from organ_segmentor import OrganMask, OrganSegmentor
from postprocessor import MaskPostProcessor

logger = logging.getLogger(__name__)


# Tasks where the detector should run (text_only path only). "segment" alone
# means click-driven workflow — handled via the explicit click_only/combined
# modes now.
_DETECTION_TASKS = {"detect", "detect+segment"}
# Tasks where MedSAM runs on top of detection output (box-prompted seg).
_BOX_PROMPTED_SEG_TASKS = {"detect+segment"}


class OrchestratorResult(BaseModel):
    instruction: Optional[str] = None
    click_bbox: Optional[List[float]] = None
    input_mode: str                    # "click_only" | "text_only" | "combined"
    input_dimensions: str              # "2d" | "3d"
    task_params: Optional[TaskParams] = None
    decision: Optional[str] = None     # "proceed" | "flag_for_review" | "reject"
    status: str                        # see below
    organ_mask: Optional[OrganMask] = None
    detections: Optional[List[Detection]] = None
    pathology_mask: Optional[SegmentationResult] = None
    z_mid: Optional[int] = None
    measurements: Optional[Dict[str, Any]] = None
    classification: Optional[Any] = None   # placeholder for v2.0 classifier
    report: Optional[Any] = None           # placeholder for report generator
    message: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

# Status taxonomy:
#   "completed"     — pipeline ran end-to-end for the chosen mode
#   "segment_only"  — text_only proceed, but task is segment-only so detection skipped
#   "needs_review"  — text_only router flagged for human confirmation
#   "rejected"      — text_only router rejected (organ/pathology missing or non-BTCV)


def _classify_image(image: np.ndarray) -> str:
    """Return '2d' or '3d' based on shape. 3D = ndim==3 and shape[2] > 3."""
    if image.ndim == 2:
        return "2d"
    if image.ndim == 3:
        return "3d" if image.shape[2] > 3 else "2d"
    raise ValueError(
        f"Unsupported image: expected 2D (H,W), RGB (H,W,3), or 3D (H,W,D); "
        f"got ndim={image.ndim}, shape={image.shape}"
    )


def _norm_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s or None


def _norm_bbox(b: Optional[List[float]]) -> Optional[List[float]]:
    if b is None:
        return None
    if len(b) != 4:
        raise ValueError(f"click_bbox must be [x1,y1,x2,y2] (length 4), got {b}")
    return [float(v) for v in b]


class Orchestrator:
    def __init__(
        self,
        organ_segmentor: Optional[OrganSegmentor] = None,
        detector: Optional[MONAIDetector] = None,
        segmentor: Optional[MedSAMSegmentor] = None,
        postprocessor: Optional[MaskPostProcessor] = None,
    ):
        self.organ_segmentor = organ_segmentor or OrganSegmentor()
        self.detector = detector or MONAIDetector()
        self.segmentor = segmentor or MedSAMSegmentor()
        self.postprocessor = postprocessor or MaskPostProcessor()

    def run(
        self,
        *,
        instruction: Optional[str] = None,
        click_bbox: Optional[List[float]] = None,
        ct_slice: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        _llm_output: Optional[str] = None,
    ) -> OrchestratorResult:
        """Dispatch to one of three input-mode pipelines.

        At least one of `instruction` or `click_bbox` must be provided.
        `ct_slice` is required whenever `click_bbox` is supplied (the click
        always refers to a 2D slice). For text-only, either `volume` (3D
        production path) or `ct_slice` (2D fallback) must be supplied.

        `_llm_output` is a test seam — when set, the parser uses it instead
        of calling Ollama.
        """
        instruction = _norm_text(instruction)
        click_bbox = _norm_bbox(click_bbox)

        if instruction is None and click_bbox is None:
            raise ValueError(
                "Must provide at least one of `instruction` or `click_bbox`"
            )

        if click_bbox is not None and instruction is None:
            return self._run_click_only(click_bbox, ct_slice, voxel_spacing)
        if instruction is not None and click_bbox is None:
            return self._run_text_only(
                instruction, volume, ct_slice, voxel_spacing, _llm_output,
            )
        return self._run_combined(
            instruction, click_bbox, ct_slice, voxel_spacing, _llm_output,
        )

    # ==================================================================
    # click_only — pure MedSAM
    # ==================================================================
    def _run_click_only(
        self,
        click_bbox: List[float],
        ct_slice: Optional[np.ndarray],
        voxel_spacing: Tuple[float, float, float],
    ) -> OrchestratorResult:
        if ct_slice is None:
            raise ValueError(
                "click_only mode requires `ct_slice` — the slice the click was made on"
            )
        dims = _classify_image(ct_slice)
        if dims != "2d":
            raise ValueError(
                f"click_only mode requires a 2D slice, got shape {ct_slice.shape}"
            )

        pathology_mask = self.segmentor.segment(ct_slice, click_bbox, label="lesion")
        measurements = self.postprocessor.process(
            pathology_mask.mask, voxel_spacing=voxel_spacing, ct_image=ct_slice,
        )
        return OrchestratorResult(
            click_bbox=click_bbox,
            input_mode="click_only",
            input_dimensions="2d",
            status="completed",
            pathology_mask=pathology_mask,
            measurements=measurements,
            message="click_only: MedSAM segmented the click bbox; LLM/detector skipped.",
        )

    # ==================================================================
    # combined — parse for context, but trust the click for segmentation
    # ==================================================================
    def _run_combined(
        self,
        instruction: str,
        click_bbox: List[float],
        ct_slice: Optional[np.ndarray],
        voxel_spacing: Tuple[float, float, float],
        _llm_output: Optional[str],
    ) -> OrchestratorResult:
        if ct_slice is None:
            raise ValueError(
                "combined mode requires `ct_slice` — the slice the click was made on"
            )
        dims = _classify_image(ct_slice)
        if dims != "2d":
            raise ValueError(
                f"combined mode requires a 2D slice, got shape {ct_slice.shape}"
            )

        # Parse for metadata enrichment only — the click is authoritative,
        # so we proceed regardless of what the router says about the text.
        params = parse_instruction(instruction, _llm_output=_llm_output)
        decision = confidence_router(params)
        label = (params.pathology or "lesion").strip().lower() or "lesion"

        pathology_mask = self.segmentor.segment(ct_slice, click_bbox, label=label)
        measurements = self.postprocessor.process(
            pathology_mask.mask, voxel_spacing=voxel_spacing, ct_image=ct_slice,
        )
        return OrchestratorResult(
            instruction=instruction,
            click_bbox=click_bbox,
            input_mode="combined",
            input_dimensions="2d",
            task_params=params,
            decision=decision,
            status="completed",
            pathology_mask=pathology_mask,
            measurements=measurements,
            message=("combined: MedSAM segmented the click bbox; "
                     "LLM provided context but organ-seg/detector skipped."),
        )

    # ==================================================================
    # text_only — full pipeline
    # ==================================================================
    def _run_text_only(
        self,
        instruction: str,
        volume: Optional[np.ndarray],
        ct_slice: Optional[np.ndarray],
        voxel_spacing: Tuple[float, float, float],
        _llm_output: Optional[str],
    ) -> OrchestratorResult:
        if volume is None and ct_slice is None:
            raise ValueError(
                "text_only mode requires `volume` (3D production path) "
                "or `ct_slice` (2D fallback)"
            )
        if volume is not None:
            dims = _classify_image(volume)
            if dims != "3d":
                raise ValueError(
                    f"`volume` must be 3D (H,W,D) with D>3, got shape {volume.shape}"
                )
            return self._text_only_3d(
                instruction, volume, voxel_spacing, _llm_output,
            )
        return self._text_only_2d(
            instruction, ct_slice, voxel_spacing, _llm_output,
        )

    def _early_return(
        self, instruction: str, params: TaskParams, decision: str, dims: str,
    ) -> Optional[OrchestratorResult]:
        if decision == "reject":
            logger.info("Rejecting %r: %s", instruction, params.model_dump())
            return OrchestratorResult(
                instruction=instruction, input_mode="text_only",
                input_dimensions=dims, task_params=params, decision=decision,
                status="rejected",
                message=("Parser could not identify a BTCV-13 organ + pathology. "
                         "Ask the radiologist to clarify before retrying."),
            )
        if decision == "flag_for_review":
            return OrchestratorResult(
                instruction=instruction, input_mode="text_only",
                input_dimensions=dims, task_params=params, decision=decision,
                status="needs_review",
                message=("Low parse confidence. Confirm parsed parameters "
                         "with the radiologist before dispatching to AI models."),
            )
        if params.task not in _DETECTION_TASKS:
            return OrchestratorResult(
                instruction=instruction, input_mode="text_only",
                input_dimensions=dims, task_params=params, decision=decision,
                status="segment_only",
                message=("Task is segmentation-only; skipping detection. "
                         "Awaiting click input — call run() again in click_only mode."),
            )
        return None

    def _text_only_3d(
        self,
        instruction: str,
        volume: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        _llm_output: Optional[str],
    ) -> OrchestratorResult:
        params = parse_instruction(instruction, _llm_output=_llm_output)
        decision = confidence_router(params)
        early = self._early_return(instruction, params, decision, "3d")
        if early is not None:
            return early

        organ_mask = self.organ_segmentor.segment(volume, params.organ)
        detections = self.detector.detect(
            volume, params, is_3d=True, organ_mask=organ_mask.mask,
        )

        pathology_mask: Optional[SegmentationResult] = None
        z_mid: Optional[int] = None
        measurements: Optional[Dict[str, Any]] = None
        if params.task in _BOX_PROMPTED_SEG_TASKS and detections:
            top = detections[0]
            if len(top.bbox) != 6:
                raise ValueError(
                    f"3D pipeline expected 6-element bbox, got {top.bbox}"
                )
            x1, y1, z1, x2, y2, z2 = top.bbox
            z_mid = int((z1 + z2) // 2)
            slice_2d = volume[:, :, z_mid]
            pathology_mask = self.segmentor.segment(
                slice_2d, [x1, y1, x2, y2], label=top.label,
            )
            measurements = self.postprocessor.process(
                pathology_mask.mask, voxel_spacing=voxel_spacing,
                ct_image=slice_2d,
            )

        return OrchestratorResult(
            instruction=instruction, input_mode="text_only",
            input_dimensions="3d", task_params=params, decision=decision,
            status="completed",
            organ_mask=organ_mask,
            detections=detections,
            pathology_mask=pathology_mask,
            z_mid=z_mid,
            measurements=measurements,
            message=self._completed_message(
                detections, pathology_mask, dims="3d", z_mid=z_mid,
            ),
        )

    def _text_only_2d(
        self,
        instruction: str,
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        _llm_output: Optional[str],
    ) -> OrchestratorResult:
        params = parse_instruction(instruction, _llm_output=_llm_output)
        decision = confidence_router(params)
        early = self._early_return(instruction, params, decision, "2d")
        if early is not None:
            return early

        # 2D fallback: no organ segmentation possible from a single slice.
        detections = self.detector.detect(image, params, is_3d=False)

        pathology_mask: Optional[SegmentationResult] = None
        measurements: Optional[Dict[str, Any]] = None
        if params.task in _BOX_PROMPTED_SEG_TASKS and detections:
            top = detections[0]
            pathology_mask = self.segmentor.segment(image, top.bbox, label=top.label)
            measurements = self.postprocessor.process(
                pathology_mask.mask, voxel_spacing=voxel_spacing, ct_image=image,
            )

        return OrchestratorResult(
            instruction=instruction, input_mode="text_only",
            input_dimensions="2d", task_params=params, decision=decision,
            status="completed",
            detections=detections,
            pathology_mask=pathology_mask,
            measurements=measurements,
            message=self._completed_message(
                detections, pathology_mask, dims="2d",
            ),
        )

    # ==================================================================
    @staticmethod
    def _completed_message(
        detections: Optional[List[Detection]],
        pathology_mask: Optional[SegmentationResult],
        *,
        dims: str,
        z_mid: Optional[int] = None,
    ) -> str:
        n = len(detections) if detections else 0
        msg = f"Detector returned {n} candidate(s)"
        if dims == "3d":
            msg += " (3D bbox within organ mask)"
            if z_mid is not None:
                msg += f"; pathology segmented at z={z_mid}"
        elif pathology_mask is not None:
            msg += "; pathology segmented on top candidate"
        return msg + "."
