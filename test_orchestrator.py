"""Tests for orchestrator.py — three-mode dispatch + post-processing.

All tests inject synthetic LLM output via the `_llm_output` test seam, so
nothing here hits Ollama and the suite runs deterministically in <2s.
"""
import json
import unittest

import numpy as np

from detector import Detection, MONAIDetector
from organ_segmentor import OrganMask, OrganSegmentor
from orchestrator import Orchestrator, OrchestratorResult
from postprocessor import MaskPostProcessor
from segmentor import MedSAMSegmentor, SegmentationResult


def _llm(**fields) -> str:
    """Build a JSON payload matching the parser's expected schema."""
    base = dict(
        organ="liver", pathology="lesion", task="detect+segment",
        region=None, urgency="routine", parse_confidence="high",
    )
    base.update(fields)
    return json.dumps(base)


def _fake_volume(h: int = 64, w: int = 64, d: int = 32) -> np.ndarray:
    return np.zeros((h, w, d), dtype=np.float32)


def _fake_slice(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)


_MEASUREMENT_KEYS = {
    "cleaned_mask", "volume_cm3", "longest_diameter_mm",
    "mean_hu", "std_hu", "sphericity", "surface_area_mm2",
}


# ============================================================================
# Input validation
# ============================================================================

class InputValidationTests(unittest.TestCase):
    def test_neither_instruction_nor_click_raises(self):
        with self.assertRaises(ValueError):
            Orchestrator().run(ct_slice=_fake_slice())

    def test_click_without_ct_slice_raises(self):
        with self.assertRaises(ValueError):
            Orchestrator().run(click_bbox=[0, 0, 10, 10])

    def test_combined_without_ct_slice_raises(self):
        with self.assertRaises(ValueError):
            Orchestrator().run(
                instruction="find lesion in liver",
                click_bbox=[0, 0, 10, 10],
                _llm_output=_llm(),
            )

    def test_text_only_without_image_raises(self):
        with self.assertRaises(ValueError):
            Orchestrator().run(
                instruction="find lesion in liver", _llm_output=_llm(),
            )

    def test_malformed_click_bbox_raises(self):
        with self.assertRaises(ValueError):
            Orchestrator().run(
                click_bbox=[0, 0, 10],   # only 3 elements
                ct_slice=_fake_slice(),
            )

    def test_click_only_with_volume_input_raises(self):
        # click_only requires a 2D ct_slice; passing a 3D volume there errors.
        with self.assertRaises(ValueError):
            Orchestrator().run(
                click_bbox=[0, 0, 10, 10], ct_slice=_fake_volume(),
            )


# ============================================================================
# Mode 1: click_only
# ============================================================================

class ClickOnlyModeTests(unittest.TestCase):
    def test_basic_click_returns_mask_and_measurements(self):
        result = Orchestrator().run(
            click_bbox=[10, 10, 40, 40],
            ct_slice=_fake_slice(),
            voxel_spacing=(0.7, 0.7, 5.0),
        )
        self.assertIsInstance(result, OrchestratorResult)
        self.assertEqual(result.input_mode, "click_only")
        self.assertEqual(result.input_dimensions, "2d")
        self.assertEqual(result.status, "completed")
        # MedSAM ran:
        self.assertIsInstance(result.pathology_mask, SegmentationResult)
        self.assertEqual(result.pathology_mask.mask.shape, (64, 64))
        # Post-processing ran:
        self.assertIsNotNone(result.measurements)
        self.assertEqual(set(result.measurements), _MEASUREMENT_KEYS)
        # No parsing happened:
        self.assertIsNone(result.task_params)
        self.assertIsNone(result.decision)
        # No organ-seg / detection happened:
        self.assertIsNone(result.organ_mask)
        self.assertIsNone(result.detections)
        # Placeholders:
        self.assertIsNone(result.classification)
        self.assertIsNone(result.report)

    def test_click_only_does_not_invoke_parser_or_detector(self):
        # If parser/detector/organ-seg were called, they'd fail loudly.
        class FailingOrganSeg(OrganSegmentor):
            def segment(self, *a, **k):
                raise AssertionError("organ_segmentor must not run in click_only")

        class FailingDetector(MONAIDetector):
            def detect(self, *a, **k):
                raise AssertionError("detector must not run in click_only")

        orch = Orchestrator(
            organ_segmentor=FailingOrganSeg(), detector=FailingDetector(),
        )
        # No _llm_output passed — if the parser ran, it'd try to hit Ollama
        # (and we assert the result regardless to confirm we never went
        # through the parser path).
        result = orch.run(
            click_bbox=[5, 5, 25, 25], ct_slice=_fake_slice(),
        )
        self.assertEqual(result.input_mode, "click_only")
        self.assertIsNone(result.task_params)

    def test_voxel_spacing_propagates_to_volume(self):
        # 2D mask of N voxels, spacing (1, 1, 5) → volume = N × 5 mm³ / 1000
        # in cm³. The post-processor uses the cleaned mask, so we just check
        # volume_cm3 is positive and scales with spacing.
        small = Orchestrator().run(
            click_bbox=[20, 20, 30, 30], ct_slice=_fake_slice(),
            voxel_spacing=(1.0, 1.0, 1.0),
        )
        big = Orchestrator().run(
            click_bbox=[20, 20, 30, 30], ct_slice=_fake_slice(),
            voxel_spacing=(1.0, 1.0, 10.0),
        )
        self.assertGreater(big.measurements["volume_cm3"],
                           small.measurements["volume_cm3"])


# ============================================================================
# Mode 2: text_only — full pipeline
# ============================================================================

class TextOnlyModeTests(unittest.TestCase):
    def setUp(self):
        self.orch = Orchestrator()
        self.volume = _fake_volume()

    def test_text_only_3d_runs_all_stages(self):
        result = self.orch.run(
            instruction="find a lesion in the liver",
            volume=self.volume,
            _llm_output=_llm(organ="liver", pathology="lesion"),
        )
        self.assertEqual(result.input_mode, "text_only")
        self.assertEqual(result.input_dimensions, "3d")
        self.assertEqual(result.status, "completed")
        # Every stage ran:
        self.assertIsNotNone(result.task_params)
        self.assertIsNotNone(result.organ_mask)
        self.assertIsNotNone(result.detections)
        self.assertIsNotNone(result.pathology_mask)
        self.assertIsNotNone(result.measurements)
        self.assertEqual(set(result.measurements), _MEASUREMENT_KEYS)
        # 3D bboxes:
        for det in result.detections:
            self.assertEqual(len(det.bbox), 6)

    def test_text_only_2d_skips_organ_seg(self):
        result = self.orch.run(
            instruction="find a lesion in the liver",
            ct_slice=_fake_slice(),
            _llm_output=_llm(organ="liver", pathology="lesion"),
        )
        self.assertEqual(result.input_mode, "text_only")
        self.assertEqual(result.input_dimensions, "2d")
        self.assertEqual(result.status, "completed")
        self.assertIsNone(result.organ_mask)
        self.assertIsNotNone(result.detections)
        for det in result.detections:
            self.assertEqual(len(det.bbox), 4)
        self.assertIsNotNone(result.pathology_mask)
        self.assertIsNotNone(result.measurements)

    def test_text_only_detect_only_skips_medsam_and_postproc(self):
        result = self.orch.run(
            instruction="locate calcifications in aorta",
            volume=self.volume,
            _llm_output=_llm(organ="aorta", pathology="calcification", task="detect"),
        )
        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.detections)
        self.assertIsNone(result.pathology_mask)
        self.assertIsNone(result.measurements)

    def test_text_only_segment_only(self):
        result = self.orch.run(
            instruction="segment the spleen",
            volume=self.volume,
            _llm_output=_llm(organ="spleen", pathology="lesion", task="segment"),
        )
        self.assertEqual(result.status, "segment_only")
        self.assertIsNone(result.organ_mask)
        self.assertIsNone(result.detections)
        self.assertIsNone(result.pathology_mask)

    def test_text_only_non_btcv_organ_rejected(self):
        result = self.orch.run(
            instruction="find tumor in brain",
            volume=self.volume,
            _llm_output=_llm(organ="brain", pathology="tumor"),
        )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.task_params.parse_confidence, "low")
        self.assertIsNone(result.detections)
        self.assertIsNone(result.measurements)

    def test_text_only_low_confidence_flagged(self):
        result = self.orch.run(
            instruction="maybe a lesion in the liver",
            volume=self.volume,
            _llm_output=_llm(parse_confidence="low"),
        )
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.detections)


# ============================================================================
# Mode 3: combined
# ============================================================================

class CombinedModeTests(unittest.TestCase):
    def test_combined_parses_for_context_but_uses_click(self):
        result = Orchestrator().run(
            instruction="find a lesion in the liver",
            click_bbox=[10, 10, 30, 30],
            ct_slice=_fake_slice(),
            _llm_output=_llm(organ="liver", pathology="lesion"),
        )
        self.assertEqual(result.input_mode, "combined")
        self.assertEqual(result.input_dimensions, "2d")
        self.assertEqual(result.status, "completed")
        # Parser ran for context:
        self.assertIsNotNone(result.task_params)
        self.assertEqual(result.task_params.organ, "liver")
        self.assertEqual(result.decision, "proceed")
        # MedSAM ran with the click bbox (label propagates from parsed pathology):
        self.assertIsNotNone(result.pathology_mask)
        self.assertEqual(result.pathology_mask.label, "lesion")
        self.assertEqual(result.pathology_mask.bbox, [10.0, 10.0, 30.0, 30.0])
        # Post-processing ran:
        self.assertIsNotNone(result.measurements)
        self.assertEqual(set(result.measurements), _MEASUREMENT_KEYS)
        # No organ-seg / detection happened:
        self.assertIsNone(result.organ_mask)
        self.assertIsNone(result.detections)

    def test_combined_skips_organ_seg_and_detector(self):
        class FailingOrganSeg(OrganSegmentor):
            def segment(self, *a, **k):
                raise AssertionError("organ_segmentor must not run in combined")

        class FailingDetector(MONAIDetector):
            def detect(self, *a, **k):
                raise AssertionError("detector must not run in combined")

        orch = Orchestrator(
            organ_segmentor=FailingOrganSeg(), detector=FailingDetector(),
        )
        result = orch.run(
            instruction="find a lesion in the liver",
            click_bbox=[5, 5, 25, 25],
            ct_slice=_fake_slice(),
            _llm_output=_llm(),
        )
        self.assertEqual(result.input_mode, "combined")
        self.assertEqual(result.status, "completed")

    def test_combined_proceeds_even_when_text_would_be_rejected(self):
        # The click is authoritative — even if the text alone would be
        # rejected (non-BTCV organ), combined mode still segments.
        result = Orchestrator().run(
            instruction="find tumor in brain",
            click_bbox=[10, 10, 30, 30],
            ct_slice=_fake_slice(),
            _llm_output=_llm(organ="brain", pathology="tumor"),
        )
        self.assertEqual(result.input_mode, "combined")
        self.assertEqual(result.status, "completed")
        # Decision is recorded but doesn't gate execution.
        self.assertEqual(result.decision, "reject")
        self.assertIsNotNone(result.pathology_mask)


# ============================================================================
# Determinism — important for stable CI
# ============================================================================

class DeterminismTests(unittest.TestCase):
    def test_text_only_3d_same_input_same_output(self):
        orch = Orchestrator()
        vol = _fake_volume()
        a = orch.run(instruction="find a lesion in the liver",
                     volume=vol, _llm_output=_llm())
        b = orch.run(instruction="find a lesion in the liver",
                     volume=vol, _llm_output=_llm())
        self.assertEqual(
            [d.model_dump() for d in a.detections],
            [d.model_dump() for d in b.detections],
        )
        np.testing.assert_array_equal(a.organ_mask.mask, b.organ_mask.mask)
        self.assertEqual(a.measurements["volume_cm3"],
                         b.measurements["volume_cm3"])

    def test_click_only_same_input_same_output(self):
        orch = Orchestrator()
        a = orch.run(click_bbox=[5, 5, 25, 25], ct_slice=_fake_slice())
        b = orch.run(click_bbox=[5, 5, 25, 25], ct_slice=_fake_slice())
        np.testing.assert_array_equal(a.pathology_mask.mask, b.pathology_mask.mask)
        # Scalar measurements compare cleanly; cleaned_mask is an ndarray and
        # needs np.testing for equality.
        for k in ("volume_cm3", "longest_diameter_mm", "mean_hu",
                  "std_hu", "sphericity", "surface_area_mm2"):
            self.assertEqual(a.measurements[k], b.measurements[k])
        np.testing.assert_array_equal(
            a.measurements["cleaned_mask"], b.measurements["cleaned_mask"],
        )


# ============================================================================
# Custom-component injection — orchestrator must accept overrides
# ============================================================================

class InjectionTests(unittest.TestCase):
    def test_detector_receives_organ_mask_in_text_only_3d(self):
        captured = {}

        class CapturingDetector(MONAIDetector):
            def detect(self, image, params, is_3d=False, organ_mask=None):
                captured["is_3d"] = is_3d
                captured["organ_mask_shape"] = (
                    organ_mask.shape if organ_mask is not None else None
                )
                return [Detection(
                    bbox=[0, 0, 0, 5, 5, 5], label=params.pathology, score=0.9,
                )]

        orch = Orchestrator(detector=CapturingDetector())
        orch.run(instruction="find a lesion in the liver",
                 volume=_fake_volume(), _llm_output=_llm())
        self.assertTrue(captured["is_3d"])
        self.assertEqual(captured["organ_mask_shape"], _fake_volume().shape)

    def test_custom_postprocessor_used(self):
        class CountingPostProc(MaskPostProcessor):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def process(self, mask, voxel_spacing=(1.0, 1.0, 1.0), ct_image=None):
                self.calls += 1
                return {
                    "cleaned_mask": mask,
                    "volume_cm3": 42.0, "longest_diameter_mm": 1.0,
                    "mean_hu": 50.0, "std_hu": 10.0,
                    "sphericity": 0.5, "surface_area_mm2": 100.0,
                }

        pp = CountingPostProc()
        orch = Orchestrator(postprocessor=pp)
        # click_only → post-processor must be called once
        orch.run(click_bbox=[5, 5, 25, 25], ct_slice=_fake_slice())
        self.assertEqual(pp.calls, 1)
        # text_only detect+segment → another call
        orch.run(instruction="find a lesion in the liver",
                 volume=_fake_volume(), _llm_output=_llm())
        self.assertEqual(pp.calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
