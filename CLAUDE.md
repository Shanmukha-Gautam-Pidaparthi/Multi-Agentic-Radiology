# Project Context: Medical AI Platform — Abdominal CT / BTCV

## What this project is

An end-to-end AI platform for medical image interpretation, supervised by Prisha Meenakshi.

**Current focus (2026-05-05):** Abdominal CT pathology detection, fine-tuned on the BTCV
(Beyond the Cranial Vault) dataset — 13 abdominal organs. Brain MRI is dropped from the
near-term roadmap; we may revisit later, but everything in this repo now assumes CT.

This repo covers stages 1–6 of the pipeline:
1. LLM instruction parser ([parser.py](parser.py))
2. CT preprocessing ([preprocessor.py](preprocessor.py))
3. Stage-1 organ segmentation ([organ_segmentor.py](organ_segmentor.py))
4. Stage-2 pathology detection ([detector.py](detector.py))
5. Pathology mask refinement ([segmentor.py](segmentor.py))
6. Mask post-processing & measurements ([postprocessor.py](postprocessor.py))

A pipeline orchestrator ([orchestrator.py](orchestrator.py)) wires them together
and dispatches across three input modes (click_only / text_only / combined).
A separate DICOM metadata reader ([dicom_reader.py](dicom_reader.py)) cross-checks
the LLM output against the scan header.

---

## Overall pipeline (current)

```
                                       ┌─ click_bbox (no text) ──────┐
Free-text instruction ──► [LLM Parser] │                             │
                               │       │                             ▼
                               ▼       │                       [MedSAMSegmentor]
                       [Confidence     │                             │
                          Router]      │                             ▼
                               │       │                       [PostProcessor]
                       reject / flag   │                             │
                               │       │                             │
                          proceed      │                             │
                               │       │                             │
NIfTI volume ─► [CT Preprocessor]      │                             │
                               │       │ instruction + click_bbox    │
                               ▼       └─────────►   parse for       │
                    [Stage-1 OrganSegmentor]         context only,   │
                               │ organ_mask          then MedSAM ────┤
                               ▼                                     │
                  [Stage-2 MONAIDetector] (mask-constrained)         │
                               │ 3D bbox [x1,y1,z1,x2,y2,z2]         │
                               ▼                                     │
                     z_mid axial slice + 2D bbox                     │
                               │                                     │
                               ▼                                     │
                     [MedSAMSegmentor]  (box-prompted)               │
                               │ 2D pathology mask                   │
                               ▼                                     │
                     [PostProcessor]                                 │
                               │ measurements (volume, diameter, …)  │
                               └─────────► Report generator ◄────────┘
                                               (downstream)
```

### Three input modes (orchestrator dispatch)

| Mode         | Inputs                            | Pipeline                                                           |
|--------------|-----------------------------------|---------------------------------------------------------------------|
| click_only   | click_bbox + ct_slice             | MedSAM → PostProcessor                                              |
| text_only    | instruction + (volume or ct_slice)| Parser → Router → OrganSeg → Detector → MedSAM → PostProcessor      |
| combined     | instruction + click_bbox + ct_slice | Parser (context only) → MedSAM(click_bbox) → PostProcessor        |

In `combined` mode the click is authoritative — the parsed `decision` is recorded
on the result for traceability but does NOT gate execution, so a click is always
honored even if the text alone would have been rejected (e.g. non-BTCV organ).

### Key design decisions confirmed with Prisha
- **Modality:** CT (confirmed). Brain MRI dropped for now.
- **Dataset:** BTCV (confirmed). The 13-organ closed vocabulary is enforced everywhere.
- **Two-stage pipeline:** organ-first, pathology-second. The detector only searches
  inside the organ mask — drastically cuts false positives in neighboring anatomy.
- **3D detection, 2D segmentation:** the detector returns a 3D bbox; we drop to the
  midpoint axial slice and run MedSAM on 2D. Real 3D pathology segmentation can replace
  this later if the intern's MedSAM-3D experiments pan out.
- **The LLM only parses** — it does not choose models. Model selection is rule-based.
- **Closed organ vocabulary:** if the radiologist asks about a non-BTCV organ
  (e.g. brain, lung, heart), the parser flags `parse_confidence='low'` and the router
  rejects. We do not silently degrade.
- **Confidence routing:** `proceed | flag_for_review | reject`. Low confidence pauses
  for human confirmation; reject blocks dispatch entirely.

---

## The 13 BTCV organs

The `organ` field in `TaskParams` is constrained to this closed set:

```
spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach,
aorta, inferior vena cava, portal vein, pancreas, right adrenal gland,
left adrenal gland
```

A `model_validator` on `TaskParams` downgrades `parse_confidence` to `"low"` for any
organ outside this set. The `confidence_router` then hard-rejects so the AI models
are never invoked on out-of-scope anatomy.

---

## TaskParams schema

```python
class TaskParams(BaseModel):
    organ: str                # one of the 13 BTCV organs (validated)
    pathology: str            # "lesion", "mass", "tumor", "cyst", "calcification", ...
    task: str                 # "detect" | "segment" | "detect+segment"
    region: Optional[str]     # sub-region if mentioned ("head" of pancreas, etc.)
    urgency: Optional[str]    # "urgent" | "routine"
    parse_confidence: Optional[str]   # "high" | "medium" | "low"
```

### Abbreviations expander (parser.ABBREV)
Whole-word substitutions before the prompt is sent to the LLM:
`rk → right kidney`, `lk → left kidney`, `ivc → inferior vena cava`,
`gb → gallbladder`, `panc → pancreas`, `rag → right adrenal gland`,
`lag → left adrenal gland`, `mets → metastasis`, `ca → carcinoma`,
`hcc → hepatocellular carcinoma`.

---

## Module reference

### [parser.py](parser.py)
- LLM call via Ollama (`llama3.1:8b`). System prompt locks output to BTCV vocabulary.
- `parse_instruction(text) -> TaskParams` — never raises; on parse failure returns
  a fallback `TaskParams(organ='unknown', pathology='unknown', parse_confidence='low')`.
- `confidence_router(params) -> "proceed" | "flag_for_review" | "reject"`.
- `BTCV_ORGANS` is the canonical 13-organ frozenset, also imported by `organ_segmentor`.

### [preprocessor.py](preprocessor.py)
- `CTPreprocessor.preprocess(nifti_path) -> (volume_3d, axial_slices)`.
- Loads NIfTI via `nibabel`, soft-tissue HU window `[-150, 250]`, normalize to `[0, 1]`,
  resize in-plane to `512×512` preserving depth.
- **Stub mode:** if `nibabel` is not installed, returns a synthetic `(512, 512, 64)`
  volume so the rest of the pipeline is testable. A loud warning fires on every
  fallback. Real production must have `nibabel` installed.
- Resampling currently uses pure-numpy nearest-neighbor — **TODO:** swap to
  `scipy.ndimage.zoom` or `monai.transforms.Resized` for proper interpolation.

### [organ_segmentor.py](organ_segmentor.py)
- `OrganSegmentor.segment(volume_3d, organ_name) -> OrganMask`.
- `OrganMask` carries `organ`, `bbox` (3D), `voxel_count`, `mask` (3D bool ndarray).
- **Stub:** generates an ellipsoidal blob at a hard-coded normalized anchor per organ
  (rough anatomical positions, not clinically accurate). Used only for plumbing and
  tests. Real wiring: replace `_stub_organ_mask` with the intern's BTCV-fine-tuned
  MedSAM weights. See module docstring for swap notes.

### [detector.py](detector.py)
- `MONAIDetector.detect(image, params, is_3d=False, organ_mask=None) -> List[Detection]`.
- `is_3d=True` returns 6-element bboxes `[x1,y1,z1,x2,y2,z2]`; otherwise 4-element.
- `organ_mask` constrains the search region. The stub uses the mask's bbox; the real
  detector will crop the volume by the mask before inference.
- Pathology label vocabulary (`ABDOMINAL_PATHOLOGY_LABELS`):
  `lesion, cyst, mass, tumor, calcification`. The stub takes the label from
  `params.pathology` directly; falls back to picking from this list if pathology is
  empty/unknown.
- **Real-MONAI swap notes** are listed at the top of the module — read them before
  wiring weights.

### [segmentor.py](segmentor.py)
- `MedSAMSegmentor.segment(image_2d, bbox_2d, label) -> SegmentationResult`.
- Box-prompted 2D segmentation. Stub draws an ellipse inscribed in the bbox.
- Same callsite for both the box-prompted-from-detection path (text_only) and
  the click-prompted path (click_only / combined).

### [postprocessor.py](postprocessor.py)
- `MaskPostProcessor.process(mask, voxel_spacing, ct_image) -> dict`.
- Returns: `cleaned_mask, volume_cm3, longest_diameter_mm, mean_hu, std_hu,
  sphericity, surface_area_mm2`.
- **Stub:** volume is computed honestly from voxel count × spacing; everything
  else is deterministic-fake (seeded by mask shape + voxel count).
- **Owner: Gautam.** Real impl needs morphological cleanup, marching-cubes
  surface area, RECIST-style longest diameter, and intensity statistics from
  the original CT — see module docstring for swap notes.

### [orchestrator.py](orchestrator.py)
- `Orchestrator.run(*, instruction=None, click_bbox=None, ct_slice=None,
  volume=None, voxel_spacing=(1.0,1.0,1.0)) -> OrchestratorResult`.
- All inputs are keyword-only. At least one of `instruction` or `click_bbox`
  must be supplied; `ct_slice` is required whenever `click_bbox` is given.
- Routes to one of three input modes: `click_only`, `text_only`, `combined`.
- For `text_only`, dispatches by image dimensionality: `volume` ⇒ full 3D
  two-stage path; `ct_slice` ⇒ 2D fallback (no organ segmentation possible
  from a single slice).
- `OrchestratorResult` includes: `instruction, click_bbox, input_mode,
  input_dimensions, task_params, decision, status, organ_mask, detections,
  pathology_mask, z_mid, measurements, classification, report, message`.
- `classification` and `report` are permanent `None` placeholders for v2.0
  (downstream classifier and report generator).

---

## Stage status

| Stage | File | Status | Real-model gating |
|---|---|---|---|
| 1. LLM parser | parser.py | working | Ollama `llama3.1:8b` (real) |
| 2. CT preprocess | preprocessor.py | stub fallback | needs `nibabel` install |
| 3. Organ seg | organ_segmentor.py | **stub** | needs intern's BTCV MedSAM weights |
| 4. Pathology detect | detector.py | **stub** | needs MONAI RetinaNet weights |
| 5. Pathology seg | segmentor.py | **stub** | needs MedSAM box-prompt weights |
| 6. Post-process | postprocessor.py | **stub** | Gautam's real impl pending |
| 7. Classification | (placeholder field) | not started | v2.0 — `result.classification` reserved |
| 8. Report generator | (placeholder field) | downstream | `result.report` reserved |

Every stub flags itself with a logger.info on construction, plus comments at the
module top describing what the real swap-in needs.

---

## Open architecture questions (current)

- **3D pathology segmentation:** intern is exploring SAM-Med3D / box-prompted MedSAM
  on volumes. If those work, we could replace the "midpoint slice + 2D mask" hack
  in the orchestrator with native 3D masks.
- **Multi-organ instructions:** what should "find lesions in liver and spleen" do?
  Currently the parser is single-organ; we'd need to generalize the schema to a list.
- **Organ→detector routing:** when real weights land, do we use one detector with
  a multi-class head, or one detector per organ? The interface in `detector.py`
  supports either.
- **Combined-mode conflict UX:** when the LLM and the click disagree (parsed
  organ ≠ where the user clicked), should the report flag the discrepancy or
  just record both? Currently we record both and don't flag — pending Prisha.
- **3D click handler:** the intern's click-coordinate handler will turn a 2D
  click into a 3D voxel coordinate. Whether the 3D version replaces or
  augments the current 2D `click_bbox` interface is open.

## Resolved architecture decisions

- ~~Modality (MRI vs CT)~~ → CT confirmed (2026-05-05).
- ~~Dataset~~ → BTCV confirmed (2026-05-05).
- ~~Brain MRI focus~~ → dropped for now (2026-05-05).
- ~~Click-only mode bypass~~ → click_only strictly bypasses parser/organ-seg/detector
  (2026-05-07). Confirmation/audit happens at the report layer, not by re-running
  detection in the background.

---

## Tests

- [test_parser.py](test_parser.py) — parser, abbreviations, BTCV validator,
  confidence router, end-to-end rejection of non-BTCV organs. Run live with
  `python3 test_parser.py --live`.
- [test_orchestrator.py](test_orchestrator.py) — full two-stage pipeline,
  detect-only, segment-only, rejection, 2D fallback, bbox dimensionality,
  determinism, custom-component injection.

Both suites use the parser's `_llm_output` test seam to inject synthetic LLM
output, so they run deterministically without needing Ollama running.

---

## Parallel work: Click coordinate handler (intern)

The intern is also working on:
- Click-coordinate handler: 2D click → 3D voxel coordinate, feeds into orchestrator's
  segment-only path.
- BTCV-fine-tuned MedSAM weights for organ segmentation (this is what
  `OrganSegmentor` will load when its stub is removed).

# Please check your work and perform code review to ensure code is optimized and efficient
