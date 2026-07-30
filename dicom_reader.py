"""DICOM metadata reader — Stage 1 companion to parser.py.

Extracts the labels already stored in the DICOM header (modality, body part,
dimensions, etc.) so the orchestrator can cross-check the doctor's typed
instruction against the scan that was actually uploaded.

This file does NOT run inference — no organ segmentation, no tumor coordinates.
That is Stage 5 (AI models) and lives elsewhere in the pipeline.

INPUT FORMAT NOTE:
This module is for *real hospital input* — single-file or multi-file `.dcm`
DICOM studies as they come off PACS. NIfTI input (`.nii.gz`) from research
datasets like BTCV is handled by [preprocessor.py](preprocessor.py) instead;
NIfTI files don't carry the same header tags (BodyPartExamined, Modality, etc.)
so the cross-check below doesn't apply to them.
"""

import pydicom
from pydicom.errors import InvalidDicomError
from pydantic import BaseModel
from typing import Optional, List

from parser import TaskParams, parse_instruction


class DicomMetadata(BaseModel):
    modality: Optional[str] = None            # "MR", "CT", "US", "XR", ...
    body_part: Optional[str] = None           # "BRAIN", "CHEST", "ABDOMEN", ...
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    number_of_frames: Optional[int] = None
    pixel_spacing: Optional[List[float]] = None   # [row_mm, col_mm]
    slice_thickness: Optional[float] = None       # mm — DICOM (0018,0050)
    patient_orientation: Optional[str] = None


class DicomReadError(Exception):
    """Raised when a file cannot be read as a valid DICOM."""
    pass


def _get(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None or val == "":
        return default
    return val


def read_dicom_metadata(path: str) -> DicomMetadata:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except InvalidDicomError as e:
        raise DicomReadError(f"Not a valid DICOM file: {path}") from e
    except FileNotFoundError as e:
        raise DicomReadError(f"DICOM file not found: {path}") from e

    pixel_spacing = _get(ds, "PixelSpacing")
    if pixel_spacing is not None:
        pixel_spacing = [float(x) for x in pixel_spacing]

    slice_thickness = _get(ds, "SliceThickness")
    if slice_thickness is not None:
        slice_thickness = float(slice_thickness)

    orientation = _get(ds, "PatientOrientation")
    if orientation is not None:
        orientation = "\\".join(str(x) for x in orientation)

    return DicomMetadata(
        modality=_get(ds, "Modality"),
        body_part=_get(ds, "BodyPartExamined"),
        study_description=_get(ds, "StudyDescription"),
        series_description=_get(ds, "SeriesDescription"),
        rows=_get(ds, "Rows"),
        columns=_get(ds, "Columns"),
        number_of_frames=_get(ds, "NumberOfFrames"),
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        patient_orientation=orientation,
    )


# Map the parser's BTCV-13 organ vocabulary to DICOM BodyPartExamined values.
# DICOM uses uppercase free-text, so we check against a set of acceptable tags.
# Every BTCV organ is abdominal so "ABDOMEN" is in every set as the primary tag;
# organ-specific tags are alternatives that some PACS systems use.
_ORGAN_TO_BODY_PART = {
    "spleen":              {"ABDOMEN", "SPLEEN"},
    "right kidney":        {"ABDOMEN", "KIDNEY", "RENAL"},
    "left kidney":         {"ABDOMEN", "KIDNEY", "RENAL"},
    "gallbladder":         {"ABDOMEN", "GALLBLADDER"},
    "esophagus":           {"ABDOMEN", "ESOPHAGUS", "CHEST"},  # esophagus straddles abdomen/chest
    "liver":               {"ABDOMEN", "LIVER"},
    "stomach":             {"ABDOMEN", "STOMACH"},
    "aorta":               {"ABDOMEN", "AORTA"},
    "inferior vena cava":  {"ABDOMEN"},
    "portal vein":         {"ABDOMEN", "LIVER"},
    "pancreas":            {"ABDOMEN", "PANCREAS"},
    "right adrenal gland": {"ABDOMEN", "ADRENAL"},
    "left adrenal gland":  {"ABDOMEN", "ADRENAL"},
}


def check_instruction_matches_scan(
    params: TaskParams, dicom: DicomMetadata
) -> List[str]:
    """Cross-check the parsed instruction against the DICOM header.

    Returns a list of human-readable warnings. Empty list means no mismatch found.
    The orchestrator should surface these to the radiologist before dispatching
    the scan to the AI models — wrong-side or wrong-organ errors are exactly
    the kind of mistake this cross-check exists to catch.
    """
    warnings: List[str] = []

    if dicom.body_part:
        expected = _ORGAN_TO_BODY_PART.get(params.organ.lower())
        if expected and dicom.body_part.upper() not in expected:
            warnings.append(
                f"Instruction organ='{params.organ}' does not match "
                f"DICOM BodyPartExamined='{dicom.body_part}'"
            )

    # Modality sanity check — the entire BTCV pipeline expects abdominal CT.
    # Anything else (MR, US, XR, ...) means the downstream models are being
    # fed data they weren't trained on; flag it as a soft warning.
    if dicom.modality and dicom.modality.upper() != "CT":
        warnings.append(
            f"Modality='{dicom.modality}' is not CT — the BTCV pipeline expects "
            "abdominal CT input. Confirm before dispatching."
        )

    # Resolution check — coarse pixel spacing means downstream segmentation
    # accuracy will suffer. 5mm is generous (typical abdominal CT is well
    # under 1mm in-plane); anything beyond is worth flagging.
    if dicom.pixel_spacing and any(s > 5.0 for s in dicom.pixel_spacing):
        warnings.append(
            f"Pixel spacing {dicom.pixel_spacing} mm has a dimension above 5 mm — "
            "coarse resolution may reduce segmentation accuracy."
        )

    # Slice-thickness check — thick slices cause partial-volume effects where
    # small organs (adrenals, vessels) blend with surrounding tissue, hurting
    # segmentation. Routine abdominal CT is typically ≤ 3mm; > 5mm is a
    # quality warning.
    if dicom.slice_thickness is not None and dicom.slice_thickness > 5.0:
        warnings.append(
            f"Slice thickness {dicom.slice_thickness}mm exceeds 5mm — "
            "partial volume effects likely, segmentation accuracy may be reduced."
        )

    return warnings


def run(image_path: str, instruction: str) -> None:
    """Full Stage 1 flow: read DICOM, parse instruction, cross-check."""
    print(f"Reading DICOM: {image_path}")
    meta = read_dicom_metadata(image_path)
    print("\nDICOM metadata:")
    for k, v in meta.model_dump().items():
        print(f"  {k}: {v}")

    print(f"\nInstruction: {instruction}")
    params = parse_instruction(instruction)
    print(f"Parsed:      {params.model_dump()}")

    warnings = check_instruction_matches_scan(params, meta)
    print("\nCross-check:")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
        print("\n  --> Route to human confirmation before dispatch.")
    else:
        print("  OK — instruction matches scan. Safe to dispatch to model selector.")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Stage 1: read a DICOM and parse a radiologist's instruction."
    )
    arg_parser.add_argument(
        "--image",
        required=True,
        help="Path to a DICOM file (.dcm)",
    )
    arg_parser.add_argument(
        "--instruction",
        required=True,
        help='Radiologist instruction, e.g. "identify tumor in brain"',
    )
    args = arg_parser.parse_args()
    run(args.image, args.instruction)
