"""Structured radiology report generation - stage 8 of the pipeline.

Turns an `OrchestratorResult` into a structured report dict plus a rendered
Markdown body, so `result.report` stops being a permanent `None` placeholder.

This stage is deliberately *presentation only*: it reads what the upstream
stages produced and never re-runs a model or invents a measurement. When an
upstream stage was a stub, the report says so in `provenance` rather than
passing stub numbers off as clinical findings.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Measurement key -> (human label, unit, decimal places).
_MEASUREMENT_FIELDS = (
    ("volume_cm3", "Volume", "cm3", 3),
    ("longest_diameter_mm", "Longest diameter", "mm", 1),
    ("mean_hu", "Mean attenuation", "HU", 1),
    ("std_hu", "Attenuation SD", "HU", 1),
    ("sphericity", "Sphericity", "", 3),
    ("surface_area_mm2", "Surface area", "mm2", 1),
)

_MODE_NARRATIVE = {
    "click_only": (
        "Operator-directed box prompt. The drawn box was passed straight to "
        "the segmentation model; no text instruction, organ segmentation or "
        "automated detection contributed to this result."
    ),
    "combined": (
        "Operator-directed box prompt with an accompanying text instruction. "
        "The box is authoritative for segmentation; the parsed instruction is "
        "recorded for context only."
    ),
    "text_only": (
        "Instruction-driven pipeline. The organ was segmented first and the "
        "detector searched only within that organ mask."
    ),
}

_STATUS_IMPRESSION = {
    "rejected": (
        "No analysis performed. The instruction did not resolve to a BTCV-13 "
        "organ with an identifiable pathology target. Clarification required."
    ),
    "needs_review": (
        "Analysis paused pending human confirmation. The instruction parsed "
        "with low confidence; confirm the extracted parameters before "
        "dispatching to the AI models."
    ),
    "segment_only": (
        "Segmentation-only task recorded. Awaiting an operator box prompt - "
        "no detection was run."
    ),
}


def _fmt(value: Any, unit: str, places: int) -> str:
    if value is None:
        return "not reported"
    if isinstance(value, (int, float)):
        text = f"{round(float(value), places):g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _collect_provenance(result: Any, stub_stages: Optional[List[str]]) -> Dict[str, Any]:
    """Record which stages actually contributed, and which were stubs."""
    stages: Dict[str, str] = {}
    stages["parser"] = "ran" if result.task_params is not None else "skipped"
    stages["organ_segmentation"] = "ran" if result.organ_mask is not None else "skipped"
    stages["detection"] = "ran" if result.detections is not None else "skipped"
    stages["segmentation"] = "ran" if result.pathology_mask is not None else "skipped"
    stages["post_processing"] = "ran" if result.measurements else "skipped"
    return {
        "stages": stages,
        "stub_stages": sorted(stub_stages or []),
        "measurements_are_synthetic": bool(stub_stages),
    }


def _findings(result: Any) -> List[Dict[str, Any]]:
    """One finding per segmented target. Currently at most one."""
    seg = result.pathology_mask
    if seg is None:
        return []
    measurements = result.measurements or {}
    finding: Dict[str, Any] = {
        "label": seg.label,
        "segmentation_score": round(float(seg.score), 3),
        "prompt_bbox": [round(float(v), 1) for v in seg.bbox],
        "slice_index": result.z_mid,
        "measurements": {
            key: measurements.get(key) for key, _, _, _ in _MEASUREMENT_FIELDS
        },
    }
    if result.task_params is not None:
        finding["organ"] = result.task_params.organ
        finding["region"] = result.task_params.region
    return [finding]


def _impression(result: Any, findings: List[Dict[str, Any]]) -> str:
    if result.status in _STATUS_IMPRESSION:
        return _STATUS_IMPRESSION[result.status]
    if not findings:
        n = len(result.detections or [])
        if n:
            return (
                f"{n} candidate region(s) detected; no segmentation mask was "
                "produced for this task type."
            )
        return "No candidate region was identified."
    f = findings[0]
    organ = f.get("organ")
    site = f" in the {organ}" if organ and organ != "unknown" else ""
    diameter = f["measurements"].get("longest_diameter_mm")
    volume = f["measurements"].get("volume_cm3")
    size_bits = []
    if diameter is not None:
        size_bits.append(f"longest diameter {_fmt(diameter, 'mm', 1)}")
    if volume is not None:
        size_bits.append(f"volume {_fmt(volume, 'cm3', 3)}")
    size = f" ({', '.join(size_bits)})" if size_bits else ""
    return (
        f"Single segmented {f['label']}{site}{size}. "
        f"Segmentation confidence {f['segmentation_score']:.3f}."
    )


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = ["# Abdominal CT - AI-Assisted Analysis Report", ""]
    meta = report["metadata"]
    lines += [
        f"**Generated:** {meta['generated_at']}",
        f"**Input mode:** `{meta['input_mode']}`  |  "
        f"**Dimensions:** `{meta['input_dimensions']}`  |  "
        f"**Status:** `{meta['status']}`",
        "",
        "## Technique",
        "",
        report["technique"],
        "",
    ]

    req = report["request"]
    lines += ["## Request", ""]
    lines.append(f"- **Instruction:** {req['instruction'] or '_(none - box prompt only)_'}")
    if req["click_bbox"]:
        x1, y1, x2, y2 = req["click_bbox"]
        lines.append(
            f"- **Box prompt:** x [{x1:g} -> {x2:g}], y [{y1:g} -> {y2:g}] "
            f"({x2 - x1:g} x {y2 - y1:g} px)"
        )
    if req["parsed"]:
        p = req["parsed"]
        lines.append(
            f"- **Parsed:** organ=`{p['organ']}`, pathology=`{p['pathology']}`, "
            f"task=`{p['task']}`, confidence=`{p['parse_confidence']}`, "
            f"router=`{req['decision']}`"
        )
    lines.append("")

    lines += ["## Findings", ""]
    if not report["findings"]:
        lines += ["_No segmented finding._", ""]
    for i, f in enumerate(report["findings"], 1):
        organ = f.get("organ")
        head = f"### Finding {i}: {f['label']}"
        if organ and organ != "unknown":
            head += f" - {organ}"
        lines += [head, ""]
        if f.get("slice_index") is not None:
            lines.append(f"- Axial slice index: {f['slice_index']}")
        lines.append(f"- Segmentation confidence: {f['segmentation_score']:.3f}")
        lines += ["", "| Measurement | Value |", "| --- | --- |"]
        for key, label, unit, places in _MEASUREMENT_FIELDS:
            lines.append(f"| {label} | {_fmt(f['measurements'].get(key), unit, places)} |")
        lines.append("")

    dets = report.get("detections") or []
    if dets:
        lines += ["## Detector candidates", "", "| # | Label | Score | Bounding box |",
                  "| --- | --- | --- | --- |"]
        for i, d in enumerate(dets, 1):
            bbox = ", ".join(f"{v:g}" for v in d["bbox"])
            lines.append(f"| {i} | {d['label']} | {d['score']:.3f} | [{bbox}] |")
        lines.append("")

    lines += ["## Impression", "", report["impression"], "", "## Provenance", ""]
    for stage, state in report["provenance"]["stages"].items():
        lines.append(f"- `{stage}`: {state}")
    stubs = report["provenance"]["stub_stages"]
    lines.append("")
    if stubs:
        lines += [
            "> **NOT FOR CLINICAL USE.** The following stages ran as stubs, so "
            "the measurements above are synthetic placeholders, not "
            f"measurements of real anatomy: {', '.join(f'`{s}`' for s in stubs)}.",
            "",
        ]
    lines += [
        "> Research prototype output. AI-generated; requires review and "
        "sign-off by a qualified radiologist.",
        "",
    ]
    return "\n".join(lines)


def generate_report(
    result: Any,
    *,
    case_id: Optional[str] = None,
    stub_stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the structured report for an `OrchestratorResult`.

    Args:
      result: the `OrchestratorResult` to describe.
      case_id: optional identifier echoed into the report metadata.
      stub_stages: names of stages that ran as stubs. When non-empty the
        report is explicitly marked as carrying synthetic measurements.

    Returns a dict with `metadata`, `technique`, `request`, `findings`,
    `detections`, `impression`, `provenance` and a rendered `markdown` body.
    """
    findings = _findings(result)
    parsed = result.task_params.model_dump() if result.task_params else None

    report: Dict[str, Any] = {
        "metadata": {
            "case_id": case_id,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "input_mode": result.input_mode,
            "input_dimensions": result.input_dimensions,
            "status": result.status,
        },
        "technique": _MODE_NARRATIVE.get(result.input_mode, "Unspecified pipeline."),
        "request": {
            "instruction": result.instruction,
            "click_bbox": result.click_bbox,
            "parsed": parsed,
            "decision": result.decision,
        },
        "findings": findings,
        "detections": [
            {
                "label": d.label,
                "score": round(float(d.score), 3),
                "bbox": [round(float(v), 1) for v in d.bbox],
            }
            for d in (result.detections or [])
        ],
        "pipeline_message": result.message,
        "provenance": _collect_provenance(result, stub_stages),
    }
    report["impression"] = _impression(result, findings)
    report["markdown"] = _render_markdown(report)
    return report
