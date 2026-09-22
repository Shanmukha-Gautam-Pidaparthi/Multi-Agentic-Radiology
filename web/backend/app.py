"""FastAPI backend for the interactive box-prompt web UI.

Wraps `Orchestrator.run()` in an HTTP API and serves the static frontend.

Endpoints
  GET  /                      -> the single-page frontend
  GET  /api/health            -> component/stub status
  GET  /api/organs            -> the BTCV-13 vocabulary
  POST /api/slices            -> upload a CT slice (PNG/JPG) or NIfTI volume
  GET  /api/slices            -> list loaded cases
  GET  /api/slices/{id}/{z}   -> rendered PNG of one axial slice
  POST /api/analyze           -> box prompt (+ optional instruction) -> result
  GET  /api/reports/{id}      -> stored report as JSON / Markdown / HTML

Slices live in an in-process cache keyed by case id. The cache is bounded
and evicts least-recently-used entries, so a long-running server does not
grow without limit. Nothing is persisted: restart the server and the
uploads are gone. That is deliberate for a research prototype handling
patient imaging.
"""
import base64
import io
import logging
import os
import sys
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, field_validator

# The pipeline modules live at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orchestrator import Orchestrator, OrchestratorResult  # noqa: E402
from parser import BTCV_ORGANS, OLLAMA_AVAILABLE  # noqa: E402
from report_generator import generate_report  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

MAX_CASES = int(os.environ.get("MEDAI_MAX_CASES", "32"))
MAX_REPORTS = int(os.environ.get("MEDAI_MAX_REPORTS", "128"))
MAX_UPLOAD_BYTES = int(os.environ.get("MEDAI_MAX_UPLOAD_MB", "64")) * 1024 * 1024
DEMO_HW = 512

# Every stage below is still a stub in this repo. The report generator uses
# this list to stamp its output as synthetic. Remove a name here once the
# corresponding real weights are wired in.
STUB_STAGES = ["organ_segmentor", "detector", "segmentor", "postprocessor"]

app = FastAPI(title="Multi-Agentic Radiology - Interactive Box Prompt", version="1.0")

_orchestrator = Orchestrator()


class _LRUCache:
    """Minimal thread-safe LRU cache. Bounded so the server cannot OOM."""

    def __init__(self, maxsize: int):
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                evicted, _ = self._data.popitem(last=False)
                logger.info("Evicted %s from cache (limit %d)", evicted, self._maxsize)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())


_cases = _LRUCache(MAX_CASES)
_reports = _LRUCache(MAX_REPORTS)


# ======================================================================
# Image helpers
# ======================================================================
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Window a float array to displayable uint8 without clipping detail."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (((arr - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_to_uint8(gray), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _mask_overlay_png(mask: np.ndarray) -> str:
    """Encode a boolean mask as a translucent red RGBA PNG (base64 data URI)."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = (229, 57, 53, 150)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _demo_volume(depth: int = 64) -> np.ndarray:
    """A synthetic abdomen-ish phantom so the UI is usable with no data.

    Not anatomy - a body ellipse with a few organ-like blobs and one bright
    focal lesion, purely so there is something to draw a box on.
    """
    rng = np.random.default_rng(20260505)
    ys, xs = np.mgrid[0:DEMO_HW, 0:DEMO_HW]
    vol = np.zeros((DEMO_HW, DEMO_HW, depth), dtype=np.float32)
    for z in range(depth):
        taper = 1.0 - 0.25 * abs(z - depth / 2) / (depth / 2)
        body = (((xs - 256) / (210 * taper)) ** 2 + ((ys - 262) / (150 * taper)) ** 2) <= 1.0
        sl = np.where(body, 0.45, 0.02).astype(np.float32)
        for cx, cy, rx, ry, val in (
            (180, 215, 95, 62, 0.72),   # liver-ish
            (335, 225, 52, 40, 0.66),   # spleen-ish
            (150, 300, 34, 30, 0.58),   # kidney-ish
            (355, 300, 34, 30, 0.58),   # kidney-ish
            (256, 240, 62, 22, 0.61),   # pancreas-ish
        ):
            blob = (((xs - cx) / (rx * taper)) ** 2 + ((ys - cy) / (ry * taper)) ** 2) <= 1.0
            sl[blob] = val
        if depth // 3 <= z <= 2 * depth // 3:
            lesion = ((xs - 200) ** 2 + (ys - 230) ** 2) <= 24 ** 2
            sl[lesion] = 0.9
        sl += rng.normal(0.0, 0.012, sl.shape).astype(np.float32)
        vol[:, :, z] = np.clip(sl, 0.0, 1.0)
    return vol


def _register_case(name: str, volume: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, Any]:
    case_id = uuid.uuid4().hex[:12]
    case = {
        "case_id": case_id,
        "name": name,
        "volume": volume,
        "depth": int(volume.shape[2]),
        "height": int(volume.shape[0]),
        "width": int(volume.shape[1]),
        "voxel_spacing": spacing,
    }
    _cases.put(case_id, case)
    return case


def _case_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "name": case["name"],
        "depth": case["depth"],
        "height": case["height"],
        "width": case["width"],
        "voxel_spacing": list(case["voxel_spacing"]),
    }


def _require_case(case_id: str) -> Dict[str, Any]:
    case = _cases.get(case_id)
    if case is None:
        raise HTTPException(404, f"Unknown case_id {case_id!r}. It may have been evicted; re-upload.")
    return case


# Seed one demo case so the UI works on a fresh clone with no BTCV data.
_DEMO_CASE = _register_case("demo-phantom (synthetic)", _demo_volume(), (1.0, 1.0, 3.0))


# ======================================================================
# Models
# ======================================================================
class AnalyzeRequest(BaseModel):
    case_id: str
    slice_index: int = Field(0, ge=0)
    bbox: Optional[List[float]] = Field(None, description="[x1, y1, x2, y2] in image pixels")
    instruction: Optional[str] = None
    llm_output: Optional[str] = Field(
        None,
        description="Test seam: inject parser JSON instead of calling Ollama.",
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(n) for n in v)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have x2 > x1 and y2 > y1")
        return [x1, y1, x2, y2]


# ======================================================================
# Routes
# ======================================================================
@app.get("/api/health")
def health() -> Dict[str, Any]:
    try:
        import nibabel  # noqa: F401
        nibabel_ok = True
    except ImportError:
        nibabel_ok = False
    return {
        "status": "ok",
        "ollama_available": OLLAMA_AVAILABLE,
        "nibabel_available": nibabel_ok,
        "stub_stages": STUB_STAGES,
        "cases_loaded": len(_cases.keys()),
        "demo_case_id": _DEMO_CASE["case_id"],
        "warning": (
            "All AI stages are stubs. Measurements are synthetic placeholders, "
            "NOT clinical measurements."
        ),
    }


@app.get("/api/organs")
def organs() -> Dict[str, Any]:
    return {"btcv_organs": sorted(BTCV_ORGANS)}


@app.get("/api/slices")
def list_cases() -> Dict[str, Any]:
    cases = [_case_summary(c) for c in (_cases.get(k) for k in _cases.keys()) if c]
    return {"cases": cases}


@app.post("/api/slices")
async def upload_case(
    file: UploadFile = File(...),
    spacing_x: float = Form(1.0),
    spacing_y: float = Form(1.0),
    spacing_z: float = Form(1.0),
) -> Dict[str, Any]:
    """Accept a 2D slice (PNG/JPG/BMP/TIF) or a 3D NIfTI volume."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    name = file.filename or "upload"
    lower = name.lower()
    spacing = (float(spacing_x), float(spacing_y), float(spacing_z))

    if lower.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError:
            raise HTTPException(
                501,
                "NIfTI upload requires nibabel. Install it (pip install nibabel) "
                "or upload a 2D PNG slice instead.",
            )
        import tempfile

        suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            img = nib.load(tmp_path)
            vol = np.asarray(img.dataobj, dtype=np.float32)
            zooms = img.header.get_zooms()[:3]
            if len(zooms) == 3 and all(float(z) > 0 for z in zooms):
                spacing = tuple(float(z) for z in zooms)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Could not read NIfTI file: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if vol.ndim == 4:                      # drop a trailing singleton/time axis
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise HTTPException(400, f"Expected a 3D NIfTI volume, got shape {vol.shape}.")
        # HU soft-tissue window, matching preprocessor.py's defaults.
        vol = np.clip(vol, -150.0, 250.0)
        vol = (vol + 150.0) / 400.0
    else:
        try:
            img2d = Image.open(io.BytesIO(raw)).convert("L")
        except Exception as e:
            raise HTTPException(400, f"Could not decode image: {e}")
        arr = np.asarray(img2d, dtype=np.float32) / 255.0
        vol = arr[:, :, None]

    case = _register_case(name, np.ascontiguousarray(vol), spacing)
    logger.info("Registered case %s (%s) shape=%s", case["case_id"], name, vol.shape)
    return _case_summary(case)


@app.get("/api/slices/{case_id}/{slice_index}")
def get_slice(case_id: str, slice_index: int) -> Response:
    case = _require_case(case_id)
    if not 0 <= slice_index < case["depth"]:
        raise HTTPException(404, f"slice_index out of range 0..{case['depth'] - 1}")
    png = _png_bytes(case["volume"][:, :, slice_index])
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


def _clamp_bbox(bbox: List[float], h: int, w: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(x1, w - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "Box prompt is degenerate after clamping to the image.")
    return [x1, y1, x2, y2]


def _serialize(result: OrchestratorResult) -> Dict[str, Any]:
    """OrchestratorResult -> JSON. numpy masks become PNG data URIs."""
    payload: Dict[str, Any] = {
        "instruction": result.instruction,
        "click_bbox": result.click_bbox,
        "input_mode": result.input_mode,
        "input_dimensions": result.input_dimensions,
        "status": result.status,
        "decision": result.decision,
        "message": result.message,
        "z_mid": result.z_mid,
        "task_params": result.task_params.model_dump() if result.task_params else None,
        "detections": [
            {"label": d.label, "score": float(d.score), "bbox": [float(v) for v in d.bbox]}
            for d in (result.detections or [])
        ],
        "pathology_mask": None,
        "measurements": None,
    }
    if result.pathology_mask is not None:
        seg = result.pathology_mask
        payload["pathology_mask"] = {
            "label": seg.label,
            "score": float(seg.score),
            "bbox": [float(v) for v in seg.bbox],
            "pixel_count": int(seg.mask.sum()),
            "overlay_png": _mask_overlay_png(seg.mask),
        }
    if result.measurements:
        # cleaned_mask is a numpy array - not JSON, and already conveyed by
        # the overlay above.
        payload["measurements"] = {
            k: v for k, v in result.measurements.items() if k != "cleaned_mask"
        }
    return payload


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    """Run the pipeline for a box prompt, a text instruction, or both."""
    case = _require_case(req.case_id)
    if not 0 <= req.slice_index < case["depth"]:
        raise HTTPException(400, f"slice_index out of range 0..{case['depth'] - 1}")

    instruction = (req.instruction or "").strip() or None
    if req.bbox is None and instruction is None:
        raise HTTPException(400, "Provide a box prompt, an instruction, or both.")

    volume = case["volume"]
    ct_slice = np.ascontiguousarray(volume[:, :, req.slice_index])
    spacing = case["voxel_spacing"]

    kwargs: Dict[str, Any] = {
        "instruction": instruction,
        "voxel_spacing": spacing,
        "_llm_output": req.llm_output,
    }
    if req.bbox is not None:
        # click_only / combined - the box always refers to this 2D slice.
        kwargs["click_bbox"] = _clamp_bbox(req.bbox, case["height"], case["width"])
        kwargs["ct_slice"] = ct_slice
    elif case["depth"] > 3:
        kwargs["volume"] = volume          # text_only, full 3D two-stage path
    else:
        kwargs["ct_slice"] = ct_slice      # text_only, 2D fallback

    try:
        result = _orchestrator.run(**kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except NotImplementedError as e:
        raise HTTPException(501, f"Stage not implemented: {e}")

    report = generate_report(
        result,
        case_id=f"{case['name']} (slice {req.slice_index})",
        stub_stages=STUB_STAGES,
    )
    report_id = uuid.uuid4().hex[:12]
    _reports.put(report_id, report)

    return {
        "report_id": report_id,
        "case": _case_summary(case),
        "slice_index": req.slice_index,
        "result": _serialize(result),
        "report": report,
    }


def _require_report(report_id: str) -> Dict[str, Any]:
    report = _reports.get(report_id)
    if report is None:
        raise HTTPException(404, f"Unknown report_id {report_id!r}. It may have been evicted.")
    return report


@app.get("/api/reports/{report_id}")
def get_report(report_id: str, fmt: str = "json") -> Response:
    report = _require_report(report_id)
    fmt = fmt.lower()
    if fmt == "json":
        return JSONResponse(report)
    if fmt in ("md", "markdown"):
        return PlainTextResponse(
            report["markdown"],
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="report_{report_id}.md"'},
        )
    if fmt == "html":
        return HTMLResponse(_report_html(report))
    raise HTTPException(400, "fmt must be one of: json, md, html")


def _report_html(report: Dict[str, Any]) -> str:
    """Printable standalone report page. Markdown is escaped, never executed."""
    import html as _html

    body = _html.escape(report["markdown"])
    title = _html.escape(str(report["metadata"].get("case_id") or "Report"))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Report - {title}</title>"
        "<style>body{font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "max-width:60rem;margin:2rem auto;padding:0 1rem;background:#fff;color:#111}"
        "pre{white-space:pre-wrap;word-wrap:break-word}"
        "@media print{body{margin:0}}</style></head>"
        f"<body><pre>{body}</pre></body></html>"
    )


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:  # pragma: no cover
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)
