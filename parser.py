"""LLM Instruction Parser — Stage 1 of the abdominal CT / BTCV pipeline.

Converts a radiologist's free-text instruction into a TaskParams. The organ
field is constrained to the 13 BTCV organs; anything outside that set is
flagged with parse_confidence='low' so the router rejects it before
dispatching to the AI models.
"""
import json
import logging
from typing import Optional

from pydantic import BaseModel, ValidationError, model_validator

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    ollama = None
    OLLAMA_AVAILABLE = False
    # NOTE: stub-only mode. parse_instruction() still works when callers pass
    # `_llm_output` (tests, and the web backend's click-driven paths, never
    # touch Ollama). A live text parse without ollama installed falls back to
    # `_fallback_params()`, which the router then rejects.
    logger.warning(
        "ollama not installed - live text parsing is disabled and will fall "
        "back to low-confidence params. Install with: pip install ollama"
    )


# The 13 organs in the BTCV (Beyond the Cranial Vault) abdominal CT dataset.
# This is the closed vocabulary the parser is allowed to emit for `organ`.
BTCV_ORGANS = frozenset({
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "portal vein",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
})


UNKNOWN = "unknown"


class TaskParams(BaseModel):
    organ: str
    pathology: str
    task: str
    region: Optional[str] = None
    urgency: Optional[str] = "routine"
    parse_confidence: Optional[str] = "high"

    @model_validator(mode="after")
    def _enforce_btcv_organ(self) -> "TaskParams":
        # If the LLM picks an organ outside the BTCV closed set, downgrade
        # confidence so the router rejects it. We deliberately do NOT mutate
        # the organ string itself — preserving what the LLM said helps debug
        # prompt regressions.
        if (self.organ or "").strip().lower() not in BTCV_ORGANS:
            self.parse_confidence = "low"
        return self


# Abdominal CT shorthand. Whole-word matches only (the expander splits on
# whitespace), so "ca" → "carcinoma" doesn't accidentally rewrite "scan".
ABBREV = {
    "rk": "right kidney",
    "lk": "left kidney",
    "ivc": "inferior vena cava",
    "gb": "gallbladder",
    "panc": "pancreas",
    "rag": "right adrenal gland",
    "lag": "left adrenal gland",
    "mets": "metastasis",
    "met": "metastasis",
    "ca": "carcinoma",
    "hcc": "hepatocellular carcinoma",
}


SYSTEM_PROMPT = """You are a medical image analysis assistant for ABDOMINAL CT scans.
Your job is to parse radiologist instructions into structured JSON.

Always respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

Use this exact schema:
{
  "organ": "<one of the 13 BTCV organs, lowercase>",
  "pathology": "<lesion | mass | tumor | cyst | calcification | hemorrhage | thrombus | metastasis | ...>",
  "task": "<detect | segment | detect+segment>",
  "region": "<sub-region if mentioned, else null>",
  "urgency": "<urgent | routine>",
  "parse_confidence": "<high | medium | low>"
}

The "organ" field MUST be one of these 13 BTCV organs (lowercase, exact spelling):
  spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach,
  aorta, inferior vena cava, portal vein, pancreas, right adrenal gland,
  left adrenal gland

Rules:
- If the radiologist names an organ AND a pathology word (lesion, mass, tumor, cyst) → task is "detect+segment"
- If the instruction says "segment" or "delineate" without a mass/lesion → task is "segment" (click-driven)
- If the instruction says "detect" or "locate" without "segment" → task is "detect"
- "identify", "find", "check for" with a pathology word → task is "detect+segment"
- "kidney" without a side is ambiguous → set organ to "unknown" and parse_confidence to "low"
- If the requested organ is not in the BTCV list (e.g. brain, lung, heart) → set organ to "unknown" and parse_confidence to "low"
- urgency is "urgent" if words like "urgent", "stat", "emergency", "asap" appear, else "routine"
- region is null if no specific sub-region is mentioned

Examples:
Input: "find a lesion in the liver"
Output: {"organ": "liver", "pathology": "lesion", "task": "detect+segment", "region": null, "urgency": "routine", "parse_confidence": "high"}

Input: "check the right kidney for cysts"
Output: {"organ": "right kidney", "pathology": "cyst", "task": "detect+segment", "region": null, "urgency": "routine", "parse_confidence": "high"}

Input: "urgently identify mass in pancreas head"
Output: {"organ": "pancreas", "pathology": "mass", "task": "detect+segment", "region": "head", "urgency": "urgent", "parse_confidence": "high"}

Input: "segment the spleen"
Output: {"organ": "spleen", "pathology": "unknown", "task": "segment", "region": null, "urgency": "routine", "parse_confidence": "high"}

Input: "look for calcifications along the aorta"
Output: {"organ": "aorta", "pathology": "calcification", "task": "detect+segment", "region": null, "urgency": "routine", "parse_confidence": "high"}

Input: "find tumor in brain"
Output: {"organ": "unknown", "pathology": "tumor", "task": "detect+segment", "region": null, "urgency": "routine", "parse_confidence": "low"}
"""


def expand_abbreviations(text: str) -> str:
    """Expand abdominal CT shorthand to full organ/pathology names."""
    words = text.lower().split()
    return " ".join(ABBREV.get(w, w) for w in words)


def _extract_json(raw: str) -> dict:
    """Extract a JSON object from raw LLM output.

    Handles markdown code fences and preamble text — uses the first '{' and
    last '}' to bound the object.
    """
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                raw = stripped
                break

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in output: {raw!r}")
    return json.loads(raw[start : end + 1])


def _fallback_params() -> TaskParams:
    """Sentinel result when parsing fails — flagged for human review downstream."""
    return TaskParams(
        organ=UNKNOWN,
        pathology=UNKNOWN,
        task="detect+segment",
        region=None,
        urgency="routine",
        parse_confidence="low",
    )


def parse_instruction(
    raw_text: str, *, _llm_output: Optional[str] = None,
) -> TaskParams:
    """Parse a free-text radiologist instruction into a TaskParams.

    On unparseable LLM output, returns a fallback TaskParams with
    parse_confidence='low' rather than raising. The confidence_router
    catches these before dispatch.

    `_llm_output` is a test seam: when set, skips the Ollama call.
    """
    cleaned = expand_abbreviations(raw_text)

    if _llm_output is not None:
        raw_output = _llm_output.strip()
    elif not OLLAMA_AVAILABLE:
        logger.warning(
            "parse_instruction(%r) called without ollama installed - "
            "returning fallback params (router will reject).", raw_text,
        )
        return _fallback_params()
    else:
        # parse_instruction() is documented as never raising. A dead Ollama
        # daemon is an environment failure, not a parse failure, so degrade
        # to fallback params and let the router reject.
        try:
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned},
                ],
            )
            raw_output = response["message"]["content"].strip()
        except Exception as e:
            logger.warning("Ollama call failed for %r: %s", raw_text, e)
            return _fallback_params()

    try:
        data = _extract_json(raw_output)
        return TaskParams(**data)
    except (json.JSONDecodeError, ValueError, ValidationError, TypeError) as e:
        logger.warning(
            "Failed to parse LLM output for input %r: %s. Raw output: %r",
            raw_text, e, raw_output,
        )
        return _fallback_params()


def confidence_router(params: TaskParams) -> str:
    """Decide what to do with a parsed instruction.

    Returns one of:
      - "reject": organ/pathology missing/unknown, OR organ not in BTCV-13
      - "flag_for_review": parse_confidence is "low" but core fields are present
      - "proceed": safe to hand off to the model selector
    """
    organ = (params.organ or "").strip().lower()
    pathology = (params.pathology or "").strip().lower()

    if organ in ("", UNKNOWN) or pathology in ("", UNKNOWN):
        return "reject"
    if organ not in BTCV_ORGANS:
        # Hard reject anything outside the closed vocabulary, even if the
        # validator hasn't downgraded confidence yet.
        return "reject"
    if (params.parse_confidence or "").lower() == "low":
        return "flag_for_review"
    return "proceed"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    test_inputs = [
        "find a lesion in the liver",
        "check rk for cysts urgently",
        "segment the spleen",
        "look for calcifications along the aorta",
        "identify mass in panc",
        "check ivc for thrombus",
        "find tumor in brain",  # outside BTCV → should be rejected
    ]

    for text in test_inputs:
        result = parse_instruction(text)
        decision = confidence_router(result)
        print(f"\nInput:    {text}")
        print(f"Parsed:   {result.model_dump()}")
        print(f"Decision: {decision}")
