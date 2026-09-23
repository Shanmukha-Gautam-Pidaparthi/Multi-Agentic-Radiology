/* Interactive box-prompt UI.
 *
 * Canvas coordinates are kept in IMAGE pixel space, not CSS pixel space, so
 * the box the backend receives matches what the user drew regardless of how
 * the canvas is scaled by the layout.
 */
"use strict";

const el = (id) => document.getElementById(id);
const canvas = el("canvas");
const ctx = canvas.getContext("2d");

const state = {
  cases: [],
  caseId: null,
  depth: 1,
  width: 512,
  height: 512,
  sliceIndex: 0,
  bbox: null,          // [x1, y1, x2, y2] in image pixels
  drag: null,          // { x, y } drag origin, image pixels
  sliceImg: null,      // HTMLImageElement of the current slice
  overlayImg: null,    // HTMLImageElement of the mask overlay
  busy: false,
};

// ---------------------------------------------------------------- helpers
function setStatus(msg, kind) {
  const node = el("status");
  node.textContent = msg || "";
  node.className = "status" + (kind ? " " + kind : "");
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image failed to load"));
    img.src = src;
  });
}

// Map a pointer event to image-pixel coordinates.
function toImageCoords(evt) {
  const r = canvas.getBoundingClientRect();
  const x = ((evt.clientX - r.left) / r.width) * canvas.width;
  const y = ((evt.clientY - r.top) / r.height) * canvas.height;
  return {
    x: Math.max(0, Math.min(x, canvas.width - 1)),
    y: Math.max(0, Math.min(y, canvas.height - 1)),
  };
}

// ---------------------------------------------------------------- drawing
function redraw(previewBox) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (state.sliceImg) ctx.drawImage(state.sliceImg, 0, 0, canvas.width, canvas.height);
  if (state.overlayImg) ctx.drawImage(state.overlayImg, 0, 0, canvas.width, canvas.height);

  const box = previewBox || state.bbox;
  if (box) {
    const [x1, y1, x2, y2] = box;
    ctx.save();
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = Math.max(1.5, canvas.width / 320);
    ctx.setLineDash([7, 5]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "rgba(0,255,136,0.10)";
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
  }
}

function setBboxText() {
  el("bboxText").textContent = state.bbox
    ? `box  x:[${state.bbox[0].toFixed(0)} → ${state.bbox[2].toFixed(0)}]  ` +
      `y:[${state.bbox[1].toFixed(0)} → ${state.bbox[3].toFixed(0)}]  ` +
      `(${(state.bbox[2] - state.bbox[0]).toFixed(0)} × ${(state.bbox[3] - state.bbox[1]).toFixed(0)} px)`
    : "no box drawn";
  updateModeBadge();
}

function updateModeBadge() {
  const hasText = el("instruction").value.trim().length > 0;
  const hasBox = state.bbox !== null;
  let mode = "";
  if (hasBox && hasText) mode = "combined";
  else if (hasBox) mode = "click_only";
  else if (hasText) mode = "text_only";
  el("modeBadge").textContent = mode ? "mode: " + mode : "";
}

// ---------------------------------------------------------------- slices
async function showSlice(index) {
  if (!state.caseId) return;
  state.sliceIndex = index;
  el("sliceLabel").textContent = `${index} / ${state.depth - 1}`;
  el("sliceSlider").value = String(index);
  try {
    state.sliceImg = await loadImage(`/api/slices/${state.caseId}/${index}`);
  } catch (e) {
    setStatus("Could not load slice: " + e.message, "err");
    return;
  }
  state.overlayImg = null;   // a new slice invalidates the previous mask
  redraw();
}

function selectCase(c) {
  state.caseId = c.case_id;
  state.depth = c.depth;
  state.width = c.width;
  state.height = c.height;
  state.bbox = null;
  state.overlayImg = null;
  // Canvas is sized in IMAGE pixels so box coordinates need no rescaling.
  canvas.width = c.width;
  canvas.height = c.height;
  const slider = el("sliceSlider");
  slider.max = String(Math.max(0, c.depth - 1));
  slider.disabled = c.depth <= 1;
  setBboxText();
  showSlice(Math.floor(c.depth / 2));
}

async function refreshCases(selectId) {
  const data = await api("/api/slices");
  state.cases = data.cases;
  const sel = el("caseSelect");
  sel.innerHTML = "";
  for (const c of state.cases) {
    const opt = document.createElement("option");
    opt.value = c.case_id;
    opt.textContent = `${c.name}  [${c.width}×${c.height}×${c.depth}]`;
    sel.appendChild(opt);
  }
  const target = state.cases.find((c) => c.case_id === selectId) || state.cases[0];
  if (target) {
    sel.value = target.case_id;
    selectCase(target);
  }
}

// ---------------------------------------------------------------- report
function renderReport(payload) {
  const r = payload.report;
  const res = payload.result;
  el("reportBody").textContent = r.markdown;

  const base = `/api/reports/${payload.report_id}`;
  el("dlMd").href = base + "?fmt=md";
  el("dlHtml").href = base + "?fmt=html";
  el("dlJson").href = base + "?fmt=json";
  el("reportActions").hidden = false;

  const stats = [["Mode", res.input_mode], ["Status", res.status]];
  if (res.decision) stats.push(["Router", res.decision]);
  if (res.pathology_mask) {
    stats.push(["Label", res.pathology_mask.label]);
    stats.push(["Seg score", res.pathology_mask.score.toFixed(3)]);
    stats.push(["Mask px", res.pathology_mask.pixel_count.toLocaleString()]);
  }
  const m = res.measurements;
  if (m) {
    stats.push(["Volume", `${m.volume_cm3} cm³`]);
    stats.push(["Diameter", `${m.longest_diameter_mm} mm`]);
    stats.push(["Mean HU", `${m.mean_hu}`]);
    stats.push(["Sphericity", `${m.sphericity}`]);
  }
  el("summary").innerHTML = stats
    .map(([k, v]) => `<div class="stat"><div class="k"></div><div class="v"></div></div>`)
    .join("");
  // Fill via textContent so model/report strings are never parsed as HTML.
  const nodes = el("summary").children;
  stats.forEach(([k, v], i) => {
    nodes[i].querySelector(".k").textContent = k;
    nodes[i].querySelector(".v").textContent = v;
  });
}

async function run() {
  if (state.busy) return;
  const instruction = el("instruction").value.trim();
  const llmOutput = el("llmOutput").value.trim();
  if (!state.bbox && !instruction) {
    setStatus("Draw a box or type an instruction first.", "err");
    return;
  }
  state.busy = true;
  el("runBtn").disabled = true;
  setStatus("Running pipeline…", "busy");
  try {
    const payload = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_id: state.caseId,
        slice_index: state.sliceIndex,
        bbox: state.bbox,
        instruction: instruction || null,
        llm_output: llmOutput || null,
      }),
    });

    const mask = payload.result.pathology_mask;
    state.overlayImg = mask ? await loadImage(mask.overlay_png) : null;
    // A 3D text_only run segments at z_mid, not the slice on screen.
    if (payload.result.z_mid !== null && payload.result.z_mid !== state.sliceIndex) {
      await showSlice(payload.result.z_mid);
      state.overlayImg = mask ? await loadImage(mask.overlay_png) : null;
    }
    redraw();
    renderReport(payload);
    setStatus(payload.result.message || "Done.", "ok");
  } catch (e) {
    setStatus("Failed: " + e.message, "err");
  } finally {
    state.busy = false;
    el("runBtn").disabled = false;
  }
}

// ---------------------------------------------------------------- events
canvas.addEventListener("pointerdown", (e) => {
  if (!state.caseId) return;
  canvas.setPointerCapture(e.pointerId);
  state.drag = toImageCoords(e);
  state.overlayImg = null;
});

canvas.addEventListener("pointermove", (e) => {
  if (!state.drag) return;
  const p = toImageCoords(e);
  redraw([
    Math.min(state.drag.x, p.x), Math.min(state.drag.y, p.y),
    Math.max(state.drag.x, p.x), Math.max(state.drag.y, p.y),
  ]);
});

function endDrag(e) {
  if (!state.drag) return;
  const p = toImageCoords(e);
  const box = [
    Math.min(state.drag.x, p.x), Math.min(state.drag.y, p.y),
    Math.max(state.drag.x, p.x), Math.max(state.drag.y, p.y),
  ];
  state.drag = null;
  // Reject accidental clicks; the backend rejects degenerate boxes anyway.
  if (box[2] - box[0] < 4 || box[3] - box[1] < 4) {
    setStatus("Box too small — drag a larger region.", "err");
    redraw();
    return;
  }
  state.bbox = box;
  setBboxText();
  setStatus("");
  redraw();
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", () => { state.drag = null; redraw(); });

el("sliceSlider").addEventListener("input", (e) => showSlice(Number(e.target.value)));
el("caseSelect").addEventListener("change", (e) => {
  const c = state.cases.find((x) => x.case_id === e.target.value);
  if (c) selectCase(c);
});
el("clearBtn").addEventListener("click", () => {
  state.bbox = null;
  state.overlayImg = null;
  setBboxText();
  redraw();
});
el("instruction").addEventListener("input", updateModeBadge);
el("runBtn").addEventListener("click", run);

el("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  setStatus(`Uploading ${file.name}…`, "busy");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const c = await api("/api/slices", { method: "POST", body: fd });
    await refreshCases(c.case_id);
    setStatus(`Loaded ${c.name}.`, "ok");
  } catch (err) {
    setStatus("Upload failed: " + err.message, "err");
  } finally {
    e.target.value = "";   // allow re-uploading the same filename
  }
});

// ---------------------------------------------------------------- boot
(async function init() {
  try {
    const h = await api("/api/health");
    el("health").textContent =
      `ollama:${h.ollama_available ? "yes" : "no"} · ` +
      `nibabel:${h.nibabel_available ? "yes" : "no"} · ` +
      `stubs:${h.stub_stages.length}`;
    if (h.stub_stages.length) {
      const b = el("stubBanner");
      b.textContent = "⚠ " + h.warning + " Stub stages: " + h.stub_stages.join(", ") + ".";
      b.hidden = false;
    }
    await refreshCases();
    setStatus("Ready — draw a box on the slice.", "");
  } catch (e) {
    setStatus("Backend unreachable: " + e.message, "err");
  }
})();
