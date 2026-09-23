# Interactive Box-Prompt Web UI

A browser front-end for the abdominal CT / BTCV pipeline. Draw a bounding box
on a CT slice, optionally add a free-text instruction, and get a structured
report back.

```
  frontend/                 vanilla HTML + canvas (no build step)
        |  REST over HTTP
        v
  backend/app.py            FastAPI (uvicorn, port 8000)
        `-- orchestrator.py  -> parser / organ_segmentor / detector /
                                segmentor / postprocessor  (repo root)
        `-- report_generator.py  -> structured report + Markdown
```

## Run it

```bash
cd Multi-Agentic-Radiology
pip install -r web/requirements.txt
python -m uvicorn app:app --reload --port 8000 --app-dir web/backend
```

Open <http://localhost:8000>. API docs are at `/docs`.

Optional extras:

```bash
pip install nibabel     # upload .nii / .nii.gz volumes instead of PNG slices
pip install ollama      # live text parsing; also needs a running Ollama daemon
                        # with: ollama pull llama3.1:8b
```

Without `ollama`, text instructions fall back to low-confidence params and the
router rejects them. Use the **parser test seam** in the UI to inject parser
JSON directly and exercise the text paths anyway.

## Using it

1. **Pick a case.** A synthetic demo phantom is preloaded so the UI works on a
   fresh clone with no BTCV data. Upload your own PNG slice or NIfTI volume
   with the file picker.
2. **Draw a box** by dragging on the slice. Scrub slices with the slider.
3. **Optionally type an instruction** (e.g. `detect and segment a lesion in the liver`).
4. **Run.** The box + instruction combination picks the orchestrator mode:

   | Box | Instruction | Mode          | Pipeline                                                    |
   |-----|-------------|---------------|-------------------------------------------------------------|
   | yes | no          | `click_only`  | MedSAM -> post-process                                       |
   | yes | yes         | `combined`    | parser (context only) -> MedSAM -> post-process              |
   | no  | yes         | `text_only`   | parser -> router -> organ-seg -> detector -> MedSAM -> post  |

   In `text_only`, a case with depth > 3 takes the full 3D two-stage path and
   the viewer jumps to the segmented `z_mid` slice; a single slice takes the
   2D fallback.
5. **Read the report** and download it as Markdown, a printable HTML page, or JSON.

## API

| Method | Path                          | Purpose                               |
|--------|-------------------------------|---------------------------------------|
| GET    | `/api/health`                 | component + stub status               |
| GET    | `/api/organs`                 | the BTCV-13 vocabulary                |
| GET    | `/api/slices`                 | list loaded cases                     |
| POST   | `/api/slices`                 | upload a PNG slice or NIfTI volume    |
| GET    | `/api/slices/{id}/{z}`        | one axial slice as PNG                |
| POST   | `/api/analyze`                | box prompt / instruction -> result    |
| GET    | `/api/reports/{id}?fmt=`      | `json` \| `md` \| `html`              |

```bash
# click_only from the command line
CASE=$(curl -s localhost:8000/api/health | python -c "import sys,json;print(json.load(sys.stdin)['demo_case_id'])")
curl -s -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d "{\"case_id\":\"$CASE\",\"slice_index\":32,\"bbox\":[170,200,235,262]}" \
  | python -c "import sys,json;print(json.load(sys.stdin)['report']['markdown'])"
```

## Limitations

- **Every AI stage is still a stub.** Volume is computed honestly from voxel
  count x spacing; every other measurement is a deterministic placeholder. The
  report labels itself NOT FOR CLINICAL USE and lists the stub stages under
  `provenance`. Remove names from `STUB_STAGES` in `backend/app.py` as real
  weights land.
- **Nothing is persisted.** Uploaded cases and reports live in a bounded
  in-process LRU cache (`MEDAI_MAX_CASES`, `MEDAI_MAX_REPORTS`) and are gone on
  restart. That is deliberate for a prototype handling patient imaging.
- **No auth, single user.** Bind to localhost only; do not expose this.
- Uploads are capped at `MEDAI_MAX_UPLOAD_MB` (default 64).
