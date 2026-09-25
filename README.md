# MedAI Handoff README

This folder contains the abdominal CT / BTCV pipeline for the MedAI project.

## What this project does

The code at the repository root wires together a multi-stage medical imaging pipeline:

1. Parse a free-text radiology instruction with the LLM parser.
2. Preprocess the CT volume.
3. Segment the target organ.
4. Detect the pathology inside that organ.
5. Refine the mask and post-process the result.

The parser is the main entry point for text instructions. The current canonical logic lives in `parser.py` and is used by `orchestrator.py`.

## Key files

- `parser.py` - LLM parser, BTCV organ vocabulary, and confidence routing.
- `orchestrator.py` - Main pipeline dispatcher.
- `detector.py` - Pathology detector stub / integration point.
- `organ_segmentor.py` - Organ segmentation stub / integration point.
- `segmentor.py` - MedSAM segmentation stub / integration point.
- `postprocessor.py` - Final measurement and cleanup stage.
- `test_parser.py` - Parser tests.
- `test_orchestrator.py` - Pipeline tests.
- `CLAUDE.md` - Architecture notes and current project contract.



If integrating new model code, the safest order is:

1. Read `CLAUDE.md` first.
2. Start with `parser.py` and make sure the text instruction still maps to a valid `TaskParams` object.
3. Update `orchestrator.py` only if the downstream flow needs a new field or a different decision rule.
4. Keep `test_parser.py` and `test_orchestrator.py` passing.
5. Do not change the BTCV organ list unless the whole pipeline is being expanded on purpose.

## How to run tests

From the repository root:

```bash
python3 test_parser.py
python3 test_orchestrator.py
```

If Ollama is available and you want to run the live parser path:

```bash
python3 test_parser.py --live
```

## Setup (fresh clone)

Use a virtual environment. On Debian/Ubuntu, installing into the system
Python is blocked by PEP 668 (`error: externally-managed-environment`), and
there is no `python` on PATH — only `python3` — until a venv is activated.

```bash
sudo apt install -y python3-venv     # Debian/Ubuntu, if the next line fails
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -V                            # sanity check: prints 3.x, venv is active

# CPU-only machine? Install torch from the CPU index FIRST, or pip pulls
# ~3 GB of NVIDIA CUDA wheels. Check with: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r Final_Pipeline/requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

With a working NVIDIA driver, skip the CPU index line — the default wheels
bring CUDA and inference is far faster. Without a GPU the app still runs;
`get_device()` falls back to CPU, but a 3D sliding-window pass takes minutes
per case.

Every `python` below assumes this venv is active. Without it, use `python3`.

### Model weights

Two checkpoints ship in the repo; two exceed GitHub's 100 MB file limit and
must be downloaded into `Final_Pipeline/models/` by hand.

| File | Size | Source |
|---|---|---|
| `best_multiorgan.pth` | 19 MB | in repo |
| `best_segresnet_model.pth` | 6.1 MB | in repo |
| `best_medsam_btcv.pth` | 388 MB | [Drive](https://drive.google.com/file/d/1iDuhJQJKtxfohO0mfEJ5TlXXTVbR5tVL/view?usp=sharing) |
| `medsam_vit_b.pth` | 358 MB | [Drive](https://drive.google.com/file/d/1Gb_dIsg4I9o1f12tHKJ4jKutsOa5Zk5B/view?usp=sharing) |

Only `best_medsam_btcv.pth` is required — `medsam_vit_b.pth` is referenced
nowhere in the code and can be skipped.

```bash
cd Final_Pipeline/models
gdown 1iDuhJQJKtxfohO0mfEJ5TlXXTVbR5tVL -O best_medsam_btcv.pth

ls -lh best_medsam_btcv.pth    # expect ~388M
file best_medsam_btcv.pth      # expect "Zip archive data", NOT "HTML document"
```

That `file` check matters: a blocked download silently writes a small HTML
page under the right filename, which only surfaces later as a confusing error
inside `torch.load`.

Both Drive files must be shared as **Anyone with the link**, or collaborators
get a 403.

### Run the web UI

```bash
cd Final_Pipeline/webapp && python app.py     # http://localhost:5000
```

`GET /health` reports which models loaded and which are missing. The app starts
and serves the UI even when a checkpoint is absent — the affected stage is
skipped and reported rather than crashing the server.

### Known gaps on a fresh clone

- **`config.py` is not in this repo.** Eight scripts at the repo root
  (`step0`–`step5`, `interactive_multiorgan.py`) do `from config import ...`
  for ~40 constants. It has never been committed on any branch. Everything
  under `Final_Pipeline/` is unaffected.
- **No TotalSegmentator cache.** Organ classification returns nothing until
  `totalseg_cache/` exists — generate it with
  `Unified_Model/step1_precompute_totalseg.py`, or install `TotalSegmentator`.
  Searched paths are listed by `GET /health`; override with
  `MEDAI_TOTALSEG_DIR`.
- **No BTCV dataset.** The `btcv/` tree the `step*` scripts expect is not
  included.



## Current status

- Parser stage: working
- Other stages: mostly stubs or integration points
- Best next edit target: `parser.py` or `orchestrator.py`, depending on the student task
