# MedAI Handoff README

This folder contains the abdominal CT / BTCV pipeline for the MedAI project.

## What this project does

The code in `code/` wires together a multi-stage medical imaging pipeline:

1. Parse a free-text radiology instruction with the LLM parser.
2. Preprocess the CT volume.
3. Segment the target organ.
4. Detect the pathology inside that organ.
5. Refine the mask and post-process the result.

The parser is the main entry point for text instructions. The current canonical logic lives in `code/parser.py` and is used by `code/orchestrator.py`.

## Key files

- `code/parser.py` - LLM parser, BTCV organ vocabulary, and confidence routing.
- `code/orchestrator.py` - Main pipeline dispatcher.
- `code/detector.py` - Pathology detector stub / integration point.
- `code/organ_segmentor.py` - Organ segmentation stub / integration point.
- `code/segmentor.py` - MedSAM segmentation stub / integration point.
- `code/postprocessor.py` - Final measurement and cleanup stage.
- `code/test_parser.py` - Parser tests.
- `code/test_orchestrator.py` - Pipeline tests.
- `code/CLAUDE.md` - Architecture notes and current project contract.



If integrating new model code, the safest order is:

1. Read `code/CLAUDE.md` first.
2. Start with `code/parser.py` and make sure the text instruction still maps to a valid `TaskParams` object.
3. Update `code/orchestrator.py` only if the downstream flow needs a new field or a different decision rule.
4. Keep `code/test_parser.py` and `code/test_orchestrator.py` passing.
5. Do not change the BTCV organ list unless the whole pipeline is being expanded on purpose.

## How to run tests

From inside `medai/code`:

```bash
python3 test_parser.py
python3 test_orchestrator.py
```

If Ollama is available and you want to run the live parser path:

```bash
python3 test_parser.py --live
```

Model_Links:

best_medsam_btcv.pth
https://drive.google.com/file/d/1iDuhJQJKtxfohO0mfEJ5TlXXTVbR5tVL/view?usp=sharing 

medsam_vit_b.pth
https://drive.google.com/file/d/1Gb_dIsg4I9o1f12tHKJ4jKutsOa5Zk5B/view?usp=sharing 



## Current status

- Parser stage: working
- Other stages: mostly stubs or integration points
- Best next edit target: `code/parser.py` or `code/orchestrator.py`, depending on the student task
